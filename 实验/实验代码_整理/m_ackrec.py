import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import INTER
from guards import clip_and_check_grad
from trainer import run_method


def sym_norm(a):
    a = a + a.t()
    a.fill_diagonal_(0.0)
    d = a.sum(1).clamp(min=1e-8).pow(-0.5)
    return d.unsqueeze(1) * a * d.unsqueeze(0)


def sparsify(a, topk):
    if topk <= 0 or topk >= a.shape[1]:
        return a
    val, idx = torch.topk(a, topk, dim=1)
    out = torch.zeros_like(a)
    out.scatter_(1, idx, val)
    return out


def build_metapaths(bundle, device, topk):
    nu = bundle["n_users"]
    nc = bundle["n_concepts"]
    nt = bundle["n_taxonomy"]
    ei = bundle["train_ei"].to(device)

    uc = torch.zeros((nu, nc), device=device)
    uc[ei[0], ei[1]] = 1.0
    cuc = sparsify(uc.t() @ uc, topk)

    cont = bundle["containment"].to(device)
    tc = torch.zeros((nt, nc), device=device)
    tc[cont[0], cont[1]] = 1.0
    ctc = tc.t() @ tc

    pre = torch.zeros((nc, nc), device=device)
    for key in ("prereq_adopted", "prereq_predicted"):
        e = bundle[key].to(device)
        pre[e[0], e[1]] += 1.0

    ch = torch.zeros((nc, nc), device=device)
    e = bundle["chapter_weak"].to(device)
    ch[e[0], e[1]] += 1.0

    mats = [sym_norm(cuc), sym_norm(ctc), sym_norm(pre), sym_norm(ch)]

    deg = torch.zeros(nu, device=device).index_add_(
        0, ei[0], torch.ones(ei.shape[1], device=device))
    w = deg.clamp(min=1.0).pow(-0.5)[ei[0]].unsqueeze(1)
    return mats, (ei[0], ei[1], w)


class ACKRec(nn.Module):
    def __init__(self, n_users, n_concepts, dim, layers, dropout, mats, bip):
        super().__init__()
        self.eu = nn.Embedding(n_users, dim)
        self.ec = nn.Embedding(n_concepts, dim)
        nn.init.normal_(self.eu.weight, std=0.1)
        nn.init.normal_(self.ec.weight, std=0.1)
        self.mats = mats
        self.bip = bip
        self.n_layers = layers
        self.dropout = dropout
        self.w = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(layers)])
        self.att = nn.Linear(dim, dim)
        self.q = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.q, std=0.1)

    def views(self):
        out = []
        for a in self.mats:
            h = self.ec.weight
            acc = [h]
            for l in range(self.n_layers):
                h = a @ h
                h = F.relu(self.w[l](h))
                if self.training and self.dropout > 0:
                    h = F.dropout(h, self.dropout)
                acc.append(h)
            out.append(torch.stack(acc, 0).mean(0))
        return torch.stack(out, 0)

    def propagate(self):
        v = self.views()
        s = torch.tanh(self.att(v)).matmul(self.q).mean(-1)
        alpha = torch.softmax(s, 0)
        ec = (alpha.view(-1, 1, 1) * v).sum(0)
        src, dst, w = self.bip
        agg = torch.zeros_like(self.eu.weight)
        agg.index_add_(0, src, ec[dst] * w)
        eu = self.eu.weight + agg
        return eu, ec

    def forward(self, users, pos, neg):
        eu, ec = self.propagate()
        u = eu[users]
        return (u * ec[pos]).sum(-1) - (u * ec[neg]).sum(-1)

    @torch.no_grad()
    def score_all(self, users):
        eu, ec = self.propagate()
        return eu[users] @ ec.t()


def make_builder(dim, layers, lr, reg, dropout, topk):
    def build(bundle, device, seed):
        torch.manual_seed(seed)
        mats, bip = build_metapaths(bundle, device, topk)
        model = ACKRec(bundle["n_users"], bundle["n_concepts"], dim, layers,
                       dropout, mats, bip).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        return model, opt
    return build


def make_step(batch):
    state = {}

    def step(model, optimizer, bundle, device, epoch):
        model.train()
        if "ei" not in state:
            ei = bundle["train_ei"].to(device)
            seen = torch.zeros((bundle["n_users"], bundle["n_concepts"]),
                               dtype=torch.bool, device=device)
            seen[ei[0], ei[1]] = True
            state["ei"] = ei
            state["seen"] = seen
        ei, seen = state["ei"], state["seen"]
        nc = bundle["n_concepts"]
        n = ei.shape[1]
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            u = ei[0, idx]
            pos = ei[1, idx]
            neg = torch.randint(0, nc, (idx.shape[0],), device=device)
            bad = seen[u, neg]
            for _ in range(4):
                if not bad.any():
                    break
                neg[bad] = torch.randint(0, nc, (int(bad.sum()),), device=device)
                bad = seen[u, neg]
            loss = F.softplus(-model(u, pos, neg)).mean()
            optimizer.zero_grad()
            loss.backward()
            clip_and_check_grad(model, epoch)
            optimizer.step()
            total += float(loss.detach()) * idx.shape[0]
        return total / n
    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--reg", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--batch", type=int, default=32768)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    hyper = {"dim": args.dim, "layers": args.layers, "lr": args.lr,
             "reg": args.reg, "dropout": args.dropout, "topk": args.topk,
             "batch": args.batch, "epochs": args.epochs,
             "patience": args.patience,
             "metapaths": "C-U-C, C-T-C, prereq, chapter_weak",
             "fusion": "view-level attention"}
    run_method("ackrec", "ACKRec", INTER / "s6_graph.pt",
               make_builder(args.dim, args.layers, args.lr, args.reg,
                            args.dropout, args.topk),
               make_step(args.batch), seeds=args.seeds, epochs=args.epochs,
               patience=args.patience, verbose=args.verbose, hyper=hyper)


if __name__ == "__main__":
    main()
