import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import INTER
from guards import clip_and_check_grad
from trainer import run_method


def segment_max(values, index, n_segments):
    device = values.device
    flat = values.detach().to("cpu")
    idx = index.to("cpu")
    out = torch.full((n_segments,), float("-inf"))
    out.scatter_reduce_(0, idx, flat, reduce="amax", include_self=False)
    out = torch.nan_to_num(out, neginf=0.0, posinf=0.0)
    return out.to(device)


def build_relations(bundle, device):
    n_u = bundle["n_users"]
    n_c = bundle["n_concepts"]
    offset_c = n_u
    offset_t = n_u + n_c

    rels = []

    ui = bundle["train_ei"].to(device)
    rels.append((ui[0], ui[1] + offset_c))
    rels.append((ui[1] + offset_c, ui[0]))

    pre = torch.cat([bundle["prereq_adopted"], bundle["prereq_predicted"]], dim=1).to(device)
    if pre.shape[1]:
        rels.append((pre[0] + offset_c, pre[1] + offset_c))
        rels.append((pre[1] + offset_c, pre[0] + offset_c))

    con = bundle["containment"].to(device)
    if con.shape[1]:
        rels.append((con[0] + offset_t, con[1] + offset_c))
        rels.append((con[1] + offset_c, con[0] + offset_t))

    src = torch.cat([s for s, _ in rels])
    dst = torch.cat([d for _, d in rels])
    rid = torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(rels)])
    return src, dst, rid, len(rels), offset_c, offset_t


class KGAT(nn.Module):
    def __init__(self, bundle, dim, n_layers, device, dropout=0.1):
        super().__init__()
        self.n_users = bundle["n_users"]
        self.n_concepts = bundle["n_concepts"]
        n_tax = bundle["n_taxonomy"]
        self.n_nodes = self.n_users + self.n_concepts + n_tax
        self.n_layers = n_layers
        self.dropout = dropout

        src, dst, rid, n_rel, off_c, off_t = build_relations(bundle, device)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("rid", rid)
        self.n_rel = n_rel
        self.off_c = off_c

        self.ent = nn.Embedding(self.n_nodes, dim)
        self.rel = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        self.Wr = nn.Parameter(torch.empty(n_rel, dim, dim))
        nn.init.xavier_uniform_(self.Wr)

        self.W1 = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.W2 = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        for lin in list(self.W1) + list(self.W2):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.register_buffer("att", torch.ones(src.shape[0]))

    @torch.no_grad()
    def refresh_attention(self):
        x = self.ent.weight
        scores = torch.empty(self.src.shape[0], device=x.device)
        for r in range(self.n_rel):
            mask = self.rid == r
            if not mask.any():
                continue
            proj = self.Wr[r]
            eh = x[self.src[mask]] @ proj
            et = x[self.dst[mask]] @ proj
            scores[mask] = (et * torch.tanh(eh + self.rel.weight[r])).sum(-1)

        peak = segment_max(scores, self.dst, self.n_nodes)
        exp = torch.exp((scores - peak[self.dst]).clamp(min=-30.0, max=30.0))
        denom = torch.zeros(self.n_nodes, device=x.device)
        denom.index_add_(0, self.dst, exp)
        self.att = exp / denom[self.dst].clamp(min=1e-10)

    def propagate(self):
        x = self.ent.weight
        outs = [F.normalize(x, dim=-1)]
        weight = self.att.unsqueeze(-1)
        for layer in range(self.n_layers):
            agg = torch.zeros_like(x)
            agg.index_add_(0, self.dst, x[self.src] * weight)
            total = F.leaky_relu(self.W1[layer](x + agg), 0.2)
            inter = F.leaky_relu(self.W2[layer](x * agg), 0.2)
            x = total + inter
            if self.training and self.dropout > 0:
                x = F.dropout(x, p=self.dropout)
            outs.append(F.normalize(x, dim=-1))
        out = torch.cat(outs, dim=-1)
        return out[:self.n_users], out[self.off_c:self.off_c + self.n_concepts]

    def score_all(self, users):
        eu, ec = self.propagate()
        return eu[users] @ ec.T

    def bpr(self, users, pos, neg):
        eu, ec = self.propagate()
        u = eu[users]
        return (u * ec[pos]).sum(-1) - (u * ec[neg]).sum(-1)


def make_builder(dim, layers, lr, reg, dropout):
    def build(bundle, device, seed):
        model = KGAT(bundle, dim, layers, device, dropout=dropout).to(device)
        model.refresh_attention()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        return model, opt
    return build


def make_step(batch):
    state = {}

    def step(model, optimizer, bundle, device, epoch):
        if "ei" not in state:
            ei = bundle["train_ei"].to(device)
            seen = torch.zeros((bundle["n_users"], bundle["n_concepts"]),
                               dtype=torch.bool, device=device)
            seen[ei[0], ei[1]] = True
            state["ei"] = ei
            state["seen"] = seen
        model.refresh_attention()
        ei, seen = state["ei"], state["seen"]
        n = ei.shape[1]
        n_items = bundle["n_concepts"]
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            u = ei[0][idx]
            p = ei[1][idx]
            neg = torch.randint(0, n_items, (idx.shape[0],), device=device)
            bad = seen[u, neg]
            for _ in range(4):
                if not bad.any():
                    break
                neg[bad] = torch.randint(0, n_items, (int(bad.sum()),), device=device)
                bad = seen[u, neg]
            diff = model.bpr(u, p, neg)
            loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
            optimizer.zero_grad()
            loss.backward()
            clip_and_check_grad(model, epoch)
            optimizer.step()
            total += float(loss.detach()) * idx.shape[0]
        return total / n
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    hyper = {"dim": args.dim, "layers": args.layers, "lr": args.lr,
             "reg": args.reg, "dropout": args.dropout, "batch": args.batch,
             "epochs": args.epochs, "patience": args.patience,
             "kg_edges": "prereq_adopted+predicted, containment"}
    run_method("kgat", "KGAT", INTER / "s6_graph.pt",
               make_builder(args.dim, args.layers, args.lr, args.reg, args.dropout),
               make_step(args.batch), seeds=args.seeds, epochs=args.epochs,
               patience=args.patience, verbose=args.verbose, hyper=hyper)


if __name__ == "__main__":
    main()
