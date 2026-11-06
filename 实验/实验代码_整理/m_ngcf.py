import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import INTER
from guards import clip_and_check_grad
from trainer import run_method


class NGCF(nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers, edge_index, device,
                 msg_dropout=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.msg_dropout = msg_dropout
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.xavier_uniform_(self.emb.weight)

        self.W1 = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.W2 = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        for lin in list(self.W1) + list(self.W2):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        u = edge_index[0]
        i = edge_index[1] + n_users
        src = torch.cat([i, u])
        dst = torch.cat([u, i])
        n_nodes = n_users + n_items
        deg = torch.zeros(n_nodes, device=device)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], device=device))
        deg = deg.clamp(min=1.0)
        weight = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("weight", weight.unsqueeze(-1))
        self.n_nodes = n_nodes

    def propagate(self):
        x = self.emb.weight
        outs = [F.normalize(x, dim=-1)]
        for layer in range(self.n_layers):
            side = torch.zeros_like(x)
            side.index_add_(0, self.dst, x[self.src] * self.weight)
            bi = torch.zeros_like(x)
            bi.index_add_(0, self.dst, (x[self.src] * x[self.dst]) * self.weight)
            x = F.leaky_relu(self.W1[layer](x + side) + self.W2[layer](bi), 0.2)
            if self.training and self.msg_dropout > 0:
                x = F.dropout(x, p=self.msg_dropout)
            outs.append(F.normalize(x, dim=-1))
        out = torch.cat(outs, dim=-1)
        return out[:self.n_users], out[self.n_users:]

    def score_all(self, users):
        eu, ei = self.propagate()
        return eu[users] @ ei.T

    def bpr(self, users, pos, neg):
        eu, ei = self.propagate()
        u = eu[users]
        return (u * ei[pos]).sum(-1) - (u * ei[neg]).sum(-1)


def make_builder(dim, layers, lr, reg, dropout):
    def build(bundle, device, seed):
        model = NGCF(bundle["n_users"], bundle["n_concepts"], dim, layers,
                     bundle["train_ei"].to(device), device,
                     msg_dropout=dropout).to(device)
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
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    hyper = {"dim": args.dim, "layers": args.layers, "lr": args.lr,
             "reg": args.reg, "msg_dropout": args.dropout, "batch": args.batch,
             "epochs": args.epochs, "patience": args.patience}
    run_method("ngcf", "NGCF", INTER / "s6_graph.pt",
               make_builder(args.dim, args.layers, args.lr, args.reg, args.dropout),
               make_step(args.batch), seeds=args.seeds, epochs=args.epochs,
               patience=args.patience, verbose=args.verbose, hyper=hyper)


if __name__ == "__main__":
    main()
