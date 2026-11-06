import argparse

import torch
import torch.nn as nn

from common import INTER
from trainer import run_method


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim, n_layers, edge_index, device):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

        u = edge_index[0]
        i = edge_index[1] + n_users
        src = torch.cat([u, i])
        dst = torch.cat([i, u])
        deg = torch.zeros(n_users + n_items, device=device)
        deg.index_add_(0, src, torch.ones(src.shape[0], device=device))
        deg = deg.clamp(min=1.0)
        weight = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("weight", weight)
        self.n_nodes = n_users + n_items
        if str(device) == "cpu" and n_users * n_items <= 20_000_000:
            R = torch.zeros((n_users, n_items), dtype=torch.float32, device=device)
            R[edge_index[0], edge_index[1]] = weight[:edge_index.shape[1]]
            self.register_buffer("_R", R)
            self.register_buffer("_RT", R.t().contiguous())
        else:
            self._R = None
            self._RT = None

    def propagate(self):
        x = self.emb.weight
        if self._R is not None:
            xu, xi = x[:self.n_users], x[self.n_users:]
            au, ai = xu, xi
            for _ in range(self.n_layers):
                xu, xi = self._R @ xi, self._RT @ xu
                au = au + xu
                ai = ai + xi
            scale = float(self.n_layers + 1)
            return au / scale, ai / scale
        acc = x
        for _ in range(self.n_layers):
            msg = x[self.src] * self.weight.unsqueeze(-1)
            agg = torch.zeros_like(x)
            agg.index_add_(0, self.dst, msg)
            x = agg
            acc = acc + x
        acc = acc / (self.n_layers + 1)
        return acc[:self.n_users], acc[self.n_users:]

    def score_all(self, users):
        eu, ei = self.propagate()
        return eu[users] @ ei.T

    def bpr(self, users, pos, neg):
        eu, ei = self.propagate()
        u = eu[users]
        return (u * ei[pos]).sum(-1) - (u * ei[neg]).sum(-1)


def make_builder(dim, layers, lr, reg):
    def build(bundle, device, seed):
        model = LightGCN(bundle["n_users"], bundle["n_concepts"], dim, layers,
                         bundle["train_ei"].to(device), device).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        return model, opt
    return build


def make_step(batch):
    state = {}

    def step(model, optimizer, bundle, device, epoch):
        if "ei" not in state:
            ei = bundle["train_ei"].to(device)
            state["ei"] = ei
            seen = torch.zeros((bundle["n_users"], bundle["n_concepts"]),
                               dtype=torch.bool, device=device)
            seen[ei[0], ei[1]] = True
            state["seen"] = seen
        ei = state["ei"]
        seen = state["seen"]
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
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    hyper = {"dim": args.dim, "layers": args.layers, "lr": args.lr,
             "reg": args.reg, "batch": args.batch, "epochs": args.epochs,
             "patience": args.patience}
    run_method("lightgcn", "LightGCN", INTER / "s6_graph.pt",
               make_builder(args.dim, args.layers, args.lr, args.reg),
               make_step(args.batch), seeds=args.seeds, epochs=args.epochs,
               patience=args.patience, verbose=args.verbose, hyper=hyper)


if __name__ == "__main__":
    main()
