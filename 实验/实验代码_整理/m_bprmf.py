import argparse
import json
import random
import statistics
import time

import torch
import torch.nn as nn

from common import INTER, OUT, load_protocol, provenance, report
from evaluator import evaluate, format_metrics


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.u = nn.Embedding(n_users, dim)
        self.i = nn.Embedding(n_items, dim)
        nn.init.normal_(self.u.weight, std=0.1)
        nn.init.normal_(self.i.weight, std=0.1)

    def score_all(self, users):
        return self.u(users) @ self.i.weight.T

    def forward(self, users, pos, neg):
        eu = self.u(users)
        ep = self.i(pos)
        en = self.i(neg)
        return (eu * ep).sum(-1) - (eu * en).sum(-1)


def train_one(bundle, seed, device, dim=64, lr=0.01, reg=1e-5,
              epochs=60, batch=8192, patience=8, verbose=False):
    torch.manual_seed(seed)
    random.seed(seed)

    n_users = bundle["n_users"]
    n_items = bundle["n_concepts"]
    train_ei = bundle["train_ei"].to(device)
    users_all = train_ei[0]
    items_all = train_ei[1]
    n_inter = users_all.shape[0]

    seen = torch.zeros((n_users, n_items), dtype=torch.bool, device=device)
    seen[users_all, items_all] = True

    model = BPRMF(n_users, n_items, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    best = {"NDCG@10": -1.0}
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_inter, device=device)
        total = 0.0
        for start in range(0, n_inter, batch):
            idx = perm[start:start + batch]
            u = users_all[idx]
            p = items_all[idx]
            neg = torch.randint(0, n_items, (idx.shape[0],), device=device)
            bad = seen[u, neg]
            for _ in range(4):
                if not bad.any():
                    break
                resample = torch.randint(0, n_items, (int(bad.sum()),), device=device)
                neg[bad] = resample
                bad = seen[u, neg]
            diff = model(u, p, neg)
            loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.shape[0]

        model.eval()
        with torch.no_grad():
            metrics, _ = evaluate(lambda uu: model.score_all(uu), bundle,
                                  split="valid", device=device)
        if verbose:
            print(f"   epoch {epoch:>3}  loss={total/n_inter:.4f}  valid {format_metrics(metrics)}")
        if metrics["NDCG@10"] > best["NDCG@10"]:
            best = metrics
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_metrics, counted = evaluate(lambda uu: model.score_all(uu), bundle,
                                         split="test", device=device)
    return test_metrics, best, counted, epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    proto = load_protocol()
    seeds = args.seeds or proto["training"]["seeds"]
    device = pick_device()
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)

    print(f"── BPR-MF 基线  device={device}  dim={args.dim}  lr={args.lr}  seeds={seeds}")
    runs = []
    t0 = time.time()
    for seed in seeds:
        m, vbest, counted, ep = train_one(bundle, seed, device, dim=args.dim,
                                          lr=args.lr, epochs=args.epochs,
                                          verbose=args.verbose)
        runs.append(m)
        print(f"   seed {seed}  epochs={ep}  test {format_metrics(m)}")

    keys = sorted(runs[0])
    summary = {}
    for k in keys:
        vals = [r[k] for r in runs]
        summary[k] = {"mean": statistics.mean(vals),
                      "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}

    report("BPR-MF 测试集结果", {
        k: f"{summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}"
        for k in ("NDCG@10", "HR@10", "MRR", "Recall@20", "Precision@10")
    })
    print(f"   耗时 {time.time()-t0:.1f}s  评测学习者 {counted:,}")

    payload = {"method": "BPR-MF", "device": device, "seeds": seeds,
               "hyper": {"dim": args.dim, "lr": args.lr, "epochs": args.epochs},
               "per_seed": runs, "summary": summary, "eval_users": counted}
    with open(OUT / "r_bprmf.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    provenance("m_bprmf", {k: summary[k]["mean"] for k in ("NDCG@10", "HR@10", "MRR")},
               [OUT / "r_bprmf.json"])


if __name__ == "__main__":
    main()
