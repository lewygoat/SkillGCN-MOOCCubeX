import common  # Auto-unpacks s6_graph.pt.gz on first run
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from evaluator import evaluate

INTER = Path("../中间产物")
EMB = INTER / "嵌入"
RESULTS = Path("../结果")

BUCKETS = [
    ("低频[10,20)", 10, 20),
    ("中频[20,50)", 20, 50),
    ("高频[50+)", 50, 10**9),
]


def bucket_users(bundle):
    train_ei = bundle["train_ei"].numpy()
    test_ei = bundle["test_ei"].numpy()
    nu = bundle["n_users"]
    cnt = np.bincount(train_ei[0], minlength=nu)
    test_users = set(np.unique(test_ei[0]).tolist())
    out = {}
    for name, lo, hi in BUCKETS:
        mask = (cnt >= lo) & (cnt < hi)
        out[name] = sorted(u for u in np.nonzero(mask)[0].tolist()
                          if u in test_users)
    return out


def eval_bucket(emb, bundle, users, device):
    n_users, n_concepts = bundle["n_users"], bundle["n_concepts"]
    eu = torch.zeros((n_users, emb.shape[1]), dtype=torch.float32, device=device)
    ei = emb.to(device)
    train_seen = set(bundle["train_ei"][0].tolist())
    for u in train_seen:
        if u < n_users:
            eu[u] = ei[bundle["train_ei"][1][bundle["train_ei"][0] == u]].mean(dim=0)

    def score_fn(uu):
        return torch.mm(eu[uu], ei.t())

    modified_bundle = dict(bundle)
    test_full = bundle["test_ei"].numpy()
    user_set = set(users)
    mask = np.array([u in user_set for u in test_full[0]])
    modified_bundle["test_ei"] = torch.from_numpy(test_full[:, mask])
    m, _ = evaluate(score_fn, modified_bundle, split="test", device=device)
    return m


def main():
    device = torch.device("cpu")
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    buckets = bucket_users(bundle)
    print(f"稀疏度分层: {[(n, len(u)) for n, u in buckets.items()]}")

    seeds = [42, 43, 44, 45, 46]
    out = {"buckets": {name: len(users) for name, users in buckets.items()},
           "seeds": seeds, "configs": {}}

    for cfg in ["t1", "t3"]:
        out["configs"][cfg] = {}
        for bname in buckets:
            out["configs"][cfg][bname] = []
        for sd in seeds:
            ef = EMB / f"emb_{cfg}_s{sd}.pt"
            if not ef.exists():
                continue
            emb = torch.load(ef, weights_only=False)["concept_emb"]
            emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
            for bname, users in buckets.items():
                m = eval_bucket(emb, bundle, users, device)
                out["configs"][cfg][bname].append(
                    {"seed": sd, "NDCG@10": round(m["NDCG@10"], 5),
                     "HR@10": round(m["HR@10"], 5)})

    for cfg in out["configs"]:
        for bname in out["configs"][cfg]:
            runs = out["configs"][cfg][bname]
            if runs:
                ndcg = [r["NDCG@10"] for r in runs]
                out["configs"][cfg][bname] = {
                    "per_seed": runs,
                    "summary": {
                        "NDCG@10": {"mean": statistics.mean(ndcg),
                                   "std": statistics.stdev(ndcg) if len(ndcg) > 1 else 0.0}}}

    outf = RESULTS / "r_exp5_sparsity.json"
    outf.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n稀疏度分层完成，T1对比T3，五种子均值±标准差")
    for bname in buckets:
        n = out["buckets"][bname]
        t1 = out["configs"]["t1"][bname]["summary"]["NDCG@10"]
        t3 = out["configs"]["t3"][bname]["summary"]["NDCG@10"]
        delta = (t3["mean"] - t1["mean"]) / t1["mean"] * 100 if t1["mean"] > 0 else 0
        print(f"  {bname}({n}人): T1={t1['mean']:.5f}±{t1['std']:.5f}  "
              f"T3={t3['mean']:.5f}±{t3['std']:.5f}  "
              f"T3相对T1 {delta:+.2f}%")
    print(f"落盘 {outf}")


if __name__ == "__main__":
    main()
