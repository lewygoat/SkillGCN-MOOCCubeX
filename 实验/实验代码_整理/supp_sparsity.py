import common  # Auto-unpacks s6_graph.pt.gz on first run
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from evaluator import evaluate
from guards import GuardViolation, assert_eval_pool_clean, assert_no_leakage, check_metrics

INTER = Path("../中间产物")
EMB = INTER / "嵌入_补充"
RUNS = Path("../结果/supp_runs")
OUT = Path("../结果/supp_sparsity.json")
SEEDS = [42, 43, 44, 45, 46]
CONFIGS = ["t1", "t3"]
BUCKETS = [("低频[10,20)", 10, 20), ("中频[20,50)", 20, 50), ("高频[50+)", 50, 10**9)]
CONSISTENCY_TOL = 1e-4
MIN_FREE_GB = 5


def guard_disk(path):
    free = shutil.disk_usage(path).free / 1e9
    if free < MIN_FREE_GB:
        raise GuardViolation(f"磁盘剩余 {free:.1f} GB 低于 {MIN_FREE_GB} GB")


def bucket_users(bundle):
    train_ei = bundle["train_ei"].numpy()
    counts = np.bincount(train_ei[0], minlength=bundle["n_users"])
    test_users = set(np.unique(bundle["test_ei"].numpy()[0]).tolist())
    out = {}
    for name, lo, hi in BUCKETS:
        sel = np.nonzero((counts >= lo) & (counts < hi))[0].tolist()
        out[name] = sorted(u for u in sel if u in test_users)
    return out


def restrict_test(bundle, users):
    keep = set(users)
    test = bundle["test_ei"].numpy()
    mask = np.array([u in keep for u in test[0]], dtype=bool)
    sub = dict(bundle)
    sub["test_ei"] = torch.from_numpy(test[:, mask])
    return sub


def load_embeddings(cfg, seed):
    path = EMB / f"exp5_{cfg}_s{seed}.pt"
    if not path.exists():
        raise GuardViolation(f"缺少嵌入文件 {path.name}")
    blob = torch.load(path, weights_only=False, map_location="cpu")
    eu = blob["user_emb"].float()
    ei = blob["concept_emb"].float()
    if not torch.isfinite(eu).all() or not torch.isfinite(ei).all():
        raise GuardViolation(f"{path.name} 嵌入含NaN或Inf")
    return eu, ei


def evaluate_subset(eu, ei, bundle, users):
    sub = restrict_test(bundle, users)

    def score_fn(uu):
        return eu[uu] @ ei.T

    metrics, counted = evaluate(score_fn, sub, split="test", device="cpu")
    check_metrics(metrics, epoch=0)
    return metrics, counted


def main():
    guard_disk(OUT.parent)
    if OUT.exists():
        print(f"已存在 {OUT}，不覆盖")
        return
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    assert_no_leakage(bundle)
    buckets = bucket_users(bundle)
    all_users = sorted(set(np.unique(bundle["test_ei"].numpy()[0]).tolist()))
    sizes = {name: len(u) for name, u in buckets.items()}
    print(f"分桶人数 {sizes}，测试用户 {len(all_users)}", flush=True)

    result = {"buckets": sizes, "n_test_users": len(all_users), "seeds": SEEDS,
              "score": "user_emb @ concept_emb.T，与训练时score_all一致", "configs": {}}

    for cfg in CONFIGS:
        per_bucket = {name: [] for name in buckets}
        for seed in SEEDS:
            eu, ei = load_embeddings(cfg, seed)
            assert_eval_pool_clean(lambda uu: eu[uu] @ ei.T, bundle)
            full, counted = evaluate_subset(eu, ei, bundle, all_users)
            ref = json.loads((RUNS / f"exp5_{cfg}_s{seed}.json").read_text(encoding="utf-8"))
            gap = abs(full["NDCG@10"] - ref["test_at_best"]["NDCG@10"])
            if gap > CONSISTENCY_TOL:
                raise GuardViolation(
                    f"{cfg} s{seed} 整体NDCG@10 {full['NDCG@10']:.5f} 与训练记录 "
                    f"{ref['test_at_best']['NDCG@10']:.5f} 相差 {gap:.2e}")
            print(f"{cfg} s{seed} 整体 {full['NDCG@10']:.5f} 与训练记录一致", flush=True)
            for name, users in buckets.items():
                m, n = evaluate_subset(eu, ei, bundle, users)
                per_bucket[name].append({"seed": seed, "n": n,
                                         "NDCG@10": round(m["NDCG@10"], 5),
                                         "HR@10": round(m["HR@10"], 5),
                                         "MRR": round(m["MRR"], 5)})
        result["configs"][cfg] = per_bucket

    summary = {}
    for name in buckets:
        row = {}
        for cfg in CONFIGS:
            xs = [r["NDCG@10"] for r in result["configs"][cfg][name]]
            row[cfg] = {"mean": round(statistics.mean(xs), 5),
                        "std": round(statistics.stdev(xs), 5)}
        row["t3_vs_t1_pct"] = round((row["t3"]["mean"] - row["t1"]["mean"]) / row["t1"]["mean"] * 100, 2)
        diffs = [b["NDCG@10"] - a["NDCG@10"]
                 for a, b in zip(result["configs"]["t1"][name], result["configs"]["t3"][name])]
        row["paired_diff_mean"] = round(statistics.mean(diffs), 5)
        row["paired_diff_all_negative"] = all(d < 0 for d in diffs)
        summary[name] = row
    result["summary"] = summary

    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, OUT)
    for name, row in summary.items():
        print(f"{name}（{sizes[name]}人） T1={row['t1']['mean']:.5f}±{row['t1']['std']:.5f} "
              f"T3={row['t3']['mean']:.5f}±{row['t3']['std']:.5f} T3相对T1 {row['t3_vs_t1_pct']:+.2f}%")
    print(f"落盘 {OUT}")


if __name__ == "__main__":
    try:
        main()
    except GuardViolation as exc:
        print(f"防御中止 {exc}", file=sys.stderr)
        sys.exit(2)
