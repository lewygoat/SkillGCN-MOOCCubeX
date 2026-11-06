import json

import torch

from common import INTER, OUT, provenance, report
from evaluator import evaluate, format_metrics
from guards import assert_no_leakage, check_metrics, resolve_device


def main():
    device = resolve_device()
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    overlaps = assert_no_leakage(bundle)

    counts = torch.zeros(bundle["n_concepts"], device=device)
    counts.index_add_(0, bundle["train_ei"][1].to(device),
                      torch.ones(bundle["train_ei"].shape[1], device=device))
    scores = counts / counts.max()

    metrics, counted = evaluate(
        lambda uu: scores.unsqueeze(0).expand(uu.shape[0], -1).clone(),
        bundle, split="test", device=device)
    check_metrics(metrics, 0)

    ranked = torch.sort(counts, descending=True).values
    total = float(counts.sum())
    concentration = {
        "top10_share": float(ranked[:10].sum() / total),
        "top50_share": float(ranked[:50].sum() / total),
        "top100_share": float(ranked[:100].sum() / total),
        "hottest_learner_share": float(ranked[0] / bundle["n_users"]),
        "longtail_concepts_under_5pct": int((counts < 0.05 * bundle["n_users"]).sum()),
    }

    summary = {k: {"mean": v, "std": 0.0} for k, v in metrics.items()}
    report("流行度基线 测试集结果", {
        k: f"{metrics[k]:.4f}" for k in
        ("NDCG@10", "HR@10", "MRR", "Recall@20", "Precision@10")
    })
    report("流行度集中度", concentration)

    payload = {"method": "Most-Popular", "tag": "popularity", "device": device,
               "seeds": [], "hyper": {"deterministic": True},
               "per_seed": [metrics], "summary": summary,
               "concentration": concentration, "eval_users": counted,
               "split_overlaps": overlaps}
    path = OUT / "r_popularity.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    provenance("m_popularity", {k: metrics[k] for k in ("NDCG@10", "HR@10", "MRR")}, [path])
    print(f"   评测学习者 {counted:,}  无随机性，单次运行即确定")


if __name__ == "__main__":
    main()
