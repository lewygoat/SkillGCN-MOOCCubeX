import common  # Auto-unpacks s6_graph.pt.gz on first run
import argparse
import json
from pathlib import Path

import numpy as np
import torch

INTER = Path("../中间产物")
POOLS = ["s8_align.json", "s8_align_rk.json", "s8_align_bigdata.json"]
TAU = 0.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--out", default=str(INTER / "s9_skill_weight.npy"))
    ap.add_argument("--meta_out", default=str(INTER / "s9_skill_weight_meta.json"))
    args = ap.parse_args()

    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    concepts = bundle["concepts"]
    idx = {c: i for i, c in enumerate(concepts)}
    n = len(concepts)

    mass = np.zeros(n, dtype=np.float64)
    edge_count = np.zeros(n, dtype=np.int64)
    kept_edges = 0
    seen_edges = 0
    for fname in POOLS:
        path = INTER / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for cand in data.get("candidates", []):
            for t in cand.get("topk", []):
                seen_edges += 1
                cp = t["concept"]
                sim = float(t["sim"])
                if cp in idx and sim >= args.tau:
                    mass[idx[cp]] += sim - args.tau
                    edge_count[idx[cp]] += 1
                    kept_edges += 1

    nz = mass > 0
    scale = float(mass[nz].mean()) if nz.any() else 1.0
    weight = (mass / scale).astype(np.float32)

    np.save(args.out, weight)
    meta = {
        "rule": "sum of (sim - tau) over candidate edges with sim >= tau, normalized by mean of nonzero",
        "tau": args.tau,
        "pools": POOLS,
        "n_concepts": n,
        "nonzero": int(nz.sum()),
        "seen_edges": seen_edges,
        "kept_edges": kept_edges,
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
        "weight_mean_nonzero": float(weight[nz].mean()) if nz.any() else 0.0,
        "scale_divisor": scale,
        "note": "BAAI/bge-small-zh-v1.5 cosine similarity excess-threshold aggregation via s8_align*.json",
    }
    Path(args.meta_out).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"s9落盘 {args.out}")
    print(f"非零 {meta['nonzero']}/{n}  保留边 {kept_edges}/{seen_edges}  "
          f"权重范围 {meta['weight_min']:.4f}-{meta['weight_max']:.4f}  "
          f"非零均值 {meta['weight_mean_nonzero']:.4f}")


if __name__ == "__main__":
    main()
