import common  # Auto-unpacks s6_graph.pt.gz on first run
import json
from pathlib import Path

import numpy as np
import torch

INTER = Path("../中间产物")
EMB = INTER / "嵌入_补充"
RESULTS = Path("../结果")
EMB_SEEDS = [42, 43, 44, 45, 46]
TARGET_LEN = 10
N_SEEDS = 300
SAMPLE_SEED = 42


def load_prereq(bundle, key):
    e = bundle[key].numpy()
    return set(zip(e[0].tolist(), e[1].tolist()))


def topo_rank(cands, prereq_set, emb):
    remaining = list(cands)
    ordered = []
    guard = 0
    while remaining and guard < len(cands) + 5:
        guard += 1
        placed = False
        for c in list(remaining):
            blockers = [p for p in remaining if p != c and (p, c) in prereq_set]
            if not blockers:
                ordered.append(c)
                remaining.remove(c)
                placed = True
                break
        if not placed:
            scores = {c: emb[c] @ emb[ordered[-1]] if ordered else 0.0 for c in remaining}
            nxt = max(remaining, key=lambda c: float(scores[c]))
            ordered.append(nxt)
            remaining.remove(nxt)
    return ordered


def gen_path(seed_concept, emb, s9, prereq_set, target_len, cov_weight):
    sims = emb @ emb[seed_concept]
    score = sims + cov_weight * torch.tensor(s9, dtype=torch.float32)
    score[seed_concept] = -1e9
    k = min(target_len - 1, emb.shape[0] - 1)
    top = torch.topk(score, k).indices.tolist()
    cands = [seed_concept] + top
    return topo_rank(cands, prereq_set, emb)


def weighted_coverage(path, s9):
    if not path:
        return 0.0
    return float(np.mean([min(s9[c], 3.0) / 3.0 for c in path]))


def viol_and_evaluable(path, prereq_set):
    pairs = 0
    viol = 0
    evaluable = 0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        pairs += 1
        if (b, a) in prereq_set:
            viol += 1
        if (a, b) in prereq_set or (b, a) in prereq_set:
            evaluable += 1
    return viol, pairs, evaluable


def main():
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    s9 = np.load(INTER / "s9_skill_weight.npy")
    wn = torch.tensor(s9 / s9.max(), dtype=torch.float32).unsqueeze(1)
    prereq_main = load_prereq(bundle, "eval_prereq_positive")
    prereq_weak = load_prereq(bundle, "chapter_weak")

    rng = np.random.RandomState(SAMPLE_SEED)
    skilled = np.nonzero(s9 > 0)[0]
    seeds = rng.choice(skilled, size=min(N_SEEDS, len(skilled)), replace=False).tolist()

    report = {"n_seed_concepts": len(seeds), "emb_seeds": EMB_SEEDS,
              "target_len": TARGET_LEN}

    prov = []
    for cfg in ("t1", "t3"):
        for sd in EMB_SEEDS:
            d = torch.load(EMB / f"exp5_{cfg}_s{sd}.pt", weights_only=False)
            post = d["concept_emb"].float()
            base = d["base_emb"].float()[:post.shape[0]]
            pre_div = post / (1.0 + float(d["gate"]) * wn)
            maxdiff = float((base - pre_div).abs().max())
            prov.append({"cfg": cfg, "seed": sd, "gate": float(d["gate"]),
                         "base_vs_divided_maxabs": round(maxdiff, 6)})
    report["pregate_provenance"] = prov

    def paths_for(emb, cov_weight):
        return [gen_path(sd, emb, s9, prereq_main, TARGET_LEN, cov_weight) for sd in seeds]

    evaluable_stats = {}
    for cfg in ("t1", "t3"):
        for cw in (0.0, 0.5, 1.0):
            ev_counts = []
            main_denoms = []
            zero_eval = 0
            for sd in EMB_SEEDS:
                d = torch.load(EMB / f"exp5_{cfg}_s{sd}.pt", weights_only=False)
                emb = d["concept_emb"].float()
                for p in paths_for(emb, cw):
                    _, pairs, ev = viol_and_evaluable(p, prereq_main)
                    ev_counts.append(ev)
                    main_denoms.append(pairs)
                    if ev == 0:
                        zero_eval += 1
            n = len(ev_counts)
            evaluable_stats[f"{cfg}_cw{cw}"] = {
                "mean_evaluable_edges_per_path": round(float(np.mean(ev_counts)), 4),
                "mean_adjacent_pairs": round(float(np.mean(main_denoms)), 4),
                "share_zero_evaluable": round(zero_eval / n, 4),
                "effective_coverage": round(float(np.sum(ev_counts) / np.sum(main_denoms)), 4),
                "n_paths": n,
            }
    report["evaluable_edges"] = evaluable_stats

    def mean_cov(emb, cw):
        return float(np.mean([weighted_coverage(p, s9) for p in paths_for(emb, cw)]))

    def ci(diffs):
        a = np.asarray(diffs, dtype=float)
        m = float(a.mean())
        sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        half = 2.776 * sd / np.sqrt(len(a))
        return round(m, 5), round(m - half, 5), round(m + half, 5)

    per_seed = {"t3_post_minus_pre_none_cw0": [], "t3_minus_t1_none_cw1": [],
                "t3_minus_t1_l2_cw1": []}
    for sd in EMB_SEEDS:
        d3 = torch.load(EMB / f"exp5_t3_s{sd}.pt", weights_only=False)
        d1 = torch.load(EMB / f"exp5_t1_s{sd}.pt", weights_only=False)
        post3 = d3["concept_emb"].float()
        pre3 = post3 / (1.0 + float(d3["gate"]) * wn)
        post1 = d1["concept_emb"].float()
        l2 = lambda e: e / e.norm(dim=1, keepdim=True).clamp(min=1e-8)
        per_seed["t3_post_minus_pre_none_cw0"].append(
            mean_cov(post3, 0.0) - mean_cov(pre3, 0.0))
        per_seed["t3_minus_t1_none_cw1"].append(
            mean_cov(post3, 1.0) - mean_cov(post1, 1.0))
        per_seed["t3_minus_t1_l2_cw1"].append(
            mean_cov(l2(post3), 1.0) - mean_cov(l2(post1), 1.0))
    report["paired_diffs"] = {k: {"per_seed": [round(x, 5) for x in v],
                                   "mean_lo_hi": ci(v)} for k, v in per_seed.items()}

    RESULTS.joinpath("verify_path.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 门控前来源核验 (base_emb vs post/(1+g*wn)) ===")
    mx = max(p["base_vs_divided_maxabs"] for p in prov)
    print(f"  最大绝对差 {mx}  (接近0说明除法重构即正确门控前嵌入)")
    print("=== 可评价边统计 ===")
    for k, v in evaluable_stats.items():
        print(f"  {k}: 可评价边/路径={v['mean_evaluable_edges_per_path']} "
              f"相邻对={v['mean_adjacent_pairs']} 零可评价占比={v['share_zero_evaluable']} "
              f"有效覆盖={v['effective_coverage']}")
    print("=== 逐模型种子配对差 + 95%t置信区间 ===")
    for k, v in report["paired_diffs"].items():
        m, lo, hi = v["mean_lo_hi"]
        print(f"  {k}: 每种子{v['per_seed']} 均值{m} CI[{lo},{hi}]")


if __name__ == "__main__":
    main()
