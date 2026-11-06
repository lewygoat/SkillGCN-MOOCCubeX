import common  # Auto-unpacks s6_graph.pt.gz on first run
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import torch


INTER = Path("../中间产物")
EMB = INTER / "嵌入_补充"
RESULTS = Path("../结果")


def load_prereq(bundle, key):
    ei = bundle.get(key)
    if ei is None or ei.numel() == 0:
        return set()
    return set(zip(ei[0].tolist(), ei[1].tolist()))


def topo_rank(cands, prereq_set, emb):
    remaining = list(cands)
    ordered = []
    guard = 0
    while remaining and guard < len(cands) + 5:
        guard += 1
        placed = False
        for c in list(remaining):
            blockers = [p for p in remaining
                        if p != c and (p, c) in prereq_set]
            if not blockers:
                ordered.append(c)
                remaining.remove(c)
                placed = True
                break
        if not placed:
            scores = {c: emb[c] @ emb[ordered[-1]] if ordered else 0.0
                      for c in remaining}
            nxt = max(remaining, key=lambda c: float(scores[c]))
            ordered.append(nxt)
            remaining.remove(nxt)
    return ordered


def gen_path(seed_concept, emb, s9, prereq_set, target_len, cov_weight):
    n = emb.shape[0]
    sims = emb @ emb[seed_concept]
    score = sims + cov_weight * torch.tensor(s9, dtype=torch.float32)
    score[seed_concept] = -1e9
    k = min(target_len - 1, n - 1)
    top = torch.topk(score, k).indices.tolist()
    cands = [seed_concept] + top
    return topo_rank(cands, prereq_set, emb)


def skill_coverage(path, s9):
    if not path:
        return 0.0
    return float(np.mean([min(s9[c], 3.0) / 3.0 for c in path]))


def prereq_violation(path, prereq_set):
    if not path or len(path) < 2:
        return 0.0
    pos = {c: i for i, c in enumerate(path)}
    viol, pairs = 0, 0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        pairs += 1
        if (b, a) in prereq_set:
            viol += 1
    return float(viol / pairs) if pairs else 0.0


def false_violation(path, neg_set):
    if not path or len(path) < 2:
        return 0.0
    fv, pairs = 0, 0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        pairs += 1
        if (b, a) in neg_set:
            fv += 1
    return float(fv / pairs) if pairs else 0.0


def smoothness(path, bloom):
    if bloom is None or not path or len(path) < 2:
        return 0.0
    diffs = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if bloom[a] > 0 and bloom[b] > 0:
            diffs.append(abs(float(bloom[a]) - float(bloom[b])))
    return float(np.var(diffs)) if diffs else 0.0


def eval_config(emb, s9, prereq_main, prereq_weak, seeds, target_len, cov_weight,
                prereq_gen=None, neg_set=None, bloom=None):
    rows = {"cov": [], "viol_main": [], "viol_main_isolated": [],
            "viol_weak": [], "fviol": [], "smooth": []}
    p_gen = prereq_gen if prereq_gen is not None else prereq_main
    for seed_concept in seeds:
        path = gen_path(seed_concept, emb, s9, prereq_main, target_len, cov_weight)
        path_iso = gen_path(seed_concept, emb, s9, p_gen, target_len, cov_weight)
        rows["cov"].append(skill_coverage(path, s9))
        rows["viol_main"].append(prereq_violation(path, prereq_main))
        rows["viol_main_isolated"].append(prereq_violation(path_iso, prereq_main))
        rows["viol_weak"].append(prereq_violation(path, prereq_weak))
        rows["fviol"].append(false_violation(path, neg_set) if neg_set else 0.0)
        rows["smooth"].append(smoothness(path, bloom))
    return {k: float(np.mean(v)) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_len", type=int, default=10)
    ap.add_argument("--n_seeds", type=int, default=300)
    ap.add_argument("--sample_seed", type=int, default=42)
    args = ap.parse_args()

    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    s9 = np.load(INTER / "s9_skill_weight.npy")
    bloom_path = INTER / "s10_concept_bloom.npy"
    bloom = np.load(bloom_path) if bloom_path.exists() else None
    prereq_gen = load_prereq(bundle, "prereq_adopted")
    prereq_main = set(zip(bundle["eval_prereq_positive"].numpy()[0].tolist(),
                          bundle["eval_prereq_positive"].numpy()[1].tolist()))
    prereq_weak = load_prereq(bundle, "chapter_weak")
    neg_set = load_prereq(bundle, "eval_prereq_negative")

    rng = np.random.RandomState(args.sample_seed)
    skilled = np.nonzero(s9 > 0)[0]
    seeds = rng.choice(skilled, size=min(args.n_seeds, len(skilled)),
                       replace=False).tolist()

    cov_grid = [0.0, 0.5, 1.0]
    emb_seeds = [42, 43, 44, 45, 46]

    # 独立验证零方差推断：零嵌入（纯按技能权重s9排序）下的路径覆盖率
    zero_emb = torch.zeros((len(s9), 64), dtype=torch.float32)
    pure_s9_eval = eval_config(zero_emb, s9, prereq_main, prereq_weak, seeds,
                               args.target_len, 1.0, prereq_gen, neg_set, bloom)

    out = {
        "target_len": args.target_len, "n_seeds": len(seeds),
        "cov_weights": cov_grid, "emb_seeds": emb_seeds,
        "pure_skill_weight_only_baseline": pure_s9_eval,
        "norm_configs": {
            "none": "concept_emb不归一化",
            "l2": "concept_emb L2归一化"
        },
        "results": {},
        "paired_significance": {}
    }

    raw_series = {}
    for cfg in ["t1", "t3"]:
        out["results"][cfg] = {}
        raw_series[cfg] = {}
        for norm_mode in ["none", "l2"]:
            out["results"][cfg][norm_mode] = {}
            per_cw = {cw: {"cov": [], "viol_main": [], "viol_main_isolated": [],
                           "viol_weak": [], "fviol": [], "smooth": []} for cw in cov_grid}
            used = 0
            for sd in emb_seeds:
                ef = EMB / f"exp5_{cfg}_s{sd}.pt"
                if not ef.exists():
                    continue
                used += 1
                data = torch.load(ef, weights_only=False)
                emb = data["concept_emb"]

                if norm_mode == "l2":
                    emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)

                for cw in cov_grid:
                    r = eval_config(emb, s9, prereq_main, prereq_weak, seeds,
                                    args.target_len, cw, prereq_gen, neg_set, bloom)
                    for k, v in r.items():
                        per_cw[cw][k].append(v)

            raw_series[cfg][norm_mode] = per_cw
            for cw in cov_grid:
                out["results"][cfg][norm_mode][f"cw{cw}"] = {
                    k: {
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "per_seed": [round(float(x), 6) for x in v]
                    }
                    for k, v in per_cw[cw].items()
                }
            out["results"][cfg][norm_mode]["emb_seeds_used"] = used

    # 配对t检验（T3对比T1，覆盖表6各单元）
    from scipy.stats import ttest_rel
    for norm_mode in ["none", "l2"]:
        out["paired_significance"][norm_mode] = {}
        for cw in cov_grid:
            out["paired_significance"][norm_mode][f"cw{cw}"] = {}
            for k in ["cov", "viol_weak", "viol_main_isolated", "fviol", "smooth"]:
                v1 = raw_series["t1"][norm_mode][cw].get(k, [])
                v3 = raw_series["t3"][norm_mode][cw].get(k, [])
                if len(v1) > 1 and len(v1) == len(v3):
                    diffs = np.asarray(v3) - np.asarray(v1)
                    if np.allclose(diffs, 0.0):
                        t_stat, p_val = 0.0, 1.0
                    else:
                        res = ttest_rel(v3, v1)
                        t_stat, p_val = float(res.statistic), float(res.pvalue)
                    out["paired_significance"][norm_mode][f"cw{cw}"][k] = {
                        "diff_mean": round(float(diffs.mean()), 6),
                        "diff_std": round(float(diffs.std(ddof=1)), 6),
                        "t_stat": round(t_stat, 4),
                        "p_value": round(p_val, 6)
                    }

    outfile = RESULTS / "supp_l2norm.json"
    outfile.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"L2归一化对比实验完成（五种子均值±标准差）")
    print(f"采样种子概念 {len(seeds)}  路径长 {args.target_len}")
    for cfg in ["t1", "t3"]:
        print(f"── {cfg} ──")
        for norm_mode in ["none", "l2"]:
            print(f"  归一化模式：{out['norm_configs'][norm_mode]}")
            r = out["results"][cfg][norm_mode]["cw0.5"]
            print(f"    技能覆盖={r['cov']['mean']:.4f}±{r['cov']['std']:.4f}")
            print(f"    人工前置违背={r['viol_main']['mean']:.4f}±{r['viol_main']['std']:.4f}")
            print(f"    弱基准违背={r['viol_weak']['mean']:.4f}±{r['viol_weak']['std']:.4f}")
    print(f"落盘 {outfile}")


if __name__ == "__main__":
    main()
