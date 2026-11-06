import common  # Auto-unpacks s6_graph.pt.gz on first run
import argparse
import json
from pathlib import Path

import numpy as np
import torch

INTER = Path("../中间产物")
EMB = Path("../中间产物/嵌入")
RESULTS = Path("../结果")


def load_prereq(bundle, key):
    ei = bundle[key].numpy()
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
    hit = sum(1 for c in path if s9[c] > 0)
    return hit / len(path) if path else 0.0


def weighted_coverage(path, s9):
    if not path:
        return 0.0
    return float(np.mean([min(s9[c], 3.0) / 3.0 for c in path]))


def prereq_violation(path, prereq_set):
    pos = {c: i for i, c in enumerate(path)}
    viol = 0
    pairs = 0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        pairs += 1
        if (b, a) in prereq_set:
            viol += 1
    return viol / pairs if pairs else 0.0


def false_violation(path, neg_set):
    pairs = 0
    fv = 0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        pairs += 1
        if (b, a) in neg_set:
            fv += 1
    return fv / pairs if pairs else 0.0


def load_concept_skills(tau=0.50):
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    c2i = {c: i for i, c in enumerate(bundle["concepts"])}
    c_skills = {i: set() for i in range(len(c2i))}
    for fname in ("s8_align.json", "s8_align_rk.json", "s8_align_bigdata.json"):
        p = INTER / fname
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for cand in data.get("candidates", []):
            sid = f"{fname}:{cand.get('skill_id', '')}"
            for t in cand.get("topk", []):
                if float(t["sim"]) >= tau and t["concept"] in c2i:
                    c_skills[c2i[t["concept"]]].add(sid)
    return c_skills


def redundancy(path, s9, c_skills=None):
    if c_skills:
        seen_skills = set()
        redundant_concepts = 0
        skilled_concepts = 0
        for c in path:
            sks = c_skills.get(c, set())
            if not sks:
                continue
            skilled_concepts += 1
            if sks & seen_skills:
                redundant_concepts += 1
            seen_skills |= sks
        return redundant_concepts / skilled_concepts if skilled_concepts else 0.0
    skilled = [c for c in path if s9[c] > 0]
    if not skilled:
        return 0.0
    return 1.0 - len(set(skilled)) / len(skilled)


def smoothness(path, bloom):
    diffs = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if bloom[a] > 0 and bloom[b] > 0:
            diffs.append(abs(bloom[a] - bloom[b]))
    return float(np.var(diffs)) if diffs else 0.0


def step_efficiency(path, s9, target_mass_ratio=0.8):
    total_mass = sum(max(float(s9[c]), 0.0) for c in path)
    if total_mass <= 0:
        return len(path)
    acc = 0.0
    for i, c in enumerate(path):
        acc += max(float(s9[c]), 0.0)
        if acc >= total_mass * target_mass_ratio:
            return i + 1
    return len(path)


def eval_config(emb, s9, bloom, prereq_gen, prereq_eval, prereq_weak, neg_set, seeds,
                target_len, cov_weight, c_skills=None):
    rows = {"cov": [], "viol": [], "viol_isolated": [], "viol_weak": [], "fviol": [],
            "redun": [], "smooth": [], "stepeff": []}
    for sd in seeds:
        path = gen_path(sd, emb, s9, prereq_eval, target_len, cov_weight)
        path_iso = gen_path(sd, emb, s9, prereq_gen, target_len, cov_weight)
        rows["cov"].append(weighted_coverage(path, s9))
        rows["viol"].append(prereq_violation(path, prereq_eval))
        rows["viol_isolated"].append(prereq_violation(path_iso, prereq_eval))
        rows["viol_weak"].append(prereq_violation(path, prereq_weak))
        rows["fviol"].append(false_violation(path, neg_set))
        rows["redun"].append(redundancy(path, s9, c_skills))
        rows["smooth"].append(smoothness(path, bloom))
        rows["stepeff"].append(step_efficiency(path, s9))
    return {k: float(np.mean(v)) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_len", type=int, default=10)
    ap.add_argument("--n_seeds", type=int, default=300)
    ap.add_argument("--sample_seed", type=int, default=42)
    args = ap.parse_args()

    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    s9 = np.load(INTER / "s9_skill_weight.npy")
    bloom = np.load(INTER / "s10_concept_bloom.npy")
    c_skills = load_concept_skills()
    prereq_gen = load_prereq(bundle, "prereq_adopted")
    prereq_main = set(zip(bundle["eval_prereq_positive"].numpy()[0].tolist(),
                          bundle["eval_prereq_positive"].numpy()[1].tolist()))
    prereq_weak = load_prereq(bundle, "chapter_weak")
    neg_set = set(zip(bundle["eval_prereq_negative"].numpy()[0].tolist(),
                      bundle["eval_prereq_negative"].numpy()[1].tolist()))

    rng = np.random.RandomState(args.sample_seed)
    skilled = np.nonzero(s9 > 0)[0]
    seeds = rng.choice(skilled, size=min(args.n_seeds, len(skilled)),
                       replace=False).tolist()

    cov_grid = {"低": 0.0, "中": 0.5, "高": 1.0}
    out = {"target_len": args.target_len, "n_seeds": len(seeds),
           "prereq_main": "人工评测正例1075对", "prereq_gen_isolated": "建模采纳先修边194对（与评测集隔离）",
           "prereq_weak": "章节弱边",
           "coverage": "s9权重加权", "configs": {}}

    emb_seeds = [42, 43, 44, 45, 46]
    out["emb_seeds"] = emb_seeds
    for cfg in ["t1", "t3"]:
        out["configs"][cfg] = {}
        per_cw = {name: {k: [] for k in
                         ["cov", "viol", "viol_isolated", "viol_weak", "fviol", "redun",
                          "smooth", "stepeff"]} for name in cov_grid}
        used = 0
        for sd in emb_seeds:
            ef = EMB / f"emb_{cfg}_s{sd}.pt"
            if not ef.exists():
                continue
            used += 1
            emb = torch.load(ef, weights_only=False)["concept_emb"]
            emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
            for name, cw in cov_grid.items():
                r = eval_config(emb, s9, bloom, prereq_gen, prereq_main, prereq_weak,
                                neg_set, seeds, args.target_len, cw, c_skills)
                for k, v in r.items():
                    per_cw[name][k].append(v)
        for name in cov_grid:
            out["configs"][cfg][name] = {
                k: {"mean": float(np.mean(v)),
                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                for k, v in per_cw[name].items()}
        out["configs"][cfg]["emb_seeds_used"] = used

    outfile = RESULTS / "r_pathgen.json"
    outfile.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    print("路径级评价完成（五种子嵌入均值±标准差），T1无锚定对比T3有锚定")
    print(f"采样种子概念 {len(seeds)}  路径长 {args.target_len}")
    for cfg in ["t1", "t3"]:
        print(f"── {cfg}（嵌入种子 {out['configs'][cfg]['emb_seeds_used']}）──")
        for name in ["低", "中", "高"]:
            r = out["configs"][cfg][name]
            print(f"  覆盖权重{name}: 加权覆盖={r['cov']['mean']:.4f}±{r['cov']['std']:.4f} "
                  f"人工违背={r['viol']['mean']:.4f} "
                  f"弱基准违背={r['viol_weak']['mean']:.4f}±{r['viol_weak']['std']:.4f} "
                  f"平滑度={r['smooth']['mean']:.4f}")
    print(f"落盘 {outfile}")


if __name__ == "__main__":
    main()
