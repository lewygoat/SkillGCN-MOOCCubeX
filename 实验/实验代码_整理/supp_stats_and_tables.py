import common  # Auto-unpacks s6_graph.pt.gz on first run
#!/usr/bin/env python3
"""
补充实验统计与零重训核验汇总脚本 (supp_stats_and_tables.py)
涵盖：
1. Top-K (K=5,10,15,20) 全基线与消融指标汇总及曲线图绘制 (fig_topk_curves.png)
2. 职业技能标准分层统计、未对齐技能点占比（覆盖缺口）与剔除“大数据”字面偏置的子池权重生成
3. 掌握度代理（投入度/覆盖度/重复度）在人工正例、随机对照、人工负例下的内部一致性检验 (Delta & Cohen's d)
4. 冷启动学习者(10748人)与冷启动概念(48个)的单列评估 (s5_coldstart.jsonl.gz)
5. 第8节固定门控五种子(42-46)聚合、同轮次(Fixed-Epoch)对照与83%/17%归因五种子配对t检验
"""
import collections
import gzip
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel

from fig_common import CJK, OUT as FIG_OUT, plt

INTER = Path("../中间产物")
RESULTS = Path("../结果")
RUNS = RESULTS / "supp_runs"
NEWS9 = RESULTS / "newS9_runs"


def _ms(vals):
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": round(float(statistics.mean(vals)), 5),
        "std": round(float(statistics.stdev(vals)) if len(vals) > 1 else 0.0, 5),
        "n": len(vals),
    }


def collect_topk_and_plot():
    """提取 r_*.json 中已计算的 K=5,10,15,20 指标并绘制 Top-K 曲线"""
    methods = [
        ("LightGCN", "r_lightgcn.json"),
        ("BPR-MF", "r_bprmf.json"),
        ("NGCF", "r_ngcf.json"),
        ("Most-Popular", "r_popularity.json"),
        ("KGAT", "r_kgat.json"),
        ("ConceptGCN", "r_conceptgcn.json"),
        ("ACKRec", "r_ackrec.json"),
        ("T1-无技能层", "r_skillgcn_t1.json"),
        ("T2-权重打乱", "r_skillgcn_t2.json"),
        ("T3-全锚定", "r_skillgcn_t3.json"),
    ]
    ks = [5, 10, 15, 20]
    table = {}
    for label, fname in methods:
        p = RESULTS / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        summary = d.get("summary", {})
        per_seed = d.get("per_seed", [])
        row = {}
        for m in ("NDCG", "HR", "Recall", "Precision"):
            for k in ks:
                key = f"{m}@{k}"
                if key in summary:
                    row[key] = summary[key]
                elif per_seed and key in per_seed[0]:
                    vals = [r[key] for r in per_seed if key in r]
                    row[key] = _ms(vals)
        if "MRR" in summary:
            row["MRR"] = summary["MRR"]
        table[label] = row

    if table:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        plot_targets = [
            ("LightGCN", "#2C5F8A", "o", "-"),
            ("BPR-MF", "#5B8CB8", "s", "--"),
            ("NGCF", "#7BA3C9", "^", "-."),
            ("KGAT", "#9FB4C4", "D", ":"),
            ("T3-全锚定", "#E1743B", "v", "-"),
        ]
        for label, col, mk, ls in plot_targets:
            if label not in table:
                continue
            ndcgs = [table[label].get(f"NDCG@{k}", {}).get("mean", 0.0) for k in ks]
            recs = [table[label].get(f"Recall@{k}", {}).get("mean", 0.0) for k in ks]
            if any(ndcgs):
                ax1.plot(ks, ndcgs, color=col, marker=mk, linestyle=ls, linewidth=1.5, label=label)
            if any(recs):
                ax2.plot(ks, recs, color=col, marker=mk, linestyle=ls, linewidth=1.5, label=label)
        for ax, yl in [(ax1, "NDCG@K"), (ax2, "Recall@K")]:
            ax.set_xticks(ks)
            ax.set_xlabel("推荐列表长度 K", fontproperties=CJK, fontsize=10)
            ax.set_ylabel(yl, fontproperties=CJK, fontsize=10)
            ax.grid(True, color="#DDDDDD", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(prop=CJK, fontsize=8.5, frameon=False)
        fig.tight_layout()
        FIG_OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_OUT / "fig_topk_curves.png", bbox_inches="tight")
        plt.close(fig)
    return table


def analyze_skill_alignment_and_subpools(tau=0.50):
    """统计三套职业技能标准的覆盖缺口（未对齐技能点占比）并导出无大数据偏置的权重向量"""
    gpath = INTER / "s6_graph.pt"
    if not gpath.exists():
        return {}
    bundle = torch.load(gpath, weights_only=False)
    concepts = list(bundle["concepts"])
    c2i = {c: i for i, c in enumerate(concepts)}
    n_c = len(concepts)

    sources = [
        ("1x_web", "s7_skills.json", "s8_align.json"),
        ("rk", "s7_skills_rk.json", "s8_align_rk.json"),
        ("1x_bigdata", "s7_skills_bigdata.json", "s8_align_bigdata.json"),
    ]
    per_source = {}
    mass_by_src = {}

    for tag, s7_name, s8_name in sources:
        p8 = INTER / s8_name
        if not p8.exists():
            continue
        d8 = json.loads(p8.read_text(encoding="utf-8"))
        cands = d8.get("candidates", [])
        n_skills = len(cands)
        aligned_skills = 0
        kept_edges = 0
        concept_hits = collections.Counter()
        mass = np.zeros(n_c, dtype=np.float64)

        for cand in cands:
            hits_this_skill = 0
            for t in cand.get("topk", []):
                cp = t["concept"]
                sim = float(t["sim"])
                if cp in c2i and sim >= tau:
                    hits_this_skill += 1
                    kept_edges += 1
                    concept_hits[c2i[cp]] += 1
                    mass[c2i[cp]] += sim - tau
            if hits_this_skill > 0:
                aligned_skills += 1

        nz = mass > 0
        scale = float(mass[nz].mean()) if nz.any() else 1.0
        w_vec = (mass / scale).astype(np.float32)
        np.save(INTER / f"s9_skill_weight_{tag}.npy", w_vec)
        mass_by_src[tag] = mass

        data_concepts = [i for i, c in enumerate(concepts) if "数据" in c]
        data_weight_share = float(w_vec[data_concepts].sum() / max(w_vec.sum(), 1e-8))

        per_source[tag] = {
            "align_file": s8_name,
            "n_skills_total": n_skills,
            "n_skills_aligned": aligned_skills,
            "n_skills_unaligned": n_skills - aligned_skills,
            "unaligned_skill_ratio": round((n_skills - aligned_skills) / max(n_skills, 1), 4),
            "kept_alignment_edges": kept_edges,
            "avg_concepts_per_skill": round(kept_edges / max(n_skills, 1), 3),
            "concepts_covered": int(nz.sum()),
            "concept_coverage_ratio": round(int(nz.sum()) / max(n_c, 1), 4),
            "data_literal_weight_share": round(data_weight_share, 4),
        }

    # 生成剔除 1x_bigdata 后的组合权重（检验剔除“大数据”字面偏置后的 T3 表现）
    if "1x_web" in mass_by_src or "rk" in mass_by_src:
        m_nobd = mass_by_src.get("1x_web", np.zeros(n_c)) + mass_by_src.get("rk", np.zeros(n_c))
        nz = m_nobd > 0
        scale = float(m_nobd[nz].mean()) if nz.any() else 1.0
        w_nobd = (m_nobd / scale).astype(np.float32)
        np.save(INTER / "s9_skill_weight_no_bigdata.npy", w_nobd)
        data_concepts = [i for i, c in enumerate(concepts) if "数据" in c]
        per_source["combined_no_bigdata"] = {
            "concepts_covered": int(nz.sum()),
            "data_literal_weight_share": round(float(w_nobd[data_concepts].sum() / max(w_nobd.sum(), 1e-8)), 4),
            "saved_path": str(INTER / "s9_skill_weight_no_bigdata.npy"),
        }

    return per_source


def verify_mastery_proxy():
    """内部一致性负对照检验：投入度(engagement)、覆盖度(coverage)、重复度(repetition)"""
    raw_path = INTER / "s4_train.jsonl.gz"
    if not raw_path.exists():
        raw_path = INTER / "s5_train.jsonl.gz"
    if not raw_path.exists():
        return {}

    pos_path = INTER / "s1_eval_positive.jsonl"
    neg_path = INTER / "s1_eval_negative.jsonl"
    if not pos_path.exists() or not neg_path.exists():
        return {}

    pos_pairs = [( json.loads(l)["c1"], json.loads(l)["c2"] ) for l in open(pos_path, encoding="utf-8") if l.strip()]
    neg_pairs = [( json.loads(l)["c1"], json.loads(l)["c2"] ) for l in open(neg_path, encoding="utf-8") if l.strip()]

    user_records = collections.defaultdict(dict)
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            user_records[r["user"]][r["concept"]] = (
                float(r.get("engagement", 0.0)),
                float(r.get("coverage", 0.0)),
                float(r.get("repetition", 0.0)),
            )

    # 扣除学习者个体均值（消除个体基线投入差异）
    user_centered = {}
    all_concepts = set()
    for u, cmap in user_records.items():
        if len(cmap) < 2:
            continue
        all_concepts.update(cmap.keys())
        arr = np.array(list(cmap.values()), dtype=np.float64)
        means = arr.mean(axis=0)
        user_centered[u] = {c: (v[0] - means[0], v[1] - means[1], v[2] - means[2])
                            for c, v in cmap.items()}

    concepts_list = sorted(all_concepts)
    rng = np.random.RandomState(42)
    pos_set = set(pos_pairs)
    neg_set = set(neg_pairs)
    rand_pairs = []
    while len(rand_pairs) < len(pos_pairs) and len(concepts_list) >= 2:
        i, j = rng.choice(len(concepts_list), 2, replace=False)
        pair = (concepts_list[i], concepts_list[j])
        if pair not in pos_set and pair not in neg_set:
            rand_pairs.append(pair)

    def eval_pair_group(pairs):
        dims = {0: ("engagement", [], []), 1: ("coverage", [], []), 2: ("repetition", [], [])}
        for c1, c2 in pairs:
            pre_vals = []
            post_vals = []
            for u, cmap in user_centered.items():
                if c1 in cmap and c2 in cmap:
                    pre_vals.append(cmap[c1])
                    post_vals.append(cmap[c2])
            if len(pre_vals) < 10:
                continue
            pre_arr = np.array(pre_vals)
            post_arr = np.array(post_vals)
            for d_idx in (0, 1, 2):
                med = np.median(pre_arr[:, d_idx])
                hi = post_arr[pre_arr[:, d_idx] >= med, d_idx]
                lo = post_arr[pre_arr[:, d_idx] < med, d_idx]
                if len(hi) >= 3 and len(lo) >= 3:
                    dims[d_idx][1].append(float(hi.mean() - lo.mean()))
                    pooled_sd = math.sqrt((float(hi.var()) + float(lo.var())) / 2.0 + 1e-12)
                    dims[d_idx][2].append(float((hi.mean() - lo.mean()) / pooled_sd))
        out = {}
        for d_idx, (name, deltas, ds) in dims.items():
            out[name] = {
                "delta_mean": round(float(np.mean(deltas)), 4) if deltas else 0.0,
                "cohen_d": round(float(np.mean(ds)), 4) if ds else 0.0,
                "n_valid_pairs": len(deltas),
            }
        return out

    pos_res = eval_pair_group(pos_pairs)
    rand_res = eval_pair_group(rand_pairs)
    neg_res = eval_pair_group(neg_pairs)

    summary = {"positive_pairs": pos_res, "random_control": rand_res, "negative_control": neg_res, "net_effect_vs_random": {}}
    for dim_name in ("engagement", "coverage", "repetition"):
        d_pos = pos_res.get(dim_name, {}).get("cohen_d", 0.0)
        d_rnd = rand_res.get(dim_name, {}).get("cohen_d", 0.0)
        summary["net_effect_vs_random"][dim_name] = round(d_pos - d_rnd, 4)
    return summary


def evaluate_coldstart():
    """单列评估 s5_coldstart.jsonl.gz 中的 10748 名冷启动学习者与 48 个冷启动概念"""
    cs_path = INTER / "s5_coldstart.jsonl.gz"
    gpath = INTER / "s6_graph.pt"
    if not cs_path.exists() or not gpath.exists():
        return {}

    bundle = torch.load(gpath, weights_only=False)
    known_users = set(bundle["users"])
    known_concepts = set(bundle["concepts"])

    cat_counts = collections.Counter()
    cold_u_set = set()
    cold_c_set = set()
    cold_u_known_c_targets = collections.defaultdict(set)

    with gzip.open(cs_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            u, c = r["user"], r["concept"]
            u_in = u in known_users
            c_in = c in known_concepts
            if not u_in:
                cold_u_set.add(u)
            if not c_in:
                cold_c_set.add(c)
            if not u_in and c_in:
                cat_counts["cold_learner_known_concept"] += 1
                cold_u_known_c_targets[u].add(c)
            elif u_in and not c_in:
                cat_counts["known_learner_cold_concept"] += 1
            else:
                cat_counts["cold_learner_cold_concept"] += 1

    # 评估 Most-Popular 与纯技能权重排序在冷启动学习者（交互已知概念）上的 NDCG@10 与 HR@10
    c2i = {c: i for i, c in enumerate(bundle["concepts"])}
    train_ei = bundle["train_ei"].numpy()
    pop = np.bincount(train_ei[1], minlength=len(c2i))
    top10_pop = [bundle["concepts"][i] for i in np.argsort(-pop)[:10]]

    s9_path = INTER / "s9_skill_weight.npy"
    top10_s9 = []
    if s9_path.exists():
        s9 = np.load(s9_path)
        top10_s9 = [bundle["concepts"][i] for i in np.argsort(-s9)[:10]]

    def eval_ranking(ranked_top10):
        if not ranked_top10 or not cold_u_known_c_targets:
            return {"HR@10": 0.0, "NDCG@10": 0.0}
        hrs, ndcgs = [], []
        discounts = [1.0 / math.log2(i + 2) for i in range(10)]
        for u, truth in cold_u_known_c_targets.items():
            hits = [1.0 if c in truth else 0.0 for c in ranked_top10]
            hrs.append(1.0 if sum(hits) > 0 else 0.0)
            dcg = sum(h * d for h, d in zip(hits, discounts))
            idcg = sum(discounts[:min(len(truth), 10)])
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        return {"HR@10": round(float(np.mean(hrs)), 5), "NDCG@10": round(float(np.mean(ndcgs)), 5)}

    return {
        "coldstart_learners_count": len(cold_u_set),
        "coldstart_concepts_count": len(cold_c_set),
        "coldstart_concepts_list": sorted(cold_c_set),
        "record_breakdown": dict(cat_counts),
        "evaluable_cold_learners_with_known_concepts": len(cold_u_known_c_targets),
        "most_popular_on_cold_learners": eval_ranking(top10_pop),
        "skill_weight_on_cold_learners": eval_ranking(top10_s9),
    }


def aggregate_section8_five_seeds():
    """汇总第8节固定门控(0.05..4)、按所学门控固定重训(表7)、同轮次对照(eqlen)的五种子结果与配对检验"""
    if not RUNS.exists():
        return {}

    out = {"fixed_gate_sweep": {}, "table7_learned_gate_fixed": {}, "equal_epoch_6_comparison": {}, "same_epoch_checkpoint_comparison": {}}

    # 1. 固定门控扫描 g in {0.05, 0.1, 0.25, 0.5, 1, 2, 4}
    for g in ("0.05", "0.1", "0.25", "0.5", "1", "2", "4"):
        vals = []
        per_seed = {}
        for sd in (42, 43, 44, 45, 46):
            p = RUNS / f"gsweep_g{g}_s{sd}.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                ndcg = d["test_at_best"]["NDCG@10"]
                vals.append(ndcg)
                per_seed[str(sd)] = {"NDCG@10": ndcg, "best_epoch": d.get("best_epoch"), "epochs_ran": d.get("epochs_ran")}
        if vals:
            out["fixed_gate_sweep"][f"g={g}"] = {"summary": _ms(vals), "per_seed": per_seed}

    # 2. 按各种子终端门控值固定重训 vs 可学习T3 vs 同种子T1
    rows_t7 = []
    for sd in (42, 43, 44, 45, 46):
        p_fix = RUNS / f"gsweep_glearned_s{sd}.json"
        p_t3 = RUNS / f"exp5_t3_s{sd}.json"
        p_t1 = RUNS / f"exp5_t1_s{sd}.json"
        if p_fix.exists() and p_t3.exists() and p_t1.exists():
            df = json.loads(p_fix.read_text(encoding="utf-8"))
            d3 = json.loads(p_t3.read_text(encoding="utf-8"))
            d1 = json.loads(p_t1.read_text(encoding="utf-8"))
            rows_t7.append({
                "seed": sd,
                "gate_value": df.get("gate_value"),
                "fixed_best_epoch": df.get("best_epoch"),
                "fixed_epochs_ran": df.get("epochs_ran"),
                "learn_best_epoch": d3.get("best_epoch"),
                "learn_epochs_ran": d3.get("epochs_ran"),
                "fixed_ndcg10": df["test_at_best"]["NDCG@10"],
                "learn_t3_ndcg10": d3["test_at_best"]["NDCG@10"],
                "t1_ndcg10": d1["test_at_best"]["NDCG@10"],
            })
    if rows_t7:
        fix_v = [r["fixed_ndcg10"] for r in rows_t7]
        lrn_v = [r["learn_t3_ndcg10"] for r in rows_t7]
        t1_v = [r["t1_ndcg10"] for r in rows_t7]
        m_fix, m_lrn, m_t1 = statistics.mean(fix_v), statistics.mean(lrn_v), statistics.mean(t1_v)
        total_drop = m_t1 - m_lrn
        scale_drop = m_t1 - m_fix
        out["table7_learned_gate_fixed"] = {
            "n_seeds": len(rows_t7),
            "per_seed": rows_t7,
            "mean_fixed": round(m_fix, 5),
            "mean_learn_t3": round(m_lrn, 5),
            "mean_t1": round(m_t1, 5),
            "drop_fixed_pct": round((m_fix - m_t1) / m_t1 * 100, 2),
            "drop_learn_pct": round((m_lrn - m_t1) / m_t1 * 100, 2),
            "scaling_share_of_loss_pct": round(scale_drop / total_drop * 100, 2) if total_drop > 0 else 0.0,
            "paired_ttest_fixed_vs_t1_p": round(float(ttest_rel(fix_v, t1_v).pvalue), 6) if len(rows_t7) > 1 else None,
            "paired_ttest_learn_vs_fixed_p": round(float(ttest_rel(lrn_v, fix_v).pvalue), 6) if len(rows_t7) > 1 else None,
        }

    # 3. 固定6轮无早停对照 (eqlen_t1_s* vs eqlen_t3_s*)
    eq_rows = []
    for sd in (42, 43, 44, 45, 46):
        p1 = RUNS / f"eqlen_t1_s{sd}.json"
        p3 = RUNS / f"eqlen_t3_s{sd}.json"
        if p1.exists() and p3.exists():
            d1 = json.loads(p1.read_text(encoding="utf-8"))
            d3 = json.loads(p3.read_text(encoding="utf-8"))
            eq_rows.append({
                "seed": sd,
                "t1_ndcg10": d1["test_at_best"]["NDCG@10"],
                "t3_ndcg10": d3["test_at_best"]["NDCG@10"],
                "t1_best_epoch": d1.get("best_epoch"),
                "t3_best_epoch": d3.get("best_epoch"),
            })
    if eq_rows:
        v1 = [r["t1_ndcg10"] for r in eq_rows]
        v3 = [r["t3_ndcg10"] for r in eq_rows]
        out["equal_epoch_6_comparison"] = {
            "n_seeds": len(eq_rows),
            "per_seed": eq_rows,
            "t1_summary": _ms(v1),
            "t3_summary": _ms(v3),
            "paired_ttest_p": round(float(ttest_rel(v3, v1).pvalue), 6) if len(eq_rows) > 1 else None,
        }

    # 4. 有界与正则化门控变体汇总
    for mode in ("bounded_tanh", "bounded_sigmoid", "l2_train", "gate_l2reg", "vector", "nobigdata"):
        vals = []
        per_seed = {}
        for sd in (42, 43, 44, 45, 46):
            p = RUNS / f"variant_{mode}_s{sd}.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                ndcg = d["test_at_best"]["NDCG@10"]
                vals.append(ndcg)
                per_seed[str(sd)] = {"NDCG@10": ndcg, "gate_best": d.get("gate_best"), "best_epoch": d.get("best_epoch")}
        if vals:
            out[f"variant_{mode}"] = {"summary": _ms(vals), "per_seed": per_seed}

    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report_data = {
        "1_topk_metrics": collect_topk_and_plot(),
        "2_skill_standard_alignment_gaps": analyze_skill_alignment_and_subpools(),
        "3_mastery_proxy_consistency": verify_mastery_proxy(),
        "4_coldstart_evaluation": evaluate_coldstart(),
        "5_section8_five_seeds_and_variants": aggregate_section8_five_seeds(),
    }
    out_path = RESULTS / "supp_comprehensive_report.json"
    out_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== 补充统计与核验完成，落盘 {out_path} ===")
    for k, v in report_data.items():
        print(f"  {k}: {'已生成' if v else '待中间产物/结果文件就绪后生成'}")


if __name__ == "__main__":
    main()
