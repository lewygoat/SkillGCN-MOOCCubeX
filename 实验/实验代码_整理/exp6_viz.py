import common  # Auto-unpacks s6_graph.pt.gz on first run
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from path_gen import gen_path, load_prereq

INTER = Path("../中间产物")
EMB   = INTER / "嵌入"
OUT   = Path("../图表")
OUT.mkdir(exist_ok=True)

def _load_cjk_font():
    for fp in ("/System/Library/Fonts/STHeiti Medium.ttc",
               "C:/Windows/Fonts/msyh.ttc",
               "C:/Windows/Fonts/simhei.ttf"):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            return font_manager.FontProperties(fname=fp)
    return font_manager.FontProperties()

CJK = _load_cjk_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300

PRIMARY = "#2C5F8A"
ACCENT  = "#E1743B"
GREY    = "#9FB4C4"
LIGHT   = "#EEF4F8"
GRID    = "#DDDDDD"

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontproperties(CJK)
    ax.tick_params(labelsize=9)


def fig_skill_weight():
    s9   = np.load(INTER / "s9_skill_weight.npy")
    b    = torch.load(INTER / "s6_graph.pt", weights_only=False)
    cons = b["concepts"]
    idx  = np.argsort(s9)[::-1][:20]
    names = [cons[i] for i in idx]
    vals  = s9[idx]
    colors = [PRIMARY if v >= vals[4] else GREY for v in vals]

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, edgecolor="#333333", linewidth=0.6, height=0.72)
    for i, v in enumerate(vals):
        ax.text(v + 0.12, i, f"{v:.2f}", va="center", fontsize=8, fontproperties=CJK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontproperties=CJK)
    ax.invert_yaxis()
    ax.set_xlabel("技能锚定权重（s9，归一化前）", fontproperties=CJK, fontsize=10)
    ax.set_title("技能锚定权重最高的二十个概念", fontproperties=CJK, fontsize=11)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp6a_skill_weight.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_exp6a 落盘")


def _wrap(t, n):
    return "\n".join(t[i:i + n] for i in range(0, len(t), n))


def fig_alignment_cases():
    pools = ["s8_align.json", "s8_align_rk.json", "s8_align_bigdata.json"]
    sk_text = {}
    for fname in ["s7_skills.json", "s7_skills_rk.json", "s7_skills_bigdata.json"]:
        p = INTER / fname
        if not p.exists():
            continue
        for s in json.loads(p.read_text(encoding="utf-8")).get("skills", []):
            if s.get("skill_id"):
                sk_text[s["skill_id"]] = (s.get("text") or s.get("source", ""))

    rows = []
    for fname in pools:
        p = INTER / fname
        if not p.exists():
            continue
        for cand in json.loads(p.read_text(encoding="utf-8")).get("candidates", []):
            sid  = cand.get("skill_id", "")
            stxt = sk_text.get(sid, sid)
            for t in cand.get("topk", []):
                rows.append((float(t["sim"]), stxt, t["concept"]))

    rows.sort(key=lambda x: x[0])
    rng = np.random.RandomState(42)
    pos_pool = [r for r in rows if r[0] >= 0.515 and r[2] in r[1]]
    neg_pool = [r for r in rows if r[0] <= 0.435 and r[2] not in r[1]]
    pick = lambda pool: sorted([pool[i] for i in rng.choice(len(pool), min(10, len(pool)), replace=False)], key=lambda x: x[0])
    pos = pick(pos_pool)
    neg = pick(neg_pool)
    print("正例池", len(pos_pool), "负例池", len(neg_pool), "总对数", len(rows))
    import numpy as _np
    arr = _np.array([r[0] for r in rows])
    print("相似度分位 10/50/90", _np.percentile(arr, [10, 50, 90]).round(3), "≥0.515占比", (arr >= 0.515).mean().round(4), "≤0.435占比", (arr <= 0.435).mean().round(4))
    cases = [("正例（BGE高置信对齐样本）", pos, PRIMARY),
             ("负例（BGE低相似度负样本）", neg, ACCENT)]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 8.2))
    for ax, (title, data, color) in zip(axes, cases):
        ax.set_xlim(0, 10.5); ax.set_ylim(-0.5, len(data) - 0.5)
        ax.axis("off")
        ax.set_title(title, fontproperties=CJK, fontsize=11, pad=10)
        for i, (sim, skill, concept) in enumerate(reversed(data)):
            y = i
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.05, y - 0.40), 10.35, 0.80,
                boxstyle="round,pad=0.02", facecolor=color, alpha=0.12,
                edgecolor=color, linewidth=0.75))
            ax.text(0.2, y, _wrap(skill, 15), va="center",
                    fontproperties=CJK, fontsize=7.5)
            ax.text(4.85, y, "→", va="center", ha="center", fontsize=10, color="#555555")
            ax.text(5.30, y, _wrap(concept, 10), va="center",
                    fontproperties=CJK, fontsize=7.8)
            ax.text(10.20, y, f"sim={sim:.3f}", va="center", ha="right",
                    fontsize=8, color=color)
        ax.axvline(4.55, color="#CCCCCC", linestyle="--", linewidth=0.8)
    fig.suptitle("技能-概念对齐正例与负例", fontproperties=CJK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp6b_alignment.png", bbox_inches="tight")
    plt.close(fig)
    print("fig_exp6b 落盘")


def fig_path_comparison():
    b     = torch.load(INTER / "s6_graph.pt", weights_only=False)
    s9    = np.load(INTER / "s9_skill_weight.npy")
    cons  = b["concepts"]
    prereq_adopted = load_prereq(b, "prereq_adopted")

    emb_supp = INTER / "嵌入_补充"
    p1 = (emb_supp / "exp5_t1_s42.pt") if (emb_supp / "exp5_t1_s42.pt").exists() else (EMB / "emb_t1_s42.pt")
    p3 = (emb_supp / "exp5_t3_s42.pt") if (emb_supp / "exp5_t3_s42.pt").exists() else (EMB / "emb_t3_s42.pt")
    emb1  = torch.load(p1, weights_only=False)["concept_emb"].float()
    emb3  = torch.load(p3, weights_only=False)["concept_emb"].float()

    rng  = np.random.RandomState(99)
    skilled = np.nonzero(s9 > 0.5)[0]
    seed_c = int(rng.choice(skilled))
    target_len = 8
    cov_weight = 0.0

    path1 = gen_path(seed_c, emb1, s9, prereq_adopted, target_len, cov_weight)
    path3 = gen_path(seed_c, emb3, s9, prereq_adopted, target_len, cov_weight)

    meta_out = Path("../结果") / "r_exp6c_path_meta.json"
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps({
        "seed_concept_idx": seed_c,
        "seed_concept_name": cons[seed_c],
        "sample_rng_seed": 99,
        "emb_seed": 42,
        "norm_mode": "none (unnormalized dot-product, isolating gate scaling effect)",
        "cov_weight_t1": cov_weight,
        "cov_weight_t3": cov_weight,
        "target_len": target_len,
        "prereq_constraint": "prereq_adopted (model input edges, isolated from eval_prereq_positive)",
        "path_t1": [{"step": i + 1, "idx": int(c), "name": cons[c], "weight": round(float(s9[c]), 4)} for i, c in enumerate(path1)],
        "path_t3": [{"step": i + 1, "idx": int(c), "name": cons[c], "weight": round(float(s9[c]), 4)} for i, c in enumerate(path3)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, path, cfg, col in [(ax1, path1, "T1 无技能锚定", GREY),
                                (ax3, path3, "T3 全技能锚定", PRIMARY)]:
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, len(path) + 0.5)
        ax.axis("off")
        ax.set_title(f"{cfg}  种子概念：{cons[seed_c][:8]}",
                     fontproperties=CJK, fontsize=10)
        for step, c in enumerate(path):
            name = cons[c][:12]
            w    = min(float(s9[c]), 5.0)
            alpha = 0.25 + 0.5 * w / 5.0
            y    = len(path) - 1 - step
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.05, y - 0.35), 1.0, 0.70,
                boxstyle="round,pad=0.04",
                facecolor=col, alpha=alpha,
                edgecolor=col, linewidth=0.8))
            ax.text(0.55, y, f"{step+1}. {name}", ha="center", va="center",
                    fontproperties=CJK, fontsize=8.5)
            ax.text(1.2, y, f"w={s9[c]:.1f}", va="center",
                    fontsize=7.5, color="#555555")
            if step < len(path) - 1:
                ax.annotate("", xy=(0.55, y - 0.38),
                            xytext=(0.55, y - 0.62),
                            arrowprops=dict(arrowstyle="-|>",
                                           color="#888888", lw=0.8))
    fig.suptitle("同一种子概念下 T1 与 T3 推荐路径对比（颜色深浅代表技能权重）",
                 fontproperties=CJK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exp6c_path_compare.png", bbox_inches="tight")
    plt.close(fig)
    print(f"fig_exp6c 落盘  种子概念={cons[seed_c]}")


if __name__ == "__main__":
    fig_skill_weight()
    fig_alignment_cases()
    fig_path_comparison()
    print("实验六三张可视化全部完成")
