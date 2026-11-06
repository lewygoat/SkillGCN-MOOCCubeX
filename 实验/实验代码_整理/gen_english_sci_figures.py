# -*- coding: utf-8 -*-
"""
Generate 5 full-width multi-panel composite figures (Fig. 1 to Fig. 5, each with (a)-(b) or (a)-(c) subfigures)
at 300 DPI for the English SCI manuscript, with strict non-overlapping typography and margins.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent.parent
INTER = ROOT / "实验" / "中间产物"
RES = ROOT / "实验" / "结果"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300

PRIMARY = "#2C5F8A"
BASE_GRAY = "#9FB4C4"
ACCENT = "#E1743B"
MID = "#5B8CB8"
LIGHT = "#B8D0E0"
GRID = "#DDDDDD"
GREEN = "#3A8E5B"
PURPLE = "#7A5195"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(labelsize=8.6)


def fig1_framework_and_kg():
    """
    Fig. 1: Multi-panel (a) Overall four-stage architecture (s1-s9) and
    (b) Log-scale edge distribution of modeling KG vs. isolated evaluation benchmarks.
    """
    fig = plt.figure(figsize=(15.8, 5.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.55, 1.0], wspace=0.33)

    # Panel (a): Architecture Diagram
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 14.55)
    ax.set_ylim(0, 5.85)
    ax.axis("off")
    ax.set_title("(a) Four-Stage Skill-Anchored Graph Collaborative Filtering & Path Auditing Architecture",
                 fontsize=9.8, fontweight="bold", color="#1a365d", pad=8)

    stages = [
        (0.12, 0.20, 3.30, 5.36, "#F4F7FA", "#2C5F8A", "Stage 1: Data & Alignment\n(Pipeline s1-s9)"),
        (3.76, 0.20, 3.36, 5.36, "#F4F7FA", "#2C5F8A", "Stage 2: Skill-Anchored\nLightGCN & Gate Variants"),
        (7.46, 0.20, 3.30, 5.36, "#F4F7FA", "#2C5F8A", "Stage 3: Topology-Isolated\nPath Generation"),
        (11.10, 0.20, 3.30, 5.36, "#F4F7FA", "#E1743B", "Stage 4: Multi-Perspective\nDiagnostic Audit"),
    ]
    for x, y, w, h, fc, ec, title in stages:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06",
            facecolor=fc, edgecolor=ec, linewidth=1.2))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=ec)

    def add_box(x, y, w, h, text, fc="#FFFFFF", ec="#5B8CB8", fs=6.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor=ec, linewidth=0.85))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#222222", linespacing=1.26)

    # Stage 1 inner boxes (x=0.27, w=3.00)
    add_box(0.27, 3.82, 3.00, 0.90,
            "Occupational Skill Standards (760)\n• 1+X Web Frontend (257 reqs)\n• Software Designer (256 reqs)\n• 1+X Big Data App. (247 reqs)",
            fc="#EBF2F7", ec=PRIMARY)
    add_box(0.27, 2.69, 3.00, 0.90,
            "BGE-small-zh-v1.5 Alignment\n• Noun-phrase extraction (512-d)\n• Top-6 align (4,560; 13,680 pool)\n• Dual-rater κw = 0.7501 (247 pairs)",
            fc="#FFFFFF", ec=MID)
    add_box(0.27, 1.56, 3.00, 0.90,
            "4-Relation Course KG (576 Nodes)\n• Prereq Adopted (194)\n• Prereq Predicted (50,784)\n• Containment (617) & Weak (15,444)",
            fc="#EBF2F7", ec=PRIMARY)
    add_box(0.27, 0.40, 3.00, 0.93,
            "Learner Funnel (27,276 Core)\n• Watch ≥ 0.55 & 3-iter k-core\n• 26,535 test-eval / 10,748 cold\n• 7:1:2 split (8.28% tr / 11.58% all)",
            fc="#FFFFFF", ec=MID)

    # Stage 2 inner boxes (x=3.91, w=3.06)
    add_box(3.91, 3.82, 3.06, 0.90,
            "Skill-Anchored Concept Init.\n• T1 (No skill): g = 0\n• T2 (Shuffled w_i) / T3 (True w_i)\n  ẽ_i^(0) = e_i^(0)·(1 + φ(g)·w_i)",
            fc="#EBF2F7", ec=PRIMARY)
    add_box(3.91, 2.61, 3.06, 0.98,
            "Gate Parameterizations φ(g)\n• Unbounded: φ(g)=g & Gate L2-Reg\n• Bounded Sigmoid: φ(g) = σ(g)\n• Bounded Tanh: φ(g) = tanh(g)\n• In-training L2: L2Norm(ẽ_i^(0))",
            fc="#FFF5EE", ec=ACCENT)
    add_box(3.91, 1.50, 3.06, 0.88,
            "Linear Graph Convolution\n• L-layer symmetric Laplacian\n• Layer-mean readout e_u, e_i\n• Pairwise BPR + L2 reg. loss",
            fc="#FFFFFF", ec=MID)
    add_box(3.91, 0.40, 3.06, 0.87,
            "Cold-Start Fallback (10,748)\n• 6-baseline non-parametric audit\n• Skill-Prior TopoSort (3,412 voc.)",
            fc="#EBF7EE", ec=GREEN)

    # Stage 3 inner boxes (x=7.61, w=3.00)
    add_box(7.61, 3.82, 3.00, 0.90,
            "Topological Isolation Guard\n• Generation uses ONLY\n  prereq_adopted (194 edges)\n• Disjoint from eval_prereq_pos\n  (1,075 DAG pairs)",
            fc="#EBF2F7", ec=PRIMARY)
    add_box(7.61, 2.62, 3.00, 0.96,
            "Similarity Geometry Modes\n• Unnormalized dot product:\n  sim(u, v) = <e_u, e_v>\n• L2-normalized cosine:\n  sim(u, v) = <ê_u, ê_v>",
            fc="#FFFFFF", ec=MID)
    add_box(7.61, 1.42, 3.00, 0.96,
            "Greedy Dual-Objective Search\nScore(v|v_t) = (1-cw)sim + cw·w_v\n• Hard mask on E_adopt (194)\n• TopoSort cold-start ordering",
            fc="#FFF5EE", ec=ACCENT)
    add_box(7.61, 0.40, 3.00, 0.79,
            "Output Path P = (v_1, ..., v_10)\n30 seed concepts × 5 seeds\n(150 paths per setting)",
            fc="#FFFFFF", ec=MID)

    # Stage 4 inner boxes (x=11.25, w=3.00)
    add_box(11.25, 3.82, 3.00, 0.90,
            "1. Multi-Cutoff Ranking\n• 7 Baselines, K ∈ {5, 10, 15, 20}\n• 3 Sparsity strata\n• 11 Hyperparameter configs",
            fc="#FFFFFF", ec=ACCENT)
    add_box(11.25, 2.69, 3.00, 0.90,
            "2. 5-Seed Gate Attribution\n• Trajectory probing (g*=3.80)\n• Fixed-g* retraining:\n  83.56% static vs. 16.44% drift",
            fc="#FFF5EE", ec=ACCENT)
    add_box(11.25, 1.56, 3.00, 0.90,
            "3. 5D Path & Zero-Emb Audit\n• Cov, Viol_iso, Viol_weak,\n  FViol, Bloom Smoothness\n• Zero-Emb baseline (e_i ≡ 0)",
            fc="#FFFFFF", ec=ACCENT)
    add_box(11.25, 0.40, 3.00, 0.93,
            "4. Bounded Rec. & Cold-Start\n• Bounded Sigmoid: +72.5% rec.\n• Cold Voc.: +68.2% NDCG@10\n• Skill Cov@10: 2.81× Popularity",
            fc="#EBF7EE", ec=GREEN)

    arrow_kw = dict(arrowstyle="-|>,head_width=0.22,head_length=0.28", color="#2C5F8A", lw=1.4)
    for x0, x1, y0 in [(3.50, 3.70, 2.86), (7.20, 7.40, 2.86), (10.84, 11.04, 2.86)]:
        ax.annotate("", xy=(x1, y0), xytext=(x0, y0), arrowprops=arrow_kw)

    # Panel (b): Knowledge Graph Edge Scale & SCC Properties
    ax2 = fig.add_subplot(gs[1])
    edges = [
        ("Prereq-Predicted\n(Train KG, SCC=542)", 50784, PRIMARY),
        ("Chapter Co-occur.\n(Weak, SCC=494)", 15444, MID),
        ("Eval Prereq-Neg.\n(Isolated Benchmark)", 2011, ACCENT),
        ("Eval Prereq-Pos.\n(Isolated DAG, SCC=1)", 1075, "#E8986B"),
        ("Taxonomy Contain.\n(103 Categories)", 617, "#7BA3C9"),
        ("Prereq-Adopted\n(Path Constr., SCC=11)", 194, GREEN),
    ]
    names = [e[0] for e in edges]
    counts = [e[1] for e in edges]
    cols = [e[2] for e in edges]
    y = np.arange(len(names))
    ax2.barh(y, counts, color=cols, edgecolor="#333333", linewidth=0.75, height=0.58)
    for yi, c in zip(y, counts):
        ax2.text(c * 1.12, yi, f"{c:,}", va="center", ha="left", fontsize=8.0, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Directed Pairs / Edges (Log Scale)", fontsize=9.0)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=7.9)
    ax2.set_xlim(100, 280000)
    ax2.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()
    ax2.set_title("(b) Modeling KG vs. Isolated Evaluation Pairs",
                  fontsize=9.8, fontweight="bold", color="#1a365d", pad=8)
    style_axis(ax2)

    fig.subplots_adjust(left=0.01, right=0.985, top=0.90, bottom=0.12, wspace=0.33)
    fig.savefig(OUT / "fig1_framework_and_kg_en.png", bbox_inches="tight")
    fig.savefig(OUT / "fig1_framework_and_kg_en.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_baselines_and_topk():
    """
    Fig. 2: Multi-panel (a) 7-baseline NDCG@10 bar chart, (b) Top-K NDCG@K curves,
    and (c) Top-K HR@K curves across K in {5, 10, 15, 20} with a non-overlapping K=10 zoom inset.
    """
    comp = json.loads((RES / "supp_comprehensive_report.json").read_text(encoding="utf-8"))
    topk_data = comp["1_topk_metrics"]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14.8, 4.45))

    # Panel (a): 7 baselines at K=10
    baseline = [
        ("LightGCN", 0.17640, 0.00035),
        ("BPR-MF", 0.17102, 0.00129),
        ("NGCF", 0.16927, 0.00075),
        ("Most-Pop", 0.16554, 0.00000),
        ("KGAT", 0.16550, 0.00085),
        ("ConceptGCN", 0.16411, 0.00133),
        ("ACKRec", 0.16263, 0.00381),
    ]
    names = [b[0] for b in baseline]
    vals = [b[1] for b in baseline]
    errs = [b[2] for b in baseline]
    colors = [PRIMARY if n == "LightGCN" else BASE_GRAY for n in names]
    x = np.arange(len(names))
    ax0.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.75, width=0.58)
    for xi, (v, err) in enumerate(zip(vals, errs)):
        if err > 0:
            ax0.errorbar(xi, v, yerr=err, fmt="none", ecolor="#222222", elinewidth=0.85, capsize=2.8)
        ax0.text(xi, v + err + 0.00045, f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax0.set_ylim(0.155, 0.1825)
    ax0.set_ylabel("Test NDCG@10 (5-Seed Mean ± SD)", fontsize=9.0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(names, rotation=26, ha="right", fontsize=8.2)
    ax0.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax0.set_axisbelow(True)
    ax0.set_title("(a) Baseline Comparison at K = 10", fontsize=9.6, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax0)

    # Panels (b) and (c): Top-K NDCG and HR curves
    ks = [5, 10, 15, 20]
    methods = [
        ("LightGCN (T1)", "LightGCN", PRIMARY, "o", "-"),
        ("BPR-MF", "BPR-MF", GREEN, "s", "--"),
        ("NGCF", "NGCF", MID, "^", "-."),
        ("T2 (Shuffled)", "T2-权重打乱", PURPLE, "D", ":"),
        ("T3 (Unbounded)", "T3-全锚定", ACCENT, "v", "-"),
        ("KGAT", "KGAT", "#7BA3C9", "P", "--"),
        ("ConceptGCN", "ConceptGCN", "#A3B18A", "X", "-."),
        ("ACKRec", "ACKRec", "#888888", "*", ":"),
    ]
    hr10_zoom = []
    for label, key, col, mk, ls in methods:
        ndcgs = [topk_data[key][f"NDCG@{k}"]["mean"] for k in ks]
        hrs = [topk_data[key][f"HR@{k}"]["mean"] for k in ks]
        ax1.plot(ks, ndcgs, label=label, color=col, marker=mk, linestyle=ls,
                 linewidth=1.45, markersize=5.0, markeredgecolor="white", markeredgewidth=0.55)
        ax2.plot(ks, hrs, label=label, color=col, marker=mk, linestyle=ls,
                 linewidth=1.45, markersize=5.0, markeredgecolor="white", markeredgewidth=0.55)

    ax1.set_xlabel("Truncation Cutoff K", fontsize=9.0)
    ax1.set_ylabel("Test NDCG@K", fontsize=9.0)
    ax1.set_xticks(ks)
    ax1.set_ylim(0.122, 0.207)
    ax1.grid(True, color=GRID, linewidth=0.7)
    ax1.set_title("(b) NDCG@K Across Cutoffs K ∈ {5, 10, 15, 20}", fontsize=9.6, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax1)

    ax2.set_xlabel("Truncation Cutoff K", fontsize=9.0)
    ax2.set_ylabel("Test HR@K (Any-Hit Ratio)", fontsize=9.0)
    ax2.set_xticks(ks)
    ax2.set_ylim(0.41, 0.99)
    ax2.grid(True, color=GRID, linewidth=0.7)
    ax2.set_title("(c) Hit Ratio HR@K Across Cutoffs K ∈ {5, 10, 15, 20}", fontsize=9.6, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax2)

    handles, labels_leg = ax1.get_legend_handles_labels()
    fig.subplots_adjust(left=0.055, right=0.985, top=0.89, bottom=0.23, wspace=0.26)
    fig.legend(handles, labels_leg, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=8, frameon=False, fontsize=8.1)
    fig.savefig(OUT / "fig2_baselines_and_topk_en.png", bbox_inches="tight")
    fig.savefig(OUT / "fig2_baselines_and_topk_en.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_ablation_and_gate_dynamics():
    """
    Fig. 3: Multi-panel (a) 3-Tier ablation & bounded/L2-reg gate remedies,
    (b) 5-seed epoch-by-epoch gate trajectory with Val Peak & Early-Stop markers,
    (c) Fixed-gate sweep showing safe perturbation zone (g<=0.10) vs. checkpoint gate g*.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.0, 4.55))

    # Panel (a): Ablation & Structural Remedies (two-line labels ABOVE error bars)
    ablation = [
        ("T1 (No Skill)", 0.17640, 0.00035),
        ("T2 (Shuffled)", 0.16822, 0.00124),
        ("T3 (Unbounded)", 0.16557, 0.00116),
        ("L2-Train Norm", 0.16984, 0.00062),
        ("Gate L2-Reg", 0.17268, 0.00051),
        ("Bounded Tanh", 0.17314, 0.00049),
        ("Bounded Sigmoid", 0.17342, 0.00044),
    ]
    labels = [a[0] for a in ablation]
    vals = [a[1] for a in ablation]
    errs = [a[2] for a in ablation]
    shades = [PRIMARY, MID, LIGHT, "#7BA3C9", "#65A888", "#418B67", GREEN]
    x = np.arange(len(labels))
    ax0.bar(x, vals, yerr=errs, color=shades, edgecolor="#333333",
            linewidth=0.75, width=0.58, capsize=2.6, error_kw={"elinewidth": 0.8})
    base = vals[0]
    for xi, (v, err) in enumerate(zip(vals, errs)):
        if xi == 0:
            ax0.text(xi, v + err + 0.00045, f"{v:.4f}\n(Base)", ha="center", va="bottom",
                     fontsize=6.6, fontweight="bold", color="#1a365d", linespacing=1.12)
        else:
            dp = (v - base) / base * 100
            col_txt = "#B33A10" if xi <= 2 else "#1F5E3B"
            ax0.text(xi, v + err + 0.00045, f"{v:.4f}\n({dp:+.2f}%)", ha="center", va="bottom",
                     fontsize=6.5, fontweight="bold", color=col_txt, linespacing=1.12)
    ax0.axvline(2.5, color="#888888", linestyle="--", linewidth=0.9)
    ax0.set_xlim(-0.62, 6.62)
    ax0.set_ylim(0.156, 0.1848)
    ax0.set_ylabel("Test NDCG@10", fontsize=9.0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=26, ha="right", fontsize=7.8)
    ax0.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax0.set_axisbelow(True)
    ax0.set_title("(a) 3-Tier Ablation & Bounded Gate Remedies", fontsize=9.5, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax0)

    # Panel (b): 5-Seed Gate Trajectory (ylim raised to 15.8 so legend has large gap above curves)
    root = RES / "newS9_runs"
    styles = [(MID, "o", "-"), (PRIMARY, "s", "--"), ("#3E7CB1", "^", "-."),
              ("#7BA3C9", "D", ":"), (ACCENT, "v", "-")]
    for sd, (col, mk, ls) in zip(range(42, 47), styles):
        d = json.loads((root / f"run_t3_s{sd}.json").read_text(encoding="utf-8"))
        traj = d["gate_trajectory"]
        ax1.plot(range(1, len(traj) + 1), traj, color=col, marker=mk,
                 linestyle=ls, linewidth=1.35, markersize=4.0, label=f"Seed {sd} (g*={d['gate_final']:.2f})")
        best = int(np.argmin([abs(v - d["gate_final"]) for v in traj])) + 1
        ax1.scatter([best], [d["gate_final"]], s=65, facecolors="none",
                    edgecolors="#222222", linewidths=1.2, zorder=5)
        ax1.scatter([len(traj)], [traj[-1]], s=45, marker="x",
                    color="#B22222", linewidths=1.2, zorder=6)
    ax1.scatter([], [], s=65, facecolors="none", edgecolors="#222222",
                linewidths=1.2, label="Val Peak (best_epoch, g*)")
    ax1.scatter([], [], s=45, marker="x", color="#B22222",
                linewidths=1.2, label="Early-Stop Term. (patience=8)")
    ax1.set_xlabel("Training Epoch", fontsize=9.0)
    ax1.set_ylabel("Unbounded Scalar Gate g", fontsize=9.0)
    ax1.set_xlim(0, 15.2)
    ax1.set_ylim(0, 15.8)
    ax1.grid(True, color=GRID, linewidth=0.7)
    leg1 = ax1.legend(loc="upper left", frameon=True, framealpha=1.0,
                      facecolor="white", edgecolor="#DDDDDD", fontsize=6.8)
    leg1.set_zorder(10)
    ax1.set_title("(b) 5-Seed Unbounded Gate Drift Trajectory", fontsize=9.5, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax1)

    # Panel (c): Fixed-Gate Constant Sweep Curve (g in 0.05..4.0) & Learned-g* Lock Point
    g_vals = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00]
    ndcg_sweep = [0.17688, 0.17641, 0.17545, 0.17456, 0.17318, 0.17058, 0.16694]
    ax2.axvspan(0.0, 0.15, color=GREEN, alpha=0.14, label="Controllable Zone (g≤0.10, +0.27% at 0.05)")
    ax2.axhline(0.17640, color=GREEN, linestyle="--", linewidth=1.1, label="T1 Baseline (g=0, 0.17640)")
    ax2.plot(g_vals, ndcg_sweep, color=PRIMARY, marker="o", linewidth=1.6, markersize=5.2,
             markeredgecolor="white", markeredgewidth=0.6, zorder=4, label="Fixed Constant Gate g (5-Seed)")
    ax2.scatter([3.80], [0.16735], color=MID, marker="s", s=58, edgecolors="white", linewidths=0.7,
                zorder=6, label="Fixed Learned g*=3.80 (0.16735, -5.13%)")
    ax2.scatter([3.80], [0.16557], color=ACCENT, marker="^", s=64, edgecolors="white", linewidths=0.7,
                zorder=6, label="Learnable T3 g*=3.80 (0.16557, -6.14%)")
    ax2.annotate("83.56% Static Scaling\n16.44% Dynamic Drift", xy=(3.74, 0.16745), xytext=(1.45, 0.1743),
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1), fontsize=7.4, fontweight="bold", color=ACCENT)
    ax2.set_xlabel("Gate Scaling Magnitude g", fontsize=9.0)
    ax2.set_ylabel("Test NDCG@10", fontsize=9.0)
    ax2.set_xlim(-0.18, 4.28)
    ax2.set_ylim(0.1565, 0.1802)
    ax2.grid(True, color=GRID, linewidth=0.7)
    leg2 = ax2.legend(loc="lower left", frameon=True, framealpha=1.0,
                      facecolor="white", edgecolor="#DDDDDD", fontsize=6.7)
    leg2.set_zorder(10)
    ax2.set_title("(c) Static vs. Dynamic Loss Attribution", fontsize=9.5, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax2)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.89, bottom=0.22, wspace=0.26)
    fig.savefig(OUT / "fig3_ablation_and_gate_dynamics_en.png", bbox_inches="tight")
    fig.savefig(OUT / "fig3_ablation_and_gate_dynamics_en.pdf", bbox_inches="tight")
    plt.close(fig)


def fig4_robustness_and_sensitivity():
    """
    Fig. 4: Multi-panel (a) De-circularized cold-start (All 10,748 & Voc-Track 3,412) + Warm sparsity strata,
    (b) Embedding dimension & propagation layer sensitivity, (c) Learning rate sensitivity.
    """
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.0, 4.45))

    # Panel (a): Cold-Start Cohorts + Warm Sparsity Strata
    strata = [
        "Cold All\n(all tracks)\n(10,748)",
        "Cold Voc.\n(voc. track)\n(3,412)",
        "Warm Low\n[10, 20)\n(6,624)",
        "Warm Med\n[20, 50)\n(10,339)",
        "Warm High\n[50, +∞)\n(9,572)"
    ]
    v_base = [0.03879, 0.04036, 0.13281, 0.16079, 0.22343]
    v_skill = [0.04205, 0.06788, 0.12632, 0.15214, 0.20724]
    x = np.arange(len(strata))
    w = 0.35
    ax0.bar(x - w / 2, v_base, width=w, color=PRIMARY, edgecolor="#333333", linewidth=0.75, label="T1 (Warm) / Most-Pop (Cold)")
    ax0.bar(x + w / 2, v_skill, width=w, color=ACCENT, edgecolor="#333333", linewidth=0.75, label="T3 (Warm) / Skill-Topo (Cold)")
    deltas = ["+8.4%\n(HR+22%)", "+68.2%\n(HR+101%)", "-4.89%", "-5.38%", "-7.25%"]
    for xi, (vb, vs, dl) in enumerate(zip(v_base, v_skill, deltas)):
        top_y = max(vb, vs) + 0.0045
        col = GREEN if xi <= 1 else ACCENT
        ax0.text(xi, top_y, dl, ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=col)
    ax0.axvline(1.5, color="#888888", linestyle="--", linewidth=0.9)
    ax0.set_xlim(-0.58, 4.58)
    ax0.set_ylim(0, 0.275)
    ax0.set_ylabel("Test NDCG@10", fontsize=9.0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(strata, fontsize=6.9)
    ax0.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax0.set_axisbelow(True)
    leg0 = ax0.legend(loc="upper left", frameon=True, framealpha=1.0,
                      facecolor="white", edgecolor="#DDDDDD", fontsize=6.9)
    leg0.set_zorder(10)
    ax0.set_title("(a) Cold-Start Gain vs. Warm Sparsity Strata", fontsize=9.5, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax0)

    # Panel (b): Embedding Dimension d & Layers L
    dims = ["d=32", "d=64*", "d=128", "d=256", "L=1", "L=2*", "L=3", "L=4"]
    t1_dl = [0.17412, 0.17649, 0.17688, 0.17704, 0.17520, 0.17649, 0.17284, 0.16811]
    t3_dl = [0.16285, 0.16679, 0.16724, 0.16763, 0.16512, 0.16679, 0.16548, 0.16432]
    x1 = np.arange(len(dims))
    ax1.plot(x1[:4], t1_dl[:4], color=PRIMARY, marker="o", linewidth=1.5, label="T1 (Dim d)")
    ax1.plot(x1[:4], t3_dl[:4], color=ACCENT, marker="s", linestyle="--", linewidth=1.5, label="T3 (Dim d)")
    ax1.plot(x1[4:], t1_dl[4:], color=PRIMARY, marker="^", linewidth=1.5, label="T1 (Layers L)")
    ax1.plot(x1[4:], t3_dl[4:], color=ACCENT, marker="v", linestyle="--", linewidth=1.5, label="T3 (Layers L)")
    ax1.axvline(3.5, ymin=0.25, ymax=0.98, color="#888888", linestyle=":", linewidth=1.0)
    ax1.text(1.5, 0.1803, "Embedding Dim d", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#444444")
    ax1.text(5.5, 0.1803, "Propagation Depth L", ha="center", va="center", fontsize=7.3, fontweight="bold", color="#444444")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(dims, fontsize=7.6)
    ax1.set_xlim(-0.45, 7.45)
    ax1.set_ylim(0.1515, 0.1825)
    ax1.set_ylabel("Test NDCG@10 (Seed 42)", fontsize=9.0)
    ax1.grid(True, color=GRID, linewidth=0.7)
    leg1 = ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2,
                      columnspacing=1.2, handlelength=1.8, frameon=True, framealpha=1.0,
                      facecolor="white", edgecolor="#DDDDDD", fontsize=6.9)
    leg1.set_zorder(10)
    ax1.set_title("(b) Sensitivity to Dimension d & Depth L", fontsize=9.5, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax1)

    # Panel (c): Learning Rate eta
    lrs = ["0.001", "0.005", "0.01*", "0.05", "0.10"]
    t1_lr = [0.17395, 0.17618, 0.17649, 0.17421, 0.17108]
    t3_lr = [0.16482, 0.16654, 0.16679, 0.15654, 0.15188]
    x2 = np.arange(len(lrs))
    ax2.plot(x2, t1_lr, color=PRIMARY, marker="o", linewidth=1.6, markersize=5, label="T1 (No Skill Layer)")
    ax2.plot(x2, t3_lr, color=ACCENT, marker="s", linestyle="--", linewidth=1.6, markersize=5, label="T3 (Unbounded Gate)")
    for xi, (v1, v3) in enumerate(zip(t1_lr, t3_lr)):
        gap = (v3 - v1) / v1 * 100
        ax2.text(xi, v3 - 0.0020, f"{gap:.1f}%", ha="center", va="top", fontsize=7.4, color=ACCENT, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(lrs, fontsize=8.2)
    ax2.set_xlim(-0.48, 4.52)
    ax2.set_xlabel("Learning Rate η (* denotes default)", fontsize=9.0)
    ax2.set_ylabel("Test NDCG@10 (Seed 42)", fontsize=9.0)
    ax2.set_ylim(0.144, 0.1815)
    ax2.grid(True, color=GRID, linewidth=0.7)
    leg2 = ax2.legend(loc="lower left", frameon=True, framealpha=1.0,
                      facecolor="white", edgecolor="#DDDDDD", fontsize=7.5)
    leg2.set_zorder(10)
    ax2.set_title("(c) Sensitivity to Learning Rate η", fontsize=9.5, fontweight="bold", color="#1a365d", pad=7)
    style_axis(ax2)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.89, bottom=0.19, wspace=0.26)
    fig.savefig(OUT / "fig4_robustness_and_sensitivity_en.png", bbox_inches="tight")
    fig.savefig(OUT / "fig4_robustness_and_sensitivity_en.pdf", bbox_inches="tight")
    plt.close(fig)


def fig5_interpretability_multi():
    """
    Fig. 5: Multi-panel (a) Exact top-15 skill-weighted concepts from s9_skill_weight.npy,
    (b) BGE-small-zh-v1.5 alignment cases from appendix_bge_audit_247pairs.csv,
    (c) 8-step learning path comparison (T1 vs T3) with generous two-line box padding.
    """
    fig = plt.figure(figsize=(16.0, 5.9))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.06, 1.70, 1.24], wspace=0.18)

    # Panel (a): Top-15 concepts by s9
    ax0 = fig.add_subplot(gs[0])
    s9 = np.load(INTER / "s9_skill_weight.npy")
    idx = np.argsort(s9)[::-1][:15]
    vals = s9[idx]
    en_names = [
        "Data Structure (id=0)",
        "Service Data Unit (id=1)",
        "Data Visualization (id=91)",
        "Big Data (id=2)",
        "Data (id=3)",
        "Data Network (id=4)",
        "Data Type (id=5)",
        "Image Data (id=6)",
        "Data Encryption (id=52)",
        "Object-Oriented Prog. (id=7)",
        "Computer System (id=8)",
        "Data Model (id=87)",
        "Data Service (id=9)",
        "Data Integrity (id=10)",
        "Network Architecture (id=11)"
    ]
    colors = [ACCENT if "Data" in n else PRIMARY for n in en_names]
    y = np.arange(len(en_names))
    ax0.barh(y, vals, color=colors, edgecolor="#333333", linewidth=0.6, height=0.66)
    for i, v in enumerate(vals):
        ax0.text(v + 0.20, i, f"{v:.2f}", va="center", fontsize=7.1)
    ax0.set_yticks(y)
    ax0.set_yticklabels(en_names, fontsize=7.3)
    ax0.invert_yaxis()
    ax0.set_xlabel("Raw Skill Weight (s9)", fontsize=8.8)
    ax0.set_xlim(0, max(vals) * 1.25)
    ax0.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax0.set_axisbelow(True)
    p1 = mpatches.Patch(color=ACCENT, label="Literal 'Data' (12/15; 25.8% -> 6.0%)")
    p2 = mpatches.Patch(color=PRIMARY, label="Other Core CS Concepts (3/15)")
    ax0.legend(handles=[p1, p2], loc="upper center", bbox_to_anchor=(0.46, -0.13),
               frameon=True, framealpha=1.0, facecolor="white", edgecolor="#DDDDDD", fontsize=6.9)
    ax0.set_title("(a) Top-15 Weighted Concepts", fontsize=9.6, fontweight="bold", color="#1a365d", pad=8)
    style_axis(ax0)

    # Panel (b): Stratified Alignment Cases from appendix_bge_audit_247pairs.csv (6 Pos + 6 Neg)
    ax1 = fig.add_subplot(gs[1])
    cases = [
        (0.551, "[BigData P184] Visual report & storage", "Data Visualization", PRIMARY),
        (0.541, "[BigData P181] Engine & data encryption", "Data Encryption", PRIMARY),
        (0.525, "[1+X Web P007] Bezier curve modular UI", "Bezier Curve", PRIMARY),
        (0.525, "[Soft.Des P116] Rasterization schedule", "Rasterization", PRIMARY),
        (0.525, "[Soft.Des P123] Gaussian mixture model", "Gaussian Mixture", PRIMARY),
        (0.521, "[1+X Web P004] Cursor async event bind", "Cursor", PRIMARY),
        (0.443, "[BigData P179] Mining & service check", "Data Center", ACCENT),
        (0.429, "[Soft.Des P141] Four-way wavehand spec", "Third Normal Form", ACCENT),
        (0.428, "[1+X Web P075] Code review pipeline", "Data Visualization", ACCENT),
        (0.427, "[1+X Web P021] Circuit-breaker UI ctrl", "Dynamic Linking", ACCENT),
        (0.426, "[Soft.Des P145] Von Neumann formal spec", "Static Single Assign", ACCENT),
        (0.421, "[1+X Web P079] Markov packet parsing", "Query Optimization", ACCENT),
    ]
    ax1.set_xlim(0, 12.2)
    ax1.set_ylim(-0.65, len(cases) - 0.15)
    ax1.axis("off")
    ax1.set_title("(b) BGE Audit Positives (Blue) vs. Negatives (Orange)", fontsize=9.6, fontweight="bold", color="#1a365d", pad=8)
    for i, (sim, sk, con, col) in enumerate(reversed(cases)):
        ax1.add_patch(mpatches.FancyBboxPatch(
            (0.05, i - 0.34), 12.05, 0.68,
            boxstyle="round,pad=0.02", facecolor=col, alpha=0.11,
            edgecolor=col, linewidth=0.65))
        ax1.text(0.22, i, sk, va="center", ha="left", fontsize=6.6, color="#222222")
        ax1.text(6.65, i, "→", va="center", ha="center", fontsize=8.4, color="#555555")
        ax1.text(7.00, i, con, va="center", ha="left", fontsize=6.8, fontweight="bold", color="#222222")
        ax1.text(11.95, i, f"{sim:.3f}", va="center", ha="right", fontsize=7.0, fontweight="bold", color=col)
    ax1.axhline(5.5, color="#888888", linestyle="--", linewidth=0.85)

    # Panel (c): 8-Step Path Comparison (two-line box labels + clean arrow clearance)
    ax2 = fig.add_subplot(gs[2])
    t1_steps = [
        ("1. Distributed Lock", 0.95),
        ("2. Counter", 0.72),
        ("3. Compare & Swap", 0.60),
        ("4. Decision Tree", 0.54),
        ("5. Gaussian Mixture", 0.79),
        ("6. AVL Tree", 0.49),
        ("7. MapReduce Model", 0.58),
        ("8. Floating-Point", 0.83),
    ]
    t3_steps = [
        ("1. Distributed Lock", 0.95),
        ("2. Data Structure", 14.99),
        ("3. Deterministic Alg.", 4.76),
        ("4. Database Trigger", 0.54),
        ("5. Bipartite Match", 0.75),
        ("6. Component Dev.", 0.50),
        ("7. Normal Dist.", 1.01),
        ("8. Data", 9.27),
    ]
    ax2.set_xlim(-0.08, 2.52)
    ax2.set_ylim(-0.55, 8.65)
    ax2.axis("off")
    ax2.set_title("(c) 8-Step Path (Seed id=148, cw=0.0, T1 vs. T3)", fontsize=9.4, fontweight="bold", color="#1a365d", pad=8)
    ax2.text(0.56, 8.32, "T1: Coherent Path", ha="center", fontsize=7.6, fontweight="bold", color=PRIMARY)
    ax2.text(1.88, 8.32, "T3: Norm-Biased Path", ha="center", fontsize=7.6, fontweight="bold", color=ACCENT)

    for col_x, steps, col in [(0.00, t1_steps, PRIMARY), (1.32, t3_steps, ACCENT)]:
        cx = col_x + 0.56
        for i, (name, w) in enumerate(steps):
            y = 7.45 - i * 1.04
            alpha = 0.20 + 0.50 * min(w / 15.0 + 0.25, 1.0)
            ax2.add_patch(mpatches.FancyBboxPatch(
                (col_x, y - 0.30), 1.12, 0.60,
                boxstyle="round,pad=0.03", facecolor=col, alpha=alpha,
                edgecolor=col, linewidth=0.85))
            ax2.text(cx, y + 0.10, name, ha="center", va="center",
                     fontsize=6.3, fontweight="bold", color="#111111")
            ax2.text(cx, y - 0.13, f"(s9 = {w:.2f})", ha="center", va="center",
                     fontsize=5.9, color="#222222")
            if i < len(steps) - 1:
                ax2.annotate("", xy=(cx, y - 0.69), xytext=(cx, y - 0.35),
                             arrowprops=dict(arrowstyle="-|>,head_width=0.17,head_length=0.20", color="#555555", lw=0.85))

    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.18, wspace=0.18)
    fig.savefig(OUT / "fig5_interpretability_multi_en.png", bbox_inches="tight")
    fig.savefig(OUT / "fig5_interpretability_multi_en.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_framework_and_kg()
    fig2_baselines_and_topk()
    fig3_ablation_and_gate_dynamics()
    fig4_robustness_and_sensitivity()
    fig5_interpretability_multi()
    print("Generated 5 composite multi-panel figures (Fig. 1 to Fig. 5) in", OUT)
