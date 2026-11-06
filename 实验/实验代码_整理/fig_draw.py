from pathlib import Path
from fig_common import (plt, CJK, OUT, BASELINE, ABLATION, GATE_SEED,
                        GRAPH_EDGES)
import numpy as np

PRIMARY = "#2C5F8A"
BASE_GRAY = "#9FB4C4"
ACCENT = "#E1743B"
MID = "#5B8CB8"
LIGHT = "#B8D0E0"
GRID = "#DDDDDD"


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(labelsize=10)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontproperties(CJK)


def fig_baseline():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [b[0] for b in BASELINE]
    vals = [b[1] for b in BASELINE]
    errs = [b[2] for b in BASELINE]
    colors = [PRIMARY if n == "LightGCN" else BASE_GRAY for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, vals, yerr=errs, color=colors, edgecolor="#333333",
                  linewidth=0.8, width=0.62, capsize=3,
                  error_kw={"elinewidth": 0.8})
    ax.set_ylim(0.155, 0.1815)
    ax.set_ylabel("NDCG@10", fontproperties=CJK, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontproperties=CJK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for xi, (v, err) in enumerate(zip(vals, errs)):
        ax.text(xi, v + err + 0.0004, f"{v:.4f}", ha="center", va="bottom",
                fontsize=8, fontproperties=CJK)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_baseline.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    labels = [f"{a[0]}\n{a[1]}" for a in ABLATION]
    vals = [a[2] for a in ABLATION]
    errs = [a[3] for a in ABLATION]
    shades = [PRIMARY, MID, LIGHT]
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=shades, edgecolor="#333333",
           linewidth=0.8, width=0.55, capsize=3,
           error_kw={"elinewidth": 0.8})
    base = vals[0]
    for xi, (v, err) in enumerate(zip(vals, errs)):
        ax.text(xi, v + err + 0.0005, f"{v:.4f}", ha="center", va="bottom",
                fontsize=9, fontproperties=CJK)
        if xi > 0:
            dp = (v - base) / base * 100
            ax.text(xi, 0.1605, f"{dp:+.2f}%", ha="center", va="center",
                    fontsize=9, color=ACCENT, fontproperties=CJK)
    ax.set_ylim(0.155, 0.1815)
    ax.set_ylabel("NDCG@10", fontproperties=CJK, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=CJK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_gate_traj():
    import json
    root = Path(__file__).resolve().parent.parent / "结果" / "newS9_runs"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    styles = [(MID, "o", "-"), (PRIMARY, "s", "--"), ("#3E7CB1", "^", "-."),
              ("#7BA3C9", "D", ":"), (ACCENT, "v", "-")]
    for sd, (col, mk, ls) in zip(range(42, 47), styles):
        d = json.loads((root / f"run_t3_s{sd}.json").read_text(encoding="utf-8"))
        traj = d["gate_trajectory"]
        ax.plot(range(1, len(traj) + 1), traj, color=col, marker=mk,
                linestyle=ls, linewidth=1.4, markersize=4.5, label=f"种子{sd}")
        best = int(np.argmin([abs(v - d["gate_final"]) for v in traj])) + 1
        ax.scatter([best], [d["gate_final"]], s=70, facecolors="none",
                   edgecolors="#333333", linewidths=1.2, zorder=5)
    ax.set_xlabel("训练轮次", fontproperties=CJK, fontsize=11)
    ax.set_ylabel("门控参数 g", fontproperties=CJK, fontsize=11)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=5,
              frameon=False, prop=CJK, fontsize=9)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_gate_traj.png", bbox_inches="tight")
    plt.close(fig)


def fig_gate_final():
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    seeds = [f"种子{g[0]}" for g in GATE_SEED]
    gates = [g[1] for g in GATE_SEED]
    x = np.arange(len(seeds))
    ax.bar(x, gates, color=PRIMARY, edgecolor="#333333", linewidth=0.8,
           width=0.58)
    for xi, v in zip(x, gates):
        ax.text(xi, v + 0.1, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9, fontproperties=CJK)
    mean = float(np.mean(gates))
    ax.axhline(mean, color=MID, linestyle="--", linewidth=1.1)
    ax.text(3.0, mean + 0.16, f"五种子均值 {mean:.2f}", ha="center", color=MID, fontsize=9,
            fontproperties=CJK)
    ax.set_ylabel("门控最优值", fontproperties=CJK, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontproperties=CJK)
    ax.set_ylim(0, 7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_gate_final.png", bbox_inches="tight")
    plt.close(fig)


def fig_graph_edges():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    names = [g[0] for g in GRAPH_EDGES]
    counts = [g[1] for g in GRAPH_EDGES]
    y = np.arange(len(names))
    shades = [PRIMARY, MID, "#7BA3C9", LIGHT]
    ax.barh(y, counts, color=shades, edgecolor="#333333", linewidth=0.8,
            height=0.6)
    for yi, c in zip(y, counts):
        ax.text(c * 1.03, yi, f"{c:,}", va="center", ha="left",
                fontsize=9, fontproperties=CJK)
    ax.set_xscale("log")
    ax.set_xlabel("边数量（对数轴）", fontproperties=CJK, fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontproperties=CJK)
    ax.set_xlim(100, 120000)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_graph_edges.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_baseline()
    fig_ablation()
    fig_gate_traj()
    fig_gate_final()
    fig_graph_edges()
    print("五图全部完成")
