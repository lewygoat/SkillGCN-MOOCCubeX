import json
import statistics
from pathlib import Path

RUNS = Path("../结果/newS9_runs")
RESULTS = Path("../结果")
METRICS = ["NDCG@10", "HR@10", "MRR", "Recall@20", "Precision@10"]


def ms(v):
    return {"mean": statistics.mean(v),
            "std": statistics.stdev(v) if len(v) > 1 else 0.0}


def load_runs(cfg):
    runs = []
    for sd in [42, 43, 44, 45, 46]:
        f = RUNS / f"run_{cfg}_s{sd}.json"
        if f.exists():
            runs.append(json.loads(f.read_text(encoding="utf-8")))
    return runs


def build_ablation(cfg, note):
    runs = load_runs(cfg)
    if not runs:
        return None
    per_seed = [r["test"] for r in runs]
    summary = {}
    for mt in METRICS:
        vals = [p[mt] for p in per_seed if mt in p]
        if vals:
            summary[mt] = ms(vals)
    out = {
        "method": f"SkillAnchoredLightGCN-{cfg.upper()}",
        "tag": f"skillgcn_{cfg}_newS9",
        "seeds": [r["seed"] for r in runs],
        "per_seed": per_seed,
        "summary": summary,
        "s9_source": "BAAI/bge-v1.5 cosine alignment via s8_align.py & s9_gen.py",
        "note": note,
    }
    (RESULTS / f"r_skillgcn_{cfg}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_gate():
    runs = load_runs("t3")
    if not runs:
        return None
    per_seed = [{"seed": r["seed"], "gate_final": r["gate_final"],
                 "epochs_ran": r["epochs_ran"],
                 "test_NDCG@10": r["test"]["NDCG@10"],
                 "val_best": r["val_best_NDCG@10"],
                 "gate_trajectory": r["gate_trajectory"]} for r in runs]
    gates = [p["gate_final"] for p in per_seed]
    out = {
        "method": "SkillAnchoredLightGCN-T3-gate-newS9",
        "seeds": [p["seed"] for p in per_seed],
        "per_seed": per_seed,
        "summary": {"gate_final": ms(gates),
                    "test_NDCG@10": ms([p["test_NDCG@10"] for p in per_seed])},
        "s9_source": "BAAI/bge-v1.5 cosine alignment via s8_align.py & s9_gen.py",
    }
    (RESULTS / "r_skillgcn_t3_gate.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out["summary"]


def main():
    print("多随机种子消融结果聚合")
    for cfg, note in [("t1", "无技能层，门控恒零，s9无作用"),
                      ("t2", "技能层接入，权重随机打乱"),
                      ("t3", "跨层对齐与锚定齐备")]:
        s = build_ablation(cfg, note)
        if s:
            n = s.get("NDCG@10", {})
            print(f"  {cfg}: NDCG@10 {n.get('mean', 0):.5f}±{n.get('std', 0):.5f}  "
                  f"种子数{len(load_runs(cfg))}")
        else:
            print(f"  {cfg}: 无run文件")
    g = build_gate()
    if g:
        print(f"  gate: {g['gate_final']['mean']:.3f}±{g['gate_final']['std']:.3f}")


if __name__ == "__main__":
    main()
