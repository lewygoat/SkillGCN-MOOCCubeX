import argparse
import glob
import json
import statistics
from pathlib import Path

RESULTS = Path("../结果")
SEED_DIR = Path("../结果/supp_runs")


def ms(v):
    return {"mean": statistics.mean(v),
            "std": statistics.stdev(v) if len(v) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["t2", "t3"], default="t3")
    args = ap.parse_args()

    files = sorted(glob.glob(str(SEED_DIR / f"r_skillgcn_{args.config}_gate_s*.json")))
    per_seed = []
    for f in files:
        per_seed.append(json.loads(Path(f).read_text(encoding="utf-8")))
    per_seed.sort(key=lambda d: d["seed"])
    if not per_seed:
        print("无逐种子文件，先运行 probe_gate.py")
        return

    gates = [d["gate_final_best_ckpt"] for d in per_seed]
    tests = [d["test_NDCG@10"] for d in per_seed]
    vals = [d["val_best"] for d in per_seed]
    eps = [d["epochs_ran"] for d in per_seed]

    summary = {
        "gate_final_best_ckpt": ms(gates),
        "test_NDCG@10": ms(tests),
        "val_best": ms(vals),
        "epochs_ran": ms(eps),
    }
    payload = {
        "method": f"SkillAnchoredLightGCN-{args.config.upper()}-gateprobe",
        "config": args.config,
        "seeds": [d["seed"] for d in per_seed],
        "gate_init": 0.0,
        "per_seed": [{"seed": d["seed"],
                      "gate_final_best_ckpt": d["gate_final_best_ckpt"],
                      "gate_trajectory": d["gate_trajectory"],
                      "test_NDCG@10": d["test_NDCG@10"],
                      "val_best": d["val_best"],
                      "epochs_ran": d["epochs_ran"]} for d in per_seed],
        "summary": summary,
        "hyper": per_seed[0]["hyper"],
        "note": "逐种子独立探测后聚合",
    }
    out = SEED_DIR / f"r_skillgcn_{args.config}_gate.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    g = summary["gate_final_best_ckpt"]
    t = summary["test_NDCG@10"]
    print(f"聚合{len(per_seed)}个种子  "
          f"gate_final={g['mean']:.4f}±{g['std']:.4f}  "
          f"test_NDCG@10={t['mean']:.5f}±{t['std']:.5f}")
    for d in per_seed:
        print(f"  seed{d['seed']}  gate={d['gate_final_best_ckpt']:.4f}  "
              f"epochs={d['epochs_ran']}  test={d['test_NDCG@10']:.5f}  "
              f"val={d['val_best']:.5f}")
    print(f"落盘 {out}")


if __name__ == "__main__":
    main()
