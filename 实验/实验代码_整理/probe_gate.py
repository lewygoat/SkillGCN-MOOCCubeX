import common  # Auto-unpacks s6_graph.pt.gz on first run
import argparse
import json
import time
from pathlib import Path

import torch

from trainer import train_with_guards
from guards import Heartbeat, resolve_device
from m_skillgcn import make_builder, make_step

INTER = Path("../中间产物")
RESULTS = Path("../结果/supp_runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["t2", "t3"], default="t3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    device = resolve_device(None)
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)

    holder = {}
    real_builder = make_builder(args.dim, args.layers, args.lr, args.reg,
                                str(INTER / "s9_skill_weight.npy"), args.config)

    def builder_probe(*a, **k):
        result = real_builder(*a, **k)
        model, _optimizer = result
        holder["model"] = model
        return result

    traj = []
    real_step = make_step(args.batch)

    def step_probe(model, *a, **k):
        loss = real_step(model, *a, **k)
        traj.append(round(float(model.gate.detach().cpu().item()), 6))
        return loss

    tag = f"skillgcn_{args.config}_gateprobe"
    hb = Heartbeat(tag)
    t0 = time.time()
    m, vbest, counted, ep = train_with_guards(
        tag, args.seed, bundle, builder_probe, step_probe, device,
        epochs=args.epochs, patience=args.patience, verbose=True, hb=hb,
        resume=False)
    dt = time.time() - t0

    gate_best = round(float(holder["model"].gate.detach().cpu().item()), 6)
    out = {
        "config": args.config,
        "seed": args.seed,
        "epochs_ran": ep,
        "gate_init": 0.0,
        "gate_final_best_ckpt": gate_best,
        "gate_trajectory": traj,
        "test_NDCG@10": round(m["NDCG@10"], 5),
        "val_best": round(float(vbest["NDCG@10"]), 5),
        "wall_seconds": round(dt, 1),
        "hyper": {"dim": args.dim, "layers": args.layers, "lr": args.lr,
                  "reg": args.reg, "batch": args.batch,
                  "epochs": args.epochs, "patience": args.patience},
    }
    outfile = RESULTS / f"r_skillgcn_{args.config}_gate_s{args.seed}.json"
    outfile.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n=== gate探测完成 config={args.config} seed={args.seed} ===")
    print(f"gate最终值(best ckpt) {gate_best}  轨迹长度 {len(traj)}  "
          f"test NDCG@10 {out['test_NDCG@10']}")
    print(f"轨迹 {traj}")
    print(f"落盘 {outfile}")


if __name__ == "__main__":
    main()
