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
RUNS = Path("../结果/newS9_runs")
EMB = Path("../中间产物/嵌入")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["t1", "t2", "t3"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    EMB.mkdir(exist_ok=True)
    device = resolve_device(None)
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    holder = {}
    real_builder = make_builder(args.dim, args.layers, args.lr, args.reg,
                                str(INTER / "s9_skill_weight.npy"), args.config)

    def builder_probe(*a, **k):
        result = real_builder(*a, **k)
        model, _opt = result
        holder["model"] = model
        return result

    traj = []
    real_step = make_step(args.batch)

    def step_probe(model, *a, **k):
        loss = real_step(model, *a, **k)
        if hasattr(model, "gate"):
            traj.append(round(float(model.gate.detach().cpu().item()), 6))
        return loss

    tag = f"newS9_{args.config}"
    hb = Heartbeat(tag)
    t0 = time.time()
    m, vbest, counted, ep = train_with_guards(
        tag, args.seed, bundle, builder_probe, step_probe, device,
        epochs=args.epochs, patience=args.patience, verbose=True, hb=hb,
        resume=False)
    dt = time.time() - t0

    model = holder["model"]
    model.eval()
    gate = None
    if hasattr(model, "gate"):
        gate = round(float(model.gate.detach().cpu().item()), 6)

    rec = {
        "config": args.config, "seed": args.seed,
        "test": {k: round(float(v), 5) for k, v in m.items()},
        "val_best_NDCG@10": round(float(vbest["NDCG@10"]), 5),
        "gate_final": gate, "gate_trajectory": traj,
        "epochs_ran": ep, "wall_seconds": round(dt, 1),
        "s9_source": "BAAI/bge-v1.5 cosine alignment via s8_align.py & s9_gen.py",
    }
    (RUNS / f"run_{args.config}_s{args.seed}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.config in ("t1", "t3"):
        with torch.no_grad():
            eu, ei = model.propagate()
        torch.save({"config": args.config, "seed": args.seed,
                    "concept_emb": ei.detach().cpu(),
                    "gate": gate, "test_NDCG@10": rec["test"]["NDCG@10"]},
                   EMB / f"emb_{args.config}_s{args.seed}.pt")

    print(f"\n=== 重跑完成 {args.config} seed={args.seed} ===")
    print(f"test NDCG@10 {rec['test']['NDCG@10']}  gate {gate}  轮次 {ep}")


if __name__ == "__main__":
    main()
