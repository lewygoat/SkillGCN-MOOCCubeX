import common  # Auto-unpacks s6_graph.pt.gz on first run
import argparse
import time
from pathlib import Path

import torch

from trainer import train_with_guards
from guards import Heartbeat, resolve_device
from m_skillgcn import make_builder, make_step

INTER = Path("../中间产物")
EMB = Path("../中间产物/嵌入")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["t1", "t3"], default="t3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    EMB.mkdir(exist_ok=True)
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

    step_fn = make_step(args.batch)
    tag = f"skillgcn_{args.config}_embexport"
    hb = Heartbeat(tag)
    t0 = time.time()
    m, vbest, counted, ep = train_with_guards(
        tag, args.seed, bundle, builder_probe, step_fn, device,
        epochs=args.epochs, patience=args.patience, verbose=True, hb=hb,
        resume=False)
    dt = time.time() - t0

    model = holder["model"]
    model.eval()
    with torch.no_grad():
        eu, ei = model.propagate()
        gate = None
        if hasattr(model, "gate"):
            gate = round(float(model.gate.detach().cpu().item()), 6)
        out = {
            "config": args.config,
            "seed": args.seed,
            "concept_emb": ei.detach().cpu(),
            "user_emb": eu.detach().cpu(),
            "gate": gate,
            "test_NDCG@10": round(m["NDCG@10"], 5),
            "val_best": round(float(vbest["NDCG@10"]), 5),
            "epochs_ran": ep,
            "wall_seconds": round(dt, 1),
            "n_concepts": bundle["n_concepts"],
            "dim": args.dim,
        }
    outfile = EMB / f"emb_{args.config}_s{args.seed}.pt"
    torch.save(out, outfile)
    print(f"\n=== 嵌入导出完成 config={args.config} seed={args.seed} ===")
    print(f"概念嵌入形状 {tuple(ei.shape)}  gate {gate}  "
          f"test NDCG@10 {out['test_NDCG@10']}  轮次 {ep}")
    print(f"落盘 {outfile}")


if __name__ == "__main__":
    main()
