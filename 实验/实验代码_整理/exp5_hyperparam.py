import common  # Auto-unpacks s6_graph.pt.gz on first run
import argparse
import json
import time
from pathlib import Path

import torch

from guards import Heartbeat, resolve_device
from m_skillgcn import make_builder, make_step
from trainer import train_with_guards

INTER = Path("../中间产物")
RESULTS = Path("../结果")


def scan_dim():
    print("=== 维度扫描 ===")
    dims = [32, 64, 128, 256]
    return scan_grid("dim", dims, lambda d: (d, 2, 0.01))


def scan_layers():
    print("=== 层数扫描 ===")
    layers = [1, 2, 3, 4]
    return scan_grid("layers", layers, lambda L: (64, L, 0.01))


def scan_lr():
    print("=== 学习率扫描 ===")
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
    return scan_grid("lr", lrs, lambda r: (64, 2, r))


def scan_grid(param_name, values, make_hyper):
    device = resolve_device(None)
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    out = {param_name: [], "configs": {"t1": {}, "t3": {}}}
    seed = 42
    for val in values:
        dim, layers, lr = make_hyper(val)
        print(f"  {param_name}={val}  dim={dim} layers={layers} lr={lr}")
        for cfg in ["t1", "t3"]:
            tag = f"exp5_{param_name}_{cfg}_{val}"
            hb = Heartbeat(tag)
            builder = make_builder(dim, layers, lr, 1e-5,
                                  str(INTER / "s9_skill_weight.npy"), cfg)
            step = make_step(8192)
            t0 = time.time()
            m, vbest, counted, ep = train_with_guards(
                tag, seed, bundle, builder, step, device,
                epochs=60, patience=8, verbose=False, hb=hb, resume=False)
            dt = time.time() - t0
            ndcg = round(m["NDCG@10"], 5)
            print(f"    {cfg}: epochs={ep} test_NDCG@10={ndcg} ({dt:.1f}s)")
            out["configs"][cfg][str(val)] = {
                "value": val, "seed": seed, "epochs": ep,
                "test_NDCG@10": ndcg, "wall_s": round(dt, 1),
                "hyper": {"dim": dim, "layers": layers, "lr": lr}}
        out[param_name].append(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", choices=["dim", "layers", "lr", "all"],
                   default="all")
    args = ap.parse_args()
    results = {}
    if args.param in ("dim", "all"):
        results["dim"] = scan_dim()
    if args.param in ("layers", "all"):
        results["layers"] = scan_layers()
    if args.param in ("lr", "all"):
        results["lr"] = scan_lr()
    outf = RESULTS / "r_exp5_hyperparam.json"
    outf.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n超参扫描完成")
    for pname in results:
        r = results[pname]
        print(f"── {pname} ──")
        for val in r[pname]:
            t1 = r["configs"]["t1"][str(val)]["test_NDCG@10"]
            t3 = r["configs"]["t3"][str(val)]["test_NDCG@10"]
            winner = "T1" if t1 > t3 else ("T3" if t3 > t1 else "平")
            print(f"  {pname}={val}: T1={t1}  T3={t3}  胜出={winner}")
    print(f"落盘 {outf}")


if __name__ == "__main__":
    main()
