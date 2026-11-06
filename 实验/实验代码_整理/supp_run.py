import common  # Auto-unpacks s6_graph.pt.gz on first run
import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch

from evaluator import evaluate
from guards import (GuardViolation, Heartbeat, assert_eval_pool_clean,
                    assert_no_leakage, check_loss, check_memory, check_metrics,
                    check_params, resolve_device)
from m_skillgcn import make_builder, make_step

INTER = Path("../中间产物")
EMB = INTER / "嵌入_补充"
RUNS = Path("../结果/supp_runs")
MIN_FREE_GB = 5.0


def atomic_write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def check_disk(path):
    free = shutil.disk_usage(path).free / 1e9
    if free < MIN_FREE_GB:
        raise GuardViolation(f"磁盘剩余{free:.1f}GB低于下限{MIN_FREE_GB}GB")


def finite(value, name):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        raise GuardViolation(f"{name}为非有限值")
    return value


def pick(metrics):
    return {k: round(float(metrics[k]), 5) for k in ("NDCG@10", "HR@10", "MRR", "Recall@20", "Precision@10")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", choices=["t1", "t2", "t3"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gate-mode",
                    choices=["learn", "fixed", "bounded_tanh", "bounded_sigmoid", "l2_train", "vector"],
                    default="learn")
    ap.add_argument("--gate-value", type=float, default=0.0)
    ap.add_argument("--skill-weight", default=str(INTER / "s9_skill_weight.npy"))
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--epoch-limit", type=int, default=900)
    ap.add_argument("--curve-test", action="store_true")
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    EMB.mkdir(parents=True, exist_ok=True)
    check_disk(RUNS)
    outfile = RUNS / f"{args.name}.json"
    if outfile.exists():
        print(f"已存在 {outfile.name}，跳过")
        return

    device = resolve_device(None)
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    overlaps = assert_no_leakage(bundle)
    torch.manual_seed(args.seed)

    gate_type = args.gate_mode if args.gate_mode in ("bounded_tanh", "bounded_sigmoid", "l2_train", "vector") else "scalar"
    build = make_builder(args.dim, args.layers, args.lr, args.reg,
                         args.skill_weight, args.config, gate_type=gate_type)
    model, optimizer = build(bundle, device, args.seed)
    if args.gate_mode == "fixed":
        with torch.no_grad():
            model.gate.fill_(args.gate_value)
        model.gate.requires_grad_(False)

    assert_eval_pool_clean(lambda uu: model.score_all(uu), bundle, device=device)
    step_fn = make_step(args.batch)
    hb = Heartbeat(f"supp_{args.name}")
    hb.beat(event="start", seed=args.seed, config=args.config, gate_mode=args.gate_mode,
            gate_value=args.gate_value, split_overlaps=overlaps)

    best_valid = {"NDCG@10": -1.0}
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    curve = []
    t0 = time.time()
    last = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss = finite(float(step_fn(model, optimizer, bundle, device, epoch)), "损失")
        check_loss(loss, epoch, history)
        check_params(model, epoch)
        history.append(loss)

        model.eval()
        with torch.no_grad():
            valid, _ = evaluate(lambda uu: model.score_all(uu), bundle, split="valid", device=device)
            test = None
            if args.curve_test:
                test, _ = evaluate(lambda uu: model.score_all(uu), bundle, split="test", device=device)
        check_metrics(valid, epoch)
        if test is not None:
            check_metrics(test, epoch)
        mem = check_memory(epoch, device)
        now = time.time()
        dt = now - last
        last = now
        if dt > args.epoch_limit:
            raise GuardViolation(f"第{epoch}轮耗时{dt:.0f}s超上限{args.epoch_limit}s")

        gate_now = round(float(model.gate.detach().cpu().mean().item()), 6)
        rec = {"epoch": epoch, "loss": round(loss, 5), "gate": gate_now,
               "valid": pick(valid), "test": pick(test) if test is not None else None, "seconds": round(dt, 1)}
        curve.append(rec)
        hb.beat(event="epoch", seed=args.seed, epoch=epoch, loss=rec["loss"], gate=gate_now,
                valid_ndcg10=rec["valid"]["NDCG@10"], test_ndcg10=rec["test"]["NDCG@10"] if rec["test"] else None,
                epoch_seconds=rec["seconds"], stale=stale, mem_gb=round(mem, 2))
        print(f"epoch {epoch:>3} loss={loss:.4f} gate={gate_now:.3f} "
              f"valid={rec['valid']['NDCG@10']:.4f} {dt:.0f}s", flush=True)

        if valid["NDCG@10"] > best_valid["NDCG@10"]:
            best_valid = valid
            best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise GuardViolation("没有可用检查点")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        test_best, counted = evaluate(lambda uu: model.score_all(uu), bundle, split="test", device=device)
        eu, ei = model.propagate()
        raw = model.emb.weight.detach().cpu()
    check_metrics(test_best, best_epoch)
    gate_best = round(float(model.gate.detach().cpu().mean().item()), 6)

    atomic_save({"user_emb": eu.detach().cpu(), "concept_emb": ei.detach().cpu(),
                 "base_emb": raw, "gate": gate_best, "config": args.config,
                 "seed": args.seed, "name": args.name}, EMB / f"{args.name}.pt")

    rec = {
        "name": args.name, "config": args.config, "seed": args.seed,
        "gate_mode": args.gate_mode, "gate_value": args.gate_value,
        "hyper": {"dim": args.dim, "layers": args.layers, "lr": args.lr, "reg": args.reg,
                  "batch": args.batch, "epochs": args.epochs, "patience": args.patience},
        "best_epoch": best_epoch, "epochs_ran": len(curve), "gate_best": gate_best,
        "valid_best": pick(best_valid), "test_at_best": pick(test_best),
        "counted_users": counted, "curve": curve,
        "wall_seconds": round(time.time() - t0, 1),
        "split_overlaps": overlaps,
    }
    atomic_write(outfile, json.dumps(rec, ensure_ascii=False, indent=2))
    hb.beat(event="done", seed=args.seed, best_epoch=best_epoch,
            test_ndcg10=rec["test_at_best"]["NDCG@10"])
    print(f"完成 {args.name} best_epoch={best_epoch} test={rec['test_at_best']['NDCG@10']}", flush=True)


if __name__ == "__main__":
    main()
