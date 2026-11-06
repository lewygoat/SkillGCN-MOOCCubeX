import json
import statistics
import time

import torch

from common import OUT, load_protocol, provenance, report
from evaluator import evaluate, format_metrics
from guards import (Checkpoint, EpochTimer, GuardViolation, Heartbeat,
                    assert_eval_pool_clean, assert_no_leakage, check_loss,
                    check_memory, check_metrics, check_params, resolve_device)


def train_with_guards(tag, seed, bundle, build_model, step_fn, device,
                      epochs=60, patience=6, epoch_limit_s=900,
                      resume=True, verbose=False, hb=None):
    torch.manual_seed(seed)
    model, optimizer = build_model(bundle, device, seed)
    ckpt = Checkpoint(tag, f"seed{seed}")
    timer = EpochTimer(epoch_limit_s)

    start_epoch = 1
    best = {"NDCG@10": -1.0}
    best_state = None
    history = []

    saved = ckpt.load() if resume else None
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"] + 1
        best = saved["best"]
        best_state = saved["best_state"]
        history = saved["history"]
        print(f"   种子{seed} 自检查点续跑，从第{start_epoch}轮开始")

    assert_eval_pool_clean(lambda uu: model.score_all(uu), bundle, device=device)

    stale = 0
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_value = step_fn(model, optimizer, bundle, device, epoch)
        check_loss(loss_value, epoch, history)
        check_params(model, epoch)
        history.append(loss_value)

        model.eval()
        with torch.no_grad():
            metrics, _ = evaluate(lambda uu: model.score_all(uu), bundle,
                                  split="valid", device=device)
        check_metrics(metrics, epoch)
        mem = check_memory(epoch, device)
        dt = timer.tick(epoch)
        last_epoch = epoch

        if verbose:
            print(f"   epoch {epoch:>3}  loss={loss_value:.4f}  {dt:5.1f}s  valid {format_metrics(metrics)}")
        if hb:
            hb.beat(seed=seed, epoch=epoch, loss=round(loss_value, 5),
                    valid_ndcg10=round(metrics["NDCG@10"], 5),
                    best_ndcg10=round(max(best["NDCG@10"], 0.0), 5),
                    epoch_seconds=round(dt, 1), stale=stale,
                    mem_gb=round(mem, 2))

        if metrics["NDCG@10"] > best["NDCG@10"]:
            best = metrics
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        ckpt.save(model, optimizer, epoch, best, best_state, history)
        if stale >= patience:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        test_metrics, counted = evaluate(lambda uu: model.score_all(uu), bundle,
                                         split="test", device=device)
    check_metrics(test_metrics, last_epoch)
    ckpt.clear()
    return test_metrics, best, counted, last_epoch


def run_method(tag, label, bundle_path, build_model, step_fn,
               seeds=None, epochs=60, patience=6, verbose=False,
               device_prefer=None, hyper=None):
    proto = load_protocol()
    seeds = seeds or proto["training"]["seeds"]
    device = resolve_device(device_prefer)
    bundle = torch.load(bundle_path, weights_only=False)

    overlaps = assert_no_leakage(bundle)
    hb = Heartbeat(tag)
    hb.beat(event="start", seeds=seeds, device=device, split_overlaps=overlaps)

    print(f"── {label}  device={device}  seeds={seeds}  hyper={hyper or {}}")
    runs, failures = [], []
    t0 = time.time()
    for seed in seeds:
        try:
            m, vbest, counted, ep = train_with_guards(
                tag, seed, bundle, build_model, step_fn, device,
                epochs=epochs, patience=patience, verbose=verbose, hb=hb)
            runs.append(m)
            print(f"   seed {seed}  epochs={ep}  test {format_metrics(m)}")
            hb.beat(event="seed_done", seed=seed, epochs=ep,
                    test_ndcg10=round(m["NDCG@10"], 5))
        except GuardViolation as exc:
            failures.append({"seed": seed, "error": str(exc)})
            print(f"   seed {seed}  防御触发 {exc}")
            hb.beat(event="guard_violation", seed=seed, error=str(exc))

    if not runs:
        hb.beat(event="all_failed", failures=failures)
        raise GuardViolation(f"{label} 全部种子失败 {failures}")

    summary = {}
    for k in sorted(runs[0]):
        vals = [r[k] for r in runs]
        summary[k] = {"mean": statistics.mean(vals),
                      "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}

    report(f"{label} 测试集结果", {
        k: f"{summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}"
        for k in ("NDCG@10", "HR@10", "MRR", "Recall@20", "Precision@10")
        if k in summary
    })
    elapsed = time.time() - t0
    print(f"   耗时 {elapsed:.1f}s  成功种子 {len(runs)}/{len(seeds)}")

    payload = {"method": label, "tag": tag, "device": device, "seeds": seeds,
               "hyper": hyper or {}, "per_seed": runs, "summary": summary,
               "failures": failures, "elapsed_s": round(elapsed, 1),
               "split_overlaps": overlaps}
    path = OUT / f"r_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    provenance(tag, {k: summary[k]["mean"] for k in ("NDCG@10", "HR@10", "MRR") if k in summary},
               [path])
    hb.beat(event="done", seeds_ok=len(runs), elapsed_s=round(elapsed, 1))
    return payload
