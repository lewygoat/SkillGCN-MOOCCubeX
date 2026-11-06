import json
import math
import time
from pathlib import Path

import torch

from common import INTER, LOG


class GuardViolation(RuntimeError):
    pass


def assert_no_leakage(bundle):
    train = set(map(tuple, bundle["train_ei"].T.tolist()))
    valid = set(map(tuple, bundle["valid_ei"].T.tolist()))
    test = set(map(tuple, bundle["test_ei"].T.tolist()))
    overlaps = {
        "train_valid": len(train & valid),
        "train_test": len(train & test),
        "valid_test": len(valid & test),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise GuardViolation(f"划分交叠 {bad}")

    pos = set(map(tuple, bundle["eval_prereq_positive"].T.tolist()))
    for key in ("prereq_adopted", "prereq_predicted"):
        edges = set(map(tuple, bundle[key].T.tolist()))
        hit = len(edges & pos)
        if hit:
            raise GuardViolation(f"{key} 与评测正例重叠 {hit} 条，建模边污染评测基准")
    return overlaps


def assert_eval_pool_clean(score_fn, bundle, device="cpu", sample=64):
    users = torch.arange(min(sample, bundle["n_users"]), device=device)
    scores = score_fn(users)
    if scores.shape[1] != bundle["n_concepts"]:
        raise GuardViolation(
            f"打分维度 {tuple(scores.shape)} 与概念数 {bundle['n_concepts']} 不符")
    scores = scores.detach()
    if torch.isnan(scores).any() or torch.isinf(scores).any():
        raise GuardViolation("打分含NaN或Inf")
    spread = float(scores.std())
    if spread < 1e-8:
        raise GuardViolation(f"打分方差近零 {spread:.3e}，模型可能未学到任何区分")
    return {"score_std": spread}


def check_loss(value, epoch, history, max_ratio=50.0):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise GuardViolation(f"第{epoch}轮损失为NaN")
    if math.isinf(value):
        raise GuardViolation(f"第{epoch}轮损失为Inf")
    if history:
        floor = min(history)
        if floor > 0 and value > floor * max_ratio:
            raise GuardViolation(
                f"第{epoch}轮损失{value:.4f}超历史最低{floor:.4f}的{max_ratio}倍，训练发散")
    return True


def check_metrics(metrics, epoch, tol=1e-9):
    degenerate = []
    for key in ("NDCG@10", "HR@10", "MRR"):
        v = metrics.get(key)
        if v is None:
            continue
        if math.isnan(v):
            raise GuardViolation(f"第{epoch}轮 {key} 为NaN")
        if v <= tol:
            degenerate.append(f"{key}={v:.2e}")
        if v >= 1.0 - tol:
            degenerate.append(f"{key}={v:.6f}")
    if degenerate:
        raise GuardViolation(f"第{epoch}轮指标退化 {' '.join(degenerate)}")
    return True


def clip_and_check_grad(model, epoch, max_norm=5.0, explode_at=1e6):
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    value = float(total)
    if math.isnan(value):
        raise GuardViolation(f"第{epoch}轮梯度范数为NaN")
    if math.isinf(value) or value > explode_at:
        raise GuardViolation(f"第{epoch}轮梯度范数 {value:.3e} 爆炸")
    return value


def check_memory(epoch, device, limit_gb=24.0):
    if device == "mps":
        used = torch.mps.current_allocated_memory() / (1 << 30)
    elif device == "cuda":
        used = torch.cuda.memory_allocated() / (1 << 30)
    else:
        return 0.0
    if used > limit_gb:
        raise GuardViolation(f"第{epoch}轮显存占用 {used:.1f}GB 超上限 {limit_gb}GB")
    return used


def check_params(model, epoch, max_norm=1e4):
    for name, p in model.named_parameters():
        if torch.isnan(p).any():
            raise GuardViolation(f"第{epoch}轮参数 {name} 含NaN")
        if torch.isinf(p).any():
            raise GuardViolation(f"第{epoch}轮参数 {name} 含Inf")
        norm = float(p.detach().norm())
        if norm > max_norm:
            raise GuardViolation(f"第{epoch}轮参数 {name} 范数 {norm:.3e} 超上限")
    return True


def resolve_device(prefer=None):
    order = [prefer] if prefer else []
    order += ["mps", "cuda", "cpu"]
    for dev in order:
        if not dev:
            continue
        try:
            if dev == "mps" and not torch.backends.mps.is_available():
                continue
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            probe = torch.zeros(8, device=dev)
            _ = (probe + 1).sum()
            return dev
        except Exception:
            continue
    return "cpu"


class Checkpoint:
    def __init__(self, tag, run_id):
        self.dir = INTER / "checkpoints" / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.pt"
        self.meta = self.dir / f"{run_id}.json"

    def save(self, model, optimizer, epoch, best, best_state, history):
        tmp = self.path.with_suffix(".tmp")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": best,
            "best_state": best_state,
            "history": history,
        }, tmp)
        tmp.replace(self.path)
        with open(self.meta, "w", encoding="utf-8") as f:
            json.dump({"epoch": epoch, "best": best,
                       "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                      f, ensure_ascii=False, indent=2)

    def load(self):
        if not self.path.exists():
            return None
        try:
            return torch.load(self.path, weights_only=False)
        except Exception:
            return None

    def clear(self):
        for p in (self.path, self.meta):
            if p.exists():
                p.unlink()


class Heartbeat:
    def __init__(self, tag):
        self.path = LOG / "heartbeat.jsonl"
        self.tag = tag
        self.t0 = time.time()

    def beat(self, **fields):
        rec = {"tag": self.tag, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "elapsed_s": round(time.time() - self.t0, 1)}
        rec.update(fields)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


class EpochTimer:
    def __init__(self, limit_s=900):
        self.limit = limit_s
        self.t = time.time()

    def tick(self, epoch):
        dt = time.time() - self.t
        self.t = time.time()
        if dt > self.limit:
            raise GuardViolation(f"第{epoch}轮耗时{dt:.0f}s超上限{self.limit}s，疑似挂起")
        return dt
