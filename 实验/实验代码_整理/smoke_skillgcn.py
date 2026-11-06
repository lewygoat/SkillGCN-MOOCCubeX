import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from common import INTER
from evaluator import evaluate, format_metrics
from guards import assert_no_leakage, resolve_device
from m_lightgcn import LightGCN, make_step


def skill_coverage_at_k(score_fn, bundle, skill_mask, split, device, k=10, random_baseline=False):
    from evaluator import build_lookup
    n_users = bundle["n_users"]
    n_concepts = bundle["n_concepts"]
    train_seen = build_lookup(bundle["train_ei"], n_users)
    target = build_lookup(bundle[f"{split}_ei"], n_users)
    users = sorted(u for u, items in target.items() if items)
    mask_t = torch.tensor(skill_mask, dtype=torch.bool, device=device)
    total, n = 0.0, 0
    for start in range(0, len(users), 2048):
        chunk = users[start:start + 2048]
        if random_baseline:
            topk = torch.randint(0, n_concepts, (len(chunk), k), device=device)
        else:
            uu = torch.tensor(chunk, dtype=torch.long, device=device)
            scores = score_fn(uu)
            for row, u in enumerate(chunk):
                seen = train_seen.get(u)
                if seen:
                    scores[row, torch.tensor(sorted(seen), dtype=torch.long, device=device)] = float("-inf")
            topk = scores.topk(k, dim=1).indices
        for row in range(len(chunk)):
            hit = mask_t[topk[row]].float().mean().item()
            total += hit
            n += 1
    return total / max(n, 1)


class SkillAnchoredLightGCN(LightGCN):
    def __init__(self, n_users, n_items, dim, n_layers, edge_index, device, skill_weight):
        super().__init__(n_users, n_items, dim, n_layers, edge_index, device)
        w = torch.tensor(skill_weight, dtype=torch.float32, device=device)
        w_norm = w / w.clamp(min=1e-8).max()
        self.register_buffer("skill_w", w_norm)
        self.gate = torch.nn.Parameter(torch.zeros(1, device=device))

    def propagate(self):
        eu, ei = super().propagate()
        ei = ei * (1.0 + self.gate * self.skill_w.unsqueeze(1))
        return eu, ei


def run(tag, seed, bundle, device, skill_weight, lam, epochs=3):
    torch.manual_seed(seed)
    if lam > 0:
        model = SkillAnchoredLightGCN(bundle["n_users"], bundle["n_concepts"], 64, 2,
                                      bundle["train_ei"].to(device), device, skill_weight).to(device)
    else:
        model = LightGCN(bundle["n_users"], bundle["n_concepts"], 64, 2,
                         bundle["train_ei"].to(device), device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    step = make_step(8192)

    for ep in range(1, epochs + 1):
        model.train()
        loss = step(model, opt, bundle, device, ep)
        model.eval()
        with torch.no_grad():
            m, _ = evaluate(lambda uu: model.score_all(uu), bundle, split="valid", device=device)
        gate_val = model.gate.item() if lam > 0 else 0.0
        print(f"   [{tag}] epoch {ep}  loss={loss:.4f}  gate={gate_val:+.4f}  valid {format_metrics(m)}")

    model.eval()
    with torch.no_grad():
        test_m, _ = evaluate(lambda uu: model.score_all(uu), bundle, split="test", device=device)
        cov = skill_coverage_at_k(lambda uu: model.score_all(uu), bundle,
                                  skill_weight > 0, "test", device, k=10)
        cov_rand = skill_coverage_at_k(None, bundle, skill_weight > 0, "test", device, k=10,
                                       random_baseline=True)
    return test_m, cov, cov_rand


def main():
    device = resolve_device(None)
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    assert_no_leakage(bundle)
    skill_weight = np.load("/tmp/skill_weight_strict.npy")
    n_cov = int((skill_weight > 0).sum())
    print(f"── 模型连通性测试  device={device}  τ=0.62严格阈值")
    print(f"   技能锚定概念数={n_cov}/576={100*n_cov/576:.1f}%（随机基线的理论命中率）")

    seed = 42
    t0 = time.time()
    m1, cov1, rand1 = run("T1无技能层", seed, bundle, device, skill_weight, lam=0.0)
    m3, cov3, rand3 = run("T3有技能锚定", seed, bundle, device, skill_weight, lam=1.0)
    print(f"\n耗时 {time.time()-t0:.1f}s")
    print(f"\n=== 对比（含随机基线）===")
    print(f"  T1无技能层    NDCG@10={m1['NDCG@10']:.4f}  技能覆盖率@10={cov1:.4f}  随机基线={rand1:.4f}")
    print(f"  T3有技能锚定  NDCG@10={m3['NDCG@10']:.4f}  技能覆盖率@10={cov3:.4f}  随机基线={rand3:.4f}")
    print(f"  NDCG@10变化   {m3['NDCG@10']-m1['NDCG@10']:+.4f}")
    print(f"  覆盖率超出随机基线的幅度  T1: {cov1-rand1:+.4f}   T3: {cov3-rand3:+.4f}")


if __name__ == "__main__":
    main()
