import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import INTER
from guards import clip_and_check_grad
from m_lightgcn import LightGCN
from trainer import run_method


def load_skill_weight(path, config, seed):
    w = np.load(path).astype(np.float32)
    if config == "t1":
        return np.zeros_like(w)
    if config == "t2":
        rng = np.random.RandomState(seed)
        return rng.permutation(w)
    return w


class SkillAnchoredLightGCN(LightGCN):
    def __init__(self, n_users, n_items, dim, n_layers, edge_index, device,
                 skill_weight, gate_type="scalar"):
        super().__init__(n_users, n_items, dim, n_layers, edge_index, device)
        w = torch.tensor(skill_weight, dtype=torch.float32, device=device)
        w_norm = w / w.clamp(min=1e-8).max()
        self.register_buffer("skill_w", w_norm)
        self.gate_type = gate_type
        if gate_type == "vector":
            self.gate = nn.Parameter(torch.zeros(n_items, device=device))
        else:
            self.gate = nn.Parameter(torch.zeros(1, device=device))

    def propagate(self):
        eu, ei = super().propagate()
        w = self.skill_w.unsqueeze(1)
        if self.gate_type == "bounded_tanh":
            ei = ei * (1.0 + torch.tanh(self.gate) * w)
        elif self.gate_type == "bounded_sigmoid":
            ei = ei * (1.0 + torch.sigmoid(self.gate) * w)
        elif self.gate_type == "l2_train":
            ei = ei * (1.0 + self.gate * w)
            ei = F.normalize(ei, p=2, dim=-1)
        elif self.gate_type == "vector":
            ei = ei * (1.0 + self.gate.unsqueeze(1) * w)
        else:
            ei = ei * (1.0 + self.gate * w)
        return eu, ei


def make_builder(dim, layers, lr, reg, skill_path, config, gate_type="scalar"):
    def build(bundle, device, seed):
        w = load_skill_weight(skill_path, config, seed)
        model = SkillAnchoredLightGCN(bundle["n_users"], bundle["n_concepts"], dim, layers,
                                      bundle["train_ei"].to(device), device, w,
                                      gate_type=gate_type).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        return model, opt
    return build


def make_step(batch):
    state = {}

    def step(model, optimizer, bundle, device, epoch):
        if "ei" not in state:
            ei = bundle["train_ei"].to(device)
            state["ei"] = ei
            seen = torch.zeros((bundle["n_users"], bundle["n_concepts"]),
                               dtype=torch.bool, device=device)
            seen[ei[0], ei[1]] = True
            state["seen"] = seen
        ei = state["ei"]
        seen = state["seen"]
        n = ei.shape[1]
        n_items = bundle["n_concepts"]
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            u = ei[0][idx]
            p = ei[1][idx]
            neg = torch.randint(0, n_items, (idx.shape[0],), device=device)
            bad = seen[u, neg]
            for _ in range(4):
                if not bad.any():
                    break
                neg[bad] = torch.randint(0, n_items, (int(bad.sum()),), device=device)
                bad = seen[u, neg]
            diff = model.bpr(u, p, neg)
            loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
            optimizer.zero_grad()
            loss.backward()
            clip_and_check_grad(model, epoch)
            optimizer.step()
            total += float(loss.detach()) * idx.shape[0]
        return total / n
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--config", choices=["t1", "t2", "t3"], default="t3")
    ap.add_argument("--skill-weight", default=str(INTER / "s9_skill_weight.npy"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    hyper = {"dim": args.dim, "layers": args.layers, "lr": args.lr,
             "reg": args.reg, "batch": args.batch, "epochs": args.epochs,
             "patience": args.patience, "config": args.config,
             "skill_weight_file": args.skill_weight,
             "note": {"t1": "无技能层，gate恒为零参照",
                      "t2": "技能层节点接入但权重随机打乱，隔离额外参数增益",
                      "t3": "跨层对齐边与锚定注意力齐备"}[args.config]}
    tag = f"skillgcn_{args.config}"
    label = f"SkillAnchoredLightGCN-{args.config.upper()}"
    run_method(tag, label, INTER / "s6_graph.pt",
               make_builder(args.dim, args.layers, args.lr, args.reg,
                            args.skill_weight, args.config),
               make_step(args.batch), seeds=args.seeds, epochs=args.epochs,
               patience=args.patience, verbose=args.verbose, hyper=hyper)


if __name__ == "__main__":
    main()
