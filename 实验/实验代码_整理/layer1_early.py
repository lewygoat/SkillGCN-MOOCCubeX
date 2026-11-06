import json
import sys
import numpy as np
import torch
from common import INTER
from evaluator import evaluate
from guards import resolve_device
from m_skillgcn import make_builder, make_step

seed = int(sys.argv[1])
n_ep = 6
device = resolve_device(None)
b = torch.load(INTER / "s6_graph.pt", weights_only=False)
nu, nc = b["n_users"], b["n_concepts"]
tr = b["train_ei"]
pop = torch.tensor(np.bincount(tr[1].numpy(), minlength=nc), dtype=torch.float32)
seen = torch.zeros((nu, nc), dtype=torch.bool)
seen[tr[0], tr[1]] = True
users = torch.unique(b["test_ei"][0])
ref = pop.unsqueeze(0).expand(len(users), -1).clone()
ref[seen[users]] = float("-inf")
rt = torch.topk(ref, 10, dim=1).indices

torch.manual_seed(seed)
build = make_builder(64, 2, 0.01, 1e-5, str(INTER / "s9_skill_weight.npy"), "t1")
model, opt = build(b, device, seed)
step = make_step(8192)
out = []
for ep in range(1, n_ep + 1):
    model.train()
    step(model, opt, b, device, ep)
    model.eval()
    with torch.no_grad():
        test, _ = evaluate(lambda uu: model.score_all(uu), b, split="test", device=device)
        eu, ei = model.propagate()
        sc = (eu[users.to(device)] @ ei.T).cpu()
    sc[seen[users]] = float("-inf")
    t = torch.topk(sc, 10, dim=1).indices
    ov = np.mean([len(set(t[i].tolist()) & set(rt[i].tolist())) / 10 for i in range(len(users))])
    rec = {"epoch": ep, "test_ndcg10": test["NDCG@10"], "ov_pop": float(ov)}
    out.append(rec)
    print(seed, rec, flush=True)
json.dump(out, open(f"../结果/layer1_early_s{seed}.json", "w"), indent=2)
