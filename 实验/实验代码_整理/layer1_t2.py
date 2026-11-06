import common  # Auto-unpacks s6_graph.pt.gz on first run
import json
import numpy as np
import torch

I = "../中间产物/"
b = torch.load(I + "s6_graph.pt", weights_only=False)
nu, nc = b["n_users"], b["n_concepts"]
w = np.load(I + "s9_skill_weight.npy").astype(np.float32)
wn = w / w.max()
tr = b["train_ei"]
pop = np.bincount(tr[1].numpy(), minlength=nc).astype(np.float64)
seen = torch.zeros((nu, nc), dtype=torch.bool)
seen[tr[0], tr[1]] = True
users = torch.unique(b["test_ei"][0])


def top10(scores):
    s = scores.clone()
    s[seen[users]] = float("-inf")
    return torch.topk(s, 10, dim=1).indices


def overlap(a, ref):
    r = top10(ref)
    return float(np.mean([len(set(a[i].tolist()) & set(r[i].tolist())) / 10 for i in range(a.shape[0])]))


pop_s = torch.tensor(pop, dtype=torch.float32).unsqueeze(0).expand(len(users), -1)
w_s = torch.tensor(wn, dtype=torch.float32).unsqueeze(0).expand(len(users), -1)
out = {}
for sd in range(42, 47):
    d = torch.load(I + f"嵌入_补充/exp5_t2_s{sd}.pt", weights_only=False)
    t = top10(d["user_emb"].float()[users] @ d["concept_emb"].float().T)
    out[f"t2_s{sd}"] = {"gate": float(d["gate"]), "ov_pop": overlap(t, pop_s), "ov_w": overlap(t, w_s)}
    print(sd, out[f"t2_s{sd}"], flush=True)
v = [r["ov_pop"] for r in out.values()]
out["mean_ov_pop"] = float(np.mean(v))
out["std_ov_pop"] = float(np.std(v, ddof=1))
print(out["mean_ov_pop"], out["std_ov_pop"])
json.dump(out, open("../结果/layer1_t2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
