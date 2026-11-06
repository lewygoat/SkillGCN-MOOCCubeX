import common  # Auto-unpacks s6_graph.pt.gz on first run
import json
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

I = "../中间产物/"
b = torch.load(I + "s6_graph.pt", weights_only=False)
nu, nc = b["n_users"], b["n_concepts"]
w = np.load(I + "s9_skill_weight.npy").astype(np.float32)
wn = w / w.max()
tr = b["train_ei"]
pop = np.bincount(tr[1].numpy(), minlength=nc).astype(np.float64)

seen = torch.zeros((nu, nc), dtype=torch.bool)
seen[tr[0], tr[1]] = True
te = b["test_ei"]
users = torch.unique(te[0])

def top10(scores):
    s = scores.clone()
    s[seen[users]] = float("-inf")
    return torch.topk(s, 10, dim=1).indices

def overlap(a, ref_scores):
    r = top10(ref_scores)
    out = []
    for i in range(a.shape[0]):
        out.append(len(set(a[i].tolist()) & set(r[i].tolist())) / 10)
    return float(np.mean(out))

pop_scores = torch.tensor(pop, dtype=torch.float32).unsqueeze(0).expand(len(users), -1)
w_scores = torch.tensor(wn, dtype=torch.float32).unsqueeze(0).expand(len(users), -1)

res = {"w_pop": {}, "per_seed": {}}
rho, rp = spearmanr(wn, pop)
pr, pp = pearsonr(wn, pop)
res["w_pop"] = {"spearman": float(rho), "spearman_p": float(rp), "pearson": float(pr), "pearson_p": float(pp), "n": nc,
                "w_nonzero": int((w > 0).sum())}

for cfg in ("t1", "t3"):
    for sd in range(42, 47):
        d = torch.load(I + f"嵌入_补充/exp5_{cfg}_s{sd}.pt", weights_only=False)
        g = float(d["gate"])
        post = d["concept_emb"].float()
        scale = 1.0 + g * torch.tensor(wn).unsqueeze(1)
        pre = post / scale
        eu = d["user_emb"].float()
        sc = eu[users] @ post.T
        t = top10(sc)
        npost = post.norm(dim=1).numpy()
        npre = pre.norm(dim=1).numpy()
        row = {
            "gate": g,
            "ov_pop": overlap(t, pop_scores),
            "ov_w": overlap(t, w_scores),
            "norm_post_w_sp": float(spearmanr(npost, wn)[0]),
            "norm_pre_w_sp": float(spearmanr(npre, wn)[0]),
            "norm_post_w_pe": float(pearsonr(npost, wn)[0]),
            "norm_pre_w_pe": float(pearsonr(npre, wn)[0]),
            "norm_post_pop_sp": float(spearmanr(npost, pop)[0]),
            "norm_pre_pop_sp": float(spearmanr(npre, pop)[0]),
            "norm_post_pop_pe": float(pearsonr(npost, pop)[0]),
            "norm_pre_pop_pe": float(pearsonr(npre, pop)[0]),
        }
        res["per_seed"][f"{cfg}_s{sd}"] = row
        print(cfg, sd, {k: round(v, 4) for k, v in row.items()}, flush=True)

with open("../结果/layer1_check.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(res["w_pop"])
