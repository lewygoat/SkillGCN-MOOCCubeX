import common  # Auto-unpacks s6_graph.pt.gz on first run
import json
import numpy as np
import torch
from path_l2norm import *

bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
s9 = np.load(INTER / "s9_skill_weight.npy")
wn = torch.tensor(s9 / s9.max(), dtype=torch.float32).unsqueeze(1)
pm = set(zip(bundle["eval_prereq_positive"].numpy()[0].tolist(), bundle["eval_prereq_positive"].numpy()[1].tolist()))
pw = load_prereq(bundle, "chapter_weak")
rng = np.random.RandomState(42)
skilled = np.nonzero(s9 > 0)[0]
seeds = rng.choice(skilled, size=min(300, len(skilled)), replace=False).tolist()
out = {}
for cfg in ("t1", "t3"):
    for mode in ("post_none", "pre_none", "post_l2", "pre_l2"):
        acc = {cw: {"cov": [], "viol_main": []} for cw in (0.0, 0.5, 1.0)}
        for sd in range(42, 47):
            d = torch.load(EMB / f"exp5_{cfg}_s{sd}.pt", weights_only=False)
            post = d["concept_emb"].float()
            pre = post / (1.0 + float(d["gate"]) * wn)
            e = post if mode.startswith("post") else pre
            if mode.endswith("l2"):
                e = e / e.norm(dim=1, keepdim=True).clamp(min=1e-8)
            for cw in acc:
                r = eval_config(e, s9, pm, pw, seeds, 10, cw)
                for k in acc[cw]:
                    acc[cw][k].append(r[k])
        out[f"{cfg}_{mode}"] = {str(cw): {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in acc[cw].items()} for cw in acc}
        print(cfg, mode, {cw: round(out[f"{cfg}_{mode}"][str(cw)]["cov"][0], 4) for cw in acc})
json.dump(out, open(RESULTS / "supp_pregate.json", "w"), ensure_ascii=False, indent=2)
