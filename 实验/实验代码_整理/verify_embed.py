import common  # Auto-unpacks s6_graph.pt.gz on first run
import json
from pathlib import Path

import torch

from evaluator import evaluate

INTER = Path("../中间产物")
EMB = INTER / "嵌入_补充"
RUNS = Path("../结果/supp_runs")
RESULTS = Path("../结果")


def main():
    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    rows = []
    for config in ("t1", "t2", "t3"):
        for seed in (42, 43, 44, 45, 46):
            f = EMB / f"exp5_{config}_s{seed}.pt"
            rec = RUNS / f"exp5_{config}_s{seed}.json"
            if not f.exists() or not rec.exists():
                rows.append({"config": config, "seed": seed, "missing": True})
                continue
            d = torch.load(f, weights_only=False, map_location="cpu")
            r = json.loads(rec.read_text(encoding="utf-8"))
            uu = d["user_emb"].float()
            ci = d["concept_emb"].float()

            def score_fn(users, uu=uu, ci=ci):
                return uu[users] @ ci.T

            m, counted = evaluate(score_fn, bundle, split="test")
            stored = r["test_at_best"]["NDCG@10"]
            repro = round(m["NDCG@10"], 5)
            rows.append({
                "config": config,
                "seed": seed,
                "emb_gate": d.get("gate"),
                "rec_gate_best": r.get("gate_best"),
                "best_epoch": r.get("best_epoch"),
                "epochs_ran": r.get("epochs_ran"),
                "stored_test_NDCG@10": stored,
                "repro_NDCG@10": repro,
                "diff": round(repro - stored, 6),
                "counted": counted,
                "rec_counted": r.get("counted_users"),
            })
    out = RESULTS / "verify_embed.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = 0
    for r in rows:
        if r.get("missing"):
            print(f"{r['config']} s{r['seed']}: MISSING")
            bad += 1
            continue
        flag = "" if abs(r["diff"]) < 1e-4 and r["counted"] == r["rec_counted"] else "  <<CHECK"
        if flag:
            bad += 1
        print(f"{r['config']} s{r['seed']}: gate={r['emb_gate']}/{r['rec_gate_best']} "
              f"ep_best={r['best_epoch']} stored={r['stored_test_NDCG@10']} "
              f"repro={r['repro_NDCG@10']} diff={r['diff']} "
              f"n={r['counted']}/{r['rec_counted']}{flag}")
    print(f"\n异常行数: {bad}")


if __name__ == "__main__":
    main()
