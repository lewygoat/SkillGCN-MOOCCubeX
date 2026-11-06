import glob
import json
import re
import statistics
import sys

sys.path.insert(0, "代码")
from common import OUT

HEAD_PAT = re.compile(
    r"── SkillAnchoredLightGCN-(T\d)\s+device=(\S+)\s+seeds=(\[[^\]]*\])\s+hyper=(\{.*\})")
SEED_PAT = re.compile(
    r"^   seed (\d+)\s+epochs=(\d+)\s+test NDCG@10=([\d.]+)\s+HR@10=([\d.]+)\s+"
    r"MRR=([\d.]+)\s+Recall@20=([\d.]+)\s+Precision@10=([\d.]+)")


def scan(paths):
    records = {}
    cur = None
    for p in paths:
        for line in open(p, encoding="utf-8", errors="ignore"):
            m = HEAD_PAT.search(line)
            if m:
                cur = {"config": m.group(1), "device": m.group(2),
                      "hyper": eval(m.group(4))}
                continue
            m = SEED_PAT.match(line)
            if m and cur:
                seed = int(m.group(1))
                rec = {"seed": seed, "epochs": int(m.group(2)),
                      "metrics": {"NDCG@10": float(m.group(3)),
                                 "HR@10": float(m.group(4)),
                                 "MRR": float(m.group(5)),
                                 "Recall@20": float(m.group(6)),
                                 "Precision@10": float(m.group(7))}}
                key = (cur["config"], seed)
                records[key] = {"cur": dict(cur), "rec": rec}
    return records


def build_summary(config, records, target_seeds):
    runs, seeds_ok = [], []
    ctx = None
    for seed in target_seeds:
        key = (config, seed)
        if key not in records:
            print(f"  缺 seed{seed}")
            continue
        ctx = records[key]["cur"]
        runs.append(records[key]["rec"]["metrics"])
        seeds_ok.append(seed)
    if not runs:
        print(f"  {config} 无可用数据")
        return None
    summary = {}
    for k in sorted(runs[0]):
        vals = [r[k] for r in runs]
        summary[k] = {"mean": statistics.mean(vals),
                      "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    payload = {"method": f"SkillAnchoredLightGCN-{config}",
              "tag": f"skillgcn_{config.lower()}", "device": ctx["device"],
              "seeds": seeds_ok, "hyper": ctx["hyper"],
              "per_seed": runs, "summary": summary, "failures": [],
              "elapsed_s": None, "split_overlaps": None,
              "note": "多随机种子运行日志汇总"}
    return payload


if __name__ == "__main__":
    paths = sorted(glob.glob(sys.argv[1])) if len(sys.argv) > 1 else []
    print("扫描文件", paths)
    records = scan(paths)
    print("提取到的(config, seed)组合：", sorted(records.keys()))
    for config in ("T1", "T2", "T3"):
        payload = build_summary(config, records, [42, 43, 44, 45, 46])
        if payload:
            path = OUT / f"r_skillgcn_{config.lower()}.json"
            json.dump(payload, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            s = payload["summary"]["NDCG@10"]
            print(f"  {config} 写入 {path}  NDCG@10={s['mean']:.4f}±{s['std']:.4f}  种子{len(payload['seeds'])}")
