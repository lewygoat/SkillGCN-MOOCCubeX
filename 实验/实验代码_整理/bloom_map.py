import common  # Auto-unpacks s6_graph.pt.gz on first run
import argparse
import json
from pathlib import Path

import numpy as np
import torch

INTER = Path("../中间产物")
POOLS = ["s8_align.json", "s8_align_rk.json", "s8_align_bigdata.json"]
TAU = 0.50

BLOOM = {
    1: ["了解", "知道", "列举", "识别", "记忆", "复述", "说出", "认识", "陈述"],
    2: ["理解", "说明", "解释", "描述", "区分", "归纳", "概括", "举例", "阐述"],
    3: ["使用", "运用", "应用", "操作", "编写", "配置", "安装", "调试", "完成",
        "执行", "处理", "进行", "嵌入", "调用", "编程", "实现", "制作"],
    4: ["分析", "比较", "诊断", "调查", "检验", "区别", "检测", "排查", "测算"],
    5: ["评价", "评估", "判断", "选择", "优化", "测试", "验证", "审查", "调优"],
    6: ["设计", "开发", "构建", "创建", "规划", "集成", "搭建", "架构",
        "部署", "建立", "编制"],
}


def median_level(levels):
    s = sorted(levels)
    m = len(s)
    return int(round((s[(m - 1) // 2] + s[m // 2]) / 2))


def verb_level(text):
    best = 0
    for lv in sorted(BLOOM.keys()):
        for v in BLOOM[lv]:
            if v in text:
                best = max(best, lv)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--out", default=str(INTER / "s10_concept_bloom.npy"))
    ap.add_argument("--meta_out", default=str(INTER / "s10_concept_bloom_meta.json"))
    args = ap.parse_args()

    skills = {}
    sk_level = {}
    for fname in ["s7_skills.json", "s7_skills_rk.json", "s7_skills_bigdata.json"]:
        path = INTER / fname
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        for s in d.get("skills", []):
            sid = s.get("skill_id")
            txt = s.get("source") or s.get("text") or ""
            if sid:
                skills[sid] = txt
                sk_level[sid] = verb_level(txt)

    bundle = torch.load(INTER / "s6_graph.pt", weights_only=False)
    concepts = bundle["concepts"]
    idx = {c: i for i, c in enumerate(concepts)}
    n = len(concepts)

    concept_levels = {i: [] for i in range(n)}
    for fname in POOLS:
        path = INTER / fname
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        for cand in d.get("candidates", []):
            sid = cand.get("skill_id")
            lv = sk_level.get(sid, 0)
            if lv == 0:
                continue
            for t in cand.get("topk", []):
                if float(t["sim"]) >= args.tau and t["concept"] in idx:
                    concept_levels[idx[t["concept"]]].append(lv)

    bloom = np.zeros(n, dtype=np.float32)
    covered = 0
    for i, lvs in concept_levels.items():
        if lvs:
            bloom[i] = median_level(lvs)
            covered += 1

    np.save(args.out, bloom)
    dist = {int(lv): int((bloom == lv).sum()) for lv in range(1, 7)}
    meta = {
        "rule": "concept Bloom = median over aligned-skill action-verb levels; verb matched by dictionary longest-level; max-aggregation discarded for degenerate distribution",
        "tau": args.tau,
        "annotation_type": "weak dictionary mapping, not human annotation",
        "n_concepts": n,
        "covered": covered,
        "uncovered": n - covered,
        "level_distribution": dist,
        "skills_with_verb": int(sum(1 for v in sk_level.values() if v > 0)),
        "skills_total": len(sk_level),
    }
    Path(args.meta_out).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"布鲁姆层级落盘 {args.out}")
    print(f"覆盖概念 {covered}/{n}  层级分布 {dist}")
    print(f"技能含可识别动词 {meta['skills_with_verb']}/{meta['skills_total']}")


if __name__ == "__main__":
    main()
