import argparse
import json
import re

from common import INTER

RULES = {
    "r1_strip_numbering": r"^\s*(?:[（(]\s*\d+\s*[)）]|\d+\s*[.、]|[①②③④⑤⑥⑦⑧⑨⑩])\s*",
    "r2_split_enum": r"[、；;]|(?<![A-Za-z])和(?![A-Za-z])|以及|及其|(?<![A-Za-z])及(?![A-Za-z])",
    "r3_strip_modal": r"^(?:能够|能|会|可以|应能|掌握|熟悉|理解|了解|熟练掌握|初步掌握)",
    "r4_strip_tail": r"(?:的方法|的方式|的步骤|的流程|的原理|的概念|的基本知识|知识|要求|等)$",
    "r5_min_len": 2,
}

LEVEL_PAT = re.compile(r"(四级|三级|二级|一级|初级|中级|高级)")
TYPE_PAT = re.compile(r"(相关知识要求|知识要求|技能要求|工作内容|职业功能)")

X1_DOMAIN_PAT = re.compile(r"^(初级|中级|高级)\s*[–—-]\s*(.+)$")
X1_TASK_PAT = re.compile(r"^(\d+)\.(\d+)\s+(.+)$")
X1_ITEM_PAT = re.compile(r"^(\d+)\.(\d+)\.(\d+)\s*(.+)$")

RK_SUBJECT_PAT = re.compile(r"^考试科目\s*(\d+)\s*[：:]\s*(.+)$")
RK_L3_PAT = re.compile(r"^(\d+)\s*[.．]\s*(\d+)\s*[.．]\s*(\d+)\s*[.．]?\s*(.+)$")
RK_L2_PAT = re.compile(r"^(\d+)\s*[.．]\s*(\d+)\s*[.．]?\s*(?![\d])(.+)$")
RK_L1_PAT = re.compile(r"^(\d+)\s*[.．、]\s*(?![\d])(.+)$")
RK_SKIP_PAT = re.compile(r"^(?:来源|考试范围|考试科目|考试时间|[一二三四五六七八九十]+\s*[、.．]|附录|说明)")

RK_GENERIC = {"功能", "特性", "类型", "方法", "概念", "原理", "分类", "组成",
              "特点", "作用", "应用", "基础", "知识", "内容", "结构", "过程",
              "性能", "分析", "设计", "管理", "技术", "工具", "标准", "规范"}


def clean_ws(s):
    s = s.replace("　", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def norm(s):
    s = clean_ws(s)
    s = re.sub(RULES["r1_strip_numbering"], "", s)
    return s.strip(" .。，,;；:：")


def split_entry(text, kind):
    parts = [p for p in re.split(RULES["r2_split_enum"], text) if p]
    out = []
    for p in parts:
        p = p.strip(" .。，,;；:：")
        if kind == "S":
            p = re.sub(RULES["r3_strip_modal"], "", p).strip()
        p = re.sub(RULES["r4_strip_tail"], "", p).strip()
        if len(p) >= RULES["r5_min_len"]:
            out.append(p)
    return out


def split_depth0(s):
    parts, buf, depth = [], [], 0
    for ch in s:
        if ch in "([":
            depth += 1
            buf.append(ch)
        elif ch in ")]":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in "、，,;；" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip(" .。，,;；:：") for p in parts
            if p.strip(" .。，,;；:：")]


def split_paren_aware(line):
    out = []
    for seg in split_depth0(line):
        m = re.match(r"^([^()]*)\((.+)\)(.*)$", seg)
        if m and any(d in m.group(2) for d in "、，,;；"):
            head = m.group(1).strip(" .。，,;；:：")
            tail = m.group(3).strip(" .。，,;；:：")
            if head:
                out.append(head)
            for sub in split_depth0(m.group(2)):
                out.append(sub + tail if tail and len(sub) <= 6 else sub)
        else:
            s = seg.strip(" .。，,;；:：")
            if s.count("(") == s.count(")"):
                out.append(s)
            else:
                out.append(s.replace("(", "").replace(")", ""))
    return out


def parse_rk(lines, level):
    subject = subject_no = None
    l1_no = l1 = None
    l2_no = l2 = None
    l3_no = l3 = None
    items = []
    dropped = []
    in_scope = False

    for raw in lines:
        line = clean_ws(raw).replace("（", "(").replace("）", ")")
        if not line:
            continue
        if "考试范围" in line:
            in_scope = True
            continue

        m = RK_SUBJECT_PAT.match(line)
        if m:
            subject_no, subject = m.group(1), m.group(2)
            l1_no = l1 = l2_no = l2 = l3_no = l3 = None
            in_scope = True
            continue

        if not in_scope:
            continue
        if RK_SKIP_PAT.match(line):
            dropped.append({"stage": "skip", "text": line})
            continue

        m = RK_L3_PAT.match(line)
        if m:
            a, b, c, name = m.groups()
            l1_no, l2_no, l3_no = a, f"{a}.{b}", f"{a}.{b}.{c}"
            l3 = name.strip()
            continue

        m = RK_L2_PAT.match(line)
        if m:
            a, b, name = m.groups()
            l1_no, l2_no = a, f"{a}.{b}"
            l2, l3_no, l3 = name.strip(), None, None
            continue

        m = RK_L1_PAT.match(line)
        if m and len(line) <= 30:
            l1_no, l1 = m.group(1), m.group(2).strip()
            l2_no = l2 = l3_no = l3 = None
            continue

        if l2_no is None:
            dropped.append({"stage": "orphan", "text": line})
            continue

        unit = l3 or l2
        for frag in split_paren_aware(line):
            if frag in RK_GENERIC or len(frag) < RULES["r5_min_len"]:
                dropped.append({"stage": "generic", "text": frag})
                continue
            items.append({
                "level": level, "kind": "K",
                "subject_no": subject_no, "subject": subject,
                "area_no": l1_no, "area": l1,
                "unit_no": l3_no or l2_no, "unit": unit,
                "source": line, "text": frag,
            })

    seen = {}
    for it in items:
        key = (it["level"], it["text"])
        if key not in seen:
            seen[key] = it
    out = []
    for i, it in enumerate(seen.values()):
        it["skill_id"] = f"SK{i:04d}"
        out.append(it)
    return out, dropped


def parse_1x(lines, levels):
    level = None
    domain_no = None
    domain_name = None
    task_no = None
    task_name = None
    items = []
    dropped = []
    for raw in lines:
        line = clean_ws(raw)
        if not line:
            continue
        m = X1_DOMAIN_PAT.match(line)
        if m:
            level, domain_name = m.group(1), m.group(2)
            domain_no = None
            task_no = task_name = None
            continue
        m = X1_TASK_PAT.match(line)
        if m and len(line) <= 40:
            domain_no, task_no, task_name = m.group(1), m.group(2), m.group(3)
            continue
        m = X1_ITEM_PAT.match(line)
        if m:
            d_no, t_no, i_no, text = m.groups()
            if level not in levels:
                dropped.append({"level": level, "text": text})
                continue
            frag = text.strip(" .。，,;；:：")
            if len(frag) >= RULES["r5_min_len"]:
                items.append({
                    "level": level, "kind": "S",
                    "domain_no": d_no, "domain": domain_name,
                    "task_no": f"{d_no}.{t_no}", "task": task_name,
                    "item_no": f"{d_no}.{t_no}.{i_no}",
                    "source": text, "text": frag,
                })
            continue
        dropped.append({"level": level, "text": line})

    seen = {}
    for it in items:
        key = (it["level"], it["item_no"], it["text"])
        if key in seen:
            continue
        seen[key] = it
    out = []
    for i, it in enumerate(seen.values()):
        it["skill_id"] = f"SK{i:04d}"
        out.append(it)
    return out, dropped


def parse(lines, levels):
    level = None
    kind = None
    module = None
    items = []
    dropped = []
    for raw in lines:
        line = norm(raw)
        if not line:
            continue
        m = LEVEL_PAT.search(line)
        if m and len(line) <= 12:
            level = m.group(1)
            continue
        m = TYPE_PAT.search(line)
        if m and len(line) <= 14:
            head = m.group(1)
            kind = "K" if "知识" in head else ("S" if "技能" in head else None)
            if kind is None:
                module = line
            continue
        if level not in levels or kind is None:
            dropped.append({"level": level, "kind": kind, "text": line})
            continue
        for frag in split_entry(line, kind):
            items.append({"level": level, "kind": kind, "module": module,
                          "source": line, "text": frag})
    seen = {}
    for it in items:
        key = (it["level"], it["kind"], it["text"])
        if key in seen:
            continue
        seen[key] = it
    out = []
    for i, it in enumerate(seen.values()):
        it["skill_id"] = f"SK{i:04d}"
        out.append(it)
    return out, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--schema", choices=["gj", "1x", "rk"], default="1x")
    p.add_argument("--level", default="初级")
    p.add_argument("--levels", nargs="+", default=None)
    p.add_argument("--standard", default="Web前端开发职业技能等级标准（2021年2.0版）")
    p.add_argument("--out", default=str(INTER / "s7_skills.json"))
    args = p.parse_args()

    lines = open(args.src, encoding="utf-8").read().splitlines()
    if args.schema == "rk":
        levels = [args.level]
        items, dropped = parse_rk(lines, args.level)
    else:
        levels = args.levels or (["初级", "中级", "高级"] if args.schema == "1x"
                                 else ["四级", "三级"])
        fn = parse_1x if args.schema == "1x" else parse
        items, dropped = fn(lines, set(levels))

    stat = {"n_lines": len(lines), "n_skills": len(items),
            "n_dropped": len(dropped), "levels": levels,
            "schema": args.schema, "rules": RULES,
            "standard": args.standard}
    key_field = {"1x": "domain", "rk": "area"}.get(args.schema, "kind")
    stat["by_level_kind"] = {}
    for it in items:
        k = f"{it['level']}/{it.get(key_field) or '未分组'}"
        stat["by_level_kind"][k] = stat["by_level_kind"].get(k, 0) + 1

    json.dump({"stat": stat, "skills": items, "dropped": dropped},
              open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(stat, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
