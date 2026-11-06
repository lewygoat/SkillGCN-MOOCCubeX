import argparse
import json
from pathlib import Path

import numpy as np

INTER = Path("../中间产物")
FILES = [
    "s8_align_annotation.json",
    "s8_align_rk_annotation.json",
    "s8_align_bigdata_annotation.json",
]
LABELS = ["否", "待议", "是"]
LABEL_ORDER = {"否": 0, "待议": 1, "是": 2}

GUIDE = """
技能-概念对齐人工标注指南

一、任务说明
每条记录呈现一个职业技能标签与一门课程概念的候选对应关系。标注者需独立
判断该技能标签所描述的能力，与该课程概念所承载的知识点之间，是否构成有效
的跨层语义对齐。判断依据技能文本与概念名称本身的语义，不参考机器预测列。

二、标签集与判定标准
标签取三个值之一，禁止留空、禁止使用其他字符。

是——技能标签的核心能力与概念直接对应，学习该概念对掌握该技能有直接支撑。
例如技能\"能使用SQL进行数据查询\"与概念\"结构化查询语言\"构成直接对应。

否——技能标签与概念分属不同知识范畴，或仅有词面重叠而无实质能力关联。
例如技能\"能使用PHP基本语法\"与概念\"抽象数据类型\"，二者无直接支撑关系。

待议——技能与概念存在部分关联但难以明确归入是或否，例如概念是技能的远端
前置知识、或技能表述过宽导致覆盖面模糊。此标签用于真实存疑的情形，不作为
回避判断的默认选项。

三、边界处理
技能文本被截断时，依据可见部分的主要能力方向判断。概念名称有歧义时，取其
在信息技术类课程中的通行含义。同一技能对应多个概念时，逐条独立判断，不受
其他条目影响。

四、独立性要求
两位标注者须在互不通气的前提下各自完成全部条目，分别填入rater_a与rater_b。
完成后运行本脚本计算Cohen's Kappa，据此评估对齐判定的标注者间信度。

五、操作步骤
逐条编辑中间产物目录下三个标注文件，填写rater_a或rater_b字段，取值限
是、否、待议。两位标注者全部填毕后执行 python3 kappa_align.py 查看结果，
执行 python3 kappa_align.py --check 查看填写进度。
"""


def load_all():
    records = []
    for name in FILES:
        path = INTER / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            row["_source"] = name
            records.append(row)
    return records


def check_progress():
    print("标注填写进度")
    total_a = total_b = total_n = 0
    for name in FILES:
        path = INTER / name
        if not path.exists():
            print(f"  {name}  文件缺失")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        n = len(data)
        a = sum(1 for x in data if str(x.get("rater_a", "")).strip())
        b = sum(1 for x in data if str(x.get("rater_b", "")).strip())
        total_a += a
        total_b += b
        total_n += n
        print(f"  {name}  共{n}条  rater_a已填{a}  rater_b已填{b}")
    print(f"合计  共{total_n}条  rater_a已填{total_a}  rater_b已填{total_b}")
    if total_a < total_n or total_b < total_n:
        print("尚未填毕，两列均填满后方可计算Kappa")
    else:
        print("两列均已填毕，可执行计算")


def validate(records):
    valid = []
    bad = []
    missing = 0
    for r in records:
        a = str(r.get("rater_a", "")).strip()
        b = str(r.get("rater_b", "")).strip()
        if not a or not b:
            missing += 1
            continue
        if a not in LABEL_ORDER or b not in LABEL_ORDER:
            bad.append((r.get("_source"), r.get("skill_id"),
                        r.get("concept"), a, b))
            continue
        valid.append((a, b))
    return valid, bad, missing


def confusion(valid):
    k = len(LABELS)
    mat = np.zeros((k, k), dtype=float)
    for a, b in valid:
        mat[LABEL_ORDER[a], LABEL_ORDER[b]] += 1
    return mat


def cohen_kappa(mat, weight=None):
    n = mat.sum()
    po_mat = mat / n
    row = po_mat.sum(axis=1)
    col = po_mat.sum(axis=0)
    k = mat.shape[0]
    expected = np.outer(row, col)
    if weight is None:
        w = 1.0 - np.eye(k)
    elif weight == "linear":
        w = np.array([[abs(i - j) / (k - 1) for j in range(k)]
                      for i in range(k)])
    elif weight == "quadratic":
        w = np.array([[((i - j) / (k - 1)) ** 2 for j in range(k)]
                      for i in range(k)])
    obs_dis = (w * po_mat).sum()
    exp_dis = (w * expected).sum()
    if exp_dis == 0:
        return float("nan")
    return 1.0 - obs_dis / exp_dis


def interpret(kappa):
    if kappa != kappa:
        return "无法判定"
    if kappa < 0:
        return "低于随机"
    if kappa < 0.20:
        return "轻微一致"
    if kappa < 0.40:
        return "尚可一致"
    if kappa < 0.60:
        return "中等一致"
    if kappa < 0.80:
        return "高度一致"
    return "近乎完全一致"


def compute():
    records = load_all()
    if not records:
        print("未找到标注文件，无法计算")
        return
    valid, bad, missing = validate(records)
    print(f"记录总数 {len(records)}  有效双标注 {len(valid)}  "
          f"缺失未填 {missing}  异常取值 {len(bad)}")
    if bad:
        print("异常取值条目，须修正为 是/否/待议 后重算")
        for src, sid, con, a, b in bad[:20]:
            print(f"  {src}  {sid}  {con}  rater_a={a}  rater_b={b}")
    if not valid:
        print("无有效双标注条目，须先由两位标注者填写 rater_a 与 rater_b")
        print("填写规范见 python3 kappa_align.py --guide")
        return
    mat = confusion(valid)
    n = mat.sum()
    po = np.trace(mat) / n
    kappa = cohen_kappa(mat)
    kappa_lin = cohen_kappa(mat, "linear")
    kappa_quad = cohen_kappa(mat, "quadratic")
    print("\n混淆矩阵，行rater_a列rater_b，顺序 否/待议/是")
    header = "        " + "  ".join(f"{l:>4}" for l in LABELS)
    print(header)
    for i, l in enumerate(LABELS):
        print(f"  {l:>4}  " + "  ".join(f"{int(mat[i, j]):>4}"
                                        for j in range(len(LABELS))))
    print(f"\n观察一致率 Po {po:.4f}")
    print(f"Cohen's Kappa 未加权 {kappa:.4f}  {interpret(kappa)}")
    print(f"加权Kappa 线性 {kappa_lin:.4f}  {interpret(kappa_lin)}")
    print(f"加权Kappa 二次 {kappa_quad:.4f}  {interpret(kappa_quad)}")
    print("\n逐类别标注者间一致条目数")
    for i, l in enumerate(LABELS):
        agree = int(mat[i, i])
        total_i = int(mat[i, :].sum() + mat[:, i].sum() - mat[i, i])
        print(f"  {l}  一致{agree}  涉及{total_i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.guide:
        print(GUIDE)
    elif args.check:
        check_progress()
    else:
        compute()


if __name__ == "__main__":
    main()
