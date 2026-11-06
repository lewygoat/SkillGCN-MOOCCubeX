import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

def _load_cjk_font():
    for fp in ("/System/Library/Fonts/STHeiti Medium.ttc",
               "C:/Windows/Fonts/msyh.ttc",
               "C:/Windows/Fonts/simhei.ttf"):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            return font_manager.FontProperties(fname=fp)
    return font_manager.FontProperties()

CJK = _load_cjk_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["savefig.dpi"] = 300

OUT = Path("../图表")
OUT.mkdir(exist_ok=True)

BASELINE = [
    ("LightGCN", 0.17640, 0.00038),
    ("BPR-MF", 0.17102, 0.00127),
    ("NGCF", 0.16927, 0.00084),
    ("Most-Popular", 0.16554, 0.0),
    ("KGAT", 0.16550, 0.00091),
    ("ConceptGCN", 0.16411, 0.00138),
    ("ACKRec", 0.16263, 0.00401),
]

ABLATION = [
    ("T1", "无技能层", 0.17640, 0.00038),
    ("T2", "权重打乱", 0.16822, 0.00124),
    ("T3", "全锚定", 0.16557, 0.00115),
]

GATE_SEED = [
    (42, 2.89, 11, 0.16679, 0.16202),
    (43, 3.77, 12, 0.16569, 0.16220),
    (44, 3.77, 12, 0.16509, 0.16242),
    (45, 2.87, 11, 0.16641, 0.16289),
    (46, 5.69, 14, 0.16387, 0.16194),
]


GRAPH_EDGES = [
    ("先修-预测", 50784),
    ("章节弱关联", 15444),
    ("包含", 617),
    ("先修-采纳", 194),
]

DATASET = {
    "学习者": 27276,
    "概念": 576,
    "分类节点": 103,
    "训练交互": 1300242,
    "验证交互": 168564,
    "测试交互": 350667,
}
