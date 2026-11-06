# Official Repository: https://github.com/lewygoat/SkillGCN-MOOCCubeX
import json
import gzip
import shutil
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
ROOT = EXP.parent
DATA = EXP / "数据" / "MOOCCube"
CONF = EXP / "配置"
INTER = EXP / "中间产物"
OUT = EXP / "结果"
LOG = EXP / "日志"

for _d in (INTER, OUT, LOG):
    _d.mkdir(parents=True, exist_ok=True)

# Auto-unpack s6_graph.pt.gz on first run if s6_graph.pt is not present
_GRAPH_PT = INTER / "s6_graph.pt"
_GRAPH_GZ = INTER / "s6_graph.pt.gz"
if not _GRAPH_PT.exists() and _GRAPH_GZ.exists():
    print("[common] Extracting compressed graph artifact s6_graph.pt.gz -> s6_graph.pt ...")
    with gzip.open(_GRAPH_GZ, "rb") as _fin, open(_GRAPH_PT, "wb") as _fout:
        shutil.copyfileobj(_fin, _fout)


def load_protocol():
    with open(CONF / "protocol.json", encoding="utf-8") as f:
        return json.load(f)


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def tsv(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                yield parts


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    n = 0
    with opener(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_set(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(items)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")
    return len(items)


def read_set(path):
    with open(path, encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.strip()}


def field_of(concept_id):
    return concept_id.rsplit("_", 1)[-1] if concept_id.startswith("K_") else ""


def name_of(concept_id):
    if not concept_id.startswith("K_"):
        return concept_id
    return concept_id[2:].rsplit("_", 1)[0]


def git_commit():
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def file_digest(path, limit=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def provenance(stage, stats, outputs=()):
    rec = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "stats": stats,
        "outputs": {},
    }
    for p in outputs:
        p = Path(p)
        if p.exists():
            rec["outputs"][p.name] = {
                "bytes": p.stat().st_size,
                "sha256_head": file_digest(p),
            }
    path = LOG / "provenance.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def report(stage, stats):
    width = max((len(k) for k in stats), default=0)
    print(f"── {stage}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"   {k:<{width}}  {v:,.4f}")
        elif isinstance(v, int):
            print(f"   {k:<{width}}  {v:,}")
        else:
            print(f"   {k:<{width}}  {v}")
