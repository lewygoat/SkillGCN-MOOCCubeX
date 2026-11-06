"""One-click data unpacker and environment integrity verifier for SkillGCN-MOOCCubeX.
Repository: https://github.com/lewygoat/SkillGCN-MOOCCubeX
"""
import gzip
import json
import shutil
from pathlib import Path

import common  # Triggers auto-unpack of s6_graph.pt.gz if needed


def main():
    inter = common.INTER
    out = common.OUT
    conf = common.CONF

    print("=== SkillGCN-MOOCCubeX Reproducibility Package Check ===")
    print(f"Repository: https://github.com/lewygoat/SkillGCN-MOOCCubeX")
    graph_pt = inter / "s6_graph.pt"
    graph_gz = inter / "s6_graph.pt.gz"
    if graph_pt.exists():
        sz_mb = graph_pt.stat().st_size / (1024 * 1024)
        print(f"[OK] Unpacked graph tensor: {graph_pt.name} ({sz_mb:.2f} MB)")
    elif graph_gz.exists():
        print(f"[..] Extracting {graph_gz.name} ...")
        with gzip.open(graph_gz, "rb") as fin, open(graph_pt, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"[OK] Extracted {graph_pt.name}")

    required_inter = [
        "s7_skills.json",
        "s7_skills_rk.json",
        "s7_skills_bigdata.json",
        "s8_align.json",
        "s9_skill_weight.npy",
        "s9_skill_weight_no_bigdata.npy",
        "s10_concept_bloom.npy",
        "s5_coldstart.jsonl.gz",
    ]
    for fn in required_inter:
        p = inter / fn
        assert p.exists(), f"Missing intermediate file: {p}"
    print(f"[OK] Verified {len(required_inter)} intermediate skill & cold-start artifacts in {inter}")

    proto = conf / "protocol.json"
    assert proto.exists(), f"Missing protocol config: {proto}"
    p_data = json.loads(proto.read_text(encoding="utf-8"))
    print(f"[OK] Loaded protocol.json (seeds={p_data.get('seeds')}, dim={p_data.get('embedding_dim')})")

    json_results = list(out.rglob("*.json"))
    print(f"[OK] Found {len(json_results)} precomputed experimental JSON logs in {out}")
    print("=== All checks passed! Ready to run verification or training scripts. ===")


if __name__ == "__main__":
    main()
