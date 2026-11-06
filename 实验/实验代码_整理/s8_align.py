import argparse
import json
import os
import random

import numpy as np
import torch

from common import INTER

_LOCAL_BGE = os.path.expanduser("~/models/bge-small-zh-v1.5")
MODEL_DIR = _LOCAL_BGE if os.path.exists(_LOCAL_BGE) else "BAAI/bge-small-zh-v1.5"


def load_concepts(path):
    b = torch.load(path, weights_only=False)
    return list(b["concepts"]), int(b["n_concepts"])


def load_skills(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["skills"]


def encode_bge(texts, model_dir):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_dir)
        v = m.encode(texts, normalize_embeddings=True, batch_size=64,
                     show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)
    except ImportError:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_dir)
        mod = AutoModel.from_pretrained(model_dir)
        mod.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
                out = mod(**enc)
                cls = out.last_hidden_state[:, 0]
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                outs.append(cls.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)


def encode_tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          sublinear_tf=True, min_df=1)
    m = vec.fit_transform(texts).astype(np.float32).toarray()
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-8, None)


def build_candidates(sim, topk):
    cands = []
    for i in range(sim.shape[0]):
        row = sim[i]
        idx = np.argsort(-row)[:topk]
        cands.append([{"concept": int(j), "sim": round(float(row[j]), 6)}
                      for j in idx])
    return cands


def full_annotation_table(cands, skills, concepts):
    out = []
    for i, cs in enumerate(cands):
        for c in cs:
            out.append({"skill_id": skills[i]["skill_id"],
                        "skill_text": skills[i]["text"],
                        "concept": concepts[c["concept"]],
                        "sim": c["sim"], "rater_a": "", "rater_b": ""})
    return out


def sample_for_annotation(cands, skills, concepts, per_bin, seed):
    rng = random.Random(seed)
    bins = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.01)]
    pool = {b: [] for b in bins}
    for i, cs in enumerate(cands):
        for c in cs:
            s = c["sim"]
            for lo, hi in bins:
                if lo <= s < hi:
                    pool[(lo, hi)].append((i, c))
                    break
    out = []
    for b, items in pool.items():
        take = min(per_bin, len(items))
        for i, c in (rng.sample(items, take) if take else []):
            out.append({"stratum": f"[{b[0]:.2f},{b[1]:.2f})",
                        "skill_id": skills[i]["skill_id"],
                        "skill_text": skills[i]["text"],
                        "concept": concepts[c["concept"]],
                        "sim": c["sim"],
                        "rater_a": "", "rater_b": ""})
    rng.shuffle(out)
    return out


def cohen_kappa(a, b):
    lab = sorted(set(a) | set(b))
    n = len(a)
    if n == 0 or len(lab) < 2:
        return None
    idx = {l: i for i, l in enumerate(lab)}
    m = np.zeros((len(lab), len(lab)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(0) / n) @ (m.sum(1) / n))
    if abs(1.0 - pe) < 1e-12:
        return None
    return (po - pe) / (1.0 - pe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skills", default=str(INTER / "s7_skills.json"))
    p.add_argument("--graph", default=str(INTER / "s6_graph.pt"))
    p.add_argument("--backend", choices=["bge", "tfidf"], default="bge")
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--per-bin", type=int, default=30)
    p.add_argument("--full", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kappa-from", default=None)
    p.add_argument("--confirmed-out", default=None)
    p.add_argument("--out", default=str(INTER / "s8_align.json"))
    args = p.parse_args()

    if args.kappa_from:
        rows = json.load(open(args.kappa_from, encoding="utf-8"))
        rated = [r for r in rows if r.get("rater_a") not in (None, "")
                and r.get("rater_b") not in (None, "")]
        a = [r["rater_a"] for r in rated]
        b = [r["rater_b"] for r in rated]
        k = cohen_kappa(a, b)
        agree = sum(1 for x, y in zip(a, b) if x == y)
        confirmed = [r for r in rated
                    if r["rater_a"] == "是" and r["rater_b"] == "是"]
        kappa_val = float(k) if k is not None else None
        print(json.dumps({"n_rated": len(rated), "n_total": len(rows),
                          "kappa": kappa_val,
                          "agree_rate": round(agree / max(len(rated), 1), 4),
                          "n_confirmed_edges": len(confirmed),
                          "kappa_ok": bool(kappa_val is not None and kappa_val >= 0.7)},
                         ensure_ascii=False, indent=1))
        if args.confirmed_out and confirmed:
            json.dump(confirmed, open(args.confirmed_out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print("已确认对齐边写入", args.confirmed_out)
        return

    concepts, nc = load_concepts(args.graph)
    skills = load_skills(args.skills)
    st = [s["text"] for s in skills]

    backend = args.backend
    if backend == "bge":
        try:
            emb = encode_bge(st + concepts, args.model_dir)
        except Exception as e:
            print("bge不可用，回退tfidf:", type(e).__name__)
            backend = "tfidf"
            emb = encode_tfidf(st + concepts)
    else:
        emb = encode_tfidf(st + concepts)

    es = emb[:len(st)]
    ec = emb[len(st):]
    with np.errstate(all="ignore"):
        sim = es @ ec.T
    if not np.isfinite(sim).all():
        raise RuntimeError("相似度矩阵含非有限值")

    cands = build_candidates(sim, args.topk)
    top1 = sim.max(axis=1)

    covered_s = sum(1 for cs in cands if cs)
    n_cand_edges = sum(len(cs) for cs in cands)
    stat = {"backend": backend, "topk": args.topk,
            "n_skills": len(skills), "n_concepts": nc,
            "n_candidate_edges": n_cand_edges,
            "skill_with_candidates": round(covered_s / max(len(skills), 1), 4),
            "top1_sim_median": round(float(np.median(top1)), 6),
            "top1_sim_p10": round(float(np.percentile(top1, 10)), 6),
            "top1_sim_p90": round(float(np.percentile(top1, 90)), 6),
            "note": "对齐边由人工标注确定，本文件仅含候选池，非最终对齐结果"}

    if args.full:
        ann = full_annotation_table(cands, skills, concepts)
    else:
        ann = sample_for_annotation(cands, skills, concepts,
                                    args.per_bin, args.seed)

    out = {"stat": stat,
           "candidates": [{"skill_id": skills[i]["skill_id"],
                           "skill_text": skills[i]["text"],
                           "topk": [{"concept": concepts[c["concept"]],
                                    "sim": c["sim"]} for c in cs]}
                          for i, cs in enumerate(cands)]}
    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ap = args.out.replace(".json", "_annotation.json")
    json.dump(ann, open(ap, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(stat, ensure_ascii=False, indent=1))
    print("标注样本", len(ann), "条写入", ap)


if __name__ == "__main__":
    main()
