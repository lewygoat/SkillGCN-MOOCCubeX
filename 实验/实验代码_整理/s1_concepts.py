import collections

from common import (DATA, INTER, field_of, jsonl, load_protocol, name_of,
                   provenance, report, write_jsonl, write_set)


def benchmark_names():
    names = set()
    pos, neg = set(), set()
    for d in jsonl(DATA / "additional_information" / "prerequisite_prediction.json"):
        a, b = d.get("c1"), d.get("c2")
        if not a or not b:
            continue
        names.add(a)
        names.add(b)
        if d.get("label") == 1:
            pos.add((a, b))
        elif d.get("label") == 0:
            neg.add((a, b))
    return names, pos, neg


def main():
    proto = load_protocol()
    keep = set(proto["domain_filter"]["keep_fields"])

    names, pos, neg = benchmark_names()

    kept, name2id = {}, collections.defaultdict(set)
    field_hist = collections.Counter()
    all_cs = set()
    for d in jsonl(DATA / "entities" / "concept.json"):
        cid = d.get("id")
        if not cid:
            continue
        fld = field_of(cid)
        if fld == "计算机科学技术":
            all_cs.add(cid)
        if d.get("name") in names and fld in keep:
            kept[cid] = d.get("name")
            name2id[d["name"]].add(cid)
            field_hist[fld] += 1

    pos_in = {(a, b) for a, b in pos if a in name2id and b in name2id}
    neg_in = {(a, b) for a, b in neg if a in name2id and b in name2id}

    write_set(INTER / "s1_concept_ids.txt", kept)
    write_set(INTER / "s1_concept_ids_alignment_pool.txt", all_cs)
    write_jsonl(INTER / "s1_concept_meta.jsonl",
                ({"id": k, "name": v, "field": field_of(k)} for k, v in sorted(kept.items())))
    write_jsonl(INTER / "s1_eval_positive.jsonl",
                ({"c1": a, "c2": b} for a, b in sorted(pos_in)))
    write_jsonl(INTER / "s1_eval_negative.jsonl",
                ({"c1": a, "c2": b} for a, b in sorted(neg_in)))

    dup = {n: len(v) for n, v in name2id.items() if len(v) > 1}
    stats = {
        "benchmark_concept_names": len(names),
        "kept_concept_ids": len(kept),
        "kept_concept_names": len(name2id),
        "names_with_multiple_ids": len(dup),
        "alignment_pool_cs_concepts": len(all_cs),
        "eval_positive_all": len(pos),
        "eval_positive_in_subset": len(pos_in),
        "eval_negative_all": len(neg),
        "eval_negative_in_subset": len(neg_in),
    }
    for f, c in field_hist.most_common():
        stats[f"field_{f}"] = c

    report("阶段一 领域与概念筛选", stats)
    provenance("s1_concepts", stats, [
        INTER / "s1_concept_ids.txt",
        INTER / "s1_concept_meta.jsonl",
        INTER / "s1_eval_positive.jsonl",
        INTER / "s1_eval_negative.jsonl",
    ])


if __name__ == "__main__":
    main()
