import collections

from common import (DATA, INTER, jsonl, load_protocol, name_of, provenance,
                   report, tsv, write_jsonl, write_set)


def id_to_name():
    mapping = {}
    for rec in jsonl(INTER / "s1_concept_meta.jsonl"):
        mapping[rec["id"]] = rec["name"]
    return mapping


def taxonomy_layer(keep_ids, i2n):
    children = collections.defaultdict(set)
    parents = collections.defaultdict(set)
    for parts in tsv(DATA / "relations" / "parent-son.json"):
        children[parts[0]].add(parts[1])
        parents[parts[1]].add(parts[0])

    containment = set()
    for cid in keep_ids:
        for tax in parents.get(cid, ()):
            containment.add((tax, i2n[cid]))

    direct = {t for c in keep_ids for t in parents.get(c, ())}
    closure = set(direct)
    frontier = set(direct)
    tax_edges = set()
    depth = 0
    while frontier and depth < 10:
        nxt = set()
        for node in frontier:
            for up in parents.get(node, ()):
                tax_edges.add((up, node))
                if up not in closure:
                    nxt.add(up)
        if not nxt:
            break
        closure |= nxt
        frontier = nxt
        depth += 1

    multi = sum(1 for c in keep_ids if len(parents.get(c, ())) > 1)
    stats = {
        "taxonomy_nodes_direct": len(direct),
        "taxonomy_nodes_closure": len(closure),
        "taxonomy_to_concept_edges": len(containment),
        "taxonomy_internal_edges": len(tax_edges),
        "max_ancestor_depth": depth,
        "concepts_with_multiple_parents": multi,
    }
    return containment, tax_edges, closure, stats


def chapter_baseline(i2n, min_support, video_concepts):
    ordered = collections.Counter()
    courses_used = 0
    for c in jsonl(DATA / "entities" / "course.json"):
        chapters = c.get("chapter") or []
        videos = c.get("video_order") or []
        if not chapters or len(chapters) != len(videos):
            continue
        first = {}
        for code, vid in zip(chapters, videos):
            for name in video_concepts.get(vid, ()):
                if name not in first or code < first[name]:
                    first[name] = code
        if len(first) < 2:
            continue
        courses_used += 1
        items = sorted(first.items(), key=lambda kv: kv[1])
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][1] < items[j][1]:
                    ordered[(items[i][0], items[j][0])] += 1
    raw = set(ordered)
    no_conflict = {p for p in raw if (p[1], p[0]) not in raw}
    supported = {p for p in no_conflict if ordered[p] >= min_support}
    return supported, {
        "courses_used": courses_used,
        "chapter_pairs_raw": len(raw),
        "chapter_pairs_no_conflict": len(no_conflict),
        f"chapter_pairs_support_ge_{min_support}": len(supported),
    }


def main():
    proto = load_protocol()
    pe = proto["prerequisite_edges"]
    i2n = id_to_name()
    keep_ids = set(i2n)
    names = set(i2n.values())

    video_concepts = collections.defaultdict(set)
    vc_edges = 0
    for parts in tsv(DATA / "relations" / "video-concept.json"):
        vid, cid = parts[0], parts[1]
        if cid in keep_ids:
            video_concepts[vid].add(i2n[cid])
            vc_edges += 1

    course_concepts = collections.defaultdict(set)
    cc_edges = 0
    for parts in tsv(DATA / "relations" / "course-concept.json"):
        cid_course, cid = parts[0], parts[1]
        if cid in keep_ids:
            course_concepts[cid_course].add(i2n[cid])
            cc_edges += 1

    adopted = set()
    for parts in tsv(DATA / "relations" / "prerequisite-dependency.json"):
        a, b = name_of(parts[0]), name_of(parts[1])
        if a in names and b in names and a != b:
            adopted.add((a, b))

    thr = pe["model_input_predicted"]["threshold"]
    predicted = set()
    for d in jsonl(DATA / "additional_information" / "prerequisite_prediction.json"):
        if d.get("label") != -1:
            continue
        a, b = d.get("c1"), d.get("c2")
        if a not in names or b not in names:
            continue
        p = d.get("predict") or []
        if len(p) > 1 and p[1] > thr:
            predicted.add((a, b))

    containment, tax_edges, tax_nodes, tax_stats = taxonomy_layer(keep_ids, i2n)

    chapter, chapter_stats = chapter_baseline(
        i2n, pe["weak_baseline_chapter"]["min_course_support"], video_concepts)

    write_jsonl(INTER / "s2_video_concept.jsonl",
                ({"video": v, "concepts": sorted(cs)} for v, cs in sorted(video_concepts.items())))
    write_jsonl(INTER / "s2_course_concept.jsonl",
                ({"course": c, "concepts": sorted(cs)} for c, cs in sorted(course_concepts.items())))
    write_jsonl(INTER / "s2_edges_adopted.jsonl",
                ({"head": a, "tail": b} for a, b in sorted(adopted)))
    write_jsonl(INTER / "s2_edges_predicted.jsonl",
                ({"head": a, "tail": b} for a, b in sorted(predicted)))
    write_jsonl(INTER / "s2_edges_containment.jsonl",
                ({"taxonomy": a, "concept": b} for a, b in sorted(containment)))
    write_jsonl(INTER / "s2_edges_taxonomy_internal.jsonl",
                ({"parent": a, "child": b} for a, b in sorted(tax_edges)))
    write_set(INTER / "s2_taxonomy_nodes.txt", tax_nodes)
    write_jsonl(INTER / "s2_edges_chapter_weak.jsonl",
                ({"head": a, "tail": b} for a, b in sorted(chapter)))

    pos = {(d["c1"], d["c2"]) for d in jsonl(INTER / "s1_eval_positive.jsonl")}
    neg = {(d["c1"], d["c2"]) for d in jsonl(INTER / "s1_eval_negative.jsonl")}
    leak_adopted = len(adopted & pos)
    leak_pred = len(predicted & pos)
    adopted_clean = adopted - pos - neg
    predicted_clean = predicted - pos - neg

    write_jsonl(INTER / "s2_edges_model_input.jsonl",
                ({"head": a, "tail": b, "src": "adopted"} for a, b in sorted(adopted_clean)))
    write_jsonl(INTER / "s2_edges_model_input_predicted.jsonl",
                ({"head": a, "tail": b, "src": "predicted"} for a, b in sorted(predicted_clean)))

    stats = {
        "concept_nodes": len(names),
        "videos_with_concepts": len(video_concepts),
        "video_concept_edges": vc_edges,
        "courses_with_concepts": len(course_concepts),
        "course_concept_edges": cc_edges,
        "prereq_adopted_raw": len(adopted),
        "prereq_adopted_clean": len(adopted_clean),
        "prereq_predicted_raw": len(predicted),
        "prereq_predicted_clean": len(predicted_clean),
        "eval_positive": len(pos),
        "eval_negative": len(neg),
        "leak_removed_adopted": leak_adopted,
        "leak_removed_predicted": leak_pred,
    }
    stats.update(tax_stats)
    stats.update(chapter_stats)
    if chapter:
        agree = len(chapter & pos)
        disagree = len({(b, a) for a, b in chapter} & pos)
        hit_neg = len(chapter & neg)
        stats["chapter_vs_human_agree"] = agree
        stats["chapter_vs_human_reversed"] = disagree
        stats["chapter_hits_human_negative"] = hit_neg
        denom = agree + disagree
        stats["chapter_directional_precision"] = agree / denom if denom else 0.0
        denom2 = agree + disagree + hit_neg
        stats["chapter_precision_with_negatives"] = agree / denom2 if denom2 else 0.0

    report("阶段二 关系层构建", stats)
    provenance("s2_relations", stats, [
        INTER / "s2_edges_model_input.jsonl",
        INTER / "s2_edges_model_input_predicted.jsonl",
        INTER / "s2_edges_containment.jsonl",
        INTER / "s2_edges_taxonomy_internal.jsonl",
        INTER / "s2_edges_chapter_weak.jsonl",
        INTER / "s2_video_concept.jsonl",
    ])

    if leak_adopted or leak_pred:
        print(f"   泄漏剔除 adopted={leak_adopted} predicted={leak_pred}，建模边已与评测基准隔离")


if __name__ == "__main__":
    main()
