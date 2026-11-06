import collections
import gzip
import json

from common import INTER, OUT, load_protocol, provenance, report, write_jsonl


def load(name):
    with gzip.open(INTER / f"s4_{name}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iterative_kcore(rows, min_user, min_concept):
    users = {r["user"] for r in rows}
    concepts = {r["concept"] for r in rows}
    rounds = 0
    while True:
        rounds += 1
        udeg = collections.Counter()
        cdeg = collections.Counter()
        for r in rows:
            if r["user"] in users and r["concept"] in concepts:
                udeg[r["user"]] += 1
                cdeg[r["concept"]] += 1
        keep_u = {u for u in users if udeg[u] >= min_user}
        keep_c = {c for c in concepts if cdeg[c] >= min_concept}
        if keep_u == users and keep_c == concepts:
            break
        users, concepts = keep_u, keep_c
        if not users or not concepts:
            break
    kept = [r for r in rows if r["user"] in users and r["concept"] in concepts]
    return kept, users, concepts, rounds


def main():
    proto = load_protocol()
    thr = proto["behavior"]["effective_watch_threshold"]
    kc = proto["kcore"]

    train = list(load("train"))
    valid = list(load("valid"))
    test = list(load("test"))

    train_eff = [r for r in train if r["coverage"] >= thr]
    valid_eff = [r for r in valid if r["coverage"] >= thr]
    test_eff = [r for r in test if r["coverage"] >= thr]

    kept, users, concepts, rounds = iterative_kcore(
        train_eff, kc["min_user_degree"], kc["min_concept_degree"])

    valid_kept = [r for r in valid_eff if r["user"] in users and r["concept"] in concepts]
    test_kept = [r for r in test_eff if r["user"] in users and r["concept"] in concepts]

    cold_users = {r["user"] for r in valid_eff + test_eff} - users
    cold_concepts = {r["concept"] for r in valid_eff + test_eff} - concepts
    cold_records = [r for r in valid_eff + test_eff
                    if r["user"] in cold_users or r["concept"] in cold_concepts]

    for name, rows in (("train", kept), ("valid", valid_kept), ("test", test_kept)):
        write_jsonl(INTER / f"s5_{name}.jsonl.gz", rows)
    write_jsonl(INTER / "s5_coldstart.jsonl.gz", cold_records)

    tot_pairs = len(kept)
    stats = {
        "threshold_applied": thr,
        "train_before_threshold": len(train),
        "train_after_threshold": len(train_eff),
        "threshold_retention": len(train_eff) / len(train) if train else 0.0,
        "kcore_rounds": rounds,
        "train_after_kcore": tot_pairs,
        "kcore_retention": tot_pairs / len(train_eff) if train_eff else 0.0,
        "learners_kept": len(users),
        "concepts_kept": len(concepts),
        "pairs_per_learner": tot_pairs / len(users) if users else 0.0,
        "density": tot_pairs / (len(users) * len(concepts)) if users and concepts else 0.0,
        "valid_kept": len(valid_kept),
        "test_kept": len(test_kept),
        "coldstart_learners": len(cold_users),
        "coldstart_concepts": len(cold_concepts),
        "coldstart_records": len(cold_records),
        "test_learners_evaluable": len({r["user"] for r in test_kept}),
    }

    report("阶段五 阈值应用与训练段k-core", stats)

    edges = {"adopted": 0, "predicted": 0, "containment": 0}
    for rec in [json.loads(l) for l in open(INTER / "s2_edges_model_input.jsonl", encoding="utf-8")]:
        if rec["head"] in concepts and rec["tail"] in concepts:
            edges["adopted"] += 1
    for rec in [json.loads(l) for l in open(INTER / "s2_edges_model_input_predicted.jsonl", encoding="utf-8")]:
        if rec["head"] in concepts and rec["tail"] in concepts:
            edges["predicted"] += 1
    for rec in [json.loads(l) for l in open(INTER / "s2_edges_containment.jsonl", encoding="utf-8")]:
        if rec["concept"] in concepts:
            edges["containment"] += 1

    pos = [json.loads(l) for l in open(INTER / "s1_eval_positive.jsonl", encoding="utf-8")]
    neg = [json.loads(l) for l in open(INTER / "s1_eval_negative.jsonl", encoding="utf-8")]
    pos_in = sum(1 for d in pos if d["c1"] in concepts and d["c2"] in concepts)
    neg_in = sum(1 for d in neg if d["c1"] in concepts and d["c2"] in concepts)

    graph_stats = {
        "prereq_adopted_in_graph": edges["adopted"],
        "prereq_predicted_in_graph": edges["predicted"],
        "containment_in_graph": edges["containment"],
        "eval_positive_in_graph": pos_in,
        "eval_negative_in_graph": neg_in,
        "eval_positive_density": pos_in / (len(concepts) * (len(concepts) - 1)) if len(concepts) > 1 else 0.0,
    }
    report("阶段五 图规模核算", graph_stats)

    stats.update(graph_stats)
    with open(OUT / "s5_dataset_card.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    provenance("s5_filter", stats, [
        INTER / "s5_train.jsonl.gz",
        INTER / "s5_valid.jsonl.gz",
        INTER / "s5_test.jsonl.gz",
        INTER / "s5_coldstart.jsonl.gz",
        OUT / "s5_dataset_card.json",
    ])


if __name__ == "__main__":
    main()
