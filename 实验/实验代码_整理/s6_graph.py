import gzip
import json

import torch

from common import INTER, OUT, jsonl, load_protocol, provenance, report


def load_split(name):
    with gzip.open(INTER / f"s5_{name}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    proto = load_protocol()
    train = list(load_split("train"))
    valid = list(load_split("valid"))
    test = list(load_split("test"))

    users = sorted({r["user"] for r in train})
    concepts = sorted({r["concept"] for r in train})
    uidx = {u: i for i, u in enumerate(users)}
    cidx = {c: i for i, c in enumerate(concepts)}

    def pairs(rows):
        out = [(uidx[r["user"]], cidx[r["concept"]]) for r in rows
               if r["user"] in uidx and r["concept"] in cidx]
        return torch.tensor(out, dtype=torch.long).T if out else torch.zeros((2, 0), dtype=torch.long)

    train_ei = pairs(train)
    valid_ei = pairs(valid)
    test_ei = pairs(test)

    mastery = torch.zeros((len(users), len(concepts), 2), dtype=torch.float32)
    for r in train:
        if r["user"] in uidx and r["concept"] in cidx:
            i, j = uidx[r["user"]], cidx[r["concept"]]
            mastery[i, j, 0] = r["engagement"]
            mastery[i, j, 1] = r["coverage"]

    def concept_edges(path, head_key, tail_key):
        out = []
        for rec in jsonl(path):
            h, t = rec[head_key], rec[tail_key]
            if h in cidx and t in cidx:
                out.append((cidx[h], cidx[t]))
        return torch.tensor(sorted(set(out)), dtype=torch.long).T if out else torch.zeros((2, 0), dtype=torch.long)

    prereq_adopted = concept_edges(INTER / "s2_edges_model_input.jsonl", "head", "tail")
    prereq_pred = concept_edges(INTER / "s2_edges_model_input_predicted.jsonl", "head", "tail")
    chapter_weak = concept_edges(INTER / "s2_edges_chapter_weak.jsonl", "head", "tail")

    tax_nodes = sorted({rec["taxonomy"] for rec in jsonl(INTER / "s2_edges_containment.jsonl")})
    tidx = {t: i for i, t in enumerate(tax_nodes)}
    contain = []
    for rec in jsonl(INTER / "s2_edges_containment.jsonl"):
        if rec["concept"] in cidx:
            contain.append((tidx[rec["taxonomy"]], cidx[rec["concept"]]))
    contain_ei = torch.tensor(sorted(set(contain)), dtype=torch.long).T if contain else torch.zeros((2, 0), dtype=torch.long)

    def eval_pairs(path):
        out = []
        for rec in jsonl(path):
            if rec["c1"] in cidx and rec["c2"] in cidx:
                out.append((cidx[rec["c1"]], cidx[rec["c2"]]))
        return torch.tensor(sorted(set(out)), dtype=torch.long).T if out else torch.zeros((2, 0), dtype=torch.long)

    eval_pos = eval_pairs(INTER / "s1_eval_positive.jsonl")
    eval_neg = eval_pairs(INTER / "s1_eval_negative.jsonl")

    bundle = {
        "n_users": len(users),
        "n_concepts": len(concepts),
        "n_taxonomy": len(tax_nodes),
        "users": users,
        "concepts": concepts,
        "taxonomy": tax_nodes,
        "train_ei": train_ei,
        "valid_ei": valid_ei,
        "test_ei": test_ei,
        "mastery": mastery,
        "prereq_adopted": prereq_adopted,
        "prereq_predicted": prereq_pred,
        "containment": contain_ei,
        "chapter_weak": chapter_weak,
        "eval_prereq_positive": eval_pos,
        "eval_prereq_negative": eval_neg,
        "protocol_version": proto["protocol_version"],
    }
    torch.save(bundle, INTER / "s6_graph.pt")

    stats = {
        "n_users": len(users),
        "n_concepts": len(concepts),
        "n_taxonomy": len(tax_nodes),
        "train_interactions": int(train_ei.shape[1]),
        "valid_interactions": int(valid_ei.shape[1]),
        "test_interactions": int(test_ei.shape[1]),
        "prereq_adopted_edges": int(prereq_adopted.shape[1]),
        "prereq_predicted_edges": int(prereq_pred.shape[1]),
        "containment_edges": int(contain_ei.shape[1]),
        "chapter_weak_edges": int(chapter_weak.shape[1]),
        "eval_prereq_positive": int(eval_pos.shape[1]),
        "eval_prereq_negative": int(eval_neg.shape[1]),
        "mastery_nonzero": int((mastery[:, :, 0] > 0).sum()),
    }
    report("阶段六 图张量构建", stats)
    provenance("s6_graph", stats, [INTER / "s6_graph.pt"])

    with open(OUT / "s6_graph_card.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
