import collections
import gzip
import json

from common import INTER, OUT, load_protocol, provenance, report, write_jsonl


def load_pairs():
    path = INTER / "s3_user_concept_raw.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def temporal_split(records, ratio):
    by_user = collections.defaultdict(list)
    undated = 0
    for r in records:
        if not r.get("first_seen"):
            undated += 1
            continue
        by_user[r["user"]].append(r)

    train, valid, test = [], [], []
    single = 0
    for uid, rows in by_user.items():
        rows.sort(key=lambda x: (x["first_seen"], x["concept"]))
        n = len(rows)
        if n < 3:
            single += 1
            train.extend(rows)
            continue
        n_tr = max(1, int(n * ratio[0]))
        n_va = max(1, int(n * ratio[1]))
        if n_tr + n_va >= n:
            n_va = max(1, n - n_tr - 1)
        train.extend(rows[:n_tr])
        valid.extend(rows[n_tr:n_tr + n_va])
        test.extend(rows[n_tr + n_va:])
    return train, valid, test, {"undated_records": undated,
                                "learners_too_short": single,
                                "learners_total": len(by_user)}


def histogram(values, bins=20):
    hist = [0] * bins
    for v in values:
        idx = min(int(v * bins), bins - 1)
        hist[idx] += 1
    return hist


def find_valley(hist):
    n = len(hist)
    peak_left = max(range(n // 2), key=lambda i: hist[i])
    peak_right = max(range(n // 2, n), key=lambda i: hist[i])
    if peak_right <= peak_left:
        return None, peak_left, peak_right
    valley = min(range(peak_left, peak_right + 1), key=lambda i: hist[i])
    left_h, right_h = hist[peak_left], hist[peak_right]
    valley_h = hist[valley]
    weaker_peak = min(left_h, right_h)
    if weaker_peak == 0 or valley_h / weaker_peak > 0.6:
        return None, peak_left, peak_right
    return valley, peak_left, peak_right


def main():
    proto = load_protocol()
    ratio = proto["split"]["ratio"]

    train, valid, test, split_stats = temporal_split(load_pairs(), ratio)

    coverage = [r["coverage"] for r in train]
    hist = histogram(coverage)
    total = len(coverage) or 1
    valley, pl, pr = find_valley(hist)

    print("── 训练段完成比例直方图（仅训练段）")
    peak = max(hist) or 1
    for i, count in enumerate(hist):
        lo, hi = i / len(hist), (i + 1) / len(hist)
        bar = "█" * int(40 * count / peak)
        mark = ""
        if valley is not None and i == valley:
            mark = "  <= 谷底"
        elif i in (pl, pr):
            mark = "  <= 峰"
        print(f"   [{lo:.2f},{hi:.2f})  {count:>9,}  {100*count/total:5.2f}%  {bar}{mark}")

    if valley is not None:
        threshold = round((valley + 1) / len(hist), 3)
        basis = "bimodal_valley"
    else:
        threshold = 0.3
        basis = "unimodal_prior"

    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        write_jsonl(INTER / f"s4_{name}.jsonl.gz", rows)

    with open(OUT / "s4_coverage_histogram.json", "w", encoding="utf-8") as f:
        json.dump({"bins": len(hist), "counts": hist, "split": "train_only",
                   "valley_bin": valley, "peak_left": pl, "peak_right": pr,
                   "threshold": threshold, "basis": basis}, f,
                  ensure_ascii=False, indent=2)

    stats = {
        "train_records": len(train),
        "valid_records": len(valid),
        "test_records": len(test),
        "train_share": len(train) / (len(train) + len(valid) + len(test)),
        "threshold": threshold,
        "threshold_basis": basis,
        "peak_left_bin": pl,
        "peak_right_bin": pr,
        "valley_bin": valley if valley is not None else -1,
    }
    stats.update(split_stats)

    report("阶段四 时序划分与阈值确定", stats)
    provenance("s4_split", stats, [
        INTER / "s4_train.jsonl.gz",
        INTER / "s4_valid.jsonl.gz",
        INTER / "s4_test.jsonl.gz",
        OUT / "s4_coverage_histogram.json",
    ])
    print(f"   阈值 {threshold} 依据 {basis}，仅用训练段确定，须写入protocol.json后冻结")


if __name__ == "__main__":
    main()
