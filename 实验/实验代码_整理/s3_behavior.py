import collections
import json

from common import (DATA, INTER, jsonl, load_protocol, provenance, report,
                   write_jsonl)

CLIP_ENGAGEMENT = 3.0
MIN_SPEED = 0.5


def video_to_concepts():
    mapping = {}
    for rec in jsonl(INTER / "s2_video_concept.jsonl"):
        mapping[rec["video"]] = rec["concepts"]
    return mapping


def main():
    proto = load_protocol()
    drop_fields = set(proto["behavior"].get("drop_fields") or [])
    v2c = video_to_concepts()

    agg = collections.defaultdict(lambda: {
        "cov": 0.0, "eng": [], "rep": 0, "first": None, "last": None, "n": 0,
    })
    learners = set()
    scanned = hit = bad_duration = 0
    speeds = []

    with open(DATA / "additional_information" / "user_video_act.json", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("id")
            if not uid:
                continue
            for field in drop_fields:
                rec.pop(field, None)
            for act in rec.get("activity") or ():
                scanned += 1
                concepts = v2c.get(act.get("video_id"))
                if not concepts:
                    continue
                duration = act.get("video_duration") or 0
                if duration <= 0:
                    bad_duration += 1
                    continue
                progress = act.get("video_progress_time") or 0
                watched = act.get("local_watching_time") or 0
                count = act.get("watching_count") or 0
                speed = (progress / watched) if watched > 0 else 1.0
                speeds.append(speed)
                coverage = min(progress / duration, 1.0)
                engagement = min(watched * max(speed, MIN_SPEED) / duration, CLIP_ENGAGEMENT)
                start = act.get("local_start_time")
                end = act.get("local_end_time")
                hit += 1
                learners.add(uid)
                for concept in concepts:
                    slot = agg[(uid, concept)]
                    slot["cov"] = max(slot["cov"], coverage)
                    slot["eng"].append(engagement)
                    slot["rep"] += count
                    slot["n"] += 1
                    if start and (slot["first"] is None or start < slot["first"]):
                        slot["first"] = start
                    if end and (slot["last"] is None or end > slot["last"]):
                        slot["last"] = end

    def rows():
        for (uid, concept), slot in agg.items():
            engagement = sum(slot["eng"]) / len(slot["eng"]) if slot["eng"] else 0.0
            yield {
                "user": uid,
                "concept": concept,
                "coverage": round(slot["cov"], 6),
                "engagement": round(engagement, 6),
                "repetition": slot["rep"],
                "videos": slot["n"],
                "first_seen": slot["first"],
                "last_seen": slot["last"],
            }

    written = write_jsonl(INTER / "s3_user_concept_raw.jsonl.gz", rows())

    concepts_hit = len({c for _, c in agg})
    speeds.sort()
    stats = {
        "activities_scanned": scanned,
        "activities_hit": hit,
        "activities_bad_duration": bad_duration,
        "learners": len(learners),
        "concepts_touched": concepts_hit,
        "user_concept_pairs": written,
        "pairs_per_learner": written / len(learners) if learners else 0.0,
        "density": written / (len(learners) * concepts_hit) if learners and concepts_hit else 0.0,
        "speed_median": speeds[len(speeds) // 2] if speeds else 0.0,
        "speed_p90": speeds[int(len(speeds) * 0.9)] if speeds else 0.0,
        "name_field_dropped": "name" in drop_fields,
    }

    report("阶段三 行为层解析", stats)
    provenance("s3_behavior", stats, [INTER / "s3_user_concept_raw.jsonl.gz"])


if __name__ == "__main__":
    main()
