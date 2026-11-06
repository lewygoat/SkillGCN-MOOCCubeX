import json
import subprocess
import time
from pathlib import Path

from common import LOG, OUT


def heartbeats(tag=None, limit=400):
    path = LOG / "heartbeat.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if tag and rec.get("tag") != tag:
                continue
            rows.append(rec)
    return rows[-limit:]


SHELLS = ("zsh", "bash", "sh", "grep", "pgrep", "ps")


def alive(pattern):
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    pids = []
    for pid in r.stdout.split():
        q = subprocess.run(["ps", "-ww", "-o", "command=", "-p", pid],
                           capture_output=True, text=True)
        tokens = q.stdout.strip().split()
        if not tokens:
            continue
        exe = tokens[0].rsplit("/", 1)[-1].lower()
        if any(exe.startswith(s) for s in SHELLS):
            continue
        if any(t.rsplit("/", 1)[-1] == pattern for t in tokens[1:]):
            pids.append(pid)
    return pids


def stale_seconds(last_at):
    if not last_at:
        return None
    try:
        t = time.mktime(time.strptime(last_at, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None
    return time.time() - t


def main():
    rows = heartbeats()
    if not rows:
        print("无心跳记录")
        return

    grouped = {}
    for rec in rows:
        grouped.setdefault(rec["tag"], []).append(rec)

    tags = {}
    for tag, recs in grouped.items():
        starts = [i for i, r in enumerate(recs) if r.get("event") == "start"]
        tags[tag] = recs[starts[-1]:] if starts else recs

    print(f"=== 实验监控  {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    for tag, recs in tags.items():
        last = recs[-1]
        pids = alive(f"m_{tag}.py") or alive(f"m_{tag.split('_')[0]}.py")
        state = "运行中" if pids else "已停止"
        events = [r for r in recs if r.get("event")]
        done = [r for r in events if r.get("event") == "seed_done"]
        violations = [r for r in events if r.get("event") == "guard_violation"]
        finished = any(r.get("event") == "done" for r in events)

        print(f"── {tag}  {state}  pid={','.join(pids) if pids else '-'}")
        if finished:
            fin = [r for r in events if r.get("event") == "done"][-1]
            print(f"   已完成  成功种子 {fin.get('seeds_ok')}  耗时 {fin.get('elapsed_s')}s")
        else:
            print(f"   完成种子 {len(done)}  最近心跳 {last.get('at')}  已运行 {last.get('elapsed_s')}s")
            if "epoch" in last:
                print(f"   当前 seed={last.get('seed')} epoch={last.get('epoch')} "
                      f"loss={last.get('loss')} valid_ndcg10={last.get('valid_ndcg10')} "
                      f"best={last.get('best_ndcg10')} stale={last.get('stale')} "
                      f"epoch_s={last.get('epoch_seconds')}")
            gap = stale_seconds(last.get("at"))
            if gap is not None and gap > 900:
                print(f"   警告 心跳停滞 {gap/60:.1f} 分钟，疑似挂起或已被终止")
        for v in violations:
            print(f"   防御触发 seed={v.get('seed')} {v.get('error')}")
        for r in done:
            print(f"   seed {r.get('seed')} epochs={r.get('epochs')} test_ndcg10={r.get('test_ndcg10')}")

    print("=== 已产出结果 ===")
    for p in sorted(OUT.glob("r_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        s = d.get("summary", {})
        nd = s.get("NDCG@10", {})
        hr = s.get("HR@10", {})
        print(f"   {d.get('method','?'):<14} NDCG@10={nd.get('mean',0):.4f}±{nd.get('std',0):.4f}  "
              f"HR@10={hr.get('mean',0):.4f}  种子{len(d.get('per_seed',[]))}")


if __name__ == "__main__":
    main()
