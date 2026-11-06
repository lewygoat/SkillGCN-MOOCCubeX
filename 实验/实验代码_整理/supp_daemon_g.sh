#!/bin/bash
cd "$(dirname "$0")"
RUNS="../结果/supp_runs"
G=$1
LOG="../结果/supp_daemon_g${G}.log"
mkdir -p "$RUNS"
echo "==== 补充实验守护启动 $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$LOG"

JOBS=()
for sd in 42 43 44 45 46; do
  JOBS+=("gsweep_g${G}_s${sd}|--config t3 --seed $sd --gate-mode fixed --gate-value $G")
done

run_one() {
  local name=$1; shift
  python3 -u supp_run.py --name "$name" "$@" >> "$LOG" 2>&1 &
  local pid=$!
  local lastsz=0 idle=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    local sz=$(stat -f%z "$LOG" 2>/dev/null || echo 0)
    if [ "$sz" -eq "$lastsz" ]; then idle=$((idle+30)); else idle=0; lastsz=$sz; fi
    if [ $idle -ge 900 ]; then
      echo "停滞${idle}s，终止 $name" >> "$LOG"
      kill -9 $pid 2>/dev/null
      break
    fi
  done
  wait $pid 2>/dev/null
}

for job in "${JOBS[@]}"; do
  name="${job%%|*}"; rest="${job#*|}"
  tries=0
  while [ ! -f "$RUNS/$name.json" ] && [ $tries -lt 5 ]; do
    tries=$((tries+1))
    echo "[$(date '+%H:%M:%S')] 开始 $name 第${tries}次" >> "$LOG"
    run_one "$name" $rest
    [ -f "$RUNS/$name.json" ] || sleep 20
  done
  [ -f "$RUNS/$name.json" ] || echo "放弃 $name" >> "$LOG"
done
echo "==== 全部完成 $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$LOG"
