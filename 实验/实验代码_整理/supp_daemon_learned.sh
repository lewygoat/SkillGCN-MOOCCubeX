#!/bin/bash
cd "$(dirname "$0")"
RUNS="../结果/supp_runs"
G=learned
LOG="../结果/supp_daemon_g${G}.log"
mkdir -p "$RUNS"
echo "==== 补充实验守护启动 $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$LOG"

G45=$(python3 -c "import json,pathlib; p=pathlib.Path('../结果/newS9_runs/run_t3_s45.json'); print(json.loads(p.read_text())['gate_final'] if p.exists() else 2.87)" 2>/dev/null || echo 2.87)
G46=$(python3 -c "import json,pathlib; p=pathlib.Path('../结果/newS9_runs/run_t3_s46.json'); print(json.loads(p.read_text())['gate_final'] if p.exists() else 5.69)" 2>/dev/null || echo 5.69)
JOBS=(
  "gsweep_glearned_s42|--config t3 --seed 42 --gate-mode fixed --gate-value 2.889186"
  "gsweep_glearned_s43|--config t3 --seed 43 --gate-mode fixed --gate-value 3.7738"
  "gsweep_glearned_s44|--config t3 --seed 44 --gate-mode fixed --gate-value 3.771735"
  "gsweep_glearned_s45|--config t3 --seed 45 --gate-mode fixed --gate-value ${G45}"
  "gsweep_glearned_s46|--config t3 --seed 46 --gate-mode fixed --gate-value ${G46}"
)

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
