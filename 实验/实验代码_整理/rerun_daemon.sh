#!/bin/bash
cd "$(dirname "$0")"
RUNS="../结果/newS9_runs"
LOG="../结果/rerun_daemon.log"
GATE=6.0
mkdir -p "$RUNS"
echo "==== 多随机种子重跑守护启动 $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$LOG"

run_one() {
  local cfg=$1 sd=$2
  python3 -u rerun_all.py --config "$cfg" --seed "$sd" >> "$LOG" 2>&1 &
  local pid=$!
  local last=0 stall=0 cur
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    cur=$(stat -f %z "$LOG" 2>/dev/null || echo 0)
    if [ "$cur" -gt "$last" ]; then last=$cur; stall=0; else stall=$((stall+1)); fi
    if [ "$stall" -ge 6 ]; then
      kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
      echo "$(date '+%H:%M:%S') $cfg s$sd 六分钟无进度判饿死，杀掉重试" >> "$LOG"
      return 1
    fi
  done
  return 0
}

for attempt in $(seq 1 200); do
  pending=""
  for cfg in t1 t2 t3; do
    for sd in 42 43 44 45 46; do
      if [ ! -f "$RUNS/run_${cfg}_s${sd}.json" ]; then
        pending="$cfg $sd"
        break 2
      fi
    done
  done
  if [ -z "$pending" ]; then
    echo "$(date '+%H:%M:%S') 15个组合全部完成，守护退出" >> "$LOG"
    exit 0
  fi
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk "BEGIN{exit !($load < $GATE)}"; then
    set -- $pending
    echo "$(date '+%H:%M:%S') 负载$load 启动 $1 s$2" >> "$LOG"
    run_one "$1" "$2"
  else
    echo "$(date '+%H:%M:%S') 负载$load 偏高，等3分钟（第${attempt}轮）" >> "$LOG"
    sleep 180
  fi
done
echo "$(date '+%H:%M:%S') 达到上限，守护退出" >> "$LOG"
