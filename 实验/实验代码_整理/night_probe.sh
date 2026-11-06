#!/bin/bash
cd "$(dirname "$0")"
RES="../结果/supp_runs"
LOG="$RES/night_probe.log"
GATE=6.0
echo "==== 门控探测守护启动 $(date '+%Y-%m-%d %H:%M:%S')  负载闸门<$GATE ====" >> "$LOG"

run_one() {
  local s=$1
  python3 -u probe_gate.py --config t3 --seed "$s" >> "$LOG" 2>&1 &
  local pid=$!
  local last=0 stall=0 cur
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    cur=$(stat -f %z "$LOG" 2>/dev/null || echo 0)
    if [ "$cur" -gt "$last" ]; then
      last=$cur
      stall=0
    else
      stall=$((stall + 1))
    fi
    if [ "$stall" -ge 5 ]; then
      kill "$pid" 2>/dev/null
      sleep 2
      kill -9 "$pid" 2>/dev/null
      echo "$(date '+%H:%M:%S') seed=$s 五分钟无进度判饿死，杀掉等下窗口" >> "$LOG"
      return 1
    fi
  done
  return 0
}

for attempt in $(seq 1 80); do
  if [ -f "$RES/r_skillgcn_t3_gate_s45.json" ] && [ -f "$RES/r_skillgcn_t3_gate_s46.json" ]; then
    echo "$(date '+%H:%M:%S') 45与46均落盘，守护退出" >> "$LOG"
    exit 0
  fi
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk "BEGIN{exit !($load < $GATE)}"; then
    for s in 45 46; do
      if [ ! -f "$RES/r_skillgcn_t3_gate_s${s}.json" ]; then
        echo "$(date '+%H:%M:%S') 负载$load 空闲，启动 seed=$s" >> "$LOG"
        run_one "$s"
        break
      fi
    done
  else
    echo "$(date '+%H:%M:%S') 负载$load 偏高，等3分钟（第${attempt}轮）" >> "$LOG"
    sleep 180
  fi
done
echo "$(date '+%H:%M:%S') 达到重试上限，守护退出" >> "$LOG"
