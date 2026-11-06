#!/bin/bash
cd "$(dirname "$0")"
EMB="../中间产物/嵌入"
LOG="../结果/embed_runner.log"
GATE=6.0
mkdir -p "$EMB"
echo "==== 嵌入导出守护启动 $(date '+%Y-%m-%d %H:%M:%S') ====" >> "$LOG"

run_one() {
  local cfg=$1
  python3 -u embed_export.py --config "$cfg" --seed 42 >> "$LOG" 2>&1 &
  local pid=$!
  local last=0 stall=0 cur
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    cur=$(stat -f %z "$LOG" 2>/dev/null || echo 0)
    if [ "$cur" -gt "$last" ]; then last=$cur; stall=0; else stall=$((stall+1)); fi
    if [ "$stall" -ge 6 ]; then
      kill "$pid" 2>/dev/null; sleep 2; kill -9 "$pid" 2>/dev/null
      echo "$(date '+%H:%M:%S') $cfg 六分钟无进度判饿死，杀掉重试" >> "$LOG"
      return 1
    fi
  done
  return 0
}

for attempt in $(seq 1 60); do
  if [ -f "$EMB/emb_t1_s42.pt" ] && [ -f "$EMB/emb_t3_s42.pt" ]; then
    echo "$(date '+%H:%M:%S') T1与T3嵌入均落盘，守护退出" >> "$LOG"
    exit 0
  fi
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  if awk "BEGIN{exit !($load < $GATE)}"; then
    for cfg in t1 t3; do
      if [ ! -f "$EMB/emb_${cfg}_s42.pt" ]; then
        echo "$(date '+%H:%M:%S') 负载$load 启动 $cfg 嵌入导出" >> "$LOG"
        run_one "$cfg"
        break
      fi
    done
  else
    echo "$(date '+%H:%M:%S') 负载$load 偏高，等3分钟（第${attempt}轮）" >> "$LOG"
    sleep 180
  fi
done
echo "$(date '+%H:%M:%S') 达到上限，守护退出" >> "$LOG"
