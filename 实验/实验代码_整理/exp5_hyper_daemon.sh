#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
LOG=../结果/exp5_hyper_daemon.log
MAXRETRY=200
exec >>"$LOG" 2>&1
echo "===================================================="
echo "超参扫描守护启动  $(date '+%Y-%m-%d %H:%M:%S')"
for i in $(seq 1 $MAXRETRY); do
    LOAD1=$(uptime | awk '{print $(NF-2)}' | tr -d ',')
    LOAD15=$(uptime | awk '{print $NF}' | tr -d ',')
    GATE=$(echo "$LOAD1 < 6.0 && $LOAD15 < 4.0" | bc -l)
    echo "轮$i  负载1min=$LOAD1 15min=$LOAD15  gate=$GATE  $(date '+%H:%M:%S')"
    if [ "$GATE" -eq 1 ]; then
        if [ -f ../结果/r_exp5_hyperparam.json ]; then
            echo "超参扫描已完成，守护退出"
            exit 0
        fi
        echo "窗口开启，启动超参扫描"
        python3 exp5_hyperparam.py
        echo "超参扫描完成  $(date '+%Y-%m-%d %H:%M:%S')"
        exit 0
    fi
    if [ $i -lt $MAXRETRY ]; then
        sleep 300
    fi
done
echo "达重试上限$MAXRETRY轮，守护退出  $(date '+%Y-%m-%d %H:%M:%S')"
exit 1
