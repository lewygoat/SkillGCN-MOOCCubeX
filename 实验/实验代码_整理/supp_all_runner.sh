#!/bin/bash
# 一键补充实验执行脚本 (supp_all_runner.sh)
# 自动在当前脚本所在目录执行，支持断点续跑（已生成的 JSON 自动跳过）

set -e
cd "$(dirname "$0")"

RUNS="../结果/supp_runs"
LOG="../结果/supp_all_runner.log"
mkdir -p "$RUNS"

echo "==== 一键补充实验启动 $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "$LOG"

# 第一步：运行零重训统计、掌握度代理负对照检验、冷启动单列评估、技能标准覆盖缺口与无大数据偏置权重生成
echo "[Step 1/5] 运行零重训统计与子池权重生成 (supp_stats_and_tables.py)..." | tee -a "$LOG"
python3 supp_stats_and_tables.py | tee -a "$LOG"

# 第二步：运行固定门控扫描与终端门控固定重训实验
echo "[Step 2/5] 补齐第8节固定门控扫描与表7的种子 45、46..." | tee -a "$LOG"
for g in 0.05 0.1 0.25 0.5 1 2 4; do
  for sd in 42 43 44 45 46; do
    python3 -u supp_run.py --name "gsweep_g${g}_s${sd}" --config t3 --seed "$sd" --gate-mode fixed --gate-value "$g" | tee -a "$LOG"
  done
done

G45=$(python3 -c "import json,pathlib; p=pathlib.Path('../结果/newS9_runs/run_t3_s45.json'); print(json.loads(p.read_text())['gate_final'] if p.exists() else 2.87)" 2>/dev/null || echo 2.87)
G46=$(python3 -c "import json,pathlib; p=pathlib.Path('../结果/newS9_runs/run_t3_s46.json'); print(json.loads(p.read_text())['gate_final'] if p.exists() else 5.69)" 2>/dev/null || echo 5.69)

python3 -u supp_run.py --name "gsweep_glearned_s42" --config t3 --seed 42 --gate-mode fixed --gate-value 2.889186 | tee -a "$LOG"
python3 -u supp_run.py --name "gsweep_glearned_s43" --config t3 --seed 43 --gate-mode fixed --gate-value 3.7738   | tee -a "$LOG"
python3 -u supp_run.py --name "gsweep_glearned_s44" --config t3 --seed 44 --gate-mode fixed --gate-value 3.771735 | tee -a "$LOG"
python3 -u supp_run.py --name "gsweep_glearned_s45" --config t3 --seed 45 --gate-mode fixed --gate-value "$G45"   | tee -a "$LOG"
python3 -u supp_run.py --name "gsweep_glearned_s46" --config t3 --seed 46 --gate-mode fixed --gate-value "$G46"   | tee -a "$LOG"

# 第三步：运行有界门控(bounded_tanh)、训练期L2归一化(l2_train)、概念级向量门控(vector)及剔除大数据字面偏置(nobigdata)的五种子消融
echo "[Step 3/5] 运行有界门控、训练期L2归一化、向量门控与去大数据偏置消融..." | tee -a "$LOG"
for sd in 42 43 44 45 46; do
  python3 -u supp_run.py --name "variant_bounded_tanh_s${sd}" --config t3 --seed "$sd" --gate-mode bounded_tanh | tee -a "$LOG"
  python3 -u supp_run.py --name "variant_l2_train_s${sd}"     --config t3 --seed "$sd" --gate-mode l2_train     | tee -a "$LOG"
  python3 -u supp_run.py --name "variant_vector_s${sd}"       --config t3 --seed "$sd" --gate-mode vector       | tee -a "$LOG"
  if [ -f "../中间产物/s9_skill_weight_no_bigdata.npy" ]; then
    python3 -u supp_run.py --name "variant_nobigdata_s${sd}"  --config t3 --seed "$sd" --skill-weight "../中间产物/s9_skill_weight_no_bigdata.npy" | tee -a "$LOG"
  fi
done

# 第四步：运行路径规划评估（含先修基准隔离、平滑度、配对t检验）与可视化
echo "[Step 4/5] 运行更新后的路径级评价 (path_l2norm.py, path_gen.py) 与图8重绘 (exp6_viz.py)..." | tee -a "$LOG"
python3 path_l2norm.py | tee -a "$LOG"
python3 path_gen.py    | tee -a "$LOG"
python3 exp6_viz.py    | tee -a "$LOG"

# 第五步：汇总全部新产出的补充实验结果
echo "[Step 5/5] 最终汇总全部补充实验指标 (supp_stats_and_tables.py)..." | tee -a "$LOG"
python3 supp_stats_and_tables.py | tee -a "$LOG"

echo "==== 全部补充实验已完成 $(date '+%Y-%m-%d %H:%M:%S') ====" | tee -a "$LOG"
