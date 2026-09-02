#!/bin/zsh
# 8B v12 子集训练自动续跑循环 — 六层防护之第2层
# 退出条件(第6层): 24小时内崩溃>=3次 → 放弃本地转租卡
cd /Users/yangxiaomin/ZCodeProject/medguard-fc
LOG=train_8b_v12.log
MAX_ATTEMPTS=6
ATTEMPT=1
CRASH_TIMES=()
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  echo "=== $(date '+%m-%d %H:%M') 第 ${ATTEMPT}/${MAX_ATTEMPTS} 次尝试 ===" >> $LOG
  caffeinate -i uv run mlx_lm.lora -c configs/qwen3-8b-v12sub.yaml >> $LOG 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "=== $(date '+%m-%d %H:%M') 训练正常完成 (exit 0) ===" >> $LOG
    exit 0
  fi
  NOW=$(date +%s)
  CRASH_TIMES=(${CRASH_TIMES[@]} $NOW)
  RECENT=0
  for t in ${CRASH_TIMES[@]}; do
    if [ $((NOW - t)) -lt 86400 ]; then RECENT=$((RECENT+1)); fi
  done
  echo "=== $(date '+%m-%d %H:%M') 异常退出(code=$code), 24h内崩溃${RECENT}次 ===" >> $LOG
  if [ $RECENT -ge 3 ]; then
    echo "=== 触发退出条件: 24小时内崩溃${RECENT}次 — 停止本地训练, 转租卡方案 ===" >> $LOG
    exit 1
  fi
  echo "=== 90 秒后自动续跑(从最新存档) ===" >> $LOG
  sleep 90
  ATTEMPT=$((ATTEMPT+1))
done
echo "=== ${MAX_ATTEMPTS} 次尝试用尽 — 保留全部检查点 ===" >> $LOG
exit 1
