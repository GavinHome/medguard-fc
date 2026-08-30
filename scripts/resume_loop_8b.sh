#!/bin/zsh
# 8B 训练自动续跑循环: 崩溃后自动从最新存档恢复, 直到 2400 步完成
# 用法: 直接后台执行; 每次尝试追加日志到 train_8b_resume.log
cd /Users/yangxiaomin/ZCodeProject/medguard-fc
MAX_ATTEMPTS=6
ATTEMPT=1
while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  echo "=== $(date '+%m-%d %H:%M') 第 ${ATTEMPT}/${MAX_ATTEMPTS} 次尝试 ===" >> train_8b_resume.log
  caffeinate -i uv run mlx_lm.lora -c configs/qwen3-8b-subset.yaml \
    --adapter-path models/qwen3-8b-sft-subset \
    --resume-adapter-file models/qwen3-8b-sft-subset/adapters.safetensors \
    --iters 800 --save-every 200 --clear-cache-threshold 512 >> train_8b_resume.log 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "=== $(date '+%m-%d %H:%M') 训练正常完成 (exit 0) ===" >> train_8b_resume.log
    exit 0
  fi
  echo "=== $(date '+%m-%d %H:%M') 异常退出(code=$code), 90 秒后自动重试 ===" >> train_8b_resume.log
  sleep 90
  ATTEMPT=$((ATTEMPT+1))
done
echo "=== ${MAX_ATTEMPTS} 次尝试均失败, 停止 — 保留全部检查点待人工决策 ===" >> train_8b_resume.log
exit 1
