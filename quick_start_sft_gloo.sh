#!/bin/bash
# 快速启动 SFT 训练 - 使用 Gloo 后端（解决 NCCL 问题）

cd /data3/ChatLM-mini-Chinese

echo "=========================================="
echo "快速启动 SFT 训练（Gloo 后端）"
echo "=========================================="

# 使用 Gloo 后端
export ACCELERATE_USE_GLOO=1

echo "使用 Gloo 后端进行多GPU训练..."
echo ""

accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
