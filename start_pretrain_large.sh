#!/bin/bash

# 大数据集预训练启动脚本
# 适用于：3×20G显存GPU + 12G内存 + 1000万训练数据

echo "=========================================="
echo "大数据集预训练启动脚本"
echo "=========================================="
echo "硬件环境："
echo "  - GPU: 3×NVIDIA 3080 (20GB显存)"
echo "  - 内存: 12GB"
echo "  - 数据量: 1000万样本"
echo "=========================================="
echo ""

# 检查数据文件是否存在
TRAIN_FILE="./data/pretrain_train_10m.parquet"
VALID_FILE="./data/pretrain_valid_100k.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ 错误: 训练数据文件不存在: $TRAIN_FILE"
    echo "请先准备数据文件，或修改 config.py 中的 train_file 路径"
    exit 1
fi

if [ ! -f "$VALID_FILE" ]; then
    echo "❌ 错误: 验证数据文件不存在: $VALID_FILE"
    echo "请先准备数据文件，或修改 config.py 中的 validation_file 路径"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 张GPU"

if [ "$GPU_COUNT" -lt 3 ]; then
    echo "⚠️  警告: 检测到的GPU数量少于3张，建议使用3张GPU以获得最佳性能"
    echo "是否继续？(y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "已取消"
        exit 0
    fi
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

# 启动训练
accelerate launch \
    --multi_gpu \
    --num_processes 3 \
    ./train_low_mem.py train \
    --use_large_config=True

echo ""
echo "=========================================="
echo "训练完成或已中断"
echo "=========================================="
