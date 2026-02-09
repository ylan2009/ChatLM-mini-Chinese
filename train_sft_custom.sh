#!/bin/bash

# ============================================================================
# 灵活SFT训练脚本 - 支持自定义batch_size
# ============================================================================
# 
# 使用方法：
#   chmod +x train_sft_custom.sh
#   
#   # 使用默认batch_size=16
#   ./train_sft_custom.sh
#   
#   # 自定义batch_size
#   ./train_sft_custom.sh 20
#   ./train_sft_custom.sh 24
#   ./train_sft_custom.sh 32
# ============================================================================

# 从命令行参数获取batch_size，默认为16
BATCH_SIZE=${1:-16}

echo "============================================================================"
echo "快速SFT训练 - 自定义batch_size"
echo "============================================================================"
echo ""
echo "配置信息："
echo "  - batch_size_per_gpu: $BATCH_SIZE"
echo "  - gradient_accumulation_steps: 2"
echo "  - 实际有效batch_size: $((BATCH_SIZE * 2 * 2))"
echo "  - 混合精度: bf16"
echo "  - 训练数据: 5,000样本"
echo "  - 验证数据: 500样本"
echo ""

# 根据batch_size估算资源占用
if [ $BATCH_SIZE -le 8 ]; then
    echo "预期性能（batch_size=$BATCH_SIZE）："
    echo "  - GPU显存占用: 8-12GB/GPU"
    echo "  - 内存占用: 8-12GB"
    echo "  - 训练速度: 比低内存模式快3-4倍"
elif [ $BATCH_SIZE -le 16 ]; then
    echo "预期性能（batch_size=$BATCH_SIZE）："
    echo "  - GPU显存占用: 12-16GB/GPU"
    echo "  - 内存占用: 10-14GB"
    echo "  - 训练速度: 比低内存模式快5-6倍"
elif [ $BATCH_SIZE -le 24 ]; then
    echo "预期性能（batch_size=$BATCH_SIZE）："
    echo "  - GPU显存占用: 16-18GB/GPU"
    echo "  - 内存占用: 12-15GB"
    echo "  - 训练速度: 比低内存模式快7-8倍"
else
    echo "预期性能（batch_size=$BATCH_SIZE）："
    echo "  - GPU显存占用: 18-20GB/GPU（接近上限）"
    echo "  - 内存占用: 14-16GB（接近上限）"
    echo "  - 训练速度: 比低内存模式快8-10倍"
    echo ""
    echo "⚠️  警告：batch_size较大，可能接近硬件上限，请密切监控资源使用！"
fi

echo ""
echo "============================================================================"
echo ""

# 强制使用 Gloo 后端（避免NCCL共享内存问题）
export ACCELERATE_USE_GLOO=1

# 启动训练，传入自定义batch_size
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True \
    --batch_size_per_gpu=$BATCH_SIZE

echo ""
echo "============================================================================"
echo "训练完成！"
echo "模型保存位置: ./model_save/sft_fast/"
echo "============================================================================"
