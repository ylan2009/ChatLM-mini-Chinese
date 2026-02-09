#!/bin/bash

# ============================================================================
# 快速SFT训练脚本 - 充分利用GPU显存（20GB × 2）
# ============================================================================
# 
# 优化策略：
# - batch_size_per_gpu: 16（从1提升到16）
# - gradient_accumulation_steps: 2（从8降到2）
# - 实际有效batch_size = 16 * 2(GPU) * 2 = 64
# - 预期GPU显存占用：12-16GB/GPU（提升6-8倍）
# - 预期训练速度：提升5-6倍
#
# 使用方法：
#   chmod +x quick_start_sft_fast.sh
#   ./quick_start_sft_fast.sh
# ============================================================================

echo "============================================================================"
echo "快速SFT训练 - 充分利用GPU显存"
echo "============================================================================"
echo ""
echo "配置信息："
echo "  - batch_size_per_gpu: 16"
echo "  - gradient_accumulation_steps: 2"
echo "  - 实际有效batch_size: 64"
echo "  - 混合精度: bf16"
echo "  - 训练数据: 5,000样本"
echo "  - 验证数据: 500样本"
echo ""
echo "预期性能："
echo "  - GPU显存占用: 12-16GB/GPU"
echo "  - 内存占用: 10-14GB"
echo "  - 训练速度: 比低内存模式快5-6倍"
echo ""
echo "============================================================================"
echo ""

# 强制使用 Gloo 后端（避免NCCL共享内存问题）
export ACCELERATE_USE_GLOO=1
export NCCL_SHM_DISABLE=1

# 启动训练
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True

echo ""
echo "============================================================================"
echo "训练完成！"
echo "模型保存位置: ./model_save/sft_fast/"
echo "============================================================================"
