#!/bin/bash

# ============================================================================
# 快速SFT训练脚本 - 充分利用GPU显存（20GB × 2）
# ============================================================================
# 
# 优化策略：
# - batch_size_per_gpu: 64（充分利用GPU显存）
# - gradient_accumulation_steps: 2
# - 实际有效batch_size = 64 * 2(GPU) * 2 = 256
# - 预期GPU显存占用：16-19GB/GPU（80-95%利用率）
# - 预期训练速度：提升20-30倍
#
# 注意：修复了trainer_low_mem.py中的batch_size限制问题
# 现在会根据可用内存智能调整：
# - 可用内存<8GB：强制batch_size=1
# - 可用内存8-13GB：限制batch_size≤4
# - 可用内存>13GB：使用配置的batch_size（你的情况）
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
echo "  - batch_size_per_gpu: 64"
echo "  - gradient_accumulation_steps: 2"
echo "  - 实际有效batch_size: 256"
echo "  - 混合精度: bf16"
echo "  - 训练数据: 5,000样本"
echo "  - 验证数据: 500样本"
echo ""
echo "预期性能："
echo "  - GPU显存占用: 16-19GB/GPU（80-95%利用率）"
echo "  - 内存占用: 10-14GB"
echo "  - 每epoch步数: ~20步"
echo "  - 单epoch时长: ~5分钟"
echo "  - 总训练时长(3 epoch): ~15分钟"
echo "  - 训练速度: 比低内存模式快20-30倍"
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
