#!/bin/bash
# 使用 Gloo 后端进行 SFT 训练（解决 NCCL 共享内存问题）

echo "=========================================="
echo "使用 Gloo 后端进行 SFT 训练"
echo "=========================================="

# 方案1: 使用配置文件
echo ""
echo "方案1: 使用 Accelerate 配置文件（推荐）"
echo "----------------------------------------"
echo "accelerate launch --config_file accelerate_config_gloo.yaml \\"
echo "    ./train_low_mem.py train \\"
echo "    --is_finetune=True \\"
echo "    --use_small_config=True"
echo ""

# 方案2: 使用环境变量
echo "方案2: 使用环境变量"
echo "----------------------------------------"
echo "export ACCELERATE_USE_GLOO=1"
echo "accelerate launch --multi_gpu --num_processes 2 \\"
echo "    ./train_low_mem.py train \\"
echo "    --is_finetune=True \\"
echo "    --use_small_config=True"
echo ""

# 方案3: 禁用 NCCL 共享内存
echo "方案3: 禁用 NCCL 共享内存"
echo "----------------------------------------"
echo "export NCCL_SHM_DISABLE=1"
echo "accelerate launch --multi_gpu --num_processes 2 \\"
echo "    ./train_low_mem.py train \\"
echo "    --is_finetune=True \\"
echo "    --use_small_config=True"
echo ""

echo "=========================================="
echo "选择一个方案并执行"
echo "=========================================="

# 询问用户选择
read -p "请选择方案 (1/2/3) [默认: 1]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "执行方案1: 使用 Accelerate 配置文件"
        accelerate launch --config_file accelerate_config_gloo.yaml \
            ./train_low_mem.py train \
            --is_finetune=True \
            --use_small_config=True
        ;;
    2)
        echo ""
        echo "执行方案2: 使用环境变量 ACCELERATE_USE_GLOO"
        export ACCELERATE_USE_GLOO=1
        accelerate launch --multi_gpu --num_processes 2 \
            ./train_low_mem.py train \
            --is_finetune=True \
            --use_small_config=True
        ;;
    3)
        echo ""
        echo "执行方案3: 禁用 NCCL 共享内存"
        export NCCL_SHM_DISABLE=1
        accelerate launch --multi_gpu --num_processes 2 \
            ./train_low_mem.py train \
            --is_finetune=True \
            --use_small_config=True
        ;;
    *)
        echo "无效的选择，退出"
        exit 1
        ;;
esac
