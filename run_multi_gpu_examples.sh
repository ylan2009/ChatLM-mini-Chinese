#!/bin/bash
# -*- coding: utf-8 -*-
# 多GPU训练启动方式示例脚本

echo "=========================================="
echo "Transformers Trainer 多GPU训练启动方式示例"
echo "=========================================="
echo ""

# 设置变量
SCRIPT="train_with_transformers_trainer.py"
NUM_GPUS=2
GPU_IDS="0,1"

echo "请选择启动方式："
echo ""
echo "1. torchrun (推荐，PyTorch原生，无需accelerate)"
echo "2. torch.distributed.launch (旧版本)"
echo "3. accelerate (需要先安装: pip install accelerate)"
echo "4. DeepSpeed (需要先安装: pip install deepspeed)"
echo "5. 自动检测 (最简单，Trainer自动处理)"
echo "6. 单GPU训练"
echo ""

# ============================================
# 方式1: 使用torchrun (推荐)
# ============================================
run_with_torchrun() {
    echo "使用 torchrun 启动训练..."
    echo "命令: torchrun --nproc_per_node=$NUM_GPUS $SCRIPT"
    echo ""
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        $SCRIPT \
        --output_dir ./model_save/output_torchrun \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 5 \
        --learning_rate 0.0001 \
        --bf16 \
        --logging_steps 50 \
        --save_steps 5000 \
        --eval_steps 5000
}

# ============================================
# 方式2: 使用torch.distributed.launch (旧版本)
# ============================================
run_with_distributed_launch() {
    echo "使用 torch.distributed.launch 启动训练..."
    echo "命令: python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $SCRIPT"
    echo ""
    
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        $SCRIPT \
        --output_dir ./model_save/output_distributed \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 5 \
        --learning_rate 0.0001 \
        --bf16 \
        --logging_steps 50 \
        --save_steps 5000 \
        --eval_steps 5000
}

# ============================================
# 方式3: 使用accelerate
# ============================================
run_with_accelerate() {
    echo "使用 accelerate 启动训练..."
    echo "注意: 首次使用需要运行 'accelerate config' 进行配置"
    echo "命令: accelerate launch --multi_gpu --num_processes=$NUM_GPUS $SCRIPT"
    echo ""
    
    # 检查是否已配置accelerate
    if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
        echo "警告: 未检测到accelerate配置文件"
        echo "请先运行: accelerate config"
        echo "或使用快速配置:"
        accelerate config default
    fi
    
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --mixed_precision=bf16 \
        $SCRIPT \
        --output_dir ./model_save/output_accelerate \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 5 \
        --learning_rate 0.0001 \
        --bf16 \
        --logging_steps 50 \
        --save_steps 5000 \
        --eval_steps 5000
}

# ============================================
# 方式4: 使用DeepSpeed
# ============================================
run_with_deepspeed() {
    echo "使用 DeepSpeed 启动训练..."
    echo "命令: deepspeed --num_gpus=$NUM_GPUS $SCRIPT --deepspeed ds_config.json"
    echo ""
    
    # 创建DeepSpeed配置文件
    cat > ds_config.json << EOF
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 50,
  "wall_clock_breakdown": false
}
EOF
    
    echo "已创建 ds_config.json"
    echo ""
    
    deepspeed \
        --num_gpus=$NUM_GPUS \
        $SCRIPT \
        --deepspeed ds_config.json \
        --output_dir ./model_save/output_deepspeed \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 5 \
        --learning_rate 0.0001 \
        --bf16 \
        --logging_steps 50 \
        --save_steps 5000 \
        --eval_steps 5000
}

# ============================================
# 方式5: 自动检测 (Trainer自动处理)
# ============================================
run_with_auto_detect() {
    echo "使用自动检测启动训练..."
    echo "Trainer会自动检测可用GPU并使用DataParallel"
    echo "命令: CUDA_VISIBLE_DEVICES=$GPU_IDS python $SCRIPT"
    echo ""
    
    CUDA_VISIBLE_DEVICES=$GPU_IDS python $SCRIPT \
        --output_dir ./model_save/output_auto \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 5 \
        --learning_rate 0.0001 \
        --bf16 \
        --logging_steps 50 \
        --save_steps 5000 \
        --eval_steps 5000
}

# ============================================
# 方式6: 单GPU训练
# ============================================
run_single_gpu() {
    echo "使用单GPU启动训练..."
    echo "命令: CUDA_VISIBLE_DEVICES=0 python $SCRIPT"
    echo ""
    
    CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
        --output_dir ./model_save/output_single \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 5 \
        --learning_rate 0.0001 \
        --bf16 \
        --logging_steps 50 \
        --save_steps 5000 \
        --eval_steps 5000
}

# ============================================
# 主菜单
# ============================================
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        run_with_torchrun
        ;;
    2)
        run_with_distributed_launch
        ;;
    3)
        run_with_accelerate
        ;;
    4)
        run_with_deepspeed
        ;;
    5)
        run_with_auto_detect
        ;;
    6)
        run_single_gpu
        ;;
    *)
        echo "无效选项，请输入 1-6"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
