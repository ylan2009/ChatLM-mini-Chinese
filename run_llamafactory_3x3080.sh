#!/bin/bash
# -*- coding: utf-8 -*-
#
# LLaMA-Factory 训练启动脚本
# 硬件配置: 3×RTX 3080 20GB + 12GB RAM
#
# 使用方法:
#   bash run_llamafactory_3x3080.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "LLaMA-Factory 训练启动脚本"
echo "硬件配置: 3×RTX 3080 20GB + 12GB RAM"
echo "=========================================="
echo ""

# ============================================================================
# 配置变量
# ============================================================================
CONFIG_FILE="llamafactory_config_3x3080.yaml"
NUM_GPUS=3
GPU_IDS="0,1,2"

# ============================================================================
# 检查依赖
# ============================================================================
# echo "检查依赖..."

# # 检查 transformers 版本
# echo "检查 transformers 版本..."
# TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "0.0.0")
# REQUIRED_MIN_VERSION="4.37.0"
# REQUIRED_MAX_VERSION="5.0.0"

# # 检查版本范围
# if ! python -c "
# from packaging import version
# import transformers
# v = transformers.__version__
# min_ok = version.parse(v) >= version.parse('$REQUIRED_MIN_VERSION')
# max_ok = version.parse(v) < version.parse('$REQUIRED_MAX_VERSION')
# exit(0 if (min_ok and max_ok) else 1)
# " 2>/dev/null; then
#     echo "❌ 错误: transformers 版本不兼容"
#     echo "   当前版本: $TRANSFORMERS_VERSION"
#     echo "   需要版本: >= $REQUIRED_MIN_VERSION 且 < $REQUIRED_MAX_VERSION"
#     echo ""
#     echo "修复方法:"
#     echo "  pip uninstall transformers -y"
#     echo "  pip install transformers==4.44.0"
#     echo ""
#     echo "详细说明请查看: FIX_TRANSFORMERS_5X.md"
#     exit 1
# fi
# echo "✓ transformers 版本: $TRANSFORMERS_VERSION"

# # 检查 trl 版本（如果安装了）
# if python -c "import trl" 2>/dev/null; then
#     TRL_VERSION=$(python -c "import trl; print(trl.__version__)" 2>/dev/null || echo "unknown")
#     echo "检查 trl 版本..."
    
#     # trl 版本应该 < 0.9.0（避免与 transformers 冲突）
#     if ! python -c "
# from packaging import version
# import trl
# v = trl.__version__
# exit(0 if version.parse(v) < version.parse('0.9.0') else 1)
# " 2>/dev/null; then
#         echo "⚠️  警告: trl 版本可能不兼容"
#         echo "   当前版本: $TRL_VERSION"
#         echo "   推荐版本: < 0.9.0"
#         echo ""
#         echo "修复方法:"
#         echo "  pip uninstall trl -y"
#         echo "  pip install trl==0.8.6"
#         echo ""
#         echo "详细说明请查看: FIX_TRL_CONFLICT.md"
#         echo ""
#         read -p "是否继续? [y/N] " -n 1 -r
#         echo
#         if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#             exit 1
#         fi
#     else
#         echo "✓ trl 版本: $TRL_VERSION"
#     fi
# fi

# # 检查 LLaMA-Factory 是否安装
# if ! python -c "import llmtuner" 2>/dev/null; then
#     echo "❌ 错误: 未安装 LLaMA-Factory"
#     echo ""
#     echo "请运行以下命令安装:"
#     echo "  pip install llmtuner"
#     echo ""
#     echo "或者从源码安装:"
#     echo "  git clone https://github.com/hiyouga/LLaMA-Factory.git"
#     echo "  cd LLaMA-Factory"
#     echo "  pip install -e ."
#     exit 1
# fi

# # 检查 DeepSpeed 是否安装
# if ! python -c "import deepspeed" 2>/dev/null; then
#     echo "⚠️  警告: 未安装 DeepSpeed"
#     echo "   DeepSpeed 可以进一步优化显存使用"
#     echo "   安装命令: pip install deepspeed"
#     echo ""
#     read -p "是否继续（不使用DeepSpeed）? [y/N] " -n 1 -r
#     echo
#     if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#         exit 1
#     fi
#     # 禁用 DeepSpeed
#     sed -i.bak 's/deepspeed: ds_config_zero2.json/deepspeed: null/' "$CONFIG_FILE"
# fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查数据集信息文件
if [ ! -f "dataset_info.json" ]; then
    echo "❌ 错误: 数据集信息文件不存在: dataset_info.json"
    exit 1
fi

echo "✓ 依赖检查完成"
echo ""

# ============================================================================
# 显示配置信息
# ============================================================================
echo "训练配置:"
echo "  配置文件: $CONFIG_FILE"
echo "  GPU数量: $NUM_GPUS"
echo "  GPU ID: $GPU_IDS"
echo ""

# ============================================================================
# 设置环境变量（针对低内存优化）
# ============================================================================
echo "设置环境变量..."

# 指定使用的GPU
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# NCCL 配置（解决共享内存问题）
export NCCL_SHM_DISABLE=1  # 禁用共享内存（避免 /dev/shm 错误）
export NCCL_P2P_DISABLE=0  # 启用 P2P 通信（GPU间直接通信）
export NCCL_TIMEOUT=3600   # 1小时超时
export NCCL_IB_DISABLE=1   # 禁用InfiniBand（如果没有IB网络）
export NCCL_DEBUG=INFO     # 启用调试信息（可选，用于诊断）

# PyTorch 内存优化（使用新的环境变量名）
export PYTORCH_ALLOC_CONF=max_split_size_mb:128  # 减少显存碎片

# Python 内存优化
export PYTHONUNBUFFERED=1  # 禁用Python缓冲

echo "✓ 环境变量设置完成"
echo ""

# ============================================================================
# 显示系统信息
# ============================================================================
echo "系统信息:"
python -c "
import torch
print(f'  PyTorch版本: {torch.__version__}')
print(f'  CUDA版本: {torch.version.cuda}')
print(f'  CUDA可用: {torch.cuda.is_available()}')
print(f'  GPU数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f'    显存: {mem_total:.1f} GB')
"
echo ""

# ============================================================================
# 选择训练方式
# ============================================================================
echo "请选择训练方式:"
echo "  1) 使用 llamafactory-cli (推荐，最简单)"
echo "  2) 使用 accelerate launch (更灵活)"
echo "  3) 使用 deepspeed (最优显存利用)"
echo "  4) 使用 torchrun (标准DDP)"
echo ""
read -p "请输入选项 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "方式1: 使用 llamafactory-cli"
        echo "=========================================="
        echo ""
        
        llamafactory-cli train "$CONFIG_FILE"
        ;;
    
    2)
        echo ""
        echo "=========================================="
        echo "方式2: 使用 accelerate launch"
        echo "=========================================="
        echo ""
        
        # 检查是否有 accelerate 配置
        if [ ! -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ]; then
            echo "首次使用 accelerate，需要配置..."
            echo "建议配置:"
            echo "  - 计算环境: multi-GPU"
            echo "  - GPU数量: 3"
            echo "  - 混合精度: bf16"
            echo ""
            accelerate config
        fi
        
        # accelerate 支持 -m 参数
        accelerate launch \
            --multi_gpu \
            --num_processes=$NUM_GPUS \
            --main_process_port=29500 \
            -m llmtuner.cli train "$CONFIG_FILE"
        ;;
    
    3)
        echo ""
        echo "=========================================="
        echo "方式3: 使用 deepspeed"
        echo "=========================================="
        echo ""
        
        # 检查 DeepSpeed 配置文件
        if [ ! -f "ds_config_zero2.json" ]; then
            echo "❌ 错误: DeepSpeed配置文件不存在: ds_config_zero2.json"
            exit 1
        fi
        
        # 创建临时启动脚本（解决相对导入问题和参数解析问题）
        cat > /tmp/deepspeed_train.py << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSpeed 训练启动脚本
解决直接运行 cli.py 时的相对导入问题和 --local_rank 参数解析问题
"""
import sys
import os

# 过滤掉 DeepSpeed 自动添加的 --local_rank 参数
# 因为 llmtuner.cli.main() 不需要这个参数（它会从环境变量读取）
filtered_args = []
skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if skip_next:
        skip_next = False
        continue
    if arg.startswith('--local_rank'):
        if '=' not in arg and i < len(sys.argv) - 1:
            skip_next = True  # 跳过下一个参数（值）
        continue  # 跳过 --local_rank
    filtered_args.append(arg)

# 替换 sys.argv
sys.argv = [sys.argv[0]] + filtered_args

# 导入并运行
from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF
        
        echo "使用 DeepSpeed 启动训练..."
        echo "  配置文件: $CONFIG_FILE"
        echo "  GPU数量: $NUM_GPUS"
        echo ""
        
        deepspeed \
            --num_gpus=$NUM_GPUS \
            --master_port=29500 \
            /tmp/deepspeed_train.py train "$CONFIG_FILE"
        ;;
    
    4)
        echo ""
        echo "=========================================="
        echo "方式4: 使用 torchrun"
        echo "=========================================="
        echo ""
        
        # 创建临时启动脚本（解决相对导入问题和参数解析问题）
        cat > /tmp/torchrun_train.py << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Torchrun 训练启动脚本
解决直接运行 cli.py 时的相对导入问题和 --local_rank 参数解析问题
"""
import sys
import os

# 过滤掉 torchrun 自动添加的 --local_rank 参数
# 因为 llmtuner.cli.main() 不需要这个参数（它会从环境变量读取）
filtered_args = []
skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if skip_next:
        skip_next = False
        continue
    if arg.startswith('--local_rank'):
        if '=' not in arg and i < len(sys.argv) - 1:
            skip_next = True  # 跳过下一个参数（值）
        continue  # 跳过 --local_rank
    filtered_args.append(arg)

# 替换 sys.argv
sys.argv = [sys.argv[0]] + filtered_args

# 导入并运行
from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF
        
        echo "使用 torchrun 启动训练..."
        echo "  配置文件: $CONFIG_FILE"
        echo "  GPU数量: $NUM_GPUS"
        echo ""
        
        torchrun \
            --nproc_per_node=$NUM_GPUS \
            --master_port=29500 \
            /tmp/torchrun_train.py train "$CONFIG_FILE"
        ;;
    
    *)
        echo "❌ 无效选项: $choice"
        exit 1
        ;;
esac

# ============================================================================
# 训练完成
# ============================================================================
echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "输出目录: ./model_save/llamafactory_3x3080_output"
echo "日志目录: ./logs/llamafactory_3x3080"
echo ""
echo "查看训练日志:"
echo "  tensorboard --logdir=./logs/llamafactory_3x3080"
echo ""
