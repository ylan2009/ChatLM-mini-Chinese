#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT训练前的环境检查脚本

使用方法：
    python check_sft_ready.py
"""

import os
import sys
sys.path.extend(['.', '..'])

from config import PROJECT_ROOT, TrainConfigSFT


def check_file_exists(file_path, description):
    """检查文件是否存在"""
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {file_path}")
    return exists


def check_sft_ready():
    """检查SFT训练所需的所有文件"""
    print("=" * 80)
    print("SFT训练环境检查")
    print("=" * 80)
    
    config = TrainConfigSFT()
    all_ready = True
    
    print("\n📁 检查数据文件:")
    print("-" * 80)
    
    # 检查训练数据
    if not check_file_exists(config.train_file, "训练数据"):
        all_ready = False
        print(f"   💡 解决方案: python prepare_sft_data.py")
    
    # 检查验证数据
    if not check_file_exists(config.validation_file, "验证数据"):
        all_ready = False
        print(f"   💡 解决方案: python prepare_sft_data.py")
    
    print("\n🤖 检查模型文件:")
    print("-" * 80)
    
    # 检查预训练模型
    if not check_file_exists(config.finetune_from_ckp_file, "预训练模型"):
        all_ready = False
        print(f"   💡 解决方案: accelerate launch --multi_gpu --num_processes 2 ./train.py train")
    
    # 检查tokenizer
    if not check_file_exists(config.tokenizer_dir, "Tokenizer目录"):
        all_ready = False
        print(f"   💡 解决方案: 请先完成预训练或下载tokenizer")
    
    print("\n📂 检查输出目录:")
    print("-" * 80)
    
    # 检查输出目录
    output_dir_exists = os.path.exists(config.output_dir)
    if not output_dir_exists:
        print(f"⚠️  输出目录不存在: {config.output_dir}")
        print(f"   💡 将自动创建")
        os.makedirs(config.output_dir, exist_ok=True)
    else:
        print(f"✅ 输出目录: {config.output_dir}")
    
    print("\n⚙️  训练配置:")
    print("-" * 80)
    print(f"  训练轮数: {config.epochs}")
    print(f"  学习率: {config.learn_rate}")
    print(f"  Batch size (per GPU): {config.batch_size_per_gpu}")
    print(f"  梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"  混合精度: {config.mixed_precision}")
    print(f"  最大序列长度: {config.max_seq_len}")
    
    print("\n" + "=" * 80)
    
    if all_ready:
        print("✅ 所有检查通过！可以开始SFT训练")
        print("\n🚀 运行训练命令:")
        print("   accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True")
        return 0
    else:
        print("❌ 检查未通过，请先解决上述问题")
        print("\n📚 详细指南: docs/sft_training_guide.md")
        return 1


if __name__ == '__main__':
    exit_code = check_sft_ready()
    sys.exit(exit_code)
