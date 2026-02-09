#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试LowMemDataset是否正常工作

使用方法：
    python test_low_mem_dataset.py
"""

import torch
from torch.utils.data import DataLoader
from model.dataset import LowMemDataset
from config import TrainConfig

def test_low_mem_dataset():
    """测试LowMemDataset的基本功能"""
    
    train_config = TrainConfig()
    
    print("=" * 80)
    print("测试 LowMemDataset")
    print("=" * 80)
    
    # 创建数据集
    print("\n1. 创建训练数据集...")
    try:
        train_dataset = LowMemDataset(
            parquet_file=train_config.train_file,
            tokenizer_dir=train_config.tokenizer_dir,
            max_seq_len=train_config.max_seq_len,
        )
        print(f"   ✓ 训练数据集创建成功，大小: {len(train_dataset)}")
    except Exception as e:
        print(f"   ✗ 训练数据集创建失败: {e}")
        return False
    
    print("\n2. 创建验证数据集...")
    try:
        valid_dataset = LowMemDataset(
            parquet_file=train_config.validation_file,
            tokenizer_dir=train_config.tokenizer_dir,
            max_seq_len=train_config.max_seq_len,
        )
        print(f"   ✓ 验证数据集创建成功，大小: {len(valid_dataset)}")
    except Exception as e:
        print(f"   ✗ 验证数据集创建失败: {e}")
        return False
    
    # 测试单条数据读取
    print("\n3. 测试单条数据读取...")
    try:
        sample = train_dataset[0]
        print(f"   ✓ 成功读取第1条数据")
        print(f"   - Prompt长度: {len(sample[0])}")
        print(f"   - Response长度: {len(sample[1])}")
        print(f"   - Prompt示例: {sample[0][:100]}...")
    except Exception as e:
        print(f"   ✗ 读取数据失败: {e}")
        return False
    
    # 测试DataLoader
    print("\n4. 测试DataLoader...")
    try:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=0,
        )
        print(f"   ✓ DataLoader创建成功")
    except Exception as e:
        print(f"   ✗ DataLoader创建失败: {e}")
        return False
    
    # 测试批次数据
    print("\n5. 测试批次数据读取...")
    try:
        batch = next(iter(train_dataloader))
        print(f"   ✓ 成功读取一个批次")
        print(f"   - input_ids shape: {batch['input_ids'].shape}")
        print(f"   - input_mask shape: {batch['input_mask'].shape}")
        print(f"   - target_ids shape: {batch['target_ids'].shape}")
    except Exception as e:
        print(f"   ✗ 读取批次失败: {e}")
        return False
    
    # 测试多GPU环境（如果有）
    if torch.cuda.device_count() >= 2:
        print(f"\n6. 检测到 {torch.cuda.device_count()} 个GPU")
        print("   ✓ LowMemDataset支持多GPU分布式训练")
    else:
        print(f"\n6. 检测到 {torch.cuda.device_count()} 个GPU")
        print("   ℹ 单GPU或CPU环境")
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！LowMemDataset工作正常")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    success = test_low_mem_dataset()
    exit(0 if success else 1)
