#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采样脚本：从大数据集中采样小数据集用于快速训练验证
"""

import pandas as pd
import os
from pathlib import Path

def sample_dataset(input_file, output_file, sample_size=10000, random_state=42):
    """
    从输入数据集中随机采样指定数量的数据
    
    Args:
        input_file: 输入的parquet文件路径
        output_file: 输出的parquet文件路径
        sample_size: 采样数量，默认10000条
        random_state: 随机种子，保证可复现
    """
    print(f"开始读取数据集: {input_file}")
    
    # 读取原始数据
    df = pd.read_parquet(input_file)
    total_size = len(df)
    
    print(f"原始数据集大小: {total_size:,} 条")
    print(f"目标采样数量: {sample_size:,} 条")
    
    # 如果数据量小于采样数量，使用全部数据
    if total_size <= sample_size:
        print(f"⚠️  数据量不足，使用全部 {total_size:,} 条数据")
        sampled_df = df
    else:
        # 随机采样
        sampled_df = df.sample(n=sample_size, random_state=random_state)
        print(f"✅ 成功采样 {len(sampled_df):,} 条数据 ({len(sampled_df)/total_size*100:.1f}%)")
    
    # 保存采样后的数据
    sampled_df.to_parquet(output_file, index=False)
    print(f"✅ 已保存到: {output_file}")
    
    # 显示数据统计
    print("\n数据统计:")
    if 'prompt' in sampled_df.columns:
        print(f"  - Prompt平均长度: {sampled_df['prompt'].str.len().mean():.0f} 字符")
    if 'response' in sampled_df.columns:
        print(f"  - Response平均长度: {sampled_df['response'].str.len().mean():.0f} 字符")
    
    return sampled_df

def main():
    # 项目根目录
    project_root = Path(__file__).parent
    data_dir = project_root / 'data'
    
    # 输入输出文件路径
    train_input = data_dir / 'sft_train_dataset.parquet'
    train_output = data_dir / 'sft_train_dataset_10k.parquet'
    
    valid_input = data_dir / 'sft_valid_dataset.parquet'
    valid_output = data_dir / 'sft_valid_dataset_1k.parquet'
    
    print("=" * 60)
    print("SFT数据集采样工具")
    print("=" * 60)
    
    # 采样训练集（10000条）
    if train_input.exists():
        print("\n[1/2] 处理训练集...")
        sample_dataset(
            input_file=str(train_input),
            output_file=str(train_output),
            sample_size=10000,
            random_state=42
        )
    else:
        print(f"❌ 训练集文件不存在: {train_input}")
    
    # 采样验证集（1000条）
    if valid_input.exists():
        print("\n[2/2] 处理验证集...")
        sample_dataset(
            input_file=str(valid_input),
            output_file=str(valid_output),
            sample_size=1000,
            random_state=42
        )
    else:
        print(f"❌ 验证集文件不存在: {valid_input}")
    
    print("\n" + "=" * 60)
    print("✅ 采样完成！")
    print("=" * 60)
    print("\n下一步操作：")
    print("1. 修改 config.py 中的数据集路径")
    print("2. 运行训练命令开始训练")
    print("\n配置修改示例：")
    print("  train_file: str = PROJECT_ROOT + '/data/sft_train_dataset_10k.parquet'")
    print("  validation_file: str = PROJECT_ROOT + '/data/sft_valid_dataset_1k.parquet'")

if __name__ == '__main__':
    main()
