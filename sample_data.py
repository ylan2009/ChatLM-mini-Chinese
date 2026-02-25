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
    
    # 数据源：分别从 sft_train_dataset 和 sft_valid_dataset 采样
    train_source = data_dir / 'sft_train_dataset.parquet'
    valid_source = data_dir / 'sft_valid_dataset.parquet'
    
    # 输出文件路径（匹配 TrainConfigSFTFast 配置）
    train_output = data_dir / 'sft_train_small_train.parquet'
    valid_output = data_dir / 'sft_train_small_valid.parquet'
    
    train_size = 50000   # 训练集采样数量
    valid_size = 2000    # 验证集采样数量
    
    print("=" * 60)
    print("SFT数据集采样工具")
    print("=" * 60)
    
    # ---- 处理训练集 ----
    print(f"\n[1/2] 处理训练集...")
    if not train_source.exists():
        print(f"❌ 训练集数据源不存在: {train_source}")
        return
    
    train_df_src = pd.read_parquet(str(train_source))
    total_train = len(train_df_src)
    print(f"训练集数据源总量: {total_train:,} 条")
    
    if total_train <= train_size:
        print(f"⚠️  数据量不足，使用全部 {total_train:,} 条数据")
        train_df = train_df_src
    else:
        train_df = train_df_src.sample(n=train_size, random_state=42)
        print(f"✅ 成功采样 {len(train_df):,} 条数据 ({len(train_df)/total_train*100:.1f}%)")
    
    train_df.to_parquet(str(train_output), index=False)
    print(f"✅ 已保存到: {train_output}")
    if 'prompt' in train_df.columns:
        print(f"  - Prompt平均长度: {train_df['prompt'].str.len().mean():.0f} 字符")
    if 'response' in train_df.columns:
        print(f"  - Response平均长度: {train_df['response'].str.len().mean():.0f} 字符")
    
    # ---- 处理验证集 ----
    print(f"\n[2/2] 处理验证集...")
    if not valid_source.exists():
        print(f"❌ 验证集数据源不存在: {valid_source}")
        return
    
    valid_df_src = pd.read_parquet(str(valid_source))
    total_valid = len(valid_df_src)
    print(f"验证集数据源总量: {total_valid:,} 条")
    
    if total_valid <= valid_size:
        print(f"⚠️  数据量不足，使用全部 {total_valid:,} 条数据")
        valid_df = valid_df_src
    else:
        valid_df = valid_df_src.sample(n=valid_size, random_state=42)
        print(f"✅ 成功采样 {len(valid_df):,} 条数据 ({len(valid_df)/total_valid*100:.1f}%)")
    
    valid_df.to_parquet(str(valid_output), index=False)
    print(f"✅ 已保存到: {valid_output}")
    if 'prompt' in valid_df.columns:
        print(f"  - Prompt平均长度: {valid_df['prompt'].str.len().mean():.0f} 字符")
    if 'response' in valid_df.columns:
        print(f"  - Response平均长度: {valid_df['response'].str.len().mean():.0f} 字符")
    
    print("\n" + "=" * 60)
    print("✅ 采样完成！")
    print("=" * 60)
    print(f"\n训练集: {train_output}  ({len(train_df):,} 条)")
    print(f"验证集: {valid_output}  ({len(valid_df):,} 条)")
    print("\n下一步操作：")
    print("运行训练命令：")
    print("  accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True --use_fast_config=True")

if __name__ == '__main__':
    main()
