#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从大数据集中采样小数据集，用于16G内存的SFT训练

使用方法：
    # 从parquet文件采样
    python prepare_small_sft_data.py --input data/sft_train_dataset.parquet --output data/sft_train_small.parquet --num_samples 5000
    
    # 从json文件采样
    python prepare_small_sft_data.py --input data/alpaca_gpt4_data_zh.json --output data/sft_train_small.parquet --num_samples 5000
"""

import os
import json
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def estimate_memory_usage(num_samples, avg_seq_len=512):
    """
    估算训练时的内存使用量
    
    参数：
        num_samples: 样本数量
        avg_seq_len: 平均序列长度
    
    返回：
        dict: 包含各部分内存估算
    """
    # 每个token约4字节（int32）
    bytes_per_token = 4
    
    # 数据集内存（input + target，假设各占一半）
    dataset_mem_gb = (num_samples * avg_seq_len * 2 * bytes_per_token) / (1024**3)
    
    # 模型参数内存（约0.7GB for small T5）
    model_mem_gb = 0.7
    
    # 梯度内存（与模型参数相同）
    gradient_mem_gb = 0.7
    
    # 优化器状态（Adafactor约为模型参数的1.5倍）
    optimizer_mem_gb = 1.0
    
    # 激活值内存（batch_size=1时约1-2GB per GPU）
    activation_mem_gb = 1.5
    
    # 系统开销
    system_overhead_gb = 1.0
    
    total_per_gpu = (
        dataset_mem_gb + 
        model_mem_gb + 
        gradient_mem_gb + 
        optimizer_mem_gb + 
        activation_mem_gb + 
        system_overhead_gb
    )
    
    return {
        'dataset_gb': dataset_mem_gb,
        'model_gb': model_mem_gb,
        'gradient_gb': gradient_mem_gb,
        'optimizer_gb': optimizer_mem_gb,
        'activation_gb': activation_mem_gb,
        'system_gb': system_overhead_gb,
        'total_per_gpu_gb': total_per_gpu,
        'total_2gpu_gb': total_per_gpu * 2,
    }


def load_data(input_file):
    """加载数据文件（支持parquet和json）"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_file}")
    
    print(f"正在加载数据: {input_file}")
    
    if input_path.suffix == '.parquet':
        # 读取parquet文件
        df = pd.read_parquet(input_file)
        print(f"  - 格式: Parquet")
        print(f"  - 总样本数: {len(df):,}")
        print(f"  - 列名: {list(df.columns)}")
        return df
    
    elif input_path.suffix == '.json':
        # 读取json文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("JSON文件格式不支持，需要是列表格式")
        
        print(f"  - 格式: JSON")
        print(f"  - 总样本数: {len(df):,}")
        print(f"  - 列名: {list(df.columns)}")
        return df
    
    else:
        raise ValueError(f"不支持的文件格式: {input_path.suffix}，仅支持 .parquet 和 .json")


def sample_data(df, num_samples, random_state=42):
    """从数据中随机采样"""
    if num_samples >= len(df):
        print(f"⚠️  请求样本数({num_samples})大于等于总样本数({len(df)})，将使用全部数据")
        return df
    
    print(f"正在采样 {num_samples:,} 个样本...")
    sampled_df = df.sample(n=num_samples, random_state=random_state)
    return sampled_df.reset_index(drop=True)


def save_parquet(df, output_file):
    """保存为parquet格式"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在保存到: {output_file}")
    
    # 转换为pyarrow table并保存
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
    
    # 获取文件大小
    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"  - 文件大小: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='从大数据集中采样小数据集用于SFT训练')
    parser.add_argument('--input', type=str, required=True, 
                        help='输入文件路径（支持.parquet或.json）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出文件路径（.parquet格式）')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='采样数量（默认5000）')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='验证集比例（默认0.1，即10%%）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认42）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("准备SFT小数据集 - 适配16G内存训练")
    print("=" * 80)
    
    # 1. 加载数据
    df = load_data(args.input)
    
    # 2. 采样训练数据
    train_samples = args.num_samples
    valid_samples = int(train_samples * args.valid_ratio)
    total_samples = train_samples + valid_samples
    
    print(f"\n采样配置:")
    print(f"  - 训练集: {train_samples:,} 样本")
    print(f"  - 验证集: {valid_samples:,} 样本")
    print(f"  - 总计: {total_samples:,} 样本")
    
    # 采样总数据
    sampled_df = sample_data(df, total_samples, random_state=args.seed)
    
    # 3. 分割训练集和验证集
    train_df = sampled_df.iloc[:train_samples]
    valid_df = sampled_df.iloc[train_samples:]
    
    # 4. 保存文件
    output_path = Path(args.output)
    train_output = output_path.parent / f"{output_path.stem}_train{output_path.suffix}"
    valid_output = output_path.parent / f"{output_path.stem}_valid{output_path.suffix}"
    
    print(f"\n保存文件:")
    save_parquet(train_df, str(train_output))
    save_parquet(valid_df, str(valid_output))
    
    # 5. 估算内存使用
    print(f"\n" + "=" * 80)
    print("内存使用估算（基于平均序列长度512）:")
    print("=" * 80)
    
    mem_est = estimate_memory_usage(train_samples, avg_seq_len=512)
    
    print(f"单GPU内存分配:")
    print(f"  - 数据集: {mem_est['dataset_gb']:.2f} GB")
    print(f"  - 模型参数: {mem_est['model_gb']:.2f} GB")
    print(f"  - 梯度: {mem_est['gradient_gb']:.2f} GB")
    print(f"  - 优化器: {mem_est['optimizer_gb']:.2f} GB")
    print(f"  - 激活值: {mem_est['activation_gb']:.2f} GB")
    print(f"  - 系统开销: {mem_est['system_gb']:.2f} GB")
    print(f"  {'─' * 40}")
    print(f"  单GPU总计: {mem_est['total_per_gpu_gb']:.2f} GB")
    print(f"  双GPU总计: {mem_est['total_2gpu_gb']:.2f} GB")
    
    # 6. 给出建议
    print(f"\n" + "=" * 80)
    print("训练建议:")
    print("=" * 80)
    
    if mem_est['total_2gpu_gb'] <= 10:
        print("✅ 内存充足！可以安全训练")
        print("   推荐配置: batch_size=1, accumulation_steps=8")
    elif mem_est['total_2gpu_gb'] <= 12:
        print("⚠️  内存较紧张，建议使用ultra_low_mem模式")
        print("   推荐配置: batch_size=1, accumulation_steps=8, ultra_low_mem=True")
    else:
        print("❌ 内存可能不足！建议减少样本数量")
        recommended_samples = int(train_samples * 10 / mem_est['total_2gpu_gb'])
        print(f"   推荐样本数: {recommended_samples:,}")
    
    print(f"\n训练命令:")
    print(f"  accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True")
    
    print(f"\n✅ 完成！")
    print(f"  - 训练集: {train_output}")
    print(f"  - 验证集: {valid_output}")


if __name__ == '__main__':
    main()
