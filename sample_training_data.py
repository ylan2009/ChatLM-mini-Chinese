#!/usr/bin/env python3
"""
数据采样脚本 - 从大数据集中采样高质量样本

用法：
    # 随机采样 300万条
    python sample_training_data.py --input data/my_train_dataset.parquet --output data/my_train_dataset_3m.parquet --num_samples 3000000
    
    # 智能采样 500万条（基于文本长度、多样性）
    python sample_training_data.py --input data/my_train_dataset.parquet --output data/my_train_dataset_5m.parquet --num_samples 5000000 --smart
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def random_sample(df: pd.DataFrame, num_samples: int, seed: int = 42) -> pd.DataFrame:
    """随机采样"""
    print(f"随机采样 {num_samples:,} 条数据...")
    return df.sample(n=num_samples, random_state=seed)


def smart_sample(df: pd.DataFrame, num_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    智能采样 - 优先保留高质量样本
    
    策略：
    1. 过滤掉过短或过长的样本（质量可能较差）
    2. 计算文本多样性分数（基于字符分布）
    3. 按分数排序，保留前 num_samples 条
    """
    print(f"智能采样 {num_samples:,} 条数据...")
    
    # 1. 计算文本长度
    print("  计算文本长度...")
    df['prompt_len'] = df['prompt'].str.len()
    df['response_len'] = df['response'].str.len()
    df['total_len'] = df['prompt_len'] + df['response_len']
    
    # 2. 过滤异常样本
    print("  过滤异常样本...")
    # 过滤掉过短的样本（<10字符）
    df = df[df['total_len'] >= 10]
    # 过滤掉过长的样本（>1000字符，可能是噪声）
    df = df[df['total_len'] <= 1000]
    
    print(f"  过滤后剩余 {len(df):,} 条数据")
    
    # 3. 计算多样性分数（基于字符分布的熵）
    print("  计算多样性分数...")
    
    def calc_diversity_score(text: str) -> float:
        """计算文本多样性分数（字符分布的熵）"""
        if not text or len(text) == 0:
            return 0.0
        
        # 统计字符频率
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # 计算熵
        total = len(text)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    # 只对前 num_samples * 2 条数据计算分数（节省时间）
    sample_size = min(len(df), num_samples * 2)
    df_sample = df.sample(n=sample_size, random_state=seed)
    
    tqdm.pandas(desc="  计算分数")
    df_sample['diversity_score'] = df_sample['prompt'].progress_apply(calc_diversity_score)
    
    # 4. 按分数排序，保留前 num_samples 条
    print("  按分数排序...")
    df_sample = df_sample.sort_values('diversity_score', ascending=False)
    df_result = df_sample.head(num_samples)
    
    # 5. 删除临时列
    df_result = df_result.drop(columns=['prompt_len', 'response_len', 'total_len', 'diversity_score'])
    
    return df_result


def main():
    parser = argparse.ArgumentParser(description='数据采样脚本')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径（parquet格式）')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径（parquet格式）')
    parser.add_argument('--num_samples', type=int, required=True, help='采样数量')
    parser.add_argument('--smart', action='store_true', help='使用智能采样（默认随机采样）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    # 读取数据
    print(f"读取数据: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  原始数据量: {len(df):,} 条")
    
    # 检查采样数量
    if args.num_samples > len(df):
        print(f"⚠️  采样数量 ({args.num_samples:,}) 大于原始数据量 ({len(df):,})，使用全部数据")
        df_sampled = df
    else:
        # 采样
        if args.smart:
            df_sampled = smart_sample(df, args.num_samples, args.seed)
        else:
            df_sampled = random_sample(df, args.num_samples, args.seed)
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"保存数据: {args.output}")
    df_sampled.to_parquet(args.output, index=False)
    
    print(f"✅ 完成！采样后数据量: {len(df_sampled):,} 条")
    
    # 统计信息
    print("\n数据统计:")
    print(f"  prompt 平均长度: {df_sampled['prompt'].str.len().mean():.1f} 字符")
    print(f"  response 平均长度: {df_sampled['response'].str.len().mean():.1f} 字符")
    print(f"  文件大小: {output_path.stat().st_size / (1024**2):.1f} MB")


if __name__ == '__main__':
    main()
