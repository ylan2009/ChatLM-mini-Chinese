#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 SFT 数据集中随机挑选指定数量的数据，并分割为训练集、验证集和测试集

使用方法：
    python sample_sft_dataset.py

输入文件：
    data/sft_dataset.parquet

输出文件：
    data/sft_train_dataset.parquet
    data/sft_valid_dataset.parquet
    data/sft_test_dataset.parquet

默认配置：
    - 随机挑选：100,000 条数据
    - 分割比例：训练集 80%，验证集 10%，测试集 10%
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.extend(['.', '..'])
from config import PROJECT_ROOT
from utils.logger import Logger

# 初始化日志
log = Logger('sample_sft_dataset', std_out=True, save2file=True, file_name=None)

# 配置
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'sft_dataset.parquet')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')

# 输出文件路径
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'sft_train_dataset.parquet')
VALID_FILE = os.path.join(OUTPUT_DIR, 'sft_valid_dataset.parquet')
TEST_FILE = os.path.join(OUTPUT_DIR, 'sft_test_dataset.parquet')

# 采样和分割配置
SAMPLE_SIZE = 100000          # 随机挑选的数据量
TRAIN_RATIO = 0.80            # 训练集比例
VALID_RATIO = 0.10            # 验证集比例
TEST_RATIO = 0.10             # 测试集比例

# 随机种子（确保可复现）
SEED = 23333


def sample_and_split_dataset(
    input_file: str,
    train_file: str,
    valid_file: str,
    test_file: str,
    sample_size: int = 100000,
    train_ratio: float = 0.80,
    valid_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 23333
) -> None:
    """
    从数据集中随机挑选指定数量的数据，并分割为训练集、验证集和测试集
    
    Args:
        input_file: 输入的 parquet 文件路径
        train_file: 训练集输出文件路径
        valid_file: 验证集输出文件路径
        test_file: 测试集输出文件路径
        sample_size: 随机挑选的数据量
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 检查比例是否正确
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"分割比例之和必须等于 1.0，当前为 {total_ratio}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        log.error(f"输入文件不存在: {input_file}")
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    log.info("=" * 60)
    log.info("开始采样和分割 SFT 数据集")
    log.info("=" * 60)
    log.info(f"输入文件: {input_file}")
    log.info(f"采样数量: {sample_size:,} 条")
    log.info(f"分割比例: 训练集={train_ratio*100:.1f}%, 验证集={valid_ratio*100:.1f}%, 测试集={test_ratio*100:.1f}%")
    log.info(f"随机种子: {seed}")
    
    # 读取数据
    log.info("正在读取数据集...")
    df = pd.read_parquet(input_file)
    total_size = len(df)
    log.info(f"数据集总大小: {total_size:,} 行")
    
    # 检查数据格式
    if 'prompt' not in df.columns or 'response' not in df.columns:
        log.error(f"数据格式错误，需要包含 'prompt' 和 'response' 两列，当前列: {df.columns.tolist()}")
        raise ValueError("数据格式错误")
    
    # 检查采样数量是否超过数据集大小
    if sample_size > total_size:
        log.warning(f"采样数量 ({sample_size:,}) 超过数据集大小 ({total_size:,})，将使用全部数据")
        sample_size = total_size
    
    # 随机采样
    log.info(f"正在随机采样 {sample_size:,} 条数据...")
    sampled_df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    log.info(f"采样完成，共 {len(sampled_df):,} 条数据")
    
    # 打乱采样后的数据
    log.info("正在打乱采样数据...")
    sampled_df = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 计算分割点
    train_size = int(len(sampled_df) * train_ratio)
    valid_size = int(len(sampled_df) * valid_ratio)
    test_size = len(sampled_df) - train_size - valid_size  # 剩余部分分配给测试集
    
    # 分割数据集
    log.info("正在分割数据集...")
    
    # 分割训练集
    train_df = sampled_df.iloc[:train_size].copy()
    
    # 分割验证集
    valid_start = train_size
    valid_end = valid_start + valid_size
    valid_df = sampled_df.iloc[valid_start:valid_end].copy()
    
    # 分割测试集（剩余部分）
    test_df = sampled_df.iloc[valid_end:].copy()
    
    # 显示分割结果
    log.info("")
    log.info("分割结果:")
    log.info(f"  训练集: {len(train_df):,} 行 ({len(train_df)/len(sampled_df)*100:.2f}%)")
    log.info(f"  验证集: {len(valid_df):,} 行 ({len(valid_df)/len(sampled_df)*100:.2f}%)")
    log.info(f"  测试集: {len(test_df):,} 行 ({len(test_df)/len(sampled_df)*100:.2f}%)")
    log.info(f"  总计: {len(train_df) + len(valid_df) + len(test_df):,} 行")
    
    # 保存文件
    log.info("")
    log.info("正在保存文件...")
    
    # 如果输出文件已存在，询问是否删除
    files_to_check = [
        (train_file, "训练集"),
        (valid_file, "验证集"),
        (test_file, "测试集")
    ]
    
    should_overwrite = None
    for file_path, file_name in files_to_check:
        if os.path.exists(file_path):
            if should_overwrite is None:
                log.warning(f"输出文件已存在，是否覆盖所有文件?")
                response = input("是否删除并覆盖所有文件? (y/n): ").strip().lower()
                should_overwrite = response in ('y', 'yes')
            
            if should_overwrite:
                os.remove(file_path)
                log.info(f"已删除: {file_path} ({file_name})")
            else:
                log.error(f"文件已存在，取消操作: {file_path}")
                log.info("如需覆盖，请先手动删除输出文件或重新运行脚本并选择 'y'")
                return
    
    # 保存训练集
    log.info(f"保存训练集到: {train_file}")
    train_df.to_parquet(train_file, index=False, engine='pyarrow')
    log.info(f"  训练集已保存，大小: {len(train_df):,} 行")
    
    # 保存验证集
    log.info(f"保存验证集到: {valid_file}")
    valid_df.to_parquet(valid_file, index=False, engine='pyarrow')
    log.info(f"  验证集已保存，大小: {len(valid_df):,} 行")
    
    # 保存测试集
    log.info(f"保存测试集到: {test_file}")
    test_df.to_parquet(test_file, index=False, engine='pyarrow')
    log.info(f"  测试集已保存，大小: {len(test_df):,} 行")
    
    # 显示数据样例
    log.info("")
    log.info("数据样例:")
    log.info("-" * 60)
    
    log.info("\n训练集样例:")
    for idx, row in train_df.head(2).iterrows():
        log.info(f"  [{idx}] Prompt: {row['prompt'][:80]}...")
        log.info(f"       Response: {row['response'][:80]}...")
    
    log.info("\n验证集样例:")
    for idx, row in valid_df.head(2).iterrows():
        log.info(f"  [{idx}] Prompt: {row['prompt'][:80]}...")
        log.info(f"       Response: {row['response'][:80]}...")
    
    log.info("\n测试集样例:")
    for idx, row in test_df.head(2).iterrows():
        log.info(f"  [{idx}] Prompt: {row['prompt'][:80]}...")
        log.info(f"       Response: {row['response'][:80]}...")
    
    log.info("")
    log.info("=" * 60)
    log.info("数据集采样和分割完成！")
    log.info("=" * 60)
    log.info(f"训练集: {train_file} ({len(train_df):,} 行)")
    log.info(f"验证集: {valid_file} ({len(valid_df):,} 行)")
    log.info(f"测试集: {test_file} ({len(test_df):,} 行)")


def main():
    """
    主函数
    """
    try:
        sample_and_split_dataset(
            input_file=INPUT_FILE,
            train_file=TRAIN_FILE,
            valid_file=VALID_FILE,
            test_file=TEST_FILE,
            sample_size=SAMPLE_SIZE,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            test_ratio=TEST_RATIO,
            seed=SEED
        )
    except Exception as e:
        log.error(f"采样和分割数据集时出错: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
