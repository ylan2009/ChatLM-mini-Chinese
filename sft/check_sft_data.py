#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT数据集质量检查脚本

使用方法：
    python check_sft_data.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.extend(['.', '..'])
from config import PROJECT_ROOT


def check_data_file(file_path: str, file_name: str) -> dict:
    """
    检查单个数据文件
    
    Returns:
        dict: 包含检查结果的字典
    """
    print(f"\n{'='*80}")
    print(f"检查文件: {file_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return {"exists": False}
    
    print(f"✅ 文件存在: {file_path}")
    
    # 读取数据
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return {"exists": True, "readable": False}
    
    print(f"✅ 文件可读")
    
    # 检查列名
    columns = df.columns.tolist()
    print(f"\n📋 列名: {columns}")
    
    required_columns = ['prompt', 'response']
    missing_columns = [col for col in required_columns if col not in columns]
    
    if missing_columns:
        print(f"❌ 缺少必需的列: {missing_columns}")
        return {"exists": True, "readable": True, "valid_format": False}
    
    print(f"✅ 包含所有必需的列: {required_columns}")
    
    # 基本统计
    total_count = len(df)
    print(f"\n📊 数据统计:")
    print(f"  总数据量: {total_count:,} 条")
    
    # 检查空值
    null_prompts = df['prompt'].isnull().sum()
    null_responses = df['response'].isnull().sum()
    
    if null_prompts > 0:
        print(f"  ⚠️  空prompt数量: {null_prompts} ({null_prompts/total_count*100:.2f}%)")
    else:
        print(f"  ✅ 无空prompt")
    
    if null_responses > 0:
        print(f"  ⚠️  空response数量: {null_responses} ({null_responses/total_count*100:.2f}%)")
    else:
        print(f"  ✅ 无空response")
    
    # 检查空字符串
    empty_prompts = (df['prompt'].astype(str).str.strip() == '').sum()
    empty_responses = (df['response'].astype(str).str.strip() == '').sum()
    
    if empty_prompts > 0:
        print(f"  ⚠️  空字符串prompt数量: {empty_prompts} ({empty_prompts/total_count*100:.2f}%)")
    else:
        print(f"  ✅ 无空字符串prompt")
    
    if empty_responses > 0:
        print(f"  ⚠️  空字符串response数量: {empty_responses} ({empty_responses/total_count*100:.2f}%)")
    else:
        print(f"  ✅ 无空字符串response")
    
    # 长度统计
    df['prompt_len'] = df['prompt'].astype(str).str.len()
    df['response_len'] = df['response'].astype(str).str.len()
    
    print(f"\n📏 长度统计:")
    print(f"  Prompt长度:")
    print(f"    平均: {df['prompt_len'].mean():.1f}")
    print(f"    中位数: {df['prompt_len'].median():.1f}")
    print(f"    最小: {df['prompt_len'].min()}")
    print(f"    最大: {df['prompt_len'].max()}")
    print(f"    标准差: {df['prompt_len'].std():.1f}")
    
    print(f"  Response长度:")
    print(f"    平均: {df['response_len'].mean():.1f}")
    print(f"    中位数: {df['response_len'].median():.1f}")
    print(f"    最小: {df['response_len'].min()}")
    print(f"    最大: {df['response_len'].max()}")
    print(f"    标准差: {df['response_len'].std():.1f}")
    
    # 检查异常短的数据
    very_short_prompts = (df['prompt_len'] < 5).sum()
    very_short_responses = (df['response_len'] < 10).sum()
    
    if very_short_prompts > 0:
        print(f"  ⚠️  过短的prompt (<5字符): {very_short_prompts} ({very_short_prompts/total_count*100:.2f}%)")
    
    if very_short_responses > 0:
        print(f"  ⚠️  过短的response (<10字符): {very_short_responses} ({very_short_responses/total_count*100:.2f}%)")
    
    # 检查异常长的数据
    very_long_prompts = (df['prompt_len'] > 512).sum()
    very_long_responses = (df['response_len'] > 512).sum()
    
    if very_long_prompts > 0:
        print(f"  ⚠️  过长的prompt (>512字符): {very_long_prompts} ({very_long_prompts/total_count*100:.2f}%)")
    
    if very_long_responses > 0:
        print(f"  ⚠️  过长的response (>512字符): {very_long_responses} ({very_long_responses/total_count*100:.2f}%)")
    
    # 检查重复数据
    duplicate_count = df.duplicated(subset=['prompt', 'response']).sum()
    if duplicate_count > 0:
        print(f"  ⚠️  重复数据: {duplicate_count} ({duplicate_count/total_count*100:.2f}%)")
    else:
        print(f"  ✅ 无重复数据")
    
    # 显示样例
    print(f"\n📝 数据样例 (前3条):")
    print("-" * 80)
    for idx, row in df.head(3).iterrows():
        prompt = str(row['prompt'])[:100]
        response = str(row['response'])[:100]
        print(f"\n样例 {idx + 1}:")
        print(f"  Prompt: {prompt}...")
        print(f"  Response: {response}...")
    
    return {
        "exists": True,
        "readable": True,
        "valid_format": True,
        "total_count": total_count,
        "null_prompts": null_prompts,
        "null_responses": null_responses,
        "empty_prompts": empty_prompts,
        "empty_responses": empty_responses,
        "duplicate_count": duplicate_count,
        "very_short_prompts": very_short_prompts,
        "very_short_responses": very_short_responses,
        "very_long_prompts": very_long_prompts,
        "very_long_responses": very_long_responses,
    }


def main():
    """
    主函数
    """
    print("=" * 80)
    print("SFT数据集质量检查")
    print("=" * 80)
    
    # 检查的文件列表
    files_to_check = [
        (os.path.join(PROJECT_ROOT, 'data', 'sft_train_dataset.parquet'), "训练集"),
        (os.path.join(PROJECT_ROOT, 'data', 'sft_valid_dataset.parquet'), "验证集"),
        (os.path.join(PROJECT_ROOT, 'data', 'sft_test_dataset.parquet'), "测试集"),
    ]
    
    results = {}
    
    for file_path, file_name in files_to_check:
        result = check_data_file(file_path, file_name)
        results[file_name] = result
    
    # 总结
    print(f"\n{'='*80}")
    print("检查总结")
    print(f"{'='*80}")
    
    all_valid = True
    total_data_count = 0
    
    for file_name, result in results.items():
        if not result.get("exists"):
            print(f"❌ {file_name}: 文件不存在")
            all_valid = False
        elif not result.get("readable"):
            print(f"❌ {file_name}: 文件无法读取")
            all_valid = False
        elif not result.get("valid_format"):
            print(f"❌ {file_name}: 数据格式不正确")
            all_valid = False
        else:
            count = result.get("total_count", 0)
            total_data_count += count
            
            issues = []
            if result.get("null_prompts", 0) > 0:
                issues.append(f"{result['null_prompts']}个空prompt")
            if result.get("null_responses", 0) > 0:
                issues.append(f"{result['null_responses']}个空response")
            if result.get("empty_prompts", 0) > 0:
                issues.append(f"{result['empty_prompts']}个空字符串prompt")
            if result.get("empty_responses", 0) > 0:
                issues.append(f"{result['empty_responses']}个空字符串response")
            if result.get("duplicate_count", 0) > 0:
                issues.append(f"{result['duplicate_count']}条重复数据")
            if result.get("very_short_responses", 0) > 0:
                issues.append(f"{result['very_short_responses']}条过短response")
            
            if issues:
                print(f"⚠️  {file_name}: {count:,}条数据，发现问题: {', '.join(issues)}")
                all_valid = False
            else:
                print(f"✅ {file_name}: {count:,}条数据，无问题")
    
    print(f"\n总数据量: {total_data_count:,} 条")
    
    print(f"\n{'='*80}")
    if all_valid:
        print("✅ 所有数据集检查通过！可以开始训练")
        print("\n🚀 运行训练命令:")
        print("   accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True")
        return 0
    else:
        print("⚠️  数据集存在一些问题，建议修复后再训练")
        print("\n💡 如果问题不严重（如少量过长数据），可以继续训练")
        print("   训练时会自动截断过长的数据")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
