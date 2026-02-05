#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Belle 数据文件诊断脚本
用于检查 Belle 数据文件的列名、数据结构和样例
"""

import sys
sys.path.extend(['.', '..'])

from fastparquet import ParquetFile
from config import PROJECT_ROOT
import pandas as pd

def diagnose_belle_file(file_path: str):
    """诊断单个 Belle 数据文件"""
    print(f"\n{'='*100}")
    print(f"📁 文件: {file_path}")
    print(f"{'='*100}\n")
    
    try:
        # 读取文件
        pf = ParquetFile(file_path)
        
        # 读取第一个 row group
        first_chunk = next(pf.iter_row_groups())
        
        # 获取列名
        columns = first_chunk.columns.tolist()
        print(f"📋 列名: {columns}\n")
        
        # 显示前 5 行数据
        print(f"📊 前 5 行数据样例:\n")
        for idx in range(min(5, len(first_chunk))):
            print(f"--- 第 {idx + 1} 行 ---")
            for col in columns:
                value = first_chunk[col][idx]
                # 限制显示长度
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                print(f"  {col}: {value_str}")
            print()
        
        # 统计信息
        total_rows = 0
        for chunk in ParquetFile(file_path):
            for rows in chunk.iter_row_groups():
                total_rows += len(rows)
        
        print(f"📈 总行数: {total_rows:,}\n")
        
        # 检查列名匹配
        print(f"🔍 列名匹配检查:")
        
        prompt_candidates = ['instruction', 'prompt', 'input', 'question']
        response_candidates = ['output', 'response', 'answer', 'target']
        
        prompt_col = None
        response_col = None
        
        for col in columns:
            col_lower = col.lower()
            if col_lower in prompt_candidates:
                prompt_col = col
                print(f"  ✅ 找到 prompt 列: {col}")
            elif col_lower in response_candidates:
                response_col = col
                print(f"  ✅ 找到 response 列: {col}")
        
        if 'conversations' in columns:
            print(f"  ✅ 找到 conversations 列")
        
        if not prompt_col and not response_col and 'conversations' not in columns:
            print(f"  ❌ 警告: 没有找到匹配的列名！")
            print(f"  可用列: {columns}")
            print(f"  期望的 prompt 列名: {prompt_candidates}")
            print(f"  期望的 response 列名: {response_candidates}")
        
        print()
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    raw_data_dir = PROJECT_ROOT + '/data/raw_data/belle'
    
    # 要检查的文件
    parquet_files = [
        f'{raw_data_dir}/generated_chat_0.4M.parquet',
        f'{raw_data_dir}/train_0.5M_CN.parquet',
        f'{raw_data_dir}/train_2M_CN.parquet'
    ]
    
    print(f"\n{'#'*100}")
    print(f"# Belle 数据文件诊断")
    print(f"# 数据目录: {raw_data_dir}")
    print(f"{'#'*100}\n")
    
    for file_path in parquet_files:
        diagnose_belle_file(file_path)
    
    print(f"\n{'#'*100}")
    print(f"# 诊断完成")
    print(f"{'#'*100}\n")


if __name__ == '__main__':
    main()
