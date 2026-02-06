#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查看预训练数据集样本的简单脚本
用法:
    python view_dataset.py                                    # 查看训练集前5条
    python view_dataset.py --file data/my_test_dataset.parquet --num 10  # 查看测试集前10条
    python view_dataset.py --random --num 3                   # 随机查看3条
    python view_dataset.py --stats                            # 只显示统计信息
"""

import sys
sys.path.extend(['.', '..'])

import argparse
from pathlib import Path
from fastparquet import ParquetFile
import random
from config import PROJECT_ROOT


def format_text(text, max_length=200):
    """格式化文本，限制显示长度"""
    if text is None:
        return "[None]"
    text_str = str(text).strip()
    if len(text_str) > max_length:
        return text_str[:max_length] + "..."
    return text_str


def get_dataset_stats(file_path):
    """获取数据集统计信息"""
    print(f"\n{'='*100}")
    print(f"📊 数据集统计信息")
    print(f"{'='*100}\n")
    
    pf = ParquetFile(file_path)
    
    # 获取列名
    first_chunk = next(pf.iter_row_groups())
    columns = first_chunk.columns.tolist()
    print(f"📋 列名: {columns}")
    
    # 统计总行数
    total_rows = 0
    prompt_lengths = []
    response_lengths = []
    
    for chunk in ParquetFile(file_path):
        for rows in chunk.iter_row_groups():
            total_rows += len(rows)
            
            # 统计长度
            if 'prompt' in columns:
                for val in rows['prompt']:
                    if val:
                        prompt_lengths.append(len(str(val)))
            
            if 'response' in columns:
                for val in rows['response']:
                    if val:
                        response_lengths.append(len(str(val)))
    
    print(f"📈 总样本数: {total_rows:,}")
    
    if prompt_lengths:
        print(f"\n📝 Prompt 统计:")
        print(f"   - 平均长度: {sum(prompt_lengths)/len(prompt_lengths):.0f} 字符")
        print(f"   - 最小长度: {min(prompt_lengths)} 字符")
        print(f"   - 最大长度: {max(prompt_lengths)} 字符")
    
    if response_lengths:
        print(f"\n💬 Response 统计:")
        print(f"   - 平均长度: {sum(response_lengths)/len(response_lengths):.0f} 字符")
        print(f"   - 最小长度: {min(response_lengths)} 字符")
        print(f"   - 最大长度: {max(response_lengths)} 字符")
    
    print(f"\n{'='*100}\n")
    
    return total_rows, columns


def view_samples(file_path, num_samples=5, random_sample=False, max_text_length=200):
    """查看数据集样本"""
    print(f"\n{'='*100}")
    print(f"📁 文件: {file_path}")
    print(f"{'='*100}\n")
    
    if not Path(file_path).exists():
        print(f"❌ 错误: 文件不存在 - {file_path}")
        return
    
    try:
        pf = ParquetFile(file_path)
        
        # 获取列名和总行数
        first_chunk = next(pf.iter_row_groups())
        columns = first_chunk.columns.tolist()
        
        # 统计总行数
        total_rows = 0
        for chunk in ParquetFile(file_path):
            for rows in chunk.iter_row_groups():
                total_rows += len(rows)
        
        print(f"📋 列名: {columns}")
        print(f"📈 总样本数: {total_rows:,}\n")
        
        # 收集样本
        samples = []
        current_row = 0
        
        # 如果是随机采样，先生成随机索引
        if random_sample:
            sample_indices = sorted(random.sample(range(total_rows), min(num_samples, total_rows)))
            sample_indices_set = set(sample_indices)
            print(f"🎲 随机采样 {len(sample_indices)} 条样本\n")
        else:
            sample_indices_set = set(range(min(num_samples, total_rows)))
            print(f"📖 显示前 {min(num_samples, total_rows)} 条样本\n")
        
        # 读取样本
        for chunk in ParquetFile(file_path):
            for rows in chunk.iter_row_groups():
                for i in range(len(rows)):
                    if current_row in sample_indices_set:
                        sample = {'row_num': current_row + 1}
                        for col in columns:
                            sample[col] = rows[col][i]
                        samples.append(sample)
                        
                        if len(samples) >= num_samples:
                            break
                    current_row += 1
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
        
        # 显示样本
        print(f"{'='*100}")
        print(f"📝 数据样本")
        print(f"{'='*100}\n")
        
        for idx, sample in enumerate(samples, 1):
            print(f"{'─'*100}")
            print(f"样本 #{sample['row_num']}")
            print(f"{'─'*100}")
            
            for col in columns:
                value = sample.get(col)
                formatted_value = format_text(value, max_text_length)
                print(f"\n【{col}】")
                print(f"{formatted_value}")
            
            print()
        
        print(f"{'='*100}\n")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='查看预训练数据集样本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看训练集前5条
  python view_dataset.py
  
  # 查看测试集前10条
  python view_dataset.py --file data/my_test_dataset.parquet --num 10
  
  # 随机查看3条
  python view_dataset.py --random --num 3
  
  # 只显示统计信息
  python view_dataset.py --stats
  
  # 查看完整文本（不截断）
  python view_dataset.py --num 2 --max-length 0
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=PROJECT_ROOT + '/data/my_train_dataset.parquet',
        help='数据集文件路径 (默认: data/my_train_dataset.parquet)'
    )
    
    parser.add_argument(
        '--num', '-n',
        type=int,
        default=5,
        help='查看的样本数量 (默认: 5)'
    )
    
    parser.add_argument(
        '--random', '-r',
        action='store_true',
        help='随机采样（默认显示前N条）'
    )
    
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='只显示统计信息，不显示样本'
    )
    
    parser.add_argument(
        '--max-length', '-m',
        type=int,
        default=200,
        help='文本显示的最大长度，0表示不限制 (默认: 200)'
    )
    
    args = parser.parse_args()
    
    # 处理文件路径
    file_path = args.file
    if not file_path.startswith('/'):
        file_path = PROJECT_ROOT + '/' + file_path
    
    # 显示统计信息
    if args.stats:
        get_dataset_stats(file_path)
    else:
        # 显示样本
        max_length = None if args.max_length == 0 else args.max_length
        view_samples(file_path, args.num, args.random, max_length or 999999)


if __name__ == '__main__':
    main()
