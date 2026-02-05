#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本语料清洗脚本

功能：
1. 清洗文本文件，去除空行、特殊字符、过短/过长的行
2. 合并短行，生成适合训练的文本块
3. 输出为适合 tokenizer 训练的格式

使用方法：
    python clean_corpus.py --input ../data/wiki.txt --output ../data/my_corpus.txt
    python clean_corpus.py --input ../data/wiki.simple.txt --output ../data/my_corpus.txt --min-length 50 --max-length 10000
"""

import argparse
import os
import re
from typing import List, Iterator
from tqdm import tqdm


# 预编译正则表达式以提升性能
_WHITESPACE_PATTERN = re.compile(r'\s+')
_CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]')
_VALID_TEXT_PATTERN = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]')
_REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{20,}')


def clean_text(text: str) -> str:
    """
    清洗单行文本（优化版本）
    
    Args:
        text: 原始文本
    
    Returns:
        清洗后的文本
    """
    # 去除首尾空白
    text = text.strip()
    
    # 去除多余的空格（保留单个空格）
    text = _WHITESPACE_PATTERN.sub(' ', text)
    
    # 去除特殊控制字符
    text = _CONTROL_CHAR_PATTERN.sub('', text)
    
    return text


def is_valid_text(text: str, min_length: int = 10, max_length: int = 50000) -> bool:
    """
    检查文本是否有效（优化版本）
    
    Args:
        text: 文本
        min_length: 最小长度
        max_length: 最大长度
    
    Returns:
        是否有效
    """
    if not text:
        return False
    
    # 检查长度
    text_len = len(text)
    if text_len < min_length or text_len > max_length:
        return False
    
    # 检查是否全是空格或特殊字符
    if not _VALID_TEXT_PATTERN.search(text):
        return False
    
    # 检查是否包含过多的重复字符（可能是垃圾数据）
    if _REPEATED_CHAR_PATTERN.search(text):
        return False
    
    return True


def merge_short_lines(
    lines: List[str], 
    target_length: int = 2048,
    min_length: int = 10,
    max_length: int = 50000
) -> Iterator[str]:
    """
    合并短行，生成适合训练的文本块
    
    Args:
        lines: 文本行列表
        target_length: 目标文本块长度
        min_length: 单行最小长度
        max_length: 单行最大长度
    
    Yields:
        合并后的文本块
    """
    buffer = []
    current_length = 0
    
    for line in lines:
        # 清洗文本
        line = clean_text(line)
        
        # 跳过无效文本
        if not is_valid_text(line, min_length=min_length, max_length=max_length):
            continue
        
        # 如果单行就超过目标长度，直接输出
        if len(line) >= target_length:
            # 先输出缓冲区
            if buffer:
                yield ' '.join(buffer)
                buffer = []
                current_length = 0
            
            # 输出长行
            yield line
            continue
        
        # 累积到缓冲区
        buffer.append(line)
        current_length += len(line)
        
        # 如果达到目标长度，输出缓冲区
        if current_length >= target_length:
            yield ' '.join(buffer)
            buffer = []
            current_length = 0
    
    # 输出剩余的缓冲区
    if buffer:
        text = ' '.join(buffer)
        if is_valid_text(text, min_length=min_length):
            yield text


def clean_corpus(
    input_file: str,
    output_file: str,
    target_length: int = 2048,
    min_length: int = 10,
    max_length: int = 50000,
    encoding: str = 'utf-8',
    buffer_size: int = 10000
):
    """
    清洗语料文件（流式处理，高性能版本）
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        target_length: 目标文本块长度
        min_length: 单行最小长度
        max_length: 单行最大长度
        encoding: 文件编码
        buffer_size: 批量写入缓冲区大小
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    print(f"📖 读取输入文件: {input_file}")
    
    # 获取文件大小
    file_size = os.path.getsize(input_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📊 文件大小: {file_size_mb:.2f} MB")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 流式处理：边读边写
    print("🧹 清洗和合并文本（流式处理）...")
    
    block_count = 0
    total_chars = 0
    write_buffer = []
    
    # 使用流式迭代器
    def line_iterator():
        with open(input_file, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                yield line
    
    # 打开输出文件
    with open(output_file, 'w', encoding=encoding, buffering=8192*1024) as out_f:
        # 使用 tqdm 显示进度（基于文件大小）
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="处理进度") as pbar:
            buffer = []
            current_length = 0
            bytes_read = 0
            
            for line in line_iterator():
                bytes_read += len(line.encode(encoding))
                pbar.update(len(line.encode(encoding)))
                
                # 清洗文本
                line = clean_text(line)
                
                # 跳过无效文本
                if not is_valid_text(line, min_length=min_length, max_length=max_length):
                    continue
                
                # 如果单行就超过目标长度，直接输出
                if len(line) >= target_length:
                    # 先输出缓冲区
                    if buffer:
                        block = ' '.join(buffer)
                        write_buffer.append(block)
                        block_count += 1
                        total_chars += len(block)
                        buffer = []
                        current_length = 0
                    
                    # 输出长行
                    write_buffer.append(line)
                    block_count += 1
                    total_chars += len(line)
                    
                    # 批量写入
                    if len(write_buffer) >= buffer_size:
                        out_f.write('\n'.join(write_buffer) + '\n')
                        write_buffer = []
                    
                    continue
                
                # 累积到缓冲区
                buffer.append(line)
                current_length += len(line)
                
                # 如果达到目标长度，输出缓冲区
                if current_length >= target_length:
                    block = ' '.join(buffer)
                    write_buffer.append(block)
                    block_count += 1
                    total_chars += len(block)
                    buffer = []
                    current_length = 0
                    
                    # 批量写入
                    if len(write_buffer) >= buffer_size:
                        out_f.write('\n'.join(write_buffer) + '\n')
                        write_buffer = []
            
            # 输出剩余的缓冲区
            if buffer:
                block = ' '.join(buffer)
                if is_valid_text(block, min_length=min_length):
                    write_buffer.append(block)
                    block_count += 1
                    total_chars += len(block)
            
            # 写入剩余的数据
            if write_buffer:
                out_f.write('\n'.join(write_buffer) + '\n')
    
    print(f"✅ 生成文本块数: {block_count:,}")
    
    # 统计信息
    avg_length = total_chars / block_count if block_count > 0 else 0
    
    print(f"📊 统计信息:")
    print(f"  - 总字符数: {total_chars:,}")
    print(f"  - 平均块长度: {avg_length:.0f}")
    
    output_size = os.path.getsize(output_file)
    output_size_mb = output_size / (1024 * 1024)
    print(f"✅ 输出文件大小: {output_size_mb:.2f} MB")
    
    # 计算压缩率
    compression_ratio = (1 - output_size / file_size) * 100 if file_size > 0 else 0
    print(f"📉 数据压缩率: {compression_ratio:.2f}%")
    
    print("🎉 清洗完成！")


def preview_file(file_path: str, num_lines: int = 10):
    """
    预览文件内容
    
    Args:
        file_path: 文件路径
        num_lines: 预览行数
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"\n📖 预览文件: {file_path}")
    print("=" * 80)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            print(f"[{i+1}] {line.rstrip()}")
            print("-" * 80)
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="文本语料清洗脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python clean_corpus.py --input ../data/wiki.txt --output ../data/my_corpus.txt
  
  # 自定义参数
  python clean_corpus.py \\
    --input ../data/wiki.simple.txt \\
    --output ../data/my_corpus.txt \\
    --target-length 2048 \\
    --min-length 50 \\
    --max-length 10000
  
  # 预览输出文件
  python clean_corpus.py --input ../data/wiki.txt --output ../data/my_corpus.txt --preview
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入文件路径（例如：../data/wiki.txt）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出文件路径（例如：../data/my_corpus.txt）'
    )
    
    parser.add_argument(
        '--target-length',
        type=int,
        default=2048,
        help='目标文本块长度（默认：2048）'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='单行最小长度（默认：10）'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=50000,
        help='单行最大长度（默认：50000）'
    )
    
    parser.add_argument(
        '--encoding',
        type=str,
        default='utf-8',
        help='文件编码（默认：utf-8）'
    )
    
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=10000,
        help='批量写入缓冲区大小（默认：10000，增大可提升速度）'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='清洗完成后预览输出文件'
    )
    
    parser.add_argument(
        '--preview-lines',
        type=int,
        default=10,
        help='预览行数（默认：10）'
    )
    
    args = parser.parse_args()
    
    # 执行清洗
    clean_corpus(
        input_file=args.input,
        output_file=args.output,
        target_length=args.target_length,
        min_length=args.min_length,
        max_length=args.max_length,
        encoding=args.encoding,
        buffer_size=args.buffer_size
    )
    
    # 预览输出文件
    if args.preview:
        preview_file(args.output, num_lines=args.preview_lines)


if __name__ == '__main__':
    main()
