#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 Tokenizer 训练错误

问题：tokenizers 库在训练时出现 Rust panic 错误
解决方案：预处理数据，去除空行和无效字符

使用方法：
    python fix_tokenizer_training.py --input ../data/wiki.simple.txt --output ../data/wiki_clean.txt
"""

import argparse
import os
import re
from tqdm import tqdm


def preprocess_line(line: str) -> str:
    """
    预处理单行文本
    
    Args:
        line: 原始文本行
    
    Returns:
        处理后的文本行
    """
    # 去除首尾空白
    line = line.strip()
    
    # 去除多余的空格
    line = re.sub(r'\s+', ' ', line)
    
    # 去除特殊控制字符
    line = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', line)
    
    return line


def is_valid_line(line: str, min_length: int = 5) -> bool:
    """
    检查文本行是否有效
    
    Args:
        line: 文本行
        min_length: 最小长度
    
    Returns:
        是否有效
    """
    if not line or len(line) < min_length:
        return False
    
    # 检查是否包含有效字符（中文、英文或数字）
    if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', line):
        return False
    
    return True


def preprocess_corpus(input_file: str, output_file: str, min_length: int = 5):
    """
    预处理语料文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        min_length: 最小行长度
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
    
    # 处理文件
    print("🧹 预处理文本...")
    
    valid_lines = 0
    invalid_lines = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # 使用 tqdm 显示进度
            for line in tqdm(f_in, desc="处理进度"):
                # 预处理
                line = preprocess_line(line)
                
                # 检查有效性
                if is_valid_line(line, min_length=min_length):
                    f_out.write(line + '\n')
                    valid_lines += 1
                else:
                    invalid_lines += 1
    
    print(f"✅ 有效行数: {valid_lines:,}")
    print(f"❌ 无效行数: {invalid_lines:,}")
    print(f"📉 过滤率: {invalid_lines / (valid_lines + invalid_lines) * 100:.2f}%")
    
    output_size = os.path.getsize(output_file)
    output_size_mb = output_size / (1024 * 1024)
    print(f"✅ 输出文件大小: {output_size_mb:.2f} MB")
    
    print("🎉 预处理完成！")


def main():
    parser = argparse.ArgumentParser(
        description="修复 Tokenizer 训练错误 - 预处理语料文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预处理 wiki.simple.txt
  python fix_tokenizer_training.py \\
    --input ../data/wiki.simple.txt \\
    --output ../data/wiki_clean.txt
  
  # 然后训练 tokenizer
  python train_tokenizer.py \\
    --method t5-base \\
    --wiki-file ../data/wiki_clean.txt \\
    --output-dir ../model_save/my_tokenizer_wiki \\
    --vocab-size 40960 \\
    --batch-size 500
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入文件路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出文件路径'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=5,
        help='最小行长度（默认：5）'
    )
    
    args = parser.parse_args()
    
    # 执行预处理
    preprocess_corpus(
        input_file=args.input,
        output_file=args.output,
        min_length=args.min_length
    )


if __name__ == '__main__':
    main()
