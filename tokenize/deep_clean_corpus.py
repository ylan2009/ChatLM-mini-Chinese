#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度清洗语料文件 - 专门解决 tokenizers 库崩溃问题

这个脚本比 fix_tokenizer_training.py 更激进，会去除所有可能导致问题的字符

使用方法：
    python deep_clean_corpus.py --input ../data/my_corpus_clean.txt --output ../data/my_corpus_deep_clean.txt
"""

import argparse
import os
import re
import unicodedata
from tqdm import tqdm


def remove_zero_width_chars(text: str) -> str:
    """去除零宽字符"""
    zero_width_chars = [
        '\u200b',  # 零宽空格
        '\u200c',  # 零宽非连接符
        '\u200d',  # 零宽连接符
        '\ufeff',  # 零宽非断空格（BOM）
        '\u2060',  # 字连接符
        '\u180e',  # 蒙古文元音分隔符
    ]
    for char in zero_width_chars:
        text = text.replace(char, '')
    return text


def remove_control_chars(text: str) -> str:
    """去除所有控制字符（除了换行符和制表符）"""
    # 保留常用的空白字符：空格、制表符、换行符
    return ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in [' ', '\t', '\n'])


def remove_problematic_unicode(text: str) -> str:
    """去除可能导致问题的 Unicode 字符"""
    # 去除私有使用区字符
    text = re.sub(r'[\ue000-\uf8ff]', '', text)  # 私有使用区
    text = re.sub(r'[\U000f0000-\U000ffffd]', '', text)  # 补充私有使用区-A
    text = re.sub(r'[\U00100000-\U0010fffd]', '', text)  # 补充私有使用区-B
    
    # 去除某些特殊符号
    text = re.sub(r'[\ufff0-\uffff]', '', text)  # 特殊用途字符
    
    return text


def normalize_whitespace(text: str) -> str:
    """标准化空白字符"""
    # 将所有空白字符统一为普通空格
    text = re.sub(r'[\t\r\n\f\v]', ' ', text)
    # 去除多余空格
    text = re.sub(r' +', ' ', text)
    return text.strip()


def deep_clean_line(line: str) -> str:
    """深度清洗单行文本"""
    # 1. 去除零宽字符
    line = remove_zero_width_chars(line)
    
    # 2. 去除控制字符
    line = remove_control_chars(line)
    
    # 3. 去除问题 Unicode 字符
    line = remove_problematic_unicode(line)
    
    # 4. 标准化为 NFC 形式（推荐的 Unicode 标准化形式）
    line = unicodedata.normalize('NFC', line)
    
    # 5. 标准化空白字符
    line = normalize_whitespace(line)
    
    return line


def is_valid_line(line: str, min_length: int = 10, max_length: int = 10000) -> bool:
    """
    检查文本行是否有效
    
    Args:
        line: 文本行
        min_length: 最小长度
        max_length: 最大长度
    
    Returns:
        是否有效
    """
    if not line:
        return False
    
    # 长度检查
    if len(line) < min_length or len(line) > max_length:
        return False
    
    # 检查是否包含有效字符（中文、英文或数字）
    if not re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', line):
        return False
    
    # 检查是否包含过多的特殊字符（可能是乱码）
    special_char_count = len(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:，。！？；：、""''（）()【】\[\]《》<>]', line))
    if special_char_count > len(line) * 0.3:  # 特殊字符超过 30%
        return False
    
    return True


def deep_clean_corpus(input_file: str, output_file: str, min_length: int = 10, max_length: int = 10000):
    """
    深度清洗语料文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        min_length: 最小行长度
        max_length: 最大行长度
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
    print("🧹 深度清洗文本...")
    print("   - 去除零宽字符")
    print("   - 去除控制字符")
    print("   - 去除问题 Unicode 字符")
    print("   - 标准化为 NFC 形式")
    print("   - 标准化空白字符")
    
    valid_lines = 0
    invalid_lines = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            # 使用 tqdm 显示进度
            for line in tqdm(f_in, desc="处理进度"):
                # 深度清洗
                line = deep_clean_line(line)
                
                # 检查有效性
                if is_valid_line(line, min_length=min_length, max_length=max_length):
                    f_out.write(line + '\n')
                    valid_lines += 1
                else:
                    invalid_lines += 1
    
    print(f"\n✅ 有效行数: {valid_lines:,}")
    print(f"❌ 无效行数: {invalid_lines:,}")
    
    if valid_lines + invalid_lines > 0:
        print(f"📉 过滤率: {invalid_lines / (valid_lines + invalid_lines) * 100:.2f}%")
    
    output_size = os.path.getsize(output_file)
    output_size_mb = output_size / (1024 * 1024)
    print(f"✅ 输出文件大小: {output_size_mb:.2f} MB")
    
    print("\n🎉 深度清洗完成！")
    print(f"📁 输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="深度清洗语料文件 - 解决 tokenizers 库崩溃问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 深度清洗语料文件
  python deep_clean_corpus.py \\
    --input ../data/my_corpus_clean.txt \\
    --output ../data/my_corpus_deep_clean.txt
  
  # 然后训练 tokenizer
  python train_tokenizer.py \\
    --method t5-base \\
    --wiki-file ../data/my_corpus_deep_clean.txt \\
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
        default=10,
        help='最小行长度（默认：10）'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=10000,
        help='最大行长度（默认：10000）'
    )
    
    args = parser.parse_args()
    
    # 执行深度清洗
    deep_clean_corpus(
        input_file=args.input,
        output_file=args.output,
        min_length=args.min_length,
        max_length=args.max_length
    )


if __name__ == '__main__':
    main()
