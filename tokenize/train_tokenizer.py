#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练中文 Tokenizer 脚本

功能说明：
1. 基于 T5-base tokenizer 训练新的中文 tokenizer（推荐方式）
   - 使用 T5-base 的 tokenizer 作为基础
   - 在中文维基百科语料上训练新的词汇表
   - 适合中文为主的场景

2. 从零开始创建自定义 tokenizer
   - 字符级别 BPE tokenizer
   - 字节级别 BPE tokenizer
   - 完全自定义的 tokenizer

使用方法：
    # 方式1：基于 T5-base 训练（推荐）
    python train_tokenizer.py --method t5-base --wiki-file ../data/wiki.simple.txt --output-dir ../model_save/my_tokenizer_wiki
    
    # 方式2：从零创建字符级别 tokenizer
    python train_tokenizer.py --method char-bpe --wiki-file ../data/wiki.simple.txt --output-dir ../model_save/my_tokenizer_char
    
    # 方式3：从零创建字节级别 tokenizer
    python train_tokenizer.py --method byte-bpe --wiki-file ../data/wiki.simple.txt --output-dir ../model_save/my_tokenizer_byte
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 延迟导入，只在需要时检查
def check_transformers():
    """检查并导入 transformers 和 rich"""
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
        from rich import progress
        return AutoTokenizer, PreTrainedTokenizerFast, progress
    except ImportError:
        print("错误: 缺少必要的库，请安装: pip install transformers rich")
        sys.exit(1)

def check_tokenizers():
    """检查并导入 tokenizers"""
    try:
        import tokenizers
        from tokenizers import Tokenizer, decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace, Punctuation, Digits, ByteLevel, Metaspace
        from tokenizers.normalizers import NFKC
        return tokenizers, Tokenizer, decoders, BPE, Whitespace, Punctuation, Digits, ByteLevel, Metaspace, NFKC
    except ImportError:
        return None, None, None, None, None, None, None, None, None, None

# 全局变量，延迟初始化
AutoTokenizer = None
PreTrainedTokenizerFast = None
progress = None
tokenizers = None
Tokenizer = None
decoders = None
BPE = None
Whitespace = None
Punctuation = None
Digits = None
ByteLevel = None
Metaspace = None
NFKC = None

from config import PROJECT_ROOT


def get_wiki_corpus_iterator(wiki_file: str, min_chunk_size: int = 2048, batch_size: int = 1000) -> Iterator[List[str]]:
    """
    从维基百科文件中生成训练语料迭代器
    
    Args:
        wiki_file: 维基百科文本文件路径
        min_chunk_size: 每个文本块的最小字符数（默认2048）
        batch_size: 每次迭代返回的文本块数量（默认1000）
    
    Yields:
        包含多个文本块的列表，每个文本块至少 min_chunk_size 个字符
    """
    global progress
    if progress is None:
        _, _, progress = check_transformers()
    if not os.path.exists(wiki_file):
        raise FileNotFoundError(f"维基百科文件不存在: {wiki_file}")
    
    print(f"加载维基百科语料: {wiki_file}")
    lines = []
    with open(wiki_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总行数: {len(lines)}")
    
    def get_training_corpus():
        buffer = []
        txt = []
        len_cnt = 0
        
        for line in progress.track(lines, description="处理语料"):
            # 跳过空行
            line = line.strip()
            if not line:
                continue
            
            len_cnt += len(line)
            txt.append(line)
            
            # 当累积字符数达到最小块大小时，创建一个文本块
            if len_cnt >= min_chunk_size:
                text = ' '.join(txt)  # 使用空格连接，而不是直接拼接
                # 确保文本不为空且长度合理
                if text and len(text) >= 10:
                    buffer.append(text)
                txt = []
                len_cnt = 0
            
            # 当缓冲区达到批次大小时，返回一批数据
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
        
        # 处理剩余的文本
        if txt:
            text = ' '.join(txt)
            if text and len(text) >= 10:
                buffer.append(text)
        
        # 返回最后一批数据
        if len(buffer) > 0:
            yield buffer
    
    return get_training_corpus()


def get_parquet_corpus_iterator(parquet_file: str, batch_size: int = 1000) -> Iterator[List[str]]:
    """
    从 Parquet 文件中生成训练语料迭代器
    
    Args:
        parquet_file: Parquet 文件路径
        batch_size: 每次迭代返回的样本数量（默认1000）
    
    Yields:
        包含多个文本的列表
    """
    global progress
    if progress is None:
        _, _, progress = check_transformers()
    
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("需要 pyarrow 库: pip install pyarrow")
    
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_file}")
    
    print(f"加载 Parquet 语料: {parquet_file}")
    pf = pq.read_table(parquet_file)
    
    def get_training_corpus():
        buffer = []
        for prompt, response in progress.track(
            zip(pf['prompt'], pf['response']), 
            total=pf.num_rows,
            description="处理语料"
        ):
            # 获取实际的字符串值
            prompt_str = prompt.as_py() if prompt.as_py() else ""
            response_str = response.as_py() if response.as_py() else ""
            
            # 跳过空值
            if not prompt_str and not response_str:
                continue
            
            # 组合 prompt 和 response
            text = f"{prompt_str} {response_str}".strip()
            
            # 确保文本不为空且长度合理
            if text and len(text) >= 10:
                buffer.append(text)
            
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
        
        if buffer:
            yield buffer
    
    return get_training_corpus()


def train_t5_base_tokenizer(
    corpus_iterator: Iterator[List[str]],
    vocab_size: int = 40960,
    output_dir: str = None
):
    """训练基于 T5-base 的 tokenizer（需要 transformers）"""
    global AutoTokenizer, PreTrainedTokenizerFast
    if AutoTokenizer is None:
        AutoTokenizer, PreTrainedTokenizerFast, _ = check_transformers()
    """
    基于 T5-base tokenizer 训练新的中文 tokenizer（推荐方式）
    
    Args:
        corpus_iterator: 语料迭代器
        vocab_size: 词汇表大小（默认40960）
        output_dir: 输出目录
    
    Returns:
        训练好的 tokenizer
    """
    print("\n" + "="*60)
    print("方法: 基于 T5-base tokenizer 训练")
    print("="*60)
    
    # Step 1: 加载 T5-base 的 tokenizer
    print("\n步骤 1: 加载 T5-base tokenizer...")
    old_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    print("✓ T5-base tokenizer 加载完成")
    
    # Step 2: 训练新的 tokenizer
    print(f"\n步骤 2: 开始训练 tokenizer (词汇表大小: {vocab_size})...")
    print("注意: 这是 CPU 密集型任务，可能需要较长时间（约1小时）")
    print("      最大内存占用约 20GB，请确保有足够内存")
    
    tokenizer = old_tokenizer.train_new_from_iterator(
        corpus_iterator, 
        vocab_size=vocab_size
    )
    
    print("✓ Tokenizer 训练完成")
    
    # Step 3: 保存 tokenizer
    if output_dir:
        print(f"\n步骤 3: 保存 tokenizer 到 {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Tokenizer 已保存到 {output_dir}")
    
    return tokenizer


def train_char_level_tokenizer(
    corpus_iterator: Iterator[List[str]],
    vocab_size: int = 40960,
    output_dir: str = None
):
    """
    训练字符级别的 BPE tokenizer
    
    Args:
        corpus_iterator: 语料迭代器
        vocab_size: 词汇表大小（默认40960）
        output_dir: 输出目录
    
    Returns:
        训练好的 tokenizer
    """
    global tokenizers, Tokenizer, decoders, BPE, Punctuation, Digits, Metaspace, NFKC, PreTrainedTokenizerFast
    
    # 检查并导入必要的库
    if tokenizers is None:
        result = check_tokenizers()
        if result[0] is None:
            raise ImportError("需要 tokenizers 库: pip install tokenizers")
        tokenizers, Tokenizer, decoders, BPE, _, Punctuation, Digits, Metaspace, NFKC = result[:9]
    
    if PreTrainedTokenizerFast is None:
        _, PreTrainedTokenizerFast, _ = check_transformers()
    
    print("\n" + "="*60)
    print("方法: 字符级别 BPE tokenizer")
    print("="*60)
    
    # 创建字符级别的 BPE tokenizer
    print("\n步骤 1: 创建字符级别 BPE tokenizer...")
    model = BPE(unk_token="[UNK]")
    tokenizer_obj = Tokenizer(model)
    
    # 使用 NFKC 标准化（全角转半角等）
    tokenizer_obj.normalizer = tokenizers.normalizers.Sequence([NFKC()])
    
    # 预分割：标点符号、数字、Metaspace
    tokenizer_obj.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        Punctuation(), 
        Digits(individual_digits=True), 
        Metaspace()
    ])
    
    # 添加特殊标记
    tokenizer_obj.add_special_tokens([
        "[PAD]", "[EOS]", "[SEP]", "[BOS]", 
        "[CLS]", "[MASK]", "[UNK]"
    ])
    
    # 设置解码器
    tokenizer_obj.decoder = decoders.Metaspace()
    
    print("✓ Tokenizer 对象创建完成")
    
    # 转换为 PreTrainedTokenizerFast
    print("\n步骤 2: 转换为 PreTrainedTokenizerFast...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token='[BOS]',
        eos_token='[EOS]',
    )
    
    # 训练 tokenizer
    print(f"\n步骤 3: 开始训练 tokenizer (词汇表大小: {vocab_size})...")
    tokenizer = tokenizer.train_new_from_iterator(
        corpus_iterator, 
        vocab_size=vocab_size
    )
    
    print("✓ Tokenizer 训练完成")
    
    # 保存 tokenizer
    if output_dir:
        print(f"\n步骤 4: 保存 tokenizer 到 {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Tokenizer 已保存到 {output_dir}")
    
    return tokenizer


def train_byte_level_tokenizer(
    corpus_iterator: Iterator[List[str]],
    vocab_size: int = 40960,
    output_dir: str = None
):
    """
    训练字节级别的 BPE tokenizer
    
    Args:
        corpus_iterator: 语料迭代器
        vocab_size: 词汇表大小（默认40960）
        output_dir: 输出目录
    
    Returns:
        训练好的 tokenizer
    """
    global tokenizers, Tokenizer, decoders, BPE, ByteLevel, PreTrainedTokenizerFast
    
    # 检查并导入必要的库
    if tokenizers is None:
        result = check_tokenizers()
        if result[0] is None:
            raise ImportError("需要 tokenizers 库: pip install tokenizers")
        tokenizers, Tokenizer, decoders, BPE, _, _, _, ByteLevel, _, _ = result
    
    if PreTrainedTokenizerFast is None:
        _, PreTrainedTokenizerFast, _ = check_transformers()
    
    print("\n" + "="*60)
    print("方法: 字节级别 BPE tokenizer")
    print("="*60)
    
    # 创建字节级别的 BPE tokenizer
    print("\n步骤 1: 创建字节级别 BPE tokenizer...")
    model = BPE()  # 字节级别不需要 unk_token
    tokenizer_obj = Tokenizer(model)
    
    # 字节级别预分割
    tokenizer_obj.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )
    
    # 添加特殊标记
    tokenizer_obj.add_special_tokens([
        "[PAD]", "[EOS]", "[SEP]", "[BOS]", 
        "[CLS]", "[MASK]", "[UNK]"
    ])
    
    # 设置解码器和后处理器
    tokenizer_obj.decoder = decoders.ByteLevel(
        add_prefix_space=True, 
        use_regex=True
    )
    tokenizer_obj.post_processor = tokenizers.processors.ByteLevel(
        trim_offsets=False
    )
    
    print("✓ Tokenizer 对象创建完成")
    
    # 转换为 PreTrainedTokenizerFast
    print("\n步骤 2: 转换为 PreTrainedTokenizerFast...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        bos_token='[BOS]',
        eos_token='[EOS]',
    )
    
    # 训练 tokenizer
    print(f"\n步骤 3: 开始训练 tokenizer (词汇表大小: {vocab_size})...")
    tokenizer = tokenizer.train_new_from_iterator(
        corpus_iterator, 
        vocab_size=vocab_size
    )
    
    print("✓ Tokenizer 训练完成")
    
    # 保存 tokenizer
    if output_dir:
        print(f"\n步骤 4: 保存 tokenizer 到 {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Tokenizer 已保存到 {output_dir}")
    
    return tokenizer


def test_tokenizer(tokenizer, test_text: str = None):
    """测试 tokenizer 的功能"""
    if test_text is None:
        test_text = '这是一段中英混输的句子, （chinese and English, here are words.）'
    
    print("\n" + "="*60)
    print("测试 Tokenizer")
    print("="*60)
    print(f"测试文本: {test_text}")
    
    # Tokenize
    tokens = tokenizer.tokenize(test_text)
    print(f"\nTokenize 结果 ({len(tokens)} 个 tokens):")
    print(tokens[:20])  # 只显示前20个
    if len(tokens) > 20:
        print(f"... (共 {len(tokens)} 个 tokens)")
    
    # Encode
    ids = tokenizer.encode(test_text)
    print(f"\nEncode 结果 ({len(ids)} 个 IDs):")
    print(ids[:20])  # 只显示前20个
    if len(ids) > 20:
        print(f"... (共 {len(ids)} 个 IDs)")
    
    # Decode
    decoded = tokenizer.decode(
        ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    print(f"\nDecode 结果:")
    print(decoded)
    
    # 验证
    if decoded.strip() == test_text.strip():
        print("\n✓ 编码解码一致性检查通过")
    else:
        print("\n⚠ 编码解码结果略有差异（可能是空格处理）")


def main():
    parser = argparse.ArgumentParser(
        description='训练中文 Tokenizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 方式1：基于 T5-base 训练（推荐）
  python train_tokenizer.py --method t5-base --wiki-file ../data/my_corpus_processed.txt --output-dir ../model_save/my_tokenizer_wiki
  
  # 方式2：从零创建字符级别 tokenizer
  python train_tokenizer.py --method char-bpe --wiki-file ../data/wiki.simple.txt --output-dir ../model_save/my_tokenizer_char
  
  # 方式3：从零创建字节级别 tokenizer
  python train_tokenizer.py --method byte-bpe --wiki-file ../data/wiki.simple.txt --output-dir ../model_save/my_tokenizer_byte
  
  # 使用 Parquet 文件作为语料
  python train_tokenizer.py --method t5-base --parquet-file ../data/my_dataset.shuffle.parquet --output-dir ../model_save/my_tokenizer_wiki
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['t5-base', 'char-bpe', 'byte-bpe'],
        default='t5-base',
        help='训练方法: t5-base (推荐), char-bpe, byte-bpe'
    )
    
    parser.add_argument(
        '--wiki-file',
        type=str,
        help='维基百科文本文件路径（.txt 格式）'
    )
    
    parser.add_argument(
        '--parquet-file',
        type=str,
        help='Parquet 文件路径（包含 prompt 和 response 列）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（保存训练好的 tokenizer）'
    )
    
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=40960,
        help='词汇表大小（默认: 40960）'
    )
    
    parser.add_argument(
        '--min-chunk-size',
        type=int,
        default=2048,
        help='每个文本块的最小字符数（仅用于 wiki-file，默认: 2048）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='每次迭代的批次大小（默认: 1000）'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='训练完成后测试 tokenizer'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not args.wiki_file and not args.parquet_file:
        parser.error("必须提供 --wiki-file 或 --parquet-file 之一")
    
    if args.wiki_file and args.parquet_file:
        parser.error("不能同时提供 --wiki-file 和 --parquet-file")
    
    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'model_save', f'my_tokenizer_{args.method}')
    
    # 获取语料迭代器
    try:
        if args.wiki_file:
            corpus_iterator = get_wiki_corpus_iterator(
                args.wiki_file,
                min_chunk_size=args.min_chunk_size,
                batch_size=args.batch_size
            )
        else:
            corpus_iterator = get_parquet_corpus_iterator(
                args.parquet_file,
                batch_size=args.batch_size
            )
    except Exception as e:
        print(f"错误: 无法加载语料: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 训练 tokenizer
    try:
        if args.method == 't5-base':
            tokenizer = train_t5_base_tokenizer(
                corpus_iterator,
                vocab_size=args.vocab_size,
                output_dir=args.output_dir
            )
        elif args.method == 'char-bpe':
            tokenizer = train_char_level_tokenizer(
                corpus_iterator,
                vocab_size=args.vocab_size,
                output_dir=args.output_dir
            )
        elif args.method == 'byte-bpe':
            tokenizer = train_byte_level_tokenizer(
                corpus_iterator,
                vocab_size=args.vocab_size,
                output_dir=args.output_dir
            )
        
        # 测试 tokenizer
        if args.test:
            test_tokenizer(tokenizer)
        
        print("\n" + "="*60)
        print("训练完成！")
        print(f"Tokenizer 已保存到: {args.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n用户中断训练")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
