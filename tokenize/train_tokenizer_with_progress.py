#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带进度监控的 SentencePiece Tokenizer 训练脚本

功能：
1. 实时监控训练进度
2. 显示训练日志
3. 预估训练时间
4. 显示训练统计

使用方法：
    python train_tokenizer_with_progress.py \
        --input ../data/my_corpus_clean.txt \
        --output ../model_save/my_tokenizer_sp \
        --vocab-size 40960

    使用采样50w的数据集

    # 采样数据
    shuf ../data/my_corpus_clean.txt | head -n 500000 > ../data/my_corpus_sampled.txt

    python train_tokenizer_with_progress.py \
        --input ../data/my_corpus_sampled.txt \
        --output ../model_save/my_tokenizer_sp \
        --vocab-size 40960
"""

import os
import sys
import time
import argparse
import threading
import subprocess
from pathlib import Path


def count_lines(file_path):
    """快速统计文件行数"""
    print("📊 正在统计数据量...")
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            line_count += 1
    return line_count


def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{int(seconds)} 秒"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes} 分 {secs} 秒"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} 小时 {minutes} 分"


def monitor_progress(start_time, estimated_time):
    """监控训练进度（动画效果）"""
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    idx = 0
    
    while threading.current_thread().is_alive():
        elapsed = time.time() - start_time
        progress_pct = min(100, (elapsed / estimated_time) * 100)
        
        # 进度条
        bar_length = 40
        filled = int(bar_length * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # 显示信息
        sys.stdout.write(f'\r  {spinner[idx]} 训练中... [{bar}] {progress_pct:.1f}% | 已用时: {format_time(elapsed)}')
        sys.stdout.flush()
        
        idx = (idx + 1) % len(spinner)
        time.sleep(0.1)


def train_sentencepiece(
    input_file,
    output_dir,
    vocab_size=40960,
    model_type='unigram',
    character_coverage=0.9995
):
    """
    训练 SentencePiece tokenizer（带进度监控）
    """
    print("\n" + "="*70)
    print("🚀 SentencePiece Tokenizer 训练")
    print("="*70)
    
    # 检查输入文件
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, 'sentencepiece')
    
    # 步骤 1: 分析数据
    print("\n📊 步骤 1/5: 分析训练数据")
    print("-" * 70)
    
    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    line_count = count_lines(input_file)
    
    print(f"  ✓ 文件路径: {input_file}")
    print(f"  ✓ 文件大小: {file_size_mb:.2f} MB")
    print(f"  ✓ 数据行数: {line_count:,} 行")
    
    # 预估训练时间（粗略估计）
    estimated_minutes = max(5, int(file_size_mb / 200 * 15))
    estimated_seconds = estimated_minutes * 60
    
    print(f"  ✓ 预估时间: {estimated_minutes}-{estimated_minutes*2} 分钟")
    
    # 步骤 2: 准备训练参数
    print("\n⚙️  步骤 2/5: 准备训练参数")
    print("-" * 70)
    print(f"  • 词汇表大小: {vocab_size}")
    print(f"  • 模型类型: {model_type}")
    print(f"  • 字符覆盖率: {character_coverage}")
    print(f"  • 训练模式: 大语料库模式")
    print(f"  • 线程数: 16")
    
    # 步骤 3: 开始训练
    print("\n🔥 步骤 3/5: 训练 SentencePiece 模型")
    print("-" * 70)
    print("  💡 提示: SentencePiece 会输出详细日志，请关注下方信息\n")
    
    # 训练参数
    train_cmd = [
        'python', '-c',
        f'''
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="{input_file}",
    model_prefix="{model_prefix}",
    vocab_size={vocab_size},
    model_type="{model_type}",
    character_coverage={character_coverage},
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="[PAD]",
    unk_piece="[UNK]",
    bos_piece="[BOS]",
    eos_piece="[EOS]",
    user_defined_symbols="[CLS],[SEP],[MASK]",
    normalization_rule_name="nfkc",
    remove_extra_whitespaces=True,
    max_sentence_length=16384,
    num_threads=16,
    train_extremely_large_corpus=True
)
print("\\n✓ 训练完成！")
'''
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行训练（显示实时输出）
    try:
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时显示输出
        for line in process.stdout:
            print(f"  {line.rstrip()}")
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("训练失败")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        return None
    
    # 计算训练耗时
    elapsed_time = time.time() - start_time
    
    print("\n" + "-" * 70)
    print(f"  ✓ 模型训练完成！")
    print(f"  ⏱  训练耗时: {format_time(elapsed_time)}")
    
    # 步骤 4: 验证模型
    print("\n✅ 步骤 4/5: 验证训练结果")
    print("-" * 70)
    
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(f'{model_prefix}.model')
        
        actual_vocab_size = sp.get_piece_size()
        print(f"  ✓ 模型文件: {model_prefix}.model")
        print(f"  ✓ 词汇表大小: {actual_vocab_size}")
        
        # 测试编码
        test_text = "你好，世界！Hello, World!"
        tokens = sp.encode(test_text, out_type=str)
        print(f"  ✓ 测试编码: '{test_text}'")
        print(f"    → {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
    except Exception as e:
        print(f"  ⚠️  验证失败: {e}")
    
    # 步骤 5: 转换为 Hugging Face 格式
    print("\n🔄 步骤 5/5: 转换为 Hugging Face Tokenizer")
    print("-" * 70)
    
    try:
        from transformers import T5Tokenizer
        
        tokenizer = T5Tokenizer(
            vocab_file=f'{model_prefix}.model',
            eos_token='[EOS]',
            unk_token='[UNK]',
            pad_token='[PAD]',
            bos_token='[BOS]',
            extra_ids=0,
        )
        
        # 添加特殊 token
        special_tokens = {
            'additional_special_tokens': ['[CLS]', '[SEP]', '[MASK]']
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # 保存
        tokenizer.save_pretrained(output_dir)
        
        print(f"  ✓ 已转换为 T5Tokenizer")
        print(f"  ✓ 已保存到: {output_dir}")
        print(f"    - tokenizer_config.json")
        print(f"    - sentencepiece.model")
        print(f"    - special_tokens_map.json")
        
    except Exception as e:
        print(f"  ⚠️  转换失败: {e}")
        print(f"  💡 但 SentencePiece 模型已保存: {model_prefix}.model")
    
    # 显示总结
    print("\n" + "="*70)
    print("🎉 训练完成！")
    print("="*70)
    print(f"📊 训练统计:")
    print(f"  • 数据量: {line_count:,} 行 ({file_size_mb:.2f} MB)")
    print(f"  • 词汇表大小: {actual_vocab_size}")
    print(f"  • 训练耗时: {format_time(elapsed_time)}")
    print(f"  • 输出目录: {output_dir}")
    
    print(f"\n💡 下一步:")
    print(f"  # 快速测试")
    print(f"  python quick_test_tokenizer.py {output_dir}")
    print(f"\n  # 完整评估")
    print(f"  python evaluate_tokenizer.py --tokenizer-dir {output_dir}")
    print("="*70 + "\n")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='训练 SentencePiece Tokenizer（带进度监控）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 基本用法
  python train_tokenizer_with_progress.py \\
      --input ../data/my_corpus_clean.txt \\
      --output ../model_save/my_tokenizer_sp \\
      --vocab-size 40960

  # 使用 BPE 模型
  python train_tokenizer_with_progress.py \\
      --input ../data/my_corpus_clean.txt \\
      --output ../model_save/my_tokenizer_bpe \\
      --vocab-size 40960 \\
      --model-type bpe

  # 使用采样数据（快速训练）
  shuf ../data/my_corpus_clean.txt | head -n 500000 > ../data/sampled.txt
  python train_tokenizer_with_progress.py \\
      --input ../data/sampled.txt \\
      --output ../model_save/my_tokenizer_sampled \\
      --vocab-size 40960
        '''
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入文本文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=40960,
        help='词汇表大小（默认: 40960）'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='unigram',
        choices=['unigram', 'bpe', 'char', 'word'],
        help='模型类型（默认: unigram）'
    )
    
    parser.add_argument(
        '--character-coverage',
        type=float,
        default=0.9995,
        help='字符覆盖率（默认: 0.9995）'
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    try:
        import sentencepiece
        import transformers
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("\n请安装依赖:")
        print("  pip install sentencepiece transformers")
        sys.exit(1)
    
    # 开始训练
    try:
        train_sentencepiece(
            input_file=args.input,
            output_dir=args.output,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
