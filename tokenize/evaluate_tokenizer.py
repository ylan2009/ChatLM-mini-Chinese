#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenizer 评估工具

功能：
1. 词汇表覆盖率分析
2. 分词效果测试
3. 压缩率评估
4. 特殊字符处理
5. 中英文混合文本处理
6. 数字和标点符号处理
7. 未知词（UNK）比例
8. 平均 token 长度

使用方法：
    # 评估单个 tokenizer
    python evaluate_tokenizer.py --tokenizer-dir ../model_save/my_tokenizer_wiki
    
    # 对比多个 tokenizer
    python evaluate_tokenizer.py \
        --tokenizer-dir ../model_save/my_tokenizer_wiki \
        --compare-with ../model_save/my_tokenizer_sp \
        --compare-with ../model_save/my_tokenizer_char
    
    # 使用自定义测试文件
    python evaluate_tokenizer.py \
        --tokenizer-dir ../model_save/my_tokenizer_wiki \
        --test-file ../data/test_corpus.txt
    
    # 详细模式（显示每个样本的分词结果）
    python evaluate_tokenizer.py \
        --tokenizer-dir ../model_save/my_tokenizer_wiki \
        --verbose
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple
from collections import Counter
import json


def check_transformers():
    """检查并导入 transformers"""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except ImportError:
        return None


def load_tokenizer(tokenizer_dir: str):
    """加载 tokenizer"""
    AutoTokenizer = check_transformers()
    if AutoTokenizer is None:
        raise ImportError("需要 transformers 库: pip install transformers")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        return tokenizer
    except Exception as e:
        print(f"❌ 加载 tokenizer 失败: {e}")
        return None


def get_test_samples() -> List[Tuple[str, str]]:
    """获取测试样本（文本，类别）"""
    return [
        # 纯中文
        ("人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。", "纯中文"),
        ("机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。", "纯中文"),
        ("深度学习是机器学习的一个分支，使用多层神经网络来学习数据的表示。", "纯中文"),
        
        # 纯英文
        ("Artificial intelligence is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence.", "纯英文"),
        ("Machine learning is a subset of artificial intelligence that enables computers to learn and improve without being explicitly programmed.", "纯英文"),
        
        # 中英混合
        ("Python 是一种广泛使用的高级编程语言，特别适合 AI 和 machine learning 开发。", "中英混合"),
        ("Transformer 模型在 NLP 领域取得了巨大成功，BERT 和 GPT 都是基于 Transformer 架构。", "中英混合"),
        ("使用 PyTorch 或 TensorFlow 可以快速构建深度学习模型。", "中英混合"),
        
        # 包含数字
        ("2024年，全球AI市场规模预计将达到5000亿美元，年增长率超过30%。", "包含数字"),
        ("GPT-3有1750亿个参数，GPT-4的参数量更大。", "包含数字"),
        
        # 包含标点和特殊字符
        ("什么是AI？它能做什么？AI的未来在哪里？", "包含标点"),
        ("【重要】请注意：模型训练需要大量数据！！！", "包含标点"),
        ("Email: test@example.com, Website: https://www.example.com", "特殊字符"),
        
        # 长文本
        ("自然语言处理（Natural Language Processing，NLP）是人工智能和语言学领域的分支学科。此领域探讨如何处理及运用自然语言；自然语言处理包括多方面和步骤，基本有认知、理解、生成等部分。", "长文本"),
        
        # 短文本
        ("你好", "短文本"),
        ("AI", "短文本"),
        ("123", "短文本"),
        
        # 专业术语
        ("Transformer架构使用self-attention机制，通过multi-head attention和position encoding实现序列建模。", "专业术语"),
        ("反向传播算法（Backpropagation）是训练神经网络的核心算法，通过梯度下降优化损失函数。", "专业术语"),
    ]


def evaluate_tokenizer(
    tokenizer,
    test_samples: List[Tuple[str, str]] = None,
    verbose: bool = False
) -> Dict:
    """
    评估 tokenizer
    
    Args:
        tokenizer: 要评估的 tokenizer
        test_samples: 测试样本列表
        verbose: 是否显示详细信息
    
    Returns:
        评估结果字典
    """
    if test_samples is None:
        test_samples = get_test_samples()
    
    results = {
        'vocab_size': len(tokenizer),
        'special_tokens': {},
        'samples': [],
        'statistics': {}
    }
    
    # 1. 检查特殊 token
    special_tokens = {
        'pad_token': tokenizer.pad_token,
        'unk_token': tokenizer.unk_token,
        'bos_token': getattr(tokenizer, 'bos_token', None),
        'eos_token': getattr(tokenizer, 'eos_token', None),
        'cls_token': getattr(tokenizer, 'cls_token', None),
        'sep_token': getattr(tokenizer, 'sep_token', None),
        'mask_token': getattr(tokenizer, 'mask_token', None),
    }
    results['special_tokens'] = special_tokens
    
    # 2. 评估每个样本
    total_chars = 0
    total_tokens = 0
    total_unk = 0
    category_stats = {}
    
    for text, category in test_samples:
        # 编码
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # 统计
        num_chars = len(text)
        num_tokens = len(tokens)
        num_unk = sum(1 for t in tokens if t == tokenizer.unk_token or '[UNK]' in str(t))
        compression_ratio = num_chars / num_tokens if num_tokens > 0 else 0
        unk_ratio = num_unk / num_tokens if num_tokens > 0 else 0
        
        # 解码测试
        decoded = tokenizer.decode(token_ids)
        is_reversible = (decoded.replace(' ', '') == text.replace(' ', ''))
        
        sample_result = {
            'text': text,
            'category': category,
            'num_chars': num_chars,
            'num_tokens': num_tokens,
            'num_unk': num_unk,
            'compression_ratio': compression_ratio,
            'unk_ratio': unk_ratio,
            'is_reversible': is_reversible,
            'tokens': tokens if verbose else None,
            'decoded': decoded if verbose else None,
        }
        
        results['samples'].append(sample_result)
        
        # 累计统计
        total_chars += num_chars
        total_tokens += num_tokens
        total_unk += num_unk
        
        # 分类统计
        if category not in category_stats:
            category_stats[category] = {
                'count': 0,
                'total_chars': 0,
                'total_tokens': 0,
                'total_unk': 0,
            }
        category_stats[category]['count'] += 1
        category_stats[category]['total_chars'] += num_chars
        category_stats[category]['total_tokens'] += num_tokens
        category_stats[category]['total_unk'] += num_unk
    
    # 3. 计算总体统计
    avg_compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    avg_unk_ratio = total_unk / total_tokens if total_tokens > 0 else 0
    
    results['statistics'] = {
        'total_samples': len(test_samples),
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'total_unk': total_unk,
        'avg_compression_ratio': avg_compression_ratio,
        'avg_unk_ratio': avg_unk_ratio,
        'category_stats': category_stats,
    }
    
    return results


def print_evaluation_report(results: Dict, tokenizer_name: str = "Tokenizer"):
    """打印评估报告"""
    print("\n" + "="*80)
    print(f"📊 {tokenizer_name} 评估报告")
    print("="*80)
    
    # 1. 基本信息
    print(f"\n📚 基本信息:")
    print(f"  词汇表大小: {results['vocab_size']:,}")
    
    # 2. 特殊 token
    print(f"\n🔖 特殊 Token:")
    for name, token in results['special_tokens'].items():
        if token:
            print(f"  {name}: {token}")
    
    # 3. 总体统计
    stats = results['statistics']
    print(f"\n📈 总体统计:")
    print(f"  测试样本数: {stats['total_samples']}")
    print(f"  总字符数: {stats['total_chars']:,}")
    print(f"  总 token 数: {stats['total_tokens']:,}")
    print(f"  平均压缩率: {stats['avg_compression_ratio']:.2f} 字符/token")
    print(f"  未知词比例: {stats['avg_unk_ratio']*100:.2f}%")
    
    # 评分
    score = calculate_score(stats['avg_compression_ratio'], stats['avg_unk_ratio'])
    print(f"\n⭐ 综合评分: {score:.1f}/100")
    print_score_interpretation(score)
    
    # 4. 分类统计
    print(f"\n📊 分类统计:")
    print(f"{'类别':<12} {'样本数':<8} {'平均压缩率':<12} {'未知词比例':<12}")
    print("-" * 50)
    
    for category, cat_stats in stats['category_stats'].items():
        avg_comp = cat_stats['total_chars'] / cat_stats['total_tokens'] if cat_stats['total_tokens'] > 0 else 0
        avg_unk = cat_stats['total_unk'] / cat_stats['total_tokens'] if cat_stats['total_tokens'] > 0 else 0
        print(f"{category:<12} {cat_stats['count']:<8} {avg_comp:<12.2f} {avg_unk*100:<12.2f}%")
    
    # 5. 样本详情（如果有）
    if results['samples'] and results['samples'][0].get('tokens'):
        print(f"\n📝 样本详情:")
        for i, sample in enumerate(results['samples'][:5], 1):  # 只显示前5个
            print(f"\n  样本 {i} ({sample['category']}):")
            print(f"    原文: {sample['text'][:60]}{'...' if len(sample['text']) > 60 else ''}")
            print(f"    Token 数: {sample['num_tokens']}, 压缩率: {sample['compression_ratio']:.2f}")
            print(f"    未知词: {sample['num_unk']}, 可逆: {'✓' if sample['is_reversible'] else '✗'}")
            if sample['tokens']:
                tokens_str = ' | '.join(sample['tokens'][:10])
                if len(sample['tokens']) > 10:
                    tokens_str += ' | ...'
                print(f"    Tokens: {tokens_str}")


def calculate_score(compression_ratio: float, unk_ratio: float) -> float:
    """
    计算综合评分
    
    评分标准：
    - 压缩率：2.0-3.0 最佳（中文），1.5-2.5 最佳（英文）
    - 未知词比例：越低越好，< 1% 优秀，< 5% 良好
    """
    # 压缩率评分（满分 60）
    if 2.0 <= compression_ratio <= 3.0:
        compression_score = 60
    elif 1.5 <= compression_ratio < 2.0 or 3.0 < compression_ratio <= 3.5:
        compression_score = 50
    elif 1.0 <= compression_ratio < 1.5 or 3.5 < compression_ratio <= 4.0:
        compression_score = 40
    else:
        compression_score = 30
    
    # 未知词比例评分（满分 40）
    if unk_ratio < 0.01:  # < 1%
        unk_score = 40
    elif unk_ratio < 0.05:  # < 5%
        unk_score = 30
    elif unk_ratio < 0.10:  # < 10%
        unk_score = 20
    else:
        unk_score = 10
    
    return compression_score + unk_score


def print_score_interpretation(score: float):
    """打印评分解释"""
    if score >= 90:
        print("  🎉 优秀！Tokenizer 训练质量非常好")
    elif score >= 75:
        print("  ✅ 良好！Tokenizer 训练质量不错")
    elif score >= 60:
        print("  ⚠️  一般，Tokenizer 可能需要改进")
    else:
        print("  ❌ 较差，建议重新训练 Tokenizer")


def compare_tokenizers(tokenizer_dirs: List[str], verbose: bool = False):
    """对比多个 tokenizer"""
    print("\n" + "="*80)
    print("🔍 Tokenizer 对比分析")
    print("="*80)
    
    all_results = []
    test_samples = get_test_samples()
    
    for tokenizer_dir in tokenizer_dirs:
        print(f"\n正在评估: {tokenizer_dir}")
        tokenizer = load_tokenizer(tokenizer_dir)
        if tokenizer is None:
            continue
        
        results = evaluate_tokenizer(tokenizer, test_samples, verbose=False)
        results['tokenizer_dir'] = tokenizer_dir
        all_results.append(results)
    
    if len(all_results) < 2:
        print("\n❌ 需要至少 2 个有效的 tokenizer 才能进行对比")
        return
    
    # 打印对比表格
    print("\n" + "="*80)
    print("📊 对比结果")
    print("="*80)
    
    print(f"\n{'Tokenizer':<40} {'词汇表':<10} {'压缩率':<10} {'未知词%':<10} {'评分':<10}")
    print("-" * 80)
    
    for result in all_results:
        name = os.path.basename(result['tokenizer_dir'])
        vocab_size = result['vocab_size']
        comp_ratio = result['statistics']['avg_compression_ratio']
        unk_ratio = result['statistics']['avg_unk_ratio'] * 100
        score = calculate_score(comp_ratio, result['statistics']['avg_unk_ratio'])
        
        print(f"{name:<40} {vocab_size:<10,} {comp_ratio:<10.2f} {unk_ratio:<10.2f} {score:<10.1f}")
    
    # 找出最佳 tokenizer
    best_result = max(all_results, key=lambda r: calculate_score(
        r['statistics']['avg_compression_ratio'],
        r['statistics']['avg_unk_ratio']
    ))
    
    print(f"\n🏆 最佳 Tokenizer: {os.path.basename(best_result['tokenizer_dir'])}")
    
    # 详细报告
    if verbose:
        for result in all_results:
            print_evaluation_report(result, os.path.basename(result['tokenizer_dir']))


def evaluate_on_file(tokenizer, test_file: str, max_samples: int = 100) -> Dict:
    """在文件上评估 tokenizer"""
    print(f"\n正在从文件读取测试样本: {test_file}")
    
    test_samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if line:
                test_samples.append((line, "文件样本"))
    
    print(f"✓ 读取了 {len(test_samples)} 个样本")
    
    return evaluate_tokenizer(tokenizer, test_samples)


def main():
    parser = argparse.ArgumentParser(
        description='评估 Tokenizer 训练质量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--tokenizer-dir',
        type=str,
        required=True,
        help='Tokenizer 目录路径'
    )
    
    parser.add_argument(
        '--compare-with',
        type=str,
        action='append',
        help='要对比的其他 tokenizer 目录（可多次使用）'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        help='自定义测试文件路径'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='从测试文件读取的最大样本数（默认: 100）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息（包括每个样本的分词结果）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='保存评估结果到 JSON 文件'
    )
    
    args = parser.parse_args()
    
    # 对比模式
    if args.compare_with:
        tokenizer_dirs = [args.tokenizer_dir] + args.compare_with
        compare_tokenizers(tokenizer_dirs, args.verbose)
        return
    
    # 单个评估模式
    print(f"\n正在加载 tokenizer: {args.tokenizer_dir}")
    tokenizer = load_tokenizer(args.tokenizer_dir)
    
    if tokenizer is None:
        sys.exit(1)
    
    print("✓ Tokenizer 加载成功")
    
    # 评估
    if args.test_file:
        results = evaluate_on_file(tokenizer, args.test_file, args.max_samples)
    else:
        test_samples = get_test_samples()
        results = evaluate_tokenizer(tokenizer, test_samples, args.verbose)
    
    # 打印报告
    print_evaluation_report(results, os.path.basename(args.tokenizer_dir))
    
    # 保存结果
    if args.output:
        # 移除不能序列化的内容
        output_results = {
            'vocab_size': results['vocab_size'],
            'special_tokens': results['special_tokens'],
            'statistics': results['statistics'],
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 评估结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
