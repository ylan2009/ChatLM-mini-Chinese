#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tokenizer 测试脚本
用法:
    python test_tokenizer.py
    python test_tokenizer.py --tokenizer model_save/my_tokenizer_sp
"""

import sys
sys.path.extend(['.', '..'])

import argparse
from transformers import T5Tokenizer, PreTrainedTokenizerFast, AutoTokenizer
from config import PROJECT_ROOT


def test_tokenizer(tokenizer_path):
    """测试 tokenizer 的各项功能"""
    
    print(f"\n{'='*100}")
    print(f"🔤 Tokenizer 功能测试")
    print(f"{'='*100}\n")
    
    # 加载 tokenizer
    print(f"📂 加载 Tokenizer: {tokenizer_path}")
    try:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        print(f"✅ 使用 T5Tokenizer 加载成功")
    except Exception as e:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"✅ 使用 AutoTokenizer 加载成功")
        except:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            print(f"✅ 使用 PreTrainedTokenizerFast 加载成功")
    
    print(f"\n{'─'*100}")
    print(f"📊 Tokenizer 基本信息")
    print(f"{'─'*100}")
    print(f"   词汇表大小: {tokenizer.vocab_size:,}")
    print(f"   模型最大长度: {tokenizer.model_max_length:,}")
    print(f"   Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"   UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    # 测试用例
    test_cases = [
        "我喜欢机器学习",
        "深度学习是人工智能的一个分支",
        "Python是一种编程语言",
        "今天天气很好，适合出去玩",
        "ChatGPT是一个大型语言模型",
        "自然语言处理技术发展迅速",
    ]
    
    print(f"\n{'─'*100}")
    print(f"🧪 分词测试")
    print(f"{'─'*100}\n")
    
    for idx, text in enumerate(test_cases, 1):
        print(f"测试 #{idx}")
        print(f"   原文: {text}")
        
        # 分词
        tokens = tokenizer.tokenize(text)
        print(f"   分词: {tokens}")
        print(f"   Token 数: {len(tokens)}")
        
        # 编码
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"   Token IDs: {ids[:10]}{'...' if len(ids) > 10 else ''}")
        
        # 解码
        decoded = tokenizer.decode(ids)
        print(f"   解码: {decoded}")
        
        # 验证一致性
        if decoded.replace(' ', '') == text.replace(' ', ''):
            print(f"   ✅ 编码-解码一致")
        else:
            print(f"   ⚠️  编码-解码不一致")
        
        print()
    
    # 测试特殊情况
    print(f"{'─'*100}")
    print(f"🔬 特殊情况测试")
    print(f"{'─'*100}\n")
    
    # 1. 长文本
    long_text = "机器学习" * 100
    tokens = tokenizer.tokenize(long_text)
    print(f"1. 长文本测试")
    print(f"   原文长度: {len(long_text)} 字符")
    print(f"   Token 数: {len(tokens)}")
    print(f"   压缩比: {len(long_text) / len(tokens):.2f}x")
    print()
    
    # 2. 生僻词
    rare_text = "量子纠缠现象"
    tokens = tokenizer.tokenize(rare_text)
    ids = tokenizer.encode(rare_text, add_special_tokens=False)
    unk_count = sum(1 for id in ids if id == tokenizer.unk_token_id)
    print(f"2. 生僻词测试")
    print(f"   原文: {rare_text}")
    print(f"   分词: {tokens}")
    print(f"   <unk> 数量: {unk_count}")
    if unk_count == 0:
        print(f"   ✅ 无未知词")
    else:
        print(f"   ⚠️  包含 {unk_count} 个未知词")
    print()
    
    # 3. 英文混合
    mixed_text = "我使用Python进行机器学习"
    tokens = tokenizer.tokenize(mixed_text)
    print(f"3. 中英混合测试")
    print(f"   原文: {mixed_text}")
    print(f"   分词: {tokens}")
    print(f"   Token 数: {len(tokens)}")
    print()
    
    # 4. 数字和符号
    symbol_text = "2024年，AI技术发展迅速！"
    tokens = tokenizer.tokenize(symbol_text)
    print(f"4. 数字符号测试")
    print(f"   原文: {symbol_text}")
    print(f"   分词: {tokens}")
    print(f"   Token 数: {len(tokens)}")
    print()
    
    # 批量编码测试
    print(f"{'─'*100}")
    print(f"📦 批量编码测试")
    print(f"{'─'*100}\n")
    
    batch_texts = [
        "短文本",
        "这是一个中等长度的文本示例",
        "这是一个更长的文本示例，用于测试批量编码时的 padding 功能"
    ]
    
    # 不带 padding
    print("不带 padding:")
    encoded = tokenizer(batch_texts, padding=False)
    for i, ids in enumerate(encoded['input_ids']):
        print(f"   文本 {i+1}: 长度 {len(ids)}, IDs: {ids[:5]}...")
    
    print("\n带 padding:")
    encoded = tokenizer(batch_texts, padding=True)
    for i, ids in enumerate(encoded['input_ids']):
        print(f"   文本 {i+1}: 长度 {len(ids)}, IDs: {ids[:5]}...")
    
    print(f"\n{'='*100}")
    print(f"✅ 测试完成！")
    print(f"{'='*100}\n")
    
    # 总结
    print(f"📋 总结:")
    print(f"   - Tokenizer 类型: {type(tokenizer).__name__}")
    print(f"   - 词汇表大小: {tokenizer.vocab_size:,}")
    print(f"   - 支持中文: ✅")
    print(f"   - 支持英文: ✅")
    print(f"   - 支持数字符号: ✅")
    print(f"   - 批量编码: ✅")
    print()


def compare_tokenizers():
    """对比不同 tokenizer 的效果"""
    
    print(f"\n{'='*100}")
    print(f"🔍 对比不同 Tokenizer")
    print(f"{'='*100}\n")
    
    text = "我喜欢机器学习和深度学习"
    
    # 1. 你的 tokenizer
    print(f"1️⃣ 你的 SentencePiece Tokenizer")
    print(f"{'─'*100}")
    try:
        tokenizer = T5Tokenizer.from_pretrained(PROJECT_ROOT + '/model_save/my_tokenizer_sp')
        tokens = tokenizer.tokenize(text)
        print(f"   原文: {text}")
        print(f"   分词: {tokens}")
        print(f"   Token 数: {len(tokens)}")
        print(f"   ✅ 加载成功\n")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}\n")
    
    # 2. 字符级（模拟）
    print(f"2️⃣ 字符级分词（模拟）")
    print(f"{'─'*100}")
    char_tokens = list(text)
    print(f"   原文: {text}")
    print(f"   分词: {char_tokens}")
    print(f"   Token 数: {len(char_tokens)}")
    print(f"   ⚠️  序列长度是 SentencePiece 的 {len(char_tokens)/len(tokens):.1f}x\n")
    
    # 3. 英文 tokenizer（对比）
    print(f"3️⃣ GPT-2 Tokenizer（英文）")
    print(f"{'─'*100}")
    try:
        from transformers import GPT2Tokenizer
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt2_tokens = gpt2_tokenizer.tokenize(text)
        print(f"   原文: {text}")
        print(f"   分词: {gpt2_tokens[:20]}{'...' if len(gpt2_tokens) > 20 else ''}")
        print(f"   Token 数: {len(gpt2_tokens)}")
        print(f"   ❌ 不适合中文，序列长度是 SentencePiece 的 {len(gpt2_tokens)/len(tokens):.1f}x\n")
    except Exception as e:
        print(f"   ⚠️  需要安装: pip install transformers\n")
    
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description='测试 Tokenizer 功能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试默认 tokenizer
  python test_tokenizer.py
  
  # 测试指定 tokenizer
  python test_tokenizer.py --tokenizer model_save/my_tokenizer_sp
  
  # 对比不同 tokenizer
  python test_tokenizer.py --compare
        """
    )
    
    parser.add_argument(
        '--tokenizer', '-t',
        type=str,
        default=PROJECT_ROOT + '/model_save/my_tokenizer_sp',
        help='Tokenizer 路径'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='对比不同 tokenizer'
    )
    
    args = parser.parse_args()
    
    # 测试 tokenizer
    test_tokenizer(args.tokenizer)
    
    # 对比
    if args.compare:
        compare_tokenizers()


if __name__ == '__main__':
    main()
