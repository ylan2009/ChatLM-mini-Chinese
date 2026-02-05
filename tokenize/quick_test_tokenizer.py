#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 Tokenizer

快速检查 tokenizer 的基本功能和质量

使用方法：
    python quick_test_tokenizer.py ../model_save/my_tokenizer_wiki
"""

import sys
import os


def quick_test(tokenizer_dir: str):
    """快速测试 tokenizer"""
    
    # 导入
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("❌ 需要安装 transformers: pip install transformers")
        return False
    
    # 加载
    print(f"\n{'='*60}")
    print(f"🔍 快速测试: {os.path.basename(tokenizer_dir)}")
    print(f"{'='*60}\n")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print("✅ Tokenizer 加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    # 基本信息
    print(f"\n📚 基本信息:")
    print(f"  词汇表大小: {len(tokenizer):,}")
    print(f"  PAD token: {tokenizer.pad_token}")
    print(f"  UNK token: {tokenizer.unk_token}")
    print(f"  EOS token: {getattr(tokenizer, 'eos_token', 'N/A')}")
    
    # 测试样本
    test_cases = [
        "人工智能是计算机科学的一个分支",
        "Machine learning is a subset of AI",
        "使用 Python 进行 AI 开发",
        "2024年，AI市场规模达到5000亿美元",
    ]
    
    print(f"\n🧪 测试样本:")
    
    total_chars = 0
    total_tokens = 0
    total_unk = 0
    
    for i, text in enumerate(test_cases, 1):
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids)
        
        num_chars = len(text)
        num_tokens = len(tokens)
        num_unk = sum(1 for t in tokens if '[UNK]' in str(t) or t == tokenizer.unk_token)
        compression = num_chars / num_tokens if num_tokens > 0 else 0
        
        total_chars += num_chars
        total_tokens += num_tokens
        total_unk += num_unk
        
        print(f"\n  [{i}] {text}")
        print(f"      Tokens ({num_tokens}): {' | '.join(tokens[:8])}{'...' if len(tokens) > 8 else ''}")
        print(f"      压缩率: {compression:.2f}, UNK: {num_unk}")
        
        # 检查可逆性
        if decoded.replace(' ', '') != text.replace(' ', ''):
            print(f"      ⚠️  解码不一致: {decoded}")
    
    # 总体评估
    avg_compression = total_chars / total_tokens if total_tokens > 0 else 0
    unk_ratio = total_unk / total_tokens if total_tokens > 0 else 0
    
    print(f"\n📊 总体评估:")
    print(f"  平均压缩率: {avg_compression:.2f} 字符/token")
    print(f"  未知词比例: {unk_ratio*100:.2f}%")
    
    # 评分
    if 2.0 <= avg_compression <= 3.0:
        comp_status = "✅ 优秀"
    elif 1.5 <= avg_compression < 2.0 or 3.0 < avg_compression <= 3.5:
        comp_status = "✅ 良好"
    else:
        comp_status = "⚠️  需要改进"
    
    if unk_ratio < 0.01:
        unk_status = "✅ 优秀"
    elif unk_ratio < 0.05:
        unk_status = "✅ 良好"
    else:
        unk_status = "⚠️  需要改进"
    
    print(f"  压缩率评价: {comp_status}")
    print(f"  未知词评价: {unk_status}")
    
    # 综合评分
    if "优秀" in comp_status and "优秀" in unk_status:
        print(f"\n🎉 综合评价: 优秀！可以使用")
    elif "良好" in comp_status or "良好" in unk_status:
        print(f"\n✅ 综合评价: 良好，可以使用")
    else:
        print(f"\n⚠️  综合评价: 建议优化或重新训练")
    
    print(f"\n💡 提示: 运行完整评估获取详细报告:")
    print(f"   python evaluate_tokenizer.py --tokenizer-dir {tokenizer_dir} --verbose")
    print()
    
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python quick_test_tokenizer.py <tokenizer_dir>")
        print("示例: python quick_test_tokenizer.py ../model_save/my_tokenizer_wiki")
        sys.exit(1)
    
    tokenizer_dir = sys.argv[1]
    
    if not os.path.exists(tokenizer_dir):
        print(f"❌ 目录不存在: {tokenizer_dir}")
        sys.exit(1)
    
    success = quick_test(tokenizer_dir)
    sys.exit(0 if success else 1)
