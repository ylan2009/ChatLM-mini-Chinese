#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估性能对比脚本：对比优化前后的评估速度和GPU利用率
"""

import time
import torch
from transformers import PreTrainedTokenizerFast
from model.chat_model import TextToTextModel
from config import T5ModelConfig
from model.config_utils import get_T5_config

def benchmark_generation(model, tokenizer, batch_size=20, num_samples=100, search_type='beam'):
    """
    测试生成速度
    
    Args:
        model: 模型
        tokenizer: tokenizer
        batch_size: batch大小
        num_samples: 测试样本数
        search_type: 生成方法 ('beam' or 'greedy')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 创建测试数据
    test_text = "你好，请介绍一下人工智能的发展历史。"
    input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # 重复batch_size次
    input_ids = input_ids.repeat(batch_size, 1)
    attention_mask = attention_mask.repeat(batch_size, 1)
    
    # 预热
    with torch.no_grad():
        _ = model.my_generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_seq_len=128,
            search_type=search_type,
        )
    
    # 正式测试
    num_batches = num_samples // batch_size
    
    print(f"\n{'='*60}")
    print(f"测试配置:")
    print(f"  - 生成方法: {search_type}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - 测试样本数: {num_samples}")
    print(f"  - Batch数量: {num_batches}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_batches):
            outputs = model.my_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_seq_len=128,
                search_type=search_type,
            )
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                samples_processed = (i + 1) * batch_size
                speed = samples_processed / elapsed
                print(f"进度: {i+1}/{num_batches} batches, "
                      f"已处理: {samples_processed} 样本, "
                      f"速度: {speed:.2f} 样本/秒")
    
    total_time = time.time() - start_time
    total_samples = num_batches * batch_size
    avg_speed = total_samples / total_time
    
    print(f"\n{'='*60}")
    print(f"测试结果:")
    print(f"  - 总耗时: {total_time:.2f} 秒")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 平均速度: {avg_speed:.2f} 样本/秒")
    print(f"  - 每个样本耗时: {total_time/total_samples*1000:.2f} 毫秒")
    print(f"{'='*60}\n")
    
    return {
        'search_type': search_type,
        'batch_size': batch_size,
        'total_time': total_time,
        'total_samples': total_samples,
        'avg_speed': avg_speed,
    }

def main():
    print("\n" + "="*60)
    print("评估性能对比测试")
    print("="*60)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到GPU，无法进行测试")
        return
    
    print(f"\n✅ 检测到 {torch.cuda.device_count()} 个GPU")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 加载tokenizer
    print("\n加载tokenizer...")
    from config import TrainConfigSFT
    config = TrainConfigSFT()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
    
    # 创建模型
    print("创建模型...")
    t5_config = get_T5_config(
        T5ModelConfig(), 
        vocab_size=len(tokenizer),
        decoder_start_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = TextToTextModel(t5_config)
    
    # 测试不同配置
    results = []
    
    # 1. 原始配置：batch_size=20, beam search
    print("\n" + "="*60)
    print("测试1: 原始配置（Beam Search, batch_size=20）")
    print("="*60)
    result1 = benchmark_generation(
        model=model,
        tokenizer=tokenizer,
        batch_size=20,
        num_samples=100,
        search_type='beam'
    )
    results.append(result1)
    
    # 2. 优化配置1：batch_size=60, beam search
    print("\n" + "="*60)
    print("测试2: 增大batch size（Beam Search, batch_size=60）")
    print("="*60)
    result2 = benchmark_generation(
        model=model,
        tokenizer=tokenizer,
        batch_size=60,
        num_samples=120,
        search_type='beam'
    )
    results.append(result2)
    
    # 3. 优化配置2：batch_size=20, greedy search
    print("\n" + "="*60)
    print("测试3: 使用Greedy Search（Greedy Search, batch_size=20）")
    print("="*60)
    result3 = benchmark_generation(
        model=model,
        tokenizer=tokenizer,
        batch_size=20,
        num_samples=100,
        search_type='greedy'
    )
    results.append(result3)
    
    # 4. 最优配置：batch_size=60, greedy search
    print("\n" + "="*60)
    print("测试4: 最优配置（Greedy Search, batch_size=60）")
    print("="*60)
    result4 = benchmark_generation(
        model=model,
        tokenizer=tokenizer,
        batch_size=60,
        num_samples=120,
        search_type='greedy'
    )
    results.append(result4)
    
    # 对比结果
    print("\n" + "="*60)
    print("性能对比总结")
    print("="*60)
    
    baseline_speed = results[0]['avg_speed']
    
    print(f"\n{'配置':<30} {'速度(样本/秒)':<15} {'相对提升':<15}")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        speedup = result['avg_speed'] / baseline_speed
        print(f"测试{i}: {result['search_type']:>6}, bs={result['batch_size']:<3} "
              f"{result['avg_speed']:>10.2f}      {speedup:>8.2f}x")
    
    print("\n" + "="*60)
    print("结论:")
    print("="*60)
    
    best_result = max(results, key=lambda x: x['avg_speed'])
    speedup = best_result['avg_speed'] / baseline_speed
    
    print(f"\n最优配置: {best_result['search_type']} search, batch_size={best_result['batch_size']}")
    print(f"性能提升: {speedup:.2f}x")
    print(f"速度: {best_result['avg_speed']:.2f} 样本/秒")
    
    print("\n建议:")
    if speedup >= 5:
        print("✅ 优化效果显著！建议使用最优配置进行评估。")
    elif speedup >= 3:
        print("✅ 优化效果良好！建议使用最优配置进行评估。")
    else:
        print("⚠️  优化效果一般，可能需要进一步调整。")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
