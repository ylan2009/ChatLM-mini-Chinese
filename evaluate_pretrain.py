#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预训练模型评估脚本
用法:
    python evaluate_pretrain.py                           # 使用默认模型和测试集
    python evaluate_pretrain.py --model model_save/checkpoint-1000  # 指定模型
    python evaluate_pretrain.py --generate                # 测试文本生成
"""

import sys
sys.path.extend(['.', '..'])

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from fastparquet import ParquetFile
import numpy as np
from tqdm import tqdm
from config import PROJECT_ROOT


def calculate_perplexity(model, tokenizer, dataset_path, max_samples=100, max_length=512):
    """计算困惑度"""
    print(f"\n{'='*100}")
    print(f"📊 计算困惑度（Perplexity）")
    print(f"{'='*100}\n")
    
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_tokens = 0
    num_samples = 0
    
    # 读取数据集
    pf = ParquetFile(dataset_path)
    
    with torch.no_grad():
        for chunk in pf:
            for rows in chunk.iter_row_groups():
                for i in tqdm(range(len(rows)), desc="计算中"):
                    if num_samples >= max_samples:
                        break
                    
                    # 获取文本
                    prompt = rows['prompt'][i] if 'prompt' in rows.columns else ""
                    response = rows['response'][i] if 'response' in rows.columns else ""
                    text = f"{prompt}\n{response}"
                    
                    # Tokenize
                    inputs = tokenizer(
                        text,
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    input_ids = inputs['input_ids'].to(device)
                    
                    # 计算损失
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    
                    # 累计
                    total_loss += loss.item() * input_ids.size(1)
                    total_tokens += input_ids.size(1)
                    num_samples += 1
                
                if num_samples >= max_samples:
                    break
            
            if num_samples >= max_samples:
                break
    
    # 计算平均损失和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"\n📈 评估结果：")
    print(f"   - 样本数量: {num_samples}")
    print(f"   - 总 Token 数: {total_tokens:,}")
    print(f"   - 平均损失: {avg_loss:.4f}")
    print(f"   - 困惑度 (PPL): {perplexity:.2f}")
    
    # 评估等级
    if perplexity < 10:
        grade = "🌟 优秀"
    elif perplexity < 30:
        grade = "✅ 良好"
    elif perplexity < 100:
        grade = "⚠️  一般"
    else:
        grade = "❌ 较差"
    
    print(f"   - 评估等级: {grade}")
    print(f"\n{'='*100}\n")
    
    return perplexity, avg_loss


def generate_text(model, tokenizer, prompts=None, max_length=100, temperature=0.8, top_p=0.9):
    """测试文本生成"""
    print(f"\n{'='*100}")
    print(f"✍️  文本生成测试")
    print(f"{'='*100}\n")
    
    if prompts is None:
        # 默认测试提示
        prompts = [
            "今天天气很",
            "机器学习是",
            "从前有座山",
            "Python是一种",
            "人工智能的应用包括",
            "深度学习和传统机器学习的区别在于",
        ]
    
    model.eval()
    device = next(model.parameters()).device
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"{'─'*100}")
        print(f"测试 #{idx}")
        print(f"{'─'*100}")
        print(f"📝 提示: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🤖 生成: {generated_text}")
        print()
    
    print(f"{'='*100}\n")


def compare_checkpoints(model_paths, tokenizer_path, test_prompts):
    """对比不同 checkpoint 的效果"""
    print(f"\n{'='*100}")
    print(f"🔍 对比不同 Checkpoint")
    print(f"{'='*100}\n")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    for model_path in model_paths:
        print(f"\n📦 模型: {model_path}")
        print(f"{'─'*100}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # 生成测试
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs['input_ids'].to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_length=50,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"   提示: {prompt}")
                print(f"   生成: {generated}\n")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ 加载失败: {e}\n")
    
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description='评估预训练模型效果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算困惑度
  python evaluate_pretrain.py --perplexity
  
  # 测试文本生成
  python evaluate_pretrain.py --generate
  
  # 完整评估
  python evaluate_pretrain.py --perplexity --generate
  
  # 指定模型
  python evaluate_pretrain.py --model model_save/checkpoint-1000 --generate
  
  # 对比不同 checkpoint
  python evaluate_pretrain.py --compare
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=PROJECT_ROOT + '/model_save',
        help='模型路径 (默认: model_save)'
    )
    
    parser.add_argument(
        '--tokenizer', '-t',
        type=str,
        default=PROJECT_ROOT + '/model_save/my_tokenizer_sp',
        help='Tokenizer 路径 (默认: model_save/my_tokenizer_sp)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=PROJECT_ROOT + '/data/my_test_dataset.parquet',
        help='测试数据集路径 (默认: data/my_test_dataset.parquet)'
    )
    
    parser.add_argument(
        '--perplexity', '-p',
        action='store_true',
        help='计算困惑度'
    )
    
    parser.add_argument(
        '--generate', '-g',
        action='store_true',
        help='测试文本生成'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='对比不同 checkpoint'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='计算困惑度时使用的最大样本数 (默认: 100)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='生成文本的最大长度 (默认: 100)'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认都做
    if not (args.perplexity or args.generate or args.compare):
        args.perplexity = True
        args.generate = True
    
    print(f"\n{'='*100}")
    print(f"🚀 预训练模型评估")
    print(f"{'='*100}")
    print(f"\n📦 模型路径: {args.model}")
    print(f"🔤 Tokenizer: {args.tokenizer}")
    print(f"📊 测试数据: {args.dataset}")
    print(f"🖥️  设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 加载模型和 tokenizer
    try:
        print(f"\n⏳ 加载模型...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
        model = AutoModelForCausalLM.from_pretrained(args.model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"✅ 模型加载成功！")
        
        # 计算困惑度
        if args.perplexity:
            if Path(args.dataset).exists():
                calculate_perplexity(
                    model, 
                    tokenizer, 
                    args.dataset,
                    max_samples=args.max_samples
                )
            else:
                print(f"⚠️  测试数据集不存在: {args.dataset}")
        
        # 测试生成
        if args.generate:
            generate_text(
                model,
                tokenizer,
                max_length=args.max_length
            )
        
        # 对比 checkpoint
        if args.compare:
            # 查找所有 checkpoint
            model_dir = Path(args.model).parent
            checkpoints = sorted(model_dir.glob("checkpoint-*"))
            
            if checkpoints:
                print(f"\n找到 {len(checkpoints)} 个 checkpoint")
                test_prompts = ["今天天气很", "机器学习是"]
                compare_checkpoints(
                    [str(cp) for cp in checkpoints[-3:]],  # 最后3个
                    args.tokenizer,
                    test_prompts
                )
            else:
                print(f"⚠️  未找到 checkpoint")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
