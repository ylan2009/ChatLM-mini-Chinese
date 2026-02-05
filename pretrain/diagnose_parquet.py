#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
诊断 parquet 文件的数据质量
"""

import sys
sys.path.extend(['.', '..'])

from fastparquet import ParquetFile
from config import PROJECT_ROOT
from collections import defaultdict

def diagnose_parquet(file_path):
    """诊断 parquet 文件"""
    print(f"\n{'='*80}")
    print(f"诊断文件: {file_path}")
    print(f"{'='*80}\n")
    
    try:
        pf = ParquetFile(file_path)
        
        # 统计信息
        total_rows = 0
        empty_prompt_count = 0
        empty_response_count = 0
        both_empty_count = 0
        valid_count = 0
        
        prompt_lengths = []
        response_lengths = []
        
        # 收集样例
        empty_prompt_samples = []
        valid_samples = []
        
        print("开始扫描数据...")
        
        for chunk in pf:
            for rows in chunk.iter_row_groups():
                prompts = rows['prompt'].tolist()
                responses = rows['response'].tolist()
                
                for prompt, response in zip(prompts, responses):
                    total_rows += 1
                    
                    prompt_str = str(prompt) if prompt is not None else ""
                    response_str = str(response) if response is not None else ""
                    
                    prompt_len = len(prompt_str.strip())
                    response_len = len(response_str.strip())
                    
                    prompt_lengths.append(prompt_len)
                    response_lengths.append(response_len)
                    
                    # 统计空值
                    if prompt_len == 0:
                        empty_prompt_count += 1
                        if len(empty_prompt_samples) < 5:
                            empty_prompt_samples.append({
                                'row': total_rows,
                                'prompt': prompt_str,
                                'response': response_str[:100]
                            })
                    
                    if response_len == 0:
                        empty_response_count += 1
                    
                    if prompt_len == 0 and response_len == 0:
                        both_empty_count += 1
                    
                    if prompt_len > 0 and response_len > 0:
                        valid_count += 1
                        if len(valid_samples) < 5:
                            valid_samples.append({
                                'row': total_rows,
                                'prompt': prompt_str[:100],
                                'response': response_str[:100]
                            })
        
        # 输出统计结果
        print(f"\n{'='*80}")
        print("统计结果:")
        print(f"{'='*80}")
        print(f"总行数: {total_rows}")
        print(f"有效数据 (prompt 和 response 都不为空): {valid_count} ({valid_count/total_rows*100:.2f}%)")
        print(f"空 prompt: {empty_prompt_count} ({empty_prompt_count/total_rows*100:.2f}%)")
        print(f"空 response: {empty_response_count} ({empty_response_count/total_rows*100:.2f}%)")
        print(f"两者都为空: {both_empty_count} ({both_empty_count/total_rows*100:.2f}%)")
        
        if prompt_lengths:
            print(f"\nPrompt 长度统计:")
            print(f"  平均长度: {sum(prompt_lengths)/len(prompt_lengths):.2f}")
            print(f"  最小长度: {min(prompt_lengths)}")
            print(f"  最大长度: {max(prompt_lengths)}")
        
        if response_lengths:
            print(f"\nResponse 长度统计:")
            print(f"  平均长度: {sum(response_lengths)/len(response_lengths):.2f}")
            print(f"  最小长度: {min(response_lengths)}")
            print(f"  最大长度: {max(response_lengths)}")
        
        # 输出样例
        if empty_prompt_samples:
            print(f"\n{'='*80}")
            print("空 Prompt 样例 (前5条):")
            print(f"{'='*80}")
            for sample in empty_prompt_samples:
                print(f"\n第 {sample['row']} 行:")
                print(f"  Prompt: '{sample['prompt']}'")
                print(f"  Response: {sample['response']}")
        
        if valid_samples:
            print(f"\n{'='*80}")
            print("有效数据样例 (前5条):")
            print(f"{'='*80}")
            for sample in valid_samples:
                print(f"\n第 {sample['row']} 行:")
                print(f"  Prompt: {sample['prompt']}")
                print(f"  Response: {sample['response']}")
        
        print(f"\n{'='*80}\n")
        
        return {
            'total_rows': total_rows,
            'valid_count': valid_count,
            'empty_prompt_count': empty_prompt_count,
            'empty_response_count': empty_response_count,
        }
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # 诊断生成的 finetune 数据
    finetune_file = PROJECT_ROOT + '/data/my_finetune_data_zh.parquet'
    
    print("开始诊断...")
    result = diagnose_parquet(finetune_file)
    
    if result:
        print("\n诊断完成！")
        if result['valid_count'] == 0:
            print("\n⚠️  警告: 没有找到有效数据！")
            print("   可能的原因:")
            print("   1. 源数据的 prompt 字段全部为空")
            print("   2. 数据处理过程中出现了问题")
            print("   3. 列名映射错误")
