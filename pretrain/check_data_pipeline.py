#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据处理管道诊断脚本
用于检查数据处理流程中每一步的数据质量，找出 prompt 为空的问题出现在哪一步
"""

import sys
sys.path.extend(['.', '..'])

from fastparquet import ParquetFile
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict
from config import PROJECT_ROOT


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.stats = {
            'total_rows': 0,
            'empty_prompt': 0,
            'empty_response': 0,
            'both_empty': 0,
            'valid': 0,
            'none_prompt': 0,
            'none_response': 0,
            'whitespace_only_prompt': 0,
            'whitespace_only_response': 0,
        }
        self.samples = {
            'empty_prompt': [],
            'empty_response': [],
            'valid': [],
        }
        self.prompt_lengths = []
        self.response_lengths = []
    
    def check_parquet(self, max_samples=5, show_progress=True):
        """检查 parquet 文件"""
        print(f"\n{'='*100}")
        print(f"📁 检查文件: {self.file_path}")
        print(f"{'='*100}\n")
        
        try:
            pf = ParquetFile(self.file_path)
            
            if show_progress:
                print("🔍 开始扫描数据...")
            
            # 逐块读取数据
            for chunk in pf:
                for rows in chunk.iter_row_groups():
                    # 尝试不同的列名
                    prompts = None
                    responses = None
                    
                    # 检查可能的列名
                    columns = rows.columns.tolist()
                    
                    # 查找 prompt 列
                    for col in ['prompt', 'instruction', 'input', 'question']:
                        if col in columns:
                            prompts = rows[col].tolist()
                            break
                    
                    # 查找 response 列
                    for col in ['response', 'output', 'answer', 'target']:
                        if col in columns:
                            responses = rows[col].tolist()
                            break
                    
                    if prompts is None or responses is None:
                        print(f"⚠️  警告: 无法找到 prompt/response 列")
                        print(f"   可用列: {columns}")
                        return None
                    
                    # 分析每一行
                    for prompt, response in zip(prompts, responses):
                        self.stats['total_rows'] += 1
                        
                        # 检查 None 值
                        if prompt is None:
                            self.stats['none_prompt'] += 1
                            prompt = ""
                        if response is None:
                            self.stats['none_response'] += 1
                            response = ""
                        
                        # 转换为字符串
                        prompt_str = str(prompt)
                        response_str = str(response)
                        
                        # 检查空字符串
                        prompt_len = len(prompt_str)
                        response_len = len(response_str)
                        
                        prompt_stripped_len = len(prompt_str.strip())
                        response_stripped_len = len(response_str.strip())
                        
                        self.prompt_lengths.append(prompt_stripped_len)
                        self.response_lengths.append(response_stripped_len)
                        
                        # 统计各种情况
                        if prompt_stripped_len == 0:
                            self.stats['empty_prompt'] += 1
                            if prompt_len > 0:
                                self.stats['whitespace_only_prompt'] += 1
                            
                            if len(self.samples['empty_prompt']) < max_samples:
                                self.samples['empty_prompt'].append({
                                    'row': self.stats['total_rows'],
                                    'prompt': repr(prompt_str[:100]),
                                    'response': response_str[:100],
                                    'prompt_is_none': prompt is None,
                                })
                        
                        if response_stripped_len == 0:
                            self.stats['empty_response'] += 1
                            if response_len > 0:
                                self.stats['whitespace_only_response'] += 1
                            
                            if len(self.samples['empty_response']) < max_samples:
                                self.samples['empty_response'].append({
                                    'row': self.stats['total_rows'],
                                    'prompt': prompt_str[:100],
                                    'response': repr(response_str[:100]),
                                    'response_is_none': response is None,
                                })
                        
                        if prompt_stripped_len == 0 and response_stripped_len == 0:
                            self.stats['both_empty'] += 1
                        
                        if prompt_stripped_len > 0 and response_stripped_len > 0:
                            self.stats['valid'] += 1
                            if len(self.samples['valid']) < max_samples:
                                self.samples['valid'].append({
                                    'row': self.stats['total_rows'],
                                    'prompt': prompt_str[:100],
                                    'response': response_str[:100],
                                })
                        
                        # 显示进度
                        if show_progress and self.stats['total_rows'] % 100000 == 0:
                            print(f"   已处理 {self.stats['total_rows']:,} 行...")
            
            if show_progress:
                print(f"✅ 扫描完成！共处理 {self.stats['total_rows']:,} 行\n")
            
            return self.stats
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_report(self):
        """打印检查报告"""
        total = self.stats['total_rows']
        if total == 0:
            print("⚠️  没有数据")
            return
        
        print(f"{'='*100}")
        print(f"📊 数据质量报告: {self.file_name}")
        print(f"{'='*100}\n")
        
        # 基本统计
        print("📈 基本统计:")
        print(f"  总行数: {total:,}")
        print(f"  有效数据 (prompt 和 response 都不为空): {self.stats['valid']:,} ({self.stats['valid']/total*100:.2f}%)")
        print()
        
        # 空值统计
        print("🔍 空值统计:")
        print(f"  空 prompt: {self.stats['empty_prompt']:,} ({self.stats['empty_prompt']/total*100:.2f}%)")
        print(f"    - None 值: {self.stats['none_prompt']:,}")
        print(f"    - 仅空白字符: {self.stats['whitespace_only_prompt']:,}")
        print(f"  空 response: {self.stats['empty_response']:,} ({self.stats['empty_response']/total*100:.2f}%)")
        print(f"    - None 值: {self.stats['none_response']:,}")
        print(f"    - 仅空白字符: {self.stats['whitespace_only_response']:,}")
        print(f"  两者都为空: {self.stats['both_empty']:,} ({self.stats['both_empty']/total*100:.2f}%)")
        print()
        
        # 长度统计
        if self.prompt_lengths:
            print("📏 Prompt 长度统计:")
            print(f"  平均长度: {sum(self.prompt_lengths)/len(self.prompt_lengths):.2f}")
            print(f"  最小长度: {min(self.prompt_lengths)}")
            print(f"  最大长度: {max(self.prompt_lengths)}")
            print()
        
        if self.response_lengths:
            print("📏 Response 长度统计:")
            print(f"  平均长度: {sum(self.response_lengths)/len(self.response_lengths):.2f}")
            print(f"  最小长度: {min(self.response_lengths)}")
            print(f"  最大长度: {max(self.response_lengths)}")
            print()
        
        # 数据质量评级
        valid_rate = self.stats['valid'] / total * 100
        print("⭐ 数据质量评级:")
        if valid_rate >= 95:
            print(f"  ✅ 优秀 ({valid_rate:.2f}% 有效数据)")
        elif valid_rate >= 80:
            print(f"  ⚠️  良好 ({valid_rate:.2f}% 有效数据)")
        elif valid_rate >= 50:
            print(f"  ⚠️  一般 ({valid_rate:.2f}% 有效数据)")
        else:
            print(f"  ❌ 较差 ({valid_rate:.2f}% 有效数据)")
        print()
        
        # 显示样例
        if self.samples['empty_prompt']:
            print(f"{'='*100}")
            print(f"🔍 空 Prompt 样例 (前 {len(self.samples['empty_prompt'])} 条):")
            print(f"{'='*100}")
            for sample in self.samples['empty_prompt']:
                print(f"\n第 {sample['row']} 行:")
                print(f"  Prompt (is_none={sample.get('prompt_is_none', False)}): {sample['prompt']}")
                print(f"  Response: {sample['response']}")
        
        if self.samples['valid']:
            print(f"\n{'='*100}")
            print(f"✅ 有效数据样例 (前 {len(self.samples['valid'])} 条):")
            print(f"{'='*100}")
            for sample in self.samples['valid']:
                print(f"\n第 {sample['row']} 行:")
                print(f"  Prompt: {sample['prompt']}")
                print(f"  Response: {sample['response']}")
        
        print(f"\n{'='*100}\n")


def check_single_file(file_path: str, max_samples=5):
    """检查单个文件"""
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    checker = DataQualityChecker(file_path)
    checker.check_parquet(max_samples=max_samples)
    checker.print_report()
    
    return checker.stats


def check_pipeline(data_dir: str = None):
    """检查整个数据处理管道"""
    if data_dir is None:
        data_dir = PROJECT_ROOT + '/data'
    
    print(f"\n{'#'*100}")
    print(f"# 数据处理管道诊断")
    print(f"# 数据目录: {data_dir}")
    print(f"{'#'*100}\n")
    
    # 定义数据处理流程中的关键文件
    pipeline_files = [
        {
            'name': '原始 Belle 数据',
            'path': f'{data_dir}/raw_data/belle/Belle_open_source_0.5M.parquet',
            'description': '从 Hugging Face 下载的原始数据'
        },
        {
            'name': '去重后的数据',
            'path': f'{data_dir}/my_dataset_no_dulpticates.parquet',
            'description': 'remove_dataset_duplicate_rows 处理后的数据'
        },
        {
            'name': '处理后的微调数据',
            'path': f'{data_dir}/my_finetune_data_zh.parquet',
            'description': 'process_belle_knowledge_enhanced_dataset_for_finetune 处理后的数据'
        },
        {
            'name': 'Shuffle 后的数据',
            'path': f'{data_dir}/my_finetune_data_zh_shuffled.parquet',
            'description': 'shuffle_parquet_dataset 处理后的数据'
        },
    ]
    
    # 检查每个文件
    results = {}
    for file_info in pipeline_files:
        file_path = file_info['path']
        
        print(f"\n{'='*100}")
        print(f"🔍 步骤: {file_info['name']}")
        print(f"📝 说明: {file_info['description']}")
        print(f"{'='*100}")
        
        if not Path(file_path).exists():
            print(f"⚠️  文件不存在，跳过: {file_path}\n")
            results[file_info['name']] = None
            continue
        
        checker = DataQualityChecker(file_path)
        stats = checker.check_parquet(max_samples=3, show_progress=True)
        
        if stats:
            checker.print_report()
            results[file_info['name']] = stats
        else:
            results[file_info['name']] = None
    
    # 生成对比报告
    print(f"\n{'#'*100}")
    print(f"# 管道对比报告")
    print(f"{'#'*100}\n")
    
    print(f"{'步骤':<30} {'总行数':>15} {'有效数据':>15} {'空Prompt':>15} {'空Response':>15}")
    print(f"{'-'*100}")
    
    for file_info in pipeline_files:
        name = file_info['name']
        stats = results.get(name)
        
        if stats is None:
            print(f"{name:<30} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
        else:
            total = stats['total_rows']
            valid = stats['valid']
            empty_prompt = stats['empty_prompt']
            empty_response = stats['empty_response']
            
            valid_pct = f"{valid:,} ({valid/total*100:.1f}%)" if total > 0 else "0"
            empty_p_pct = f"{empty_prompt:,} ({empty_prompt/total*100:.1f}%)" if total > 0 else "0"
            empty_r_pct = f"{empty_response:,} ({empty_response/total*100:.1f}%)" if total > 0 else "0"
            
            print(f"{name:<30} {total:>15,} {valid_pct:>15} {empty_p_pct:>15} {empty_r_pct:>15}")
    
    print(f"\n{'#'*100}\n")
    
    # 分析问题
    print("🔍 问题分析:\n")
    
    # 检查是否有文件的空 prompt 比例突然增加
    prev_empty_rate = 0
    problem_found = False
    
    for file_info in pipeline_files:
        name = file_info['name']
        stats = results.get(name)
        
        if stats and stats['total_rows'] > 0:
            empty_rate = stats['empty_prompt'] / stats['total_rows'] * 100
            
            if empty_rate > 50:
                print(f"❌ 严重问题: '{name}' 中有 {empty_rate:.1f}% 的数据 prompt 为空！")
                print(f"   文件路径: {file_info['path']}")
                print(f"   这一步可能存在问题！\n")
                problem_found = True
            elif empty_rate > prev_empty_rate + 10:
                print(f"⚠️  警告: '{name}' 中空 prompt 比例增加了 {empty_rate - prev_empty_rate:.1f}%")
                print(f"   从 {prev_empty_rate:.1f}% 增加到 {empty_rate:.1f}%")
                print(f"   文件路径: {file_info['path']}")
                print(f"   这一步可能引入了问题！\n")
                problem_found = True
            
            prev_empty_rate = empty_rate
    
    if not problem_found:
        print("✅ 未发现明显的数据质量问题")
    
    print(f"\n{'#'*100}\n")


def main():
    parser = argparse.ArgumentParser(description='数据处理管道诊断工具')
    parser.add_argument('--file', type=str, help='检查单个文件')
    parser.add_argument('--pipeline', action='store_true', help='检查整个数据处理管道')
    parser.add_argument('--data-dir', type=str, help='数据目录路径')
    parser.add_argument('--samples', type=int, default=5, help='显示的样例数量')
    
    args = parser.parse_args()
    
    if args.file:
        # 检查单个文件
        check_single_file(args.file, max_samples=args.samples)
    elif args.pipeline:
        # 检查整个管道
        check_pipeline(data_dir=args.data_dir)
    else:
        # 默认检查整个管道
        print("未指定参数，默认检查整个数据处理管道")
        print("使用 --file <文件路径> 检查单个文件")
        print("使用 --pipeline 检查整个管道\n")
        check_pipeline(data_dir=args.data_dir)


if __name__ == '__main__':
    main()
