#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 HuggingFace 下载 BELLE SFT 数据集并处理成统一的 parquet 格式

数据集来源：
- BelleGroup/generated_chat_0.4M
- BelleGroup/train_0.5M_CN
- BelleGroup/train_2M_CN

处理后的数据集格式：parquet 文件，包含两列：prompt 和 response

使用方法：
    python download_belle_sft_dataset.py

输出文件：
    data/sft_dataset.parquet

数据清洗规则：
    1. 剔除翻译任务（包含"翻译"、"translate"等关键词）
    2. 删除表格类任务（包含"表格"或"-----"）
    3. 过滤长度超过 512 的数据
    4. 过滤 response 长度小于 10 的数据
    5. 去除重复的标点符号
    6. 去重（基于 prompt 和 response）
"""

import os
import sys
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import re

# 添加项目路径
sys.path.extend(['.', '..'])
from config import PROJECT_ROOT
from utils.logger import Logger

# 初始化日志
log = Logger('download_belle_sft', std_out=True, save2file=True, file_name=None)

# 配置
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sft_dataset.parquet')

# 要下载的数据集列表
DATASETS = [
    "BelleGroup/generated_chat_0.4M",
    "BelleGroup/train_0.5M_CN",
    "BelleGroup/train_2M_CN"
]

# 数据清洗参数
MAX_LEN = 512  # 最大长度限制
MIN_RESPONSE_LEN = 10  # response 最小长度


def remove_duplicate_punctuation(sentence: str) -> str:
    """
    删除句子中重复的标点符号、重复的空格
    参考：utils/raw_data_process.py
    """
    punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！""''@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
    
    # 将空格（全角空格）替换为逗号
    sentence = re.sub(' |　', '，', sentence)
    
    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]
        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1
    
    return ans


def clean_data(prompt: str, response: str) -> tuple:
    """
    清洗单条数据
    
    Returns:
        (prompt, response) 或 (None, None) 如果数据无效
    """
    # 检查空值
    if not prompt or not response:
        return None, None
    
    # 转换为字符串
    prompt = str(prompt).strip()
    response = str(response).strip()
    
    # 剔除翻译任务
    prompt_lower = prompt.lower()
    if 'translate' in prompt_lower:
        return None, None
    
    for word in ('翻译', '英译', '译英', '中译', '译中', '汉译', '译汉'):
        if word in prompt:
            return None, None
    
    # 删除表格类任务
    if '表格' in prompt or '-----' in prompt or '-----' in response:
        return None, None
    
    # 长度检查
    if len(prompt) > MAX_LEN or len(response) > MAX_LEN:
        return None, None
    
    if len(response) < MIN_RESPONSE_LEN:
        return None, None
    
    # 清理重复标点
    prompt = remove_duplicate_punctuation(prompt)
    response = remove_duplicate_punctuation(response)
    
    # 再次检查长度（清理后可能变短）
    if len(response) < MIN_RESPONSE_LEN:
        return None, None
    
    return prompt, response


def process_dataset(dataset_name: str) -> pd.DataFrame:
    """
    下载并处理单个数据集
    
    Args:
        dataset_name: HuggingFace 数据集名称
        
    Returns:
        处理后的 DataFrame，包含 prompt 和 response 两列
    """
    log.info(f"正在下载数据集: {dataset_name}")
    
    try:
        # 从 HuggingFace 下载数据集
        dataset = load_dataset(dataset_name, split="train")
        log.info(f"  数据集原始大小: {len(dataset)} 行")
        
        # 转换为 pandas DataFrame
        df = dataset.to_pandas()
        
        # 处理列名映射
        # BELLE 数据集通常使用 instruction/input/output 格式
        if "instruction" in df.columns:
            # 某些数据集可能有 input 列，需要合并到 prompt 中
            if "input" in df.columns:
                # 合并 instruction 和 input
                def combine_prompt(row):
                    instruction = str(row["instruction"]) if pd.notna(row["instruction"]) else ""
                    input_text = str(row["input"]) if pd.notna(row.get("input")) else ""
                    input_text = input_text.strip()
                    if input_text:
                        return instruction + "\n" + input_text
                    return instruction
                
                df["prompt"] = df.apply(combine_prompt, axis=1)
            else:
                df["prompt"] = df["instruction"].astype(str)
            
            if "output" in df.columns:
                df["response"] = df["output"].astype(str)
            else:
                log.warning(f"  数据集 {dataset_name} 没有找到 output 列，尝试其他列名...")
                # 尝试其他可能的列名
                if "response" in df.columns:
                    df["response"] = df["response"].astype(str)
                else:
                    log.error(f"  无法找到 response 列，可用列: {df.columns.tolist()}")
                    return pd.DataFrame(columns=["prompt", "response"])
        elif "prompt" in df.columns and "response" in df.columns:
            # 如果已经是 prompt/response 格式
            df["prompt"] = df["prompt"].astype(str)
            df["response"] = df["response"].astype(str)
        else:
            log.error(f"  无法识别数据集格式，可用列: {df.columns.tolist()}")
            return pd.DataFrame(columns=["prompt", "response"])
        
        # 只保留需要的列
        if "prompt" in df.columns and "response" in df.columns:
            df = df[["prompt", "response"]]
        else:
            log.error(f"  数据集 {dataset_name} 格式不正确")
            return pd.DataFrame(columns=["prompt", "response"])
        
        # 数据清洗
        log.info(f"  开始清洗数据...")
        cleaned_data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理 {dataset_name}"):
            prompt, response = clean_data(row["prompt"], row["response"])
            if prompt is not None and response is not None:
                cleaned_data.append({"prompt": prompt, "response": response})
        
        cleaned_df = pd.DataFrame(cleaned_data)
        log.info(f"  清洗后剩余: {len(cleaned_df)} 行 (保留率: {len(cleaned_df)/len(df)*100:.2f}%)")
        
        return cleaned_df
        
    except Exception as e:
        log.error(f"  处理数据集 {dataset_name} 时出错: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return pd.DataFrame(columns=["prompt", "response"])


def main():
    """
    主函数：下载、处理和合并所有数据集
    """
    log.info("=" * 60)
    log.info("开始下载和处理 BELLE SFT 数据集")
    log.info("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理所有数据集
    all_dataframes = []
    
    for dataset_name in DATASETS:
        df = process_dataset(dataset_name)
        if len(df) > 0:
            all_dataframes.append(df)
        log.info("")
    
    if not all_dataframes:
        log.error("没有成功处理任何数据集，退出")
        return
    
    # 合并所有数据集
    log.info("正在合并所有数据集...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    log.info(f"合并前总行数: {len(combined_df)}")
    
    # 去重
    log.info("正在去除重复数据...")
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["prompt", "response"])
    after_dedup = len(combined_df)
    log.info(f"去重后总行数: {after_dedup} (去除了 {before_dedup - after_dedup} 条重复数据)")
    
    # 移除空值（虽然应该已经处理过了，但再检查一次）
    combined_df = combined_df.dropna(subset=["prompt", "response"])
    log.info(f"最终数据集大小: {len(combined_df)} 行")
    
    # 保存为 parquet 文件
    log.info(f"正在保存数据集到: {OUTPUT_FILE}")
    combined_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    log.info(f"数据集已成功保存！")
    
    # 显示数据样例
    log.info("\n数据样例:")
    log.info("-" * 60)
    for idx, row in combined_df.head(3).iterrows():
        log.info(f"\n样例 {idx + 1}:")
        log.info(f"  Prompt: {row['prompt'][:100]}...")
        log.info(f"  Response: {row['response'][:100]}...")
    
    log.info("\n" + "=" * 60)
    log.info("处理完成！")
    log.info(f"数据集文件: {OUTPUT_FILE}")
    log.info(f"数据集大小: {len(combined_df)} 行")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
