#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LLaMA-Factory训练T5模型

安装依赖:
pip install transformers datasets torch_optimizer
# 可选: pip install accelerate deepspeed

多GPU训练方式（选择其一）:

方式1: 使用torchrun (推荐，无需accelerate)
    torchrun --nproc_per_node=2 train_with_llamafactory.py

方式2: 使用torch.distributed.launch (旧版本)
    python -m torch.distributed.launch --nproc_per_node=2 train_with_llamafactory.py

方式3: 使用accelerate (更灵活)
    accelerate launch --multi_gpu --num_processes=2 train_with_llamafactory.py

方式4: 使用DeepSpeed (大模型推荐)
    deepspeed --num_gpus=2 train_with_llamafactory.py --deepspeed ds_config.json

方式5: 自动检测多GPU (最简单)
    CUDA_VISIBLE_DEVICES=0,1 python train_with_llamafactory.py

单GPU训练:
    python train_with_llamafactory.py
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from torch_optimizer import Adafactor


@dataclass
class ModelArguments:
    """模型参数"""
    model_name_or_path: str = field(
        default="./model_save/ChatLM-mini-Chinese/",
        metadata={"help": "T5模型路径"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer路径，默认使用model_name_or_path"}
    )


@dataclass
class DataArguments:
    """数据参数"""
    train_file: str = field(
        default="./data/my_train_dataset.parquet",
        metadata={"help": "训练数据文件（Parquet格式）"}
    )
    validation_file: str = field(
        default="./data/my_valid_dataset.parquet",
        metadata={"help": "验证数据文件（Parquet格式）"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "数据预处理进程数"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """自定义训练参数"""
    # 基础训练参数
    output_dir: str = field(default="./model_save/llama_factory_output")
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=8)
    
    # 学习率和优化器
    learning_rate: float = field(default=0.0001)
    optim: str = field(default="adafactor")  # 使用Adafactor
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1024)
    
    # 混合精度和优化
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    max_grad_norm: float = field(default=1.0)
    
    # 日志和保存
    logging_steps: int = field(default=50)
    save_steps: int = field(default=5000)
    save_total_limit: int = field(default=8)
    
    # 评估
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=5000)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    
    # 分布式训练
    ddp_timeout: int = field(default=1800)
    dataloader_num_workers: int = field(default=0)  # 低内存模式禁用
    dataloader_pin_memory: bool = field(default=False)  # 低内存模式禁用


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    数据预处理函数
    
    假设Parquet文件包含以下列：
    - input: 输入文本
    - target: 目标文本
    """
    # Tokenize输入
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_seq_length,
        truncation=True,
        padding=False,  # 使用DataCollator动态padding
    )
    
    # Tokenize目标
    labels = tokenizer(
        examples["target"],
        max_length=max_seq_length,
        truncation=True,
        padding=False,
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds):
    """
    计算评估指标（可以添加BLEU等指标）
    """
    import numpy as np
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    predictions, labels = eval_preds
    
    # 解码预测和标签
    # 这里需要tokenizer，可以通过闭包传入
    # 简化版本：只计算准确率
    predictions = np.argmax(predictions, axis=-1)
    
    # 移除padding
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]
    
    accuracy = (predictions == labels).mean()
    
    return {
        "accuracy": accuracy,
    }


def main():
    """主函数"""
    import torch
    from transformers import HfArgumentParser
    
    # 检测分布式训练环境
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 只在主进程打印信息
    is_main_process = local_rank in [-1, 0]
    
    if is_main_process:
        print("=" * 80)
        print("训练环境信息:")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        print(f"  当前进程: {local_rank if local_rank != -1 else 'main'}")
        print(f"  总进程数: {world_size}")
        
        # 检测使用的分布式后端
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print(f"  分布式后端: {torch.distributed.get_backend()}")
            print(f"  分布式方式: DDP (DistributedDataParallel)")
        elif torch.cuda.device_count() > 1:
            print(f"  分布式方式: DataParallel (自动检测)")
        else:
            print(f"  分布式方式: 单GPU/CPU")
        print("=" * 80)
    
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 加载tokenizer
    tokenizer_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    
    # 加载模型
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
    )
    
    # 加载数据集
    data_files = {
        "train": data_args.train_file,
        "validation": data_args.validation_file,
    }
    
    # 使用datasets库加载Parquet文件
    raw_datasets = load_dataset(
        "parquet",
        data_files=data_files,
    )
    
    # 预处理数据
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_function(
            examples, 
            tokenizer, 
            data_args.max_seq_length
        ),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing datasets",
    )
    
    # 数据整理器（动态padding）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.bf16 else None,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("=" * 80)
    print("开始训练...")
    print(f"训练样本数: {len(tokenized_datasets['train'])}")
    print(f"验证样本数: {len(tokenized_datasets['validation'])}")
    print("=" * 80)
    
    train_result = trainer.train()
    
    # 保存模型
    trainer.save_model()
    trainer.save_state()
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 评估
    print("=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("训练完成！")


if __name__ == "__main__":
    main()