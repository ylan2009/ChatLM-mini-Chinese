#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真正的 LLaMA-Factory 训练T5模型

这个脚本使用 LLaMA-Factory 的官方 API

安装依赖:
pip install llmtuner  # LLaMA-Factory的包名

使用方法:
python train_with_real_llamafactory.py

或使用accelerate:
accelerate launch --multi_gpu --num_processes=2 train_with_real_llamafactory.py
"""

import os
import yaml
from pathlib import Path


def create_llamafactory_config():
    """
    创建 LLaMA-Factory 配置文件
    
    LLaMA-Factory 使用 YAML 配置文件来定义训练参数
    """
    config = {
        # ========== 模型配置 ==========
        "model_name_or_path": "./model_save/ChatLM-mini-Chinese/",
        "finetuning_type": "full",  # full=全量微调, lora=LoRA微调
        
        # ========== 数据配置 ==========
        "dataset": "custom_t5_dataset",  # 数据集名称（需要在dataset_info.json中定义）
        "template": "default",  # 模板类型
        "cutoff_len": 512,  # 最大序列长度
        "preprocessing_num_workers": 4,
        
        # ========== 训练阶段 ==========
        "stage": "pt",  # pt=预训练, sft=监督微调, rm=奖励模型, ppo=强化学习
        "do_train": True,
        "do_eval": True,
        
        # ========== 输出配置 ==========
        "output_dir": "./model_save/llamafactory_output",
        "overwrite_output_dir": True,
        "overwrite_cache": False,
        
        # ========== 训练超参数 ==========
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps": 8,
        "learning_rate": 0.0001,
        "num_train_epochs": 5,
        "max_steps": -1,  # -1表示使用num_train_epochs
        
        # ========== 优化器配置 ==========
        "optim": "adafactor",  # adamw_torch, adafactor
        "lr_scheduler_type": "cosine",
        "warmup_steps": 1024,
        "max_grad_norm": 1.0,
        
        # ========== 混合精度 ==========
        "bf16": True,  # 使用BF16混合精度
        "fp16": False,
        
        # ========== 内存优化 ==========
        "gradient_checkpointing": True,  # 梯度检查点（节省显存）
        
        # ========== 日志配置 ==========
        "logging_steps": 50,
        "save_steps": 5000,
        "save_total_limit": 8,
        
        # ========== 评估配置 ==========
        "evaluation_strategy": "steps",
        "eval_steps": 5000,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        
        # ========== 分布式训练 ==========
        "ddp_timeout": 1800,
        "dataloader_num_workers": 0,  # 低内存模式
        "dataloader_pin_memory": False,
        
        # ========== DeepSpeed配置（可选） ==========
        "deepspeed": None,  # 如果需要可以指定DeepSpeed配置文件路径
        
        # ========== 其他配置 ==========
        "report_to": ["tensorboard"],  # 日志记录工具
        "plot_loss": True,  # 绘制损失曲线
    }
    
    return config


def create_dataset_info():
    """
    创建数据集信息文件
    
    LLaMA-Factory 需要一个 dataset_info.json 文件来定义数据集
    """
    dataset_info = {
        "custom_t5_dataset": {
            "file_name": "data/my_train_dataset.parquet",  # 训练数据
            "file_format": "parquet",  # 数据格式
            "columns": {
                "prompt": "input",  # 输入列名
                "response": "target",  # 输出列名
            },
            "split": "train",  # 数据集分割
        },
        "custom_t5_dataset_eval": {
            "file_name": "data/my_valid_dataset.parquet",  # 验证数据
            "file_format": "parquet",
            "columns": {
                "prompt": "input",
                "response": "target",
            },
            "split": "validation",
        }
    }
    
    return dataset_info


def main():
    """主函数"""
    print("=" * 80)
    print("使用真正的 LLaMA-Factory 训练")
    print("=" * 80)
    
    # 创建配置文件
    config = create_llamafactory_config()
    config_path = Path("llamafactory_config.yaml")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"✓ 已创建配置文件: {config_path}")
    
    # 创建数据集信息文件
    dataset_info = create_dataset_info()
    dataset_info_path = Path("dataset_info.json")
    
    import json
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已创建数据集信息文件: {dataset_info_path}")
    
    # 使用 LLaMA-Factory 的 API
    try:
        from llmtuner import run_exp
        
        print("\n" + "=" * 80)
        print("开始训练...")
        print("=" * 80 + "\n")
        
        # 运行训练
        run_exp(args={"config_file": str(config_path)})
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        
    except ImportError:
        print("\n" + "!" * 80)
        print("错误: 未安装 LLaMA-Factory")
        print("请运行: pip install llmtuner")
        print("!" * 80)
        print("\n或者使用命令行方式:")
        print(f"  llamafactory-cli train {config_path}")
        print("!" * 80)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
