#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试TrainConfigSFTUltra配置

使用方法：
    python test_ultra_config.py
"""

from config import TrainConfigSFTUltra

def test_ultra_config():
    """测试超高性能配置"""
    print("=" * 80)
    print("测试 TrainConfigSFTUltra 配置")
    print("=" * 80)
    
    # 创建配置实例
    config = TrainConfigSFTUltra()
    
    # 打印关键配置参数
    print(f"batch_size_per_gpu: {config.batch_size_per_gpu}")
    print(f"gradient_accumulation_steps: {config.gradient_accumulation_steps}")
    print(f"dataloader_num_workers: {config.dataloader_num_workers}")
    print(f"dataloader_pin_memory: {config.dataloader_pin_memory}")
    print(f"use_torch_compile: {config.use_torch_compile}")
    print(f"compile_mode: {config.compile_mode}")
    print(f"gradient_checkpointing: {config.gradient_checkpointing}")
    print(f"gradient_clip_algo: {config.gradient_clip_algo}")
    
    # 计算有效batch_size
    num_gpus = 3  # 你的机器有3个GPU
    effective_batch_size = config.batch_size_per_gpu * num_gpus * config.gradient_accumulation_steps
    print(f"有效batch_size: {effective_batch_size} (3 GPU × {config.batch_size_per_gpu} × {config.gradient_accumulation_steps})")
    
    print("=" * 80)
    print("✅ 配置测试完成！")
    print("=" * 80)

if __name__ == '__main__':
    test_ultra_config()