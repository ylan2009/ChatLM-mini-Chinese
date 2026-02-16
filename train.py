#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本 - 支持预训练和SFT微调

使用方法：
    # 预训练（使用默认配置）
    accelerate launch --multi_gpu --num_processes 2 ./train.py train
    
    # 大数据集预训练（使用TrainConfigPretrainLarge配置 - 1000万数据，3×20G显存GPU）
    accelerate launch --multi_gpu --num_processes 3 ./train.py train --use_large_config=True
    
    # SFT微调（使用TrainConfigSFT配置）
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True
    
    # 预训练（自定义学习率和训练轮数）
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --epochs=10 --learn_rate=0.0002
    
    # SFT微调（自定义学习率和训练轮数，推荐使用较小学习率）
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True --epochs=5 --learn_rate=1e-5
    
    # 从断点继续训练
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_keep_training=True

参数说明：
    train: 执行训练函数
    --is_keep_training: 是否从断点处加载状态继续训练（默认: False）
    --is_finetune: 是否微调，微调会冻结encoder和embedding（默认: False）
    --use_large_config: 是否使用TrainConfigPretrainLarge配置（大数据集预训练）（默认: False）
    --epochs: 训练轮数，如果指定则覆盖TrainConfig中的默认值
    --learn_rate: 学习率，如果指定则覆盖TrainConfig中的默认值
                  注意：SFT微调时建议使用更小的学习率，如 1e-5
"""

import fire

from config import TrainConfig, TrainConfigSFT, TrainConfigPretrainLarge, T5ModelConfig
from model.trainer import ChatTrainer


class TrainWrapper:
    """训练包装类，用于支持配置选择"""
    
    def __init__(self):
        self.model_config = T5ModelConfig()
    
    def train(self, is_keep_training: bool = False, is_finetune: bool = False, use_large_config: bool = False, **kwargs):
        """
        训练函数
        
        参数：
            is_keep_training: 是否从断点继续训练
            is_finetune: 是否进行SFT微调
            use_large_config: 是否使用大数据集预训练配置（TrainConfigPretrainLarge）
            **kwargs: 其他参数（如epochs, learn_rate等）
        """
        # 根据参数选择配置
        if use_large_config:
            print("=" * 80)
            print("使用 TrainConfigPretrainLarge 配置（大数据集预训练）")
            print("=" * 80)
            train_config = TrainConfigPretrainLarge()
        elif is_finetune:
            print("=" * 80)
            print("使用 TrainConfigSFT 配置（SFT微调）")
            print("=" * 80)
            train_config = TrainConfigSFT()
        else:
            print("=" * 80)
            print("使用 TrainConfig 配置（预训练）")
            print("=" * 80)
            train_config = TrainConfig()
        
        # 如果有自定义参数，覆盖配置
        if kwargs:
            print("\n自定义参数:")
            for key, value in kwargs.items():
                if hasattr(train_config, key):
                    old_value = getattr(train_config, key)
                    setattr(train_config, key, value)
                    print(f"  {key}: {old_value} -> {value}")
                else:
                    print(f"  警告: 配置中不存在参数 '{key}'，已忽略")
            print()
        
        # 创建训练器并开始训练
        chat_trainer = ChatTrainer(train_config=train_config, model_config=self.model_config)
        chat_trainer.train(is_keep_training=is_keep_training, is_finetune=is_finetune)
    
    def test(self, best_epoch: int = 0, use_large_config: bool = False):
        """
        测试函数
        
        参数：
            best_epoch: 要测试的epoch编号
            use_large_config: 是否使用大数据集预训练配置
        """
        if use_large_config:
            train_config = TrainConfigPretrainLarge()
        else:
            train_config = TrainConfig()
        
        chat_trainer = ChatTrainer(train_config=train_config, model_config=self.model_config)
        chat_trainer.test(best_epoch=best_epoch)


if __name__ == '__main__':
    wrapper = TrainWrapper()
    fire.Fire(component=wrapper)