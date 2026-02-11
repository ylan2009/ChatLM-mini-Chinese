#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本 - 支持预训练和SFT微调

使用方法：
    # 预训练（使用默认配置）
    accelerate launch --multi_gpu --num_processes 2 ./train.py train
    
    # SFT微调（使用默认配置）
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True
    
    # 预训练（自定义学习率和训练轮数）
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --epochs=10 --learn_rate=0.0002
    
    # SFT微调（自定义学习率和训练轮数，推荐使用较小学习率）
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True --epochs=5 --learn_rate=1e-5
    
    # 从断点继续训练
    accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_keep_training=True

    accelerate launch --multi_gpu --num_processes 3 ./train.py train --is_keep_training=True

参数说明：
    train: 执行训练函数
    --is_keep_training: 是否从断点处加载状态继续训练（默认: False）
    --is_finetune: 是否微调，微调会冻结encoder和embedding（默认: False）
    --epochs: 训练轮数，如果指定则覆盖TrainConfig中的默认值（默认: 8）
    --learn_rate: 学习率，如果指定则覆盖TrainConfig中的默认值（默认: 0.0001）
                  注意：SFT微调时建议使用更小的学习率，如 1e-5
"""

import fire

from config import TrainConfig, TrainConfigSFT, T5ModelConfig
from model.trainer import ChatTrainer


if __name__ == '__main__':
    # 默认使用 TrainConfig 进行预训练
    # 如果需要使用 TrainConfigSFT 进行 SFT 微调，可以修改这里
    train_config = TrainConfig()
    model_config = T5ModelConfig()

    chat_trainer = ChatTrainer(train_config=train_config, model_config=model_config)

    # 解析命令行参数，执行指定函数
    # fire.Fire 会自动将命令行参数映射到函数参数
    # 例如：--epochs=5 会传递给 train(epochs=5)
    # 注意：SFT微调时建议设置 is_finetune=True
    fire.Fire(component=chat_trainer)