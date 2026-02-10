#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低内存训练脚本 - 针对16G内存环境优化

使用方法：
    # 预训练（使用默认配置）
    accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train
    
    # 大数据集预训练（使用TrainConfigPretrainLarge配置 - 1000万数据，3×20G显存GPU）
    accelerate launch --multi_gpu --num_processes 3 ./train_low_mem.py train --use_large_config=True
    
    # SFT微调（使用TrainConfigSFT配置）
    accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True
    
    # SFT微调（使用TrainConfigSFTSmall配置 - 小数据集）
    accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True --use_small_config=True

    # SFT微调（使用TrainConfigSFTFast配置 - 高性能）
    accelerate launch --multi_gpu --num_processes 3 ./train_low_mem.py train --is_finetune=True --use_fast_config=True
    
    # 预训练（自定义学习率和训练轮数）
    accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --epochs=10 --learn_rate=0.0002
    
    # SFT微调（自定义学习率和训练轮数，推荐使用较小学习率）
    accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True --epochs=5 --learn_rate=1e-5
    
    # 从断点继续训练
    accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_keep_training=True

参数说明：
    train: 执行训练函数
    --is_keep_training: 是否从断点处加载状态继续训练（默认: False）
    --is_finetune: 是否微调，微调会冻结encoder和embedding（默认: False）
    --use_small_config: 是否使用TrainConfigSFTSmall配置（小数据集配置）（默认: False）
    --use_large_config: 是否使用TrainConfigPretrainLarge配置（大数据集预训练，1000万数据）（默认: False）
    --epochs: 训练轮数，如果指定则覆盖TrainConfig中的默认值（默认: 8）
    --learn_rate: 学习率，如果指定则覆盖TrainConfig中的默认值（默认: 0.0001）
                  注意：SFT微调时建议使用更小的学习率，如 1e-5

优化说明：
    本脚本针对16G内存环境进行了以下优化：
    1. 强制关闭数据集内存缓存（keep_in_memory=False）
    2. 使用更小的batch_size（最大2）
    3. 增加梯度累积步数（16）来补偿小batch size
    4. 禁用DataLoader的num_workers，避免多进程内存开销
    5. 定期清理GPU和CPU缓存
    6. 添加内存使用监控

配置说明：
    - TrainConfig: 预训练配置
    - TrainConfigPretrainLarge: 大数据集预训练配置（1000万数据，3×20G显存GPU）
    - TrainConfigSFT: SFT微调配置（标准数据集）
    - TrainConfigSFTSmall: SFT微调配置（小数据集，适合16G内存）
    - TrainConfigSFTFast: SFT微调配置（高性能，充分利用GPU显存）
"""

# ============================================================================
# 【重要】NCCL 环境变量配置 - 解决共享内存问题
# ============================================================================
import os

# 方案1: 使用 socket 通信替代共享内存（推荐）
# 这会稍微降低通信速度，但能避免共享内存问题
os.environ.setdefault('NCCL_SHM_DISABLE', '0')  # 0=启用共享内存, 1=禁用

# 方案2: 增加 NCCL 超时时间（避免初始化超时）
os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30分钟超时

# 方案3: 启用 NCCL 调试信息（如果需要诊断问题）
# os.environ.setdefault('NCCL_DEBUG', 'INFO')  # 取消注释以启用调试

# 方案4: 设置 NCCL 使用的网络接口（如果有多个网卡）
# os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth0')  # 根据实际网卡名称修改

# 方案5: 使用 Gloo 后端替代 NCCL（兼容性更好，但速度稍慢）
# 如果 NCCL 问题无法解决，可以尝试这个
# os.environ.setdefault('ACCELERATE_USE_GLOO', '1')

print("=" * 80)
print("NCCL 环境变量配置:")
print(f"  NCCL_SHM_DISABLE: {os.environ.get('NCCL_SHM_DISABLE', 'not set')}")
print(f"  NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT', 'not set')}")
print("=" * 80)
# ============================================================================

import fire

from config import TrainConfig, TrainConfigSFT, TrainConfigSFTSmall, TrainConfigSFTFast, TrainConfigPretrainLarge, T5ModelConfig
from model.trainer_low_mem import ChatTrainerLowMem


class TrainWrapper:
    """训练包装类，用于支持配置选择"""
    
    def __init__(self):
        self.model_config = T5ModelConfig()
    
    def train(self, is_keep_training: bool = False, is_finetune: bool = False, use_small_config: bool = False, use_fast_config: bool = False, use_large_config: bool = False, **kwargs):
        """
        训练函数
        
        参数：
            is_keep_training: 是否从断点继续训练
            is_finetune: 是否进行SFT微调
            use_small_config: 是否使用小数据集配置（TrainConfigSFTSmall - 低内存）
            use_fast_config: 是否使用高性能配置（TrainConfigSFTFast - 充分利用GPU显存）
            use_large_config: 是否使用大数据集预训练配置（TrainConfigPretrainLarge - 1000万数据）
            **kwargs: 其他参数（如epochs, learn_rate等）
        """
        # 根据参数选择配置
        if use_large_config:
            # 使用大数据集预训练配置
            print("=" * 80)
            print("使用 TrainConfigPretrainLarge 配置（大数据集预训练 - 1000万数据）")
            print("=" * 80)
            train_config = TrainConfigPretrainLarge()
        elif use_fast_config:
            # 使用高性能配置
            print("=" * 80)
            print("使用 TrainConfigSFTFast 配置（高性能 - 充分利用GPU显存）")
            print("=" * 80)
            train_config = TrainConfigSFTFast()
        elif use_small_config:
            # 使用小数据集配置
            print("=" * 80)
            print("使用 TrainConfigSFTSmall 配置（小数据集 - 适合16G内存）")
            print("=" * 80)
            train_config = TrainConfigSFTSmall()
        elif is_finetune:
            # 使用标准SFT配置
            print("=" * 80)
            print("使用 TrainConfigSFT 配置（标准SFT微调）")
            print("=" * 80)
            train_config = TrainConfigSFT()
        else:
            # 使用预训练配置
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
        chat_trainer = ChatTrainerLowMem(train_config=train_config, model_config=self.model_config)
        chat_trainer.train(is_keep_training=is_keep_training, is_finetune=is_finetune)
    
    def test(self, best_epoch: int = 0, use_small_config: bool = False):
        """
        测试函数
        
        参数：
            best_epoch: 要测试的epoch编号
            use_small_config: 是否使用小数据集配置
        """
        # 根据参数选择配置
        if use_small_config:
            train_config = TrainConfigSFTSmall()
        else:
            train_config = TrainConfigSFT()
        
        chat_trainer = ChatTrainerLowMem(train_config=train_config, model_config=self.model_config)
        chat_trainer.test(best_epoch=best_epoch)


if __name__ == '__main__':
    # 使用包装类支持配置选择
    wrapper = TrainWrapper()
    
    # 解析命令行参数，执行指定函数
    # fire.Fire 会自动将命令行参数映射到函数参数
    # 例如：
    #   train --is_finetune=True --use_small_config=True
    #   train --is_finetune=True --epochs=5 --learn_rate=1e-5
    fire.Fire(component=wrapper)
