# 使用小数据集进行SFT训练 - 快速指南

## 🎯 目标

使用 `data/sft_train_small_train.parquet` 和 `data/sft_train_small_valid.parquet` 进行SFT训练。

## ✅ 已完成的修改

### 1. 配置文件修改

在 `config.py` 中，`TrainConfigSFTSmall` 配置类已经指向小数据集：

```python
class TrainConfigSFTSmall:
    # 数据文件路径
    train_file: str = PROJECT_ROOT + '/data/sft_train_small_train.parquet'
    validation_file: str = PROJECT_ROOT + '/data/sft_train_small_valid.parquet'
    
    # 其他优化配置
    epochs: int = 3                    # 小数据集3个epoch
    batch_size_per_gpu: int = 1        # 极致低内存
    gradient_accumulation_steps: int = 8  # 梯度累积
```

### 2. 训练脚本修改

`train_low_mem.py` 已支持通过命令行参数选择配置：

- 添加了 `--use_small_config` 参数
- 添加了配置选择逻辑
- 支持自定义参数覆盖

## 🚀 使用方法

### 方法1：使用小数据集配置（推荐）

```bash
# 使用TrainConfigSFTSmall配置进行SFT训练
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

**说明**：
- `--is_finetune=True`：启用SFT微调模式（冻结encoder和embedding）
- `--use_small_config=True`：使用 `TrainConfigSFTSmall` 配置
- 自动使用 `data/sft_train_small_train.parquet` 和 `data/sft_train_small_valid.parquet`

### 方法2：自定义参数

```bash
# 使用小数据集配置，并自定义训练轮数和学习率
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True \
    --epochs=5 \
    --learn_rate=3e-5
```

### 方法3：从断点继续训练

```bash
# 从上次中断的地方继续训练
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True \
    --is_keep_training=True
```

## 📊 配置对比

| 配置项 | TrainConfigSFT | TrainConfigSFTSmall |
|-------|----------------|---------------------|
| 训练数据 | sft_train_dataset_10k.parquet | sft_train_small_train.parquet |
| 验证数据 | sft_valid_dataset_1k.parquet | sft_train_small_valid.parquet |
| Batch size | 20 | 1 |
| 梯度累积 | 6 | 8 |
| Epochs | 20 | 3 |
| 内存占用 | ~12-16GB | ~8-10GB |
| 适用场景 | 标准训练 | 16G内存环境 |

## 🔍 验证配置

在训练开始前，脚本会打印使用的配置：

```
================================================================================
使用 TrainConfigSFTSmall 配置（小数据集 - 适合16G内存）
================================================================================

自定义参数:
  (如果有自定义参数会在这里显示)

================================================================================
低内存模式训练 - 针对16G内存优化
================================================================================
cpu memory available: 13.15 GB, disk space available: 44.79 GB
使用LowMemDataset: 支持多GPU + 低内存模式，按需从磁盘读取数据
...
```

## 📝 完整训练流程

### 步骤1：确认数据文件存在

```bash
ls -lh data/sft_train_small_*.parquet
```

应该看到：
- `data/sft_train_small_train.parquet`
- `data/sft_train_small_valid.parquet`

### 步骤2：确认预训练模型存在

```bash
ls -lh model_save/chat_small_t5.best.bin
```

如果不存在，需要先完成预训练。

### 步骤3：开始训练

```bash
# 使用小数据集配置
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

### 步骤4：监控训练（另一个终端）

```bash
# 监控内存和GPU
watch -n 2 'free -h && echo "---GPU---" && nvidia-smi'

# 查看训练日志
tail -f logs/*.log
```

## 🎓 训练参数说明

### 必需参数

- `train`：执行训练函数
- `--is_finetune=True`：启用SFT微调模式

### 可选参数

- `--use_small_config=True`：使用小数据集配置（推荐）
- `--is_keep_training=True`：从断点继续训练
- `--epochs=N`：自定义训练轮数
- `--learn_rate=X`：自定义学习率
- `--batch_size_per_gpu=N`：自定义batch size

### 参数优先级

命令行参数 > 配置类默认值

例如：
```bash
# 使用TrainConfigSFTSmall，但将epochs改为5
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True \
    --epochs=5
```

## 📈 预期效果

使用小数据集（假设5000样本）训练3个epoch：

| 指标 | 预期值 |
|-----|-------|
| 训练时长 | 6-12小时 |
| 内存占用 | 8-10GB（双GPU） |
| 训练Loss | 0.5-1.0 |
| 验证BLEU | 0.3-0.5 |
| 模型大小 | ~700MB |

## ⚠️ 常见问题

### Q1: 如何确认使用了正确的配置？

**A**: 训练开始时会打印配置信息：
```
使用 TrainConfigSFTSmall 配置（小数据集 - 适合16G内存）
```

### Q2: 如何修改数据文件路径？

**A**: 有两种方法：

方法1：修改 `config.py` 中的 `TrainConfigSFTSmall`：
```python
train_file: str = PROJECT_ROOT + '/data/你的训练文件.parquet'
validation_file: str = PROJECT_ROOT + '/data/你的验证文件.parquet'
```

方法2：通过命令行参数（需要修改train_low_mem.py支持）

### Q3: 内存还是不够怎么办？

**A**: 尝试以下方法：
1. 减少数据量（使用更小的数据集）
2. 使用单GPU训练：`python train_low_mem.py train --is_finetune=True --use_small_config=True`
3. 减少 `max_seq_len`（在config.py中修改）

### Q4: 如何查看训练进度？

**A**: 
- 终端会显示实时进度条
- 查看日志：`tail -f logs/*.log`
- 查看保存的模型：`ls -lh model_save/sft_small/`

## 🎉 总结

现在 `ChatTrainerLowMem` 已经完全支持使用小数据集进行SFT训练：

1. ✅ 配置文件已更新（`TrainConfigSFTSmall`）
2. ✅ 训练脚本已支持配置选择（`--use_small_config`）
3. ✅ 数据文件路径已指向 `sft_train_small_*.parquet`
4. ✅ 所有优化已就绪（低内存模式、梯度累积等）

**立即开始训练**：
```bash
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

祝训练顺利！🚀
