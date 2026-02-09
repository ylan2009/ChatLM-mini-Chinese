# SFT小数据集训练指南 - 16G内存优化版

## 📋 概述

本指南帮助你在**16G内存**环境下，从大数据集中采样合适数量的数据进行SFT（Supervised Fine-Tuning）训练。

## 🎯 内存与数据量对应关系

根据你的内存限制（16G，可用约13G），推荐的数据量：

| 可用内存 | 推荐训练样本数 | 验证样本数 | 预期内存占用 | 训练时长（估算） |
|---------|--------------|-----------|-------------|----------------|
| 13GB    | 3,000-5,000  | 300-500   | ~8-10GB     | 2-4小时/epoch  |
| 16GB    | 5,000-8,000  | 500-800   | ~10-12GB    | 3-6小时/epoch  |
| 20GB+   | 10,000+      | 1,000+    | ~12GB+      | 6+小时/epoch   |

**推荐配置（16G内存）**：
- ✅ **训练样本：5,000**
- ✅ **验证样本：500**
- ✅ **总计：5,500样本**

## 🚀 快速开始

### 步骤1：准备小数据集

你有两个数据源可选：

#### 选项A：从现有的SFT训练集采样（推荐）

```bash
# 从sft_train_dataset.parquet采样5000个样本
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_small.parquet \
    --num_samples 5000 \
    --valid_ratio 0.1
```

#### 选项B：从JSON文件采样

```bash
# 从alpaca_gpt4_data_zh.json采样5000个样本
python prepare_small_sft_data.py \
    --input data/alpaca_gpt4_data_zh.json \
    --output data/sft_small.parquet \
    --num_samples 5000 \
    --valid_ratio 0.1
```

**输出文件**：
- `data/sft_small_train.parquet` - 训练集（5,000样本）
- `data/sft_small_valid.parquet` - 验证集（500样本）

### 步骤2：修改配置文件

编辑 `config.py`，修改SFT配置：

```python
class TrainConfigSFT(TrainConfig):
    # 修改数据文件路径
    train_file = './data/sft_small_train.parquet'
    validation_file = './data/sft_small_valid.parquet'
    
    # 其他配置保持不变
    batch_size_per_gpu = 1
    gradient_accumulation_steps = 8
    epochs = 3  # 小数据集可以训练更多epoch
```

### 步骤3：开始训练

```bash
# 使用低内存模式训练
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True
```

## 📊 不同样本数的内存估算

脚本会自动估算内存使用：

```bash
# 测试3000样本
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_3k.parquet \
    --num_samples 3000

# 测试5000样本
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_5k.parquet \
    --num_samples 5000

# 测试8000样本
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_8k.parquet \
    --num_samples 8000
```

## 🔧 高级选项

### 自定义验证集比例

```bash
# 使用20%作为验证集
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_small.parquet \
    --num_samples 5000 \
    --valid_ratio 0.2
```

### 指定随机种子（保证可复现）

```bash
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_small.parquet \
    --num_samples 5000 \
    --seed 123
```

## 💡 训练优化建议

### 1. 小数据集训练策略

- **增加epoch数**：小数据集可以训练更多epoch（3-5个）
- **使用数据增强**：可以考虑添加数据增强技术
- **早停策略**：监控验证集BLEU分数，避免过拟合

### 2. 内存优化配置

在 `config.py` 中：

```python
class TrainConfigSFT(TrainConfig):
    # 极致低内存配置
    batch_size_per_gpu = 1          # 最小batch size
    gradient_accumulation_steps = 8  # 梯度累积补偿
    max_seq_len = 512               # 限制序列长度
    
    # 训练配置
    epochs = 3                      # 小数据集多训练几轮
    save_steps = 200                # 更频繁保存
    logging_steps = 50              # 更频繁记录
```

### 3. 监控内存使用

训练时在另一个终端监控：

```bash
# 监控系统内存
watch -n 2 'free -h'

# 监控GPU内存
watch -n 2 'nvidia-smi'
```

## 📈 预期效果

使用5,000样本训练的预期效果：

| 指标 | 预期值 | 说明 |
|-----|-------|------|
| 训练时长 | 2-4小时/epoch | 取决于GPU性能 |
| 内存占用 | 8-10GB | 双GPU总计 |
| BLEU分数 | 0.3-0.5 | 3个epoch后 |
| 模型大小 | ~700MB | 单个checkpoint |

## ⚠️ 常见问题

### Q1: 内存还是不够怎么办？

**解决方案**：
1. 减少样本数到3,000
2. 减少 `max_seq_len` 到 256
3. 使用单GPU训练（虽然更慢）

```bash
# 单GPU训练
python train_low_mem.py train --is_finetune=True
```

### Q2: 训练速度太慢？

**解决方案**：
1. 增加 `batch_size_per_gpu` 到 2（如果内存允许）
2. 减少 `gradient_accumulation_steps` 到 4
3. 使用更少的样本（3,000）

### Q3: 如何验证数据质量？

```python
# 查看采样后的数据
import pandas as pd

df = pd.read_parquet('data/sft_small_train.parquet')
print(f"样本数: {len(df)}")
print(f"列名: {df.columns.tolist()}")
print(f"\n前3个样本:")
print(df.head(3))
```

### Q4: 小数据集会过拟合吗？

**是的**，小数据集容易过拟合。建议：
1. 监控训练集和验证集的BLEU分数差异
2. 使用早停（early stopping）
3. 训练3-5个epoch后停止
4. 保存验证集BLEU最高的模型

## 📝 完整示例

```bash
# 1. 准备数据（5000样本）
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_5k.parquet \
    --num_samples 5000 \
    --valid_ratio 0.1

# 2. 修改config.py中的文件路径
# train_file = './data/sft_5k_train.parquet'
# validation_file = './data/sft_5k_valid.parquet'

# 3. 开始训练
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True

# 4. 监控内存（另一个终端）
watch -n 2 'free -h && echo "---GPU---" && nvidia-smi'
```

## 🎓 数据量选择建议

根据你的目标选择合适的数据量：

| 目标 | 推荐样本数 | 说明 |
|-----|-----------|------|
| 快速验证流程 | 1,000-2,000 | 快速测试，1小时内完成 |
| **平衡训练（推荐）** | **5,000** | **效果与速度平衡** |
| 追求更好效果 | 8,000-10,000 | 需要更多内存和时间 |
| 完整训练 | 20,000+ | 需要32GB+内存 |

## ✅ 总结

对于你的16G内存环境：

1. ✅ **推荐使用5,000训练样本 + 500验证样本**
2. ✅ **使用 `prepare_small_sft_data.py` 脚本采样**
3. ✅ **使用 `train_low_mem.py` 进行训练**
4. ✅ **预期内存占用：8-10GB**
5. ✅ **预期训练时长：2-4小时/epoch**

祝训练顺利！🚀
