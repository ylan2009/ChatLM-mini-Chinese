# 16G内存SFT训练完整方案

## 📋 问题分析

你的环境：
- **总内存**：16GB
- **可用内存**：13GB+
- **目标**：从大数据集中选择合适数量的数据进行SFT训练
- **数据源**：`my_finetune_data_zh.parquet` 或 `sft_train.json`

## ✅ 解决方案

我已经为你准备了完整的解决方案，包括：

### 1. 数据采样工具 - [`prepare_small_sft_data.py`](prepare_small_sft_data.py)

**功能**：
- ✅ 支持从 `.parquet` 或 `.json` 文件采样
- ✅ 自动分割训练集和验证集
- ✅ 自动估算内存使用量
- ✅ 给出训练建议

**使用方法**：
```bash
# 从parquet文件采样5000个样本（推荐）
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_5k.parquet \
    --num_samples 5000 \
    --valid_ratio 0.1

# 从json文件采样
python prepare_small_sft_data.py \
    --input data/alpaca_gpt4_data_zh.json \
    --output data/sft_5k.parquet \
    --num_samples 5000
```

### 2. 优化配置类 - [`config.py`](config.py)

新增 `TrainConfigSFTSmall` 配置类，专门针对16G内存优化：

```python
class TrainConfigSFTSmall:
    epochs: int = 3                    # 小数据集3个epoch
    batch_size_per_gpu: int = 1        # 极致低内存
    gradient_accumulation_steps: int = 8  # 梯度累积补偿
    
    # 数据文件（使用采样后的小数据集）
    train_file = './data/sft_5k_train.parquet'
    validation_file = './data/sft_5k_valid.parquet'
```

### 3. 一键启动脚本 - [`quick_start_sft_small.sh`](quick_start_sft_small.sh)

**最简单的使用方式**：
```bash
# 使用默认配置（5000样本）
./quick_start_sft_small.sh

# 自定义样本数
./quick_start_sft_small.sh --samples 3000

# 从JSON文件采样
./quick_start_sft_small.sh --input data/alpaca_gpt4_data_zh.json --samples 5000
```

脚本会自动完成：
1. ✅ 数据采样
2. ✅ 更新配置文件
3. ✅ 启动训练

### 4. 详细指南 - [`SFT_SMALL_DATASET_GUIDE.md`](SFT_SMALL_DATASET_GUIDE.md)

包含：
- 内存与数据量对应关系
- 详细使用步骤
- 高级优化选项
- 常见问题解答

## 🎯 推荐配置（16G内存）

根据你的内存限制，我推荐：

| 配置项 | 推荐值 | 说明 |
|-------|--------|------|
| **训练样本数** | **5,000** | 平衡效果与内存 |
| **验证样本数** | **500** | 10%验证集 |
| **Batch size** | **1** | 每GPU |
| **梯度累积** | **8** | 有效batch=16 |
| **Epochs** | **3** | 避免过拟合 |
| **预期内存** | **8-10GB** | 双GPU总计 |
| **训练时长** | **2-4小时/epoch** | 取决于GPU |

## 🚀 快速开始（3步）

### 方式A：使用一键脚本（最简单）

```bash
# 1. 给脚本执行权限（首次需要）
chmod +x quick_start_sft_small.sh

# 2. 运行脚本
./quick_start_sft_small.sh

# 就这么简单！脚本会自动完成所有步骤
```

### 方式B：手动执行（更灵活）

```bash
# 1. 准备数据（5000样本）
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_5k.parquet \
    --num_samples 5000

# 2. 修改 config.py
# 将 TrainConfigSFTSmall 中的文件路径改为：
#   train_file = PROJECT_ROOT + '/data/sft_5k_train.parquet'
#   validation_file = PROJECT_ROOT + '/data/sft_5k_valid.parquet'

# 3. 开始训练
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True
```

## 📊 不同样本数的选择

| 样本数 | 内存占用 | 训练时长 | 适用场景 |
|-------|---------|---------|---------|
| 3,000 | ~6-8GB | 1-2h/epoch | 快速验证流程 |
| **5,000** | **~8-10GB** | **2-4h/epoch** | **推荐：平衡效果与速度** |
| 8,000 | ~10-12GB | 4-6h/epoch | 追求更好效果 |
| 10,000+ | ~12GB+ | 6+h/epoch | 需要更多内存 |

## 💡 选择数据源建议

你有两个数据源可选：

### 选项1：`sft_train_dataset.parquet`（推荐）

**优点**：
- ✅ 已经是parquet格式，读取更快
- ✅ 数据已经预处理好
- ✅ 与现有训练流程兼容

**使用**：
```bash
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_5k.parquet \
    --num_samples 5000
```

### 选项2：`alpaca_gpt4_data_zh.json`

**优点**：
- ✅ 高质量GPT-4生成数据
- ✅ 适合对话任务

**使用**：
```bash
python prepare_small_sft_data.py \
    --input data/alpaca_gpt4_data_zh.json \
    --output data/sft_5k.parquet \
    --num_samples 5000
```

**建议**：如果你不确定，**优先选择 `sft_train_dataset.parquet`**，因为它已经过预处理。

## 🔧 内存优化技巧

如果5000样本还是内存不够：

### 1. 减少样本数
```bash
./quick_start_sft_small.sh --samples 3000
```

### 2. 减少序列长度
修改 `config.py` 中的 `TrainConfigSFTSmall`：
```python
max_seq_len: int = 256  # 从512降到256
```

### 3. 使用单GPU
```bash
python train_low_mem.py train --is_finetune=True
```

## 📈 预期效果

使用5000样本训练3个epoch后：

| 指标 | 预期值 |
|-----|-------|
| 训练Loss | 0.5-1.0 |
| 验证BLEU | 0.3-0.5 |
| 模型大小 | ~700MB |
| 总训练时长 | 6-12小时 |

## 🎓 训练监控

### 监控内存使用
```bash
# 终端1：训练
./quick_start_sft_small.sh

# 终端2：监控
watch -n 2 'free -h && echo "---GPU---" && nvidia-smi'
```

### 查看训练日志
```bash
tail -f logs/*.log
```

### 查看训练进度
训练过程中会显示：
- 当前epoch和step
- 实时loss
- 内存使用情况
- 预计剩余时间

## ⚠️ 注意事项

1. **数据质量 > 数据数量**：5000条高质量数据比10000条低质量数据效果更好
2. **避免过拟合**：小数据集容易过拟合，建议只训练3-5个epoch
3. **保存最佳模型**：系统会自动保存验证集BLEU最高的模型
4. **磁盘空间**：确保至少有5GB可用磁盘空间（用于保存checkpoint）

## 📝 完整示例

```bash
# 场景：从sft_train_dataset.parquet采样5000条数据进行SFT训练

# 步骤1：查看可用数据
ls -lh data/*.parquet

# 步骤2：采样数据
python prepare_small_sft_data.py \
    --input data/sft_train_dataset.parquet \
    --output data/sft_5k.parquet \
    --num_samples 5000 \
    --valid_ratio 0.1

# 步骤3：查看采样结果
python -c "
import pandas as pd
df = pd.read_parquet('data/sft_5k_train.parquet')
print(f'训练集样本数: {len(df)}')
print(f'列名: {df.columns.tolist()}')
print(f'前3个样本:')
print(df.head(3))
"

# 步骤4：开始训练
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True

# 步骤5：监控训练（另一个终端）
watch -n 2 'free -h && echo "---GPU---" && nvidia-smi'
```

## 🎉 总结

对于你的16G内存环境：

1. ✅ **推荐使用5,000训练样本**
2. ✅ **优先选择 `sft_train_dataset.parquet` 作为数据源**
3. ✅ **使用 `quick_start_sft_small.sh` 一键启动**
4. ✅ **预期内存占用：8-10GB**
5. ✅ **预期训练时长：6-12小时（3个epoch）**

现在就开始吧！🚀

```bash
./quick_start_sft_small.sh
```

如有问题，请查看 [`SFT_SMALL_DATASET_GUIDE.md`](SFT_SMALL_DATASET_GUIDE.md) 获取更多帮助。
