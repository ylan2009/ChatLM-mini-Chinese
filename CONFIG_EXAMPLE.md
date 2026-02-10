# 大数据集预训练配置示例

## 📁 数据文件准备

### 1. 数据格式要求

数据文件必须是 **Parquet 格式**，包含以下两列：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `prompt` | string | 输入文本（问题/上下文） | "什么是机器学习？" |
| `response` | string | 输出文本（答案/回复） | "机器学习是人工智能的一个分支..." |

### 2. 数据文件路径配置

在 [`config.py`](config.py) 中找到 `TrainConfigPretrainLarge` 类，修改以下路径：

```python
@dataclass
class TrainConfigPretrainLarge:
    # ... 其他配置 ...
    
    # 修改为你的数据文件路径
    train_file: str = PROJECT_ROOT + '/data/pretrain_train_10m.parquet'      # 1000万训练数据
    validation_file: str = PROJECT_ROOT + '/data/pretrain_valid_100k.parquet'  # 10万验证数据
    test_file: str = PROJECT_ROOT + '/data/pretrain_test.parquet'             # 测试数据（可选）
```

### 3. 数据准备示例

#### 方法A: 从CSV转换为Parquet

```python
import pandas as pd

# 读取CSV文件
df = pd.read_csv('your_data.csv')

# 确保包含 prompt 和 response 列
# df = df[['prompt', 'response']]

# 保存为Parquet格式
df.to_parquet('data/pretrain_train_10m.parquet', index=False)
```

#### 方法B: 从JSON转换为Parquet

```python
import pandas as pd
import json

# 读取JSON文件
with open('your_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为DataFrame
df = pd.DataFrame(data)

# 确保包含 prompt 和 response 列
# 如果列名不同，需要重命名：
# df = df.rename(columns={'question': 'prompt', 'answer': 'response'})

# 保存为Parquet格式
df.to_parquet('data/pretrain_train_10m.parquet', index=False)
```

#### 方法C: 从JSONL转换为Parquet

```python
import pandas as pd

# 读取JSONL文件
data = []
with open('your_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# 转换为DataFrame
df = pd.DataFrame(data)

# 保存为Parquet格式
df.to_parquet('data/pretrain_train_10m.parquet', index=False)
```

### 4. 数据分割示例

如果你有一个大文件，需要分割为训练集和验证集：

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
df = pd.read_parquet('your_large_data.parquet')

# 分割数据：90%训练，10%验证
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

# 保存
train_df.to_parquet('data/pretrain_train_10m.parquet', index=False)
valid_df.to_parquet('data/pretrain_valid_100k.parquet', index=False)

print(f"训练集大小: {len(train_df)}")
print(f"验证集大小: {len(valid_df)}")
```

### 5. 数据质量检查

```python
import pandas as pd

# 读取数据
df = pd.read_parquet('data/pretrain_train_10m.parquet')

# 检查数据
print("数据形状:", df.shape)
print("\n列名:", df.columns.tolist())
print("\n前5行:")
print(df.head())

# 检查空值
print("\n空值统计:")
print(df.isnull().sum())

# 检查文本长度分布
df['prompt_len'] = df['prompt'].str.len()
df['response_len'] = df['response'].str.len()
print("\nprompt长度统计:")
print(df['prompt_len'].describe())
print("\nresponse长度统计:")
print(df['response_len'].describe())
```

## 🔧 配置参数说明

### 核心参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `epochs` | 3 | 训练轮数 | 大数据集3-5个epoch足够 |
| `batch_size_per_gpu` | 32 | 每张GPU的batch size | 显存不足时降到24或16 |
| `gradient_accumulation_steps` | 2 | 梯度累积步数 | 内存不足时增加到4 |
| `max_seq_len` | 192 | 最大序列长度 | 预训练192足够，可降到128 |
| `learn_rate` | 0.0001 | 学习率 | loss不下降时降低到5e-5 |
| `warmup_steps` | 1024 | 预热步数 | 可增加到2048 |
| `save_steps` | 5000 | 保存间隔 | 可调整为2000-10000 |
| `logging_steps` | 100 | 日志间隔 | 可调整为50-200 |

### 路径参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tokenizer_dir` | `model_save/my_tokenizer_sp/` | tokenizer路径 |
| `train_file` | `data/pretrain_train_10m.parquet` | 训练数据路径 |
| `validation_file` | `data/pretrain_valid_100k.parquet` | 验证数据路径 |
| `model_file` | `model_save/pretrain_large/chat_small_t5.{}.bin` | 模型保存路径 |
| `train_state_dir` | `model_save/pretrain_large/train_latest_state` | 训练状态保存路径 |

## 📊 资源占用预估

### 内存占用

| 配置 | 预估内存占用 | 说明 |
|------|-------------|------|
| batch_size=32, grad_accum=2 | 8-10GB | 推荐配置 |
| batch_size=24, grad_accum=2 | 7-9GB | 内存紧张时 |
| batch_size=16, grad_accum=4 | 6-8GB | 极低内存 |

### 显存占用

| 配置 | 预估显存占用/GPU | 说明 |
|------|-----------------|------|
| max_seq_len=192, batch_size=32 | 16-18GB | 推荐配置 |
| max_seq_len=192, batch_size=24 | 12-14GB | 显存紧张时 |
| max_seq_len=128, batch_size=32 | 10-12GB | 短序列 |

### 训练时间

| 数据量 | 配置 | 每epoch耗时 | 3 epochs总耗时 |
|--------|------|------------|---------------|
| 1000万 | batch_size=32×3×2=192 | 7-11小时 | 21-33小时 |
| 1000万 | batch_size=24×3×2=144 | 9-14小时 | 27-42小时 |
| 500万 | batch_size=32×3×2=192 | 3.5-5.5小时 | 10.5-16.5小时 |

## 🚀 快速开始

### 1. 准备数据

```bash
# 创建数据目录
mkdir -p data

# 将你的数据转换为parquet格式（参考上面的示例）
python prepare_data.py
```

### 2. 修改配置

编辑 [`config.py`](config.py)，修改 `TrainConfigPretrainLarge` 中的数据路径。

### 3. 启动训练

```bash
# 方法1: 使用快速启动脚本
./start_pretrain_large.sh

# 方法2: 手动启动
accelerate launch --multi_gpu --num_processes 3 ./train_low_mem.py train --use_large_config=True
```

### 4. 监控训练

```bash
# 查看训练日志
tail -f logs/chat_trainer_low_mem_*.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

## 🔍 常见问题

### Q1: 如何修改数据路径？

**A**: 编辑 [`config.py`](config.py)，找到 `TrainConfigPretrainLarge` 类，修改 `train_file` 和 `validation_file` 路径。

### Q2: 数据量不是1000万怎么办？

**A**: 配置会自动适应数据量，无需修改。只需确保数据格式正确即可。

### Q3: 如何调整batch_size？

**A**: 
```bash
# 方法1: 命令行参数
accelerate launch --multi_gpu --num_processes 3 ./train_low_mem.py train --use_large_config=True --batch_size_per_gpu=24

# 方法2: 修改config.py中的batch_size_per_gpu
```

### Q4: 如何从断点继续训练？

**A**:
```bash
accelerate launch --multi_gpu --num_processes 3 ./train_low_mem.py train --use_large_config=True --is_keep_training=True
```

### Q5: 如何修改序列长度？

**A**:
```bash
# 方法1: 命令行参数
accelerate launch --multi_gpu --num_processes 3 ./train_low_mem.py train --use_large_config=True --max_seq_len=128

# 方法2: 修改config.py中的max_seq_len
```

## 📝 配置模板

### 小数据集（100万样本）

```python
epochs: int = 5
batch_size_per_gpu: int = 32
gradient_accumulation_steps: int = 2
max_seq_len: int = 192
save_steps: int = 2000
```

### 中等数据集（500万样本）

```python
epochs: int = 4
batch_size_per_gpu: int = 32
gradient_accumulation_steps: int = 2
max_seq_len: int = 192
save_steps: int = 3000
```

### 大数据集（1000万样本）- 默认配置

```python
epochs: int = 3
batch_size_per_gpu: int = 32
gradient_accumulation_steps: int = 2
max_seq_len: int = 192
save_steps: int = 5000
```

### 超大数据集（5000万样本）

```python
epochs: int = 2
batch_size_per_gpu: int = 32
gradient_accumulation_steps: int = 2
max_seq_len: int = 192
save_steps: int = 10000
```

## 🎯 性能优化建议

### 1. 数据存储优化
- ✅ 将数据文件放在SSD上（而非HDD）
- ✅ 使用Parquet格式（比CSV快3-5倍）
- ✅ 预先清洗数据，去除空值和异常值

### 2. 训练速度优化
- ✅ 使用混合精度训练（bf16）
- ✅ 增大batch_size（充分利用GPU）
- ✅ 减少save_steps（减少IO开销）

### 3. 内存优化
- ✅ 启用ultra_low_mem模式
- ✅ 禁用num_workers
- ✅ 定期清理缓存

### 4. 显存优化
- ✅ 缩短max_seq_len
- ✅ 使用梯度累积
- ✅ 使用混合精度训练

## 📞 技术支持

如有问题，请查看：
- [优化指南](OPTIMIZATION_GUIDE.md)
- [训练脚本](train_low_mem.py)
- [配置文件](config.py)
