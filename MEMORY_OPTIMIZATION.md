# 内存优化指南 - 将训练内存控制在10G左右

## 🎯 目标
在16G内存的机器上，将训练时的内存占用控制在10G左右，留出足够的系统内存。

## 📊 内存占用分析

### 内存消耗来源
1. **模型参数** (~2-3GB)
2. **优化器状态** (~2-3GB)
3. **梯度** (~1-2GB)
4. **数据加载** (~1-3GB)
5. **PyArrow缓存** (~1-2GB)
6. **系统开销** (~1-2GB)

## 🔧 优化策略

### 1. 使用Ultra Low Memory模式

已自动启用！当可用内存 < 13GB时，系统会自动切换到ultra_low_mem模式：

```python
# 在 LowMemDataset 中
ultra_low_mem=True  # 每次读取时重新打开文件，避免PyArrow缓存累积
```

**效果**：减少1-2GB内存占用，但会降低约10-15%的训练速度。

### 2. 调整Batch Size和梯度累积

当前配置会根据可用内存自动调整：

| 可用内存 | batch_size | eval_batch_size | accumulation_steps |
|---------|-----------|----------------|-------------------|
| < 13GB  | 1         | 2              | 8                 |
| ≥ 13GB  | 2         | 4              | 16                |

**手动调整**（如果需要）：

编辑 `config.py`：

```python
class TrainConfig:
    batch_size_per_gpu = 1  # 改为1
    gradient_accumulation_steps = 8  # 改为8
```

### 3. 减小序列长度

编辑 `config.py`：

```python
class TrainConfig:
    max_seq_len = 256  # 从512改为256，可减少约30%内存
```

**注意**：这会影响模型处理长文本的能力。

### 4. 使用更小的模型

编辑 `config.py`：

```python
class T5ModelConfig:
    d_model = 384  # 从512改为384
    d_ff = 1536    # 从2048改为1536
    num_layers = 4  # 从6改为4
    num_heads = 6   # 从8改为6
```

**效果**：可减少约40%的内存占用，但模型能力会下降。

### 5. 禁用混合精度（如果使用）

编辑 `config.py`：

```python
class TrainConfig:
    mixed_precision = 'no'  # 从'fp16'改为'no'
```

**注意**：这会增加训练时间，但在某些情况下可以减少内存峰值。

### 6. 增加内存清理频率

已自动优化！系统会：
- 每50步清理一次内存
- 每200步强制清理并记录内存使用
- 每个epoch开始前清理内存
- 评估前清理内存

### 7. 系统级优化

#### 关闭不必要的程序
```bash
# 查看内存使用
free -h

# 关闭浏览器、IDE等占用内存的程序
```

#### 增加Swap空间（临时方案）
```bash
# 创建8GB swap文件
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 查看swap状态
swapon --show
```

**注意**：使用swap会显著降低训练速度。

## 🚀 推荐配置组合

### 配置1：极致低内存（~8-10GB）
适合可用内存 < 13GB的情况

```python
# config.py
class TrainConfig:
    batch_size_per_gpu = 1
    gradient_accumulation_steps = 8
    max_seq_len = 256
    mixed_precision = 'no'

class T5ModelConfig:
    d_model = 384
    d_ff = 1536
    num_layers = 4
    num_heads = 6
```

**训练命令**：
```bash
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True
```

### 配置2：平衡模式（~10-12GB）
适合可用内存 13-15GB的情况

```python
# config.py
class TrainConfig:
    batch_size_per_gpu = 2
    gradient_accumulation_steps = 16
    max_seq_len = 384
    mixed_precision = 'fp16'

class T5ModelConfig:
    # 使用默认配置
```

**训练命令**：
```bash
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True
```

## 📈 监控内存使用

### 训练时监控
系统会自动记录内存使用情况到日志文件：

```bash
# 查看日志
tail -f logs/chat_trainer_low_mem_*.log | grep "内存监控"
```

### 实时监控
在另一个终端运行：

```bash
# 每2秒刷新一次
watch -n 2 'free -h && echo "---GPU---" && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

## 🐛 常见问题

### Q1: 内存还是持续增长怎么办？

**A**: 尝试以下步骤：

1. 确认使用的是 `train_low_mem.py` 而不是 `train.py`
2. 检查是否有其他程序占用内存
3. 使用配置1（极致低内存模式）
4. 考虑只使用单GPU训练：
   ```bash
   python ./train_low_mem.py train --is_finetune=True
   ```

### Q2: Ultra Low Memory模式太慢了

**A**: 如果内存足够（>13GB），系统会自动关闭ultra_low_mem模式。

手动关闭（不推荐）：
```python
# 修改 trainer_low_mem.py
use_ultra_low_mem = False  # 强制关闭
```

### Q3: 训练速度太慢

**A**: 内存优化和训练速度是权衡关系：

| 优化级别 | 内存占用 | 相对速度 |
|---------|---------|---------|
| 标准模式 | ~15GB   | 100%    |
| 低内存模式 | ~12GB   | 85%     |
| 极致低内存 | ~10GB   | 70%     |

可以考虑：
- 使用更大内存的机器
- 减少数据集大小
- 使用单GPU训练（避免多进程开销）

### Q4: 如何验证优化效果？

**A**: 对比训练前后的内存使用：

```bash
# 训练前
free -h

# 训练中（另一个终端）
watch -n 2 free -h

# 查看峰值内存
grep "内存监控" logs/chat_trainer_low_mem_*.log
```

## 📝 总结

要将内存控制在10G左右，推荐：

1. ✅ 使用 `train_low_mem.py`（已自动优化）
2. ✅ 让系统自动选择ultra_low_mem模式（可用内存<13GB时）
3. ✅ 使用配置1（极致低内存模式）
4. ✅ 关闭其他占用内存的程序
5. ✅ 监控内存使用情况

**最简单的方法**：
```bash
# 1. 确保可用内存 < 13GB（系统会自动启用ultra_low_mem）
# 2. 直接运行
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True
```

系统会自动：
- 检测可用内存
- 选择合适的batch size
- 启用ultra_low_mem模式
- 频繁清理内存
- 记录内存使用情况
