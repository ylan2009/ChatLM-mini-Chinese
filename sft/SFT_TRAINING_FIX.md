# SFT训练修复总结

## 🎯 修复的问题

### 1. **trainer.py 缺少文件验证**
- ✅ 添加了训练数据文件存在性检查
- ✅ 添加了预训练模型存在性检查
- ✅ 改进了错误提示信息

### 2. **缺少数据准备流程**
- ✅ 创建了 `prepare_sft_data.py` 脚本
- ✅ 自动从alpaca数据生成SFT训练数据
- ✅ 自动划分训练集和验证集

### 3. **缺少环境检查工具**
- ✅ 创建了 `check_sft_ready.py` 脚本
- ✅ 一键检查所有必要文件是否存在

---

## 🚀 快速开始

### 步骤1：检查环境

```bash
python check_sft_ready.py
```

如果检查失败，按照提示解决问题。

### 步骤2：准备数据（如果需要）

```bash
# 1. 下载并处理alpaca数据
cd utils
python -c "from dpo_data_process import process_alpaca_gpt4_data; process_alpaca_gpt4_data()"

# 2. 转换为SFT训练格式
cd ..
python prepare_sft_data.py
```

### 步骤3：运行SFT训练

```bash
# 多GPU训练（推荐）
accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True

# 单GPU训练
accelerate launch ./train.py train --is_finetune=True

# 自定义参数
accelerate launch --multi_gpu --num_processes 2 ./train.py train \
    --is_finetune=True \
    --epochs=5 \
    --learn_rate=1e-5
```

---

## 📋 修改的文件

### 1. `model/trainer.py`
**修改内容**：
- 在 `train()` 方法开始时添加文件验证
- 改进 `is_finetune` 时的模型加载逻辑
- 添加可训练参数统计

**关键代码**：
```python
# 验证数据文件
if not os.path.exists(train_config.train_file):
    raise FileNotFoundError(
        f'训练数据文件不存在: {train_config.train_file}\n'
        f'请先运行: python prepare_sft_data.py 生成训练数据'
    )

# 验证预训练模型
if is_finetune and not os.path.exists(train_config.finetune_from_ckp_file):
    raise FileNotFoundError(
        f'预训练模型文件不存在: {train_config.finetune_from_ckp_file}\n'
        f'请先完成预训练，或检查config.py中的finetune_from_ckp_file路径是否正确'
    )
```

### 2. `prepare_sft_data.py` (新建)
**功能**：
- 从alpaca数据生成SFT训练数据
- 转换格式：`chosen` → `response`
- 自动划分训练集和验证集（95%/5%）
- 输出parquet格式

### 3. `check_sft_ready.py` (新建)
**功能**：
- 检查所有必要文件是否存在
- 显示训练配置
- 提供解决方案提示

### 4. `docs/sft_training_guide.md` (新建)
**内容**：
- 完整的SFT训练指南
- 数据准备步骤
- 常见问题解决方案
- 最佳实践建议

---

## ⚙️ 配置说明

在 `config.py` 中的 `TrainConfigSFT`：

```python
@dataclass
class TrainConfigSFT:
    epochs: int = 5                              # 训练轮数
    batch_size_per_gpu: int = 24                 # 每张GPU的batch size
    learn_rate: float = 1e-5                     # 学习率（重要！）
    gradient_accumulation_steps: int = 4         # 梯度累积
    
    # 数据文件路径
    train_file: str = PROJECT_ROOT + '/data/sft_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/sft_valid_dataset.parquet'
    
    # 预训练模型路径（必须存在！）
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'
```

**关键参数**：
- ✅ **学习率**：1e-5（比预训练小10倍）
- ✅ **训练轮数**：5 epochs（比预训练少）
- ✅ **参数冻结**：冻结encoder和embedding，只训练decoder

---

## 🔍 训练监控

### 查看日志
```bash
# 实时查看日志
tail -f logs/chat_trainer_*.log

# 查看loss变化
grep "training loss" logs/chat_trainer_*.log | tail -20
```

### 关键指标
- **Loss下降**：loss应该逐渐下降
- **BLEU分数**：BLEU应该逐渐提升（目标 > 0.3）
- **学习率**：使用OneCycleLR，学习率会先上升后下降

---

## ❓ 常见问题

### Q1: FileNotFoundError: 训练数据文件不存在
**解决方案**：
```bash
python prepare_sft_data.py
```

### Q2: FileNotFoundError: 预训练模型文件不存在
**解决方案**：
```bash
# 先完成预训练
accelerate launch --multi_gpu --num_processes 2 ./train.py train
```

### Q3: 生成的reject质量很差
**原因**：
- SFT模型没有训练好
- 使用了错误的模型（预训练模型而不是SFT模型）

**解决方案**：
- 确保SFT训练完成且收敛
- 检查模型路径是否正确

### Q4: CUDA out of memory
**解决方案**：
```python
# 在config.py中调整参数
batch_size_per_gpu: int = 16  # 减小batch size
gradient_accumulation_steps: int = 8  # 增加梯度累积
```

---

## 📚 相关文档

- [SFT训练完整指南](docs/sft_training_guide.md)
- [项目README](../README.md)

---

## ✅ 验证修复

运行以下命令验证修复是否成功：

```bash
# 1. 检查环境
python check_sft_ready.py

# 2. 如果检查通过，运行训练
accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True
```

如果遇到问题，查看错误信息，应该会有清晰的提示告诉你缺少什么文件以及如何解决。

---

## 🎓 关键要点

1. **数据准备是第一步**
   - 必须先运行 `prepare_sft_data.py`
   - 数据格式必须是 `{'prompt': ..., 'response': ...}`

2. **预训练模型是基础**
   - SFT是在预训练模型基础上微调
   - 必须先完成预训练

3. **学习率要小**
   - SFT的学习率应该是预训练的1/10
   - 推荐使用1e-5

4. **冻结部分参数**
   - 冻结encoder和embedding
   - 只训练decoder

5. **监控训练过程**
   - 查看loss是否下降
   - 查看BLEU分数是否提升
   - 测试生成效果
