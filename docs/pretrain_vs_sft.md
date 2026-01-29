# 预训练 vs SFT 微调：代码实现和区别详解

## 📋 目录
1. [代码实现是否一样？](#代码实现是否一样)
2. [预训练和微调的具体区别](#预训练和微调的具体区别)
3. [两种 SFT 实现方式](#两种-sft-实现方式)
4. [数据集格式和大小](#数据集格式和大小)
5. [总结](#总结)

---

## 🔍 代码实现是否一样？

### 对于 `train.py` 的情况

**是的，代码实现基本一样**，都使用 `ChatTrainer.train()` 方法，但通过 `is_finetune` 参数控制行为：

```python
# 预训练
accelerate launch --multi_gpu --num_processes 2 ./train.py train
# 等价于：chat_trainer.train(is_keep_training=False, is_finetune=False)

# SFT 微调
accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True
# 等价于：chat_trainer.train(is_keep_training=False, is_finetune=True)
```

### 关键代码逻辑

在 `model/trainer.py` 的 `train()` 方法中：

```python
def train(self, is_keep_training: bool=False, is_finetune: bool=False) -> None:
    # ... 初始化模型 ...
    model = TextToTextModel(t5_config)
    
    # 微调时加载预训练模型并冻结部分参数
    if is_finetune:
        model.load_state_dict(torch.load(train_config.finetune_from_ckp_file))
        
        # 冻结 encoder 和 embedding，只训练 decoder
        layers_to_freeze = [model.shared, model.encoder]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
```

---

## ⚖️ 预训练和微调的具体区别

### 1. **模型初始化**

| 项目 | 预训练 | SFT 微调 |
|------|--------|----------|
| 模型初始化 | 随机初始化 | 加载预训练权重 |
| 参数来源 | `TextToTextModel(t5_config)` | `torch.load(finetune_from_ckp_file)` |

### 2. **参数冻结**

| 项目 | 预训练 | SFT 微调 |
|------|--------|----------|
| 训练所有参数 | ✅ 是 | ❌ 否 |
| 冻结的层 | 无 | `model.shared` (embedding)<br>`model.encoder` |
| 可训练参数 | 100% | 约 33% (仅 decoder) |

**为什么冻结 encoder？**
- Encoder 负责理解输入，预训练阶段已经学到了通用表示
- Decoder 负责生成输出，微调时只需要学习如何更好地生成特定任务的内容
- 这样可以：
  - **减少显存占用**：只计算 decoder 的梯度
  - **加快训练速度**：更少的参数需要更新
  - **防止过拟合**：保留预训练的知识

### 3. **训练配置差异**

虽然代码实现一样，但**实际使用时配置应该不同**：

#### 预训练配置（`TrainConfig`）
```python
epochs: int = 8
batch_size_per_gpu: int = 16
learn_rate: float = 0.0001          # 较大学习率
div_factor: int = 50                # OneCycleLR 的除数
warmup_steps: int = 1024             # 较长预热
gradient_accumulation_steps: int = 8
max_seq_len: int = 256
```

#### SFT 微调配置（建议）
```python
epochs: int = 3-5                   # 更少轮数
batch_size_per_gpu: int = 16
learn_rate: float = 1e-5            # 更小学习率（重要！）
div_factor: int = 10                # 更小的除数
warmup_steps: int = 100-500         # 更短预热
gradient_accumulation_steps: int = 4
max_seq_len: int = 384              # 可能更长
```

**为什么学习率要更小？**
- 预训练模型已经收敛到较好的位置
- 微调只需要小幅调整，大学习率会破坏预训练权重
- 通常微调学习率是预训练的 1/10 到 1/100

### 4. **数据集差异**

| 项目 | 预训练 | SFT 微调 |
|------|--------|----------|
| 数据量 | **非常大**（百万级） | **较小**（万到十万级） |
| 数据质量 | 多样但可能噪声大 | **高质量、精标注** |
| 数据来源 | 维基百科、网页文本等 | 人工标注、精选问答等 |
| 数据格式 | Parquet: `prompt`, `response` | Parquet: `prompt`, `response` |

**注意**：虽然格式一样，但内容质量完全不同！

---

## 🔀 两种 SFT 实现方式

项目中实际上有**两种 SFT 实现方式**：

### 方式1：使用 `train.py`（你当前使用的方式）

```bash
accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True
```

**特点**：
- ✅ 使用自定义的 `ChatTrainer`
- ✅ 支持多 GPU 分布式训练
- ✅ 使用 `Accelerator` 进行混合精度训练
- ✅ 支持断点续训
- ✅ 冻结 encoder 和 embedding

**适用场景**：
- 需要分布式训练
- 需要精细控制训练过程
- 需要自定义评估指标（BLEU）

### 方式2：使用 `sft_train.py`（独立实现）

```bash
python sft_train.py
```

**特点**：
- ✅ 使用 HuggingFace 的 `Seq2SeqTrainer`
- ✅ 更简单、更标准化
- ✅ 自动支持 TensorBoard 日志
- ✅ 使用 JSON 格式数据集
- ❌ **不冻结参数**（全量微调）

**代码对比**：

```python
# sft_train.py 中的实现
model = TextToTextModel.from_pretrained(config.finetune_from_ckp_file)
# 注意：这里没有冻结任何参数！
# 所有参数都会更新

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    ...
)
```

**适用场景**：
- 单卡或小规模训练
- 需要快速实验
- 使用 HuggingFace 生态工具

---

## 📊 数据集格式和大小

### 数据格式

**两种方式都使用相同的数据格式**：

```python
# Parquet 文件结构
{
    'prompt': ['问题1', '问题2', ...],
    'response': ['回答1', '回答2', ...]
}
```

### 数据集大小对比

根据 `config.py` 中的配置：

#### 预训练数据集
```python
train_file: str = PROJECT_ROOT + '/data/my_train_dataset_3k.parquet'
validation_file: str = PROJECT_ROOT + '/data/my_valid_dataset_1k.parquet'
# 注意：文件名中的 "3k" 可能只是示例，实际可能是百万级
```

根据项目 README：
- **预训练数据集**：约 930 万条（Text-to-Text）
- **来源**：维基百科、问答数据、医疗数据、知乎等

#### SFT 微调数据集
```python
sft_train_file: str = PROJECT_ROOT + '/data/sft_train.json'
# 通常较小，几万到几十万条
```

**典型大小**：
- 预训练：100万 - 1000万条
- SFT 微调：1万 - 50万条

---

## 💡 总结

### 你的理解是否正确？

**部分正确，但需要补充**：

1. ✅ **代码实现确实一样**（对于 `train.py`）
   - 都使用 `ChatTrainer.train()` 方法
   - 通过 `is_finetune` 参数控制

2. ✅ **SFT 微调确实会冻结部分参数**
   - 冻结 `model.shared` (embedding)
   - 冻结 `model.encoder`
   - 只训练 `model.decoder`

3. ⚠️ **但不仅仅是数据集大小不同**
   - **数据质量**：SFT 数据更高质量、更精标注
   - **学习率**：SFT 通常用更小的学习率
   - **训练轮数**：SFT 通常训练更少轮数
   - **数据量**：SFT 数据量通常更小

### 关键区别总结表

| 维度 | 预训练 | SFT 微调 |
|------|--------|----------|
| **代码实现** | `ChatTrainer.train(is_finetune=False)` | `ChatTrainer.train(is_finetune=True)` |
| **模型初始化** | 随机初始化 | 加载预训练权重 |
| **参数冻结** | 无 | 冻结 encoder + embedding |
| **可训练参数** | 100% | ~33% (仅 decoder) |
| **学习率** | 较大 (1e-4) | 较小 (1e-5) |
| **训练轮数** | 较多 (8 epochs) | 较少 (3-5 epochs) |
| **数据量** | 非常大 (百万级) | 较小 (万级) |
| **数据质量** | 多样但可能有噪声 | 高质量、精标注 |
| **训练目标** | 学习通用语言表示 | 学习特定任务生成 |

### 建议

1. **预训练阶段**：
   - 使用大量、多样的数据
   - 较大学习率，训练多轮
   - 训练所有参数

2. **SFT 微调阶段**：
   - 使用高质量、精标注的数据
   - **较小学习率**（重要！）
   - 冻结 encoder，只训练 decoder
   - 训练较少轮数，避免过拟合

3. **如果使用 `train.py` 做 SFT**：
   - 记得设置 `--is_finetune=True`
   - 建议调整 `TrainConfig` 中的学习率（改为 1e-5）
   - 建议减少训练轮数（改为 3-5）

---

## 🔗 相关文件

- `train.py` - 预训练和微调的统一入口
- `sft_train.py` - 独立的 SFT 实现（使用 HuggingFace Trainer）
- `model/trainer.py` - `ChatTrainer` 类实现
- `config.py` - `TrainConfig` 和 `SFTconfig` 配置
