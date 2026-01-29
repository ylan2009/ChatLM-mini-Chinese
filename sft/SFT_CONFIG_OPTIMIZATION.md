# SFT训练配置优化方案

## 🚨 当前问题诊断

### 训练日志分析（2026-01-27）

```
epoch:4, step:500-1666, loss: 4.5-5.2
epoch:4, avg_loss: 4.771
epoch:4, cur_bleu4: 0.0494 (非常低！)
```

**关键问题**：
1. ❌ **Loss非常高**（4.5-5.2），正常应该在1.0-2.0之间
2. ❌ **Loss没有下降趋势**，在4.5-5.2之间波动
3. ❌ **BLEU4分数极低**（0.0494），说明模型生成质量很差
4. ❌ **训练到第4个epoch仍然没有收敛**

**结论**：模型根本没有学会根据prompt生成回答，这就是为什么生成DPO数据时reject完全不相关！

---

## 🎯 根本原因分析

### 1. **学习率可能太低**

当前配置：
```python
learn_rate: float = 1e-5  # 0.00001
```

**问题**：
- 预训练模型的学习率是`1e-4`（0.0001）
- SFT微调时使用`1e-5`，是预训练的1/10
- 但如果模型和数据差异较大，`1e-5`可能太保守，导致学习太慢

**证据**：
- Loss在4.5-5.2之间波动，没有明显下降
- 训练了4个epoch（约6600步），loss仍然很高

### 2. **Warmup步数太少**

当前配置：
```python
warmup_steps: int = 100  # 只有100步
```

**问题**：
- 总训练步数约：80000条 / (24 * 2 * 4) ≈ 417步/epoch
- 5个epoch总共约2085步
- Warmup只占100/2085 ≈ 4.8%，太少了！

**建议**：
- Warmup应该占总步数的10-15%
- 应该设置为200-300步

### 3. **Batch size可能太大**

当前配置：
```python
batch_size_per_gpu: int = 24
gradient_accumulation_steps: int = 4
# 实际batch size = 24 * 2 * 4 = 192
```

**问题**：
- 实际batch size = 192，非常大
- 大batch size会导致：
  - 梯度更新不频繁（每192个样本才更新一次）
  - 学习不稳定
  - 容易陷入局部最优

**建议**：
- 减小batch size或gradient_accumulation_steps
- 实际batch size控制在64-96之间

### 4. **可能需要更多epoch**

当前配置：
```python
epochs: int = 5
```

**问题**：
- 从日志看，第4个epoch的loss仍然很高
- 5个epoch可能不够

**建议**：
- 增加到8-10个epoch
- 或者训练到loss < 1.5时early stop

---

## 💡 优化方案

### 方案A：激进优化（推荐）

**适用场景**：Loss很高，需要快速收敛

```python
@dataclass
class TrainConfigSFT:
    epochs: int = 10                             # 增加到10个epoch
    batch_size_per_gpu: int = 16                 # 从24减小到16
    
    learn_rate: float = 5e-5                     # 从1e-5增加到5e-5（5倍）
    div_factor: int = 25                         # 从50减小到25，让初始学习率更高
    
    mixed_precision: str = "bf16"
    
    gradient_accumulation_steps: int = 3         # 从4减小到3
    # 实际batch size = 16 * 2 * 3 = 96（合理）
    
    warmup_steps: int = 300                      # 从100增加到300
    # 总步数约：80000/(16*2*3) ≈ 833步/epoch
    # 10个epoch约8330步，warmup占3.6%（合理）
    
    tokenizer_dir: str = PROJECT_ROOT + '/model_save/my_tokenizer_wiki/'
    model_file: str = PROJECT_ROOT + '/model_save/sft/chat_small_t5.{}.bin'
    model_config_file: str = PROJECT_ROOT + '/model_save/sft/model_config.json'
    train_file: str = PROJECT_ROOT + '/data/sft_train_dataset.parquet'
    validation_file: str = PROJECT_ROOT + '/data/sft_valid_dataset.parquet'
    test_file: str = PROJECT_ROOT + '/data/sft_test_dataset.parquet'
    
    finetune_from_ckp_file = PROJECT_ROOT + '/model_save/chat_small_t5.best.bin'
    
    train_state_dir: str = PROJECT_ROOT + '/model_save/sft/train_latest_state_sft'
    output_dir: str = PROJECT_ROOT + '/model_save/sft'
    
    logging_steps: int = 50                      # 每50步记录一次
    save_steps: int = 400                        # 从500减小到400，更频繁保存
    
    keep_latest_n_ckp: int = 5
    
    seed: int = 23333
    dataloader_buffer_size: int = 50000
    max_seq_len: int = 256
```

**预期效果**：
- Loss应该在2-3个epoch内降到2.0以下
- 5-6个epoch降到1.0-1.5
- 8-10个epoch降到0.8-1.2

---

### 方案B：保守优化

**适用场景**：担心学习率太高导致不稳定

```python
@dataclass
class TrainConfigSFT:
    epochs: int = 8                              # 增加到8个epoch
    batch_size_per_gpu: int = 20                 # 从24减小到20
    
    learn_rate: float = 3e-5                     # 从1e-5增加到3e-5（3倍）
    div_factor: int = 40                         # 从50减小到40
    
    mixed_precision: str = "bf16"
    
    gradient_accumulation_steps: int = 4         # 保持不变
    # 实际batch size = 20 * 2 * 4 = 160（稍大）
    
    warmup_steps: int = 200                      # 从100增加到200
    
    # ... 其他配置同方案A ...
    
    logging_steps: int = 50
    save_steps: int = 500                        # 保持不变
    
    keep_latest_n_ckp: int = 5
    
    seed: int = 23333
    dataloader_buffer_size: int = 50000
    max_seq_len: int = 256
```

**预期效果**：
- Loss应该在3-4个epoch内降到2.0以下
- 6-7个epoch降到1.0-1.5

---

### 方案C：渐进优化（最保险）

**适用场景**：不确定哪个参数有问题，逐步调整

**第一步**：只调整学习率和warmup

```python
learn_rate: float = 3e-5      # 增加3倍
warmup_steps: int = 200       # 增加2倍
epochs: int = 8               # 增加到8
```

**训练1-2个epoch后观察**：
- 如果loss开始下降 → 继续训练
- 如果loss仍然很高 → 进入第二步

**第二步**：调整batch size

```python
batch_size_per_gpu: int = 16           # 减小
gradient_accumulation_steps: int = 3   # 减小
```

**第三步**：如果还不行，增加学习率

```python
learn_rate: float = 5e-5      # 增加到5倍
```

---

## 📊 训练监控指标

### 正常的训练应该看到：

**Epoch 1**：
```
step 50:  loss: 3.5-4.0
step 100: loss: 2.8-3.5
step 200: loss: 2.0-2.8
step 400: loss: 1.5-2.2
```

**Epoch 2-3**：
```
avg_loss: 1.2-1.8
```

**Epoch 5+**：
```
avg_loss: 0.8-1.2
bleu4: 0.15-0.25
```

### 如果看到以下情况，说明有问题：

❌ **Loss不下降**（一直在4.0+）
- 学习率太低
- Batch size太大
- 数据有问题

❌ **Loss突然爆炸**（变成NaN或>10）
- 学习率太高
- 梯度爆炸

❌ **Loss震荡**（上下波动很大）
- Batch size太小
- 学习率太高

---

## 🚀 立即执行步骤

### 1. 备份当前配置

```bash
cp config.py config.py.backup
```

### 2. 应用优化配置

**推荐使用方案A（激进优化）**，因为当前loss太高，需要快速收敛。

修改 `config.py` 中的 `TrainConfigSFT`：

```python
@dataclass
class TrainConfigSFT:
    epochs: int = 10                             # ← 改这里
    batch_size_per_gpu: int = 16                 # ← 改这里
    
    learn_rate: float = 5e-5                     # ← 改这里
    div_factor: int = 25                         # ← 改这里
    
    mixed_precision: str = "bf16"
    
    gradient_accumulation_steps: int = 3         # ← 改这里
    
    warmup_steps: int = 300                      # ← 改这里
    
    # ... 其他保持不变 ...
    
    save_steps: int = 400                        # ← 改这里
```

### 3. 清理旧的训练状态

```bash
# 删除旧的训练状态，从头开始训练
rm -rf /Users/twrong/git/code/ChatLM-mini-Chinese/model_save/sft/train_latest_state_sft
```

### 4. 开始训练

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese
accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True
```

### 5. 实时监控

**终端1**：查看训练进度
```bash
tail -f logs/chat_trainer-*.log | grep "training loss"
```

**终端2**：查看epoch总结
```bash
tail -f logs/chat_trainer-*.log | grep "epoch log"
```

### 6. 判断训练效果

**第1个epoch结束后**（约30-40分钟）：

查看avg_loss：
```bash
grep "epoch log" logs/chat_trainer-*.log | tail -1
```

**期望看到**：
```
epoch:1, avg_loss: 2.0-2.5  # 如果是这个范围，说明配置OK
```

**如果看到**：
```
epoch:1, avg_loss: 4.0+     # 仍然很高，需要进一步调整
```

---

## 🔧 进一步调整

### 如果方案A训练后loss仍然很高（>3.0）

**可能原因**：
1. 数据格式有问题
2. 预训练模型质量不好
3. 需要更高的学习率

**解决方案**：

#### 1. 检查数据格式

```python
import pandas as pd

# 读取训练数据
df = pd.read_parquet('/Users/twrong/git/code/ChatLM-mini-Chinese/data/sft_train_dataset.parquet')

# 查看前几条
print(df.head())

# 检查是否有空值
print(df.isnull().sum())

# 检查prompt和response长度
print(df['prompt'].str.len().describe())
print(df['response'].str.len().describe())
```

**期望看到**：
- 没有空值
- prompt长度：10-200字符
- response长度：20-500字符

#### 2. 尝试更高的学习率

```python
learn_rate: float = 1e-4  # 增加到10倍（与预训练相同）
```

#### 3. 检查预训练模型

```bash
# 查看预训练模型文件大小
ls -lh /Users/twrong/git/code/ChatLM-mini-Chinese/model_save/chat_small_t5.best.bin
```

**应该看到**：约700-800MB

如果文件太小或不存在，说明预训练模型有问题。

---

## 📈 预期训练时间

**使用方案A配置**：
- 每个epoch：约30-40分钟（2张GPU）
- 10个epoch：约5-7小时
- 建议：晚上开始训练，第二天早上查看结果

**训练完成后**：
- Loss应该降到0.8-1.2
- BLEU4应该在0.15-0.25
- 手动测试模型应该能生成相关回答

---

## ✅ 成功标准

### 训练成功的标志：

1. ✅ **Loss < 1.5**（最好 < 1.0）
2. ✅ **BLEU4 > 0.15**
3. ✅ **手动测试模型能生成相关回答**

### 手动测试方法：

```python
from model.infer import ChatBot
from config import InferConfig

config = InferConfig()
config.model_dir = '/Users/twrong/git/code/ChatLM-mini-Chinese/model_save/sft/chat_small_t5.best.bin'
config.tokenizer_dir = '/Users/twrong/git/code/ChatLM-mini-Chinese/model_save/my_tokenizer_wiki/'

chatbot = ChatBot(config)

# 测试
test_cases = [
    "列出5种不同的水果。",
    "解释什么是人工智能。",
    "翻译成英语：你好，世界！",
    "1+1等于多少？",
    "写一首关于春天的诗。"
]

for prompt in test_cases:
    response = chatbot.chat(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print("-" * 50)
```

**期望结果**：
- 回答应该与prompt相关
- 回答应该是合理的中文
- 不应该是完全不相关的内容

---

## 🎯 关键要点总结

1. **当前问题**：Loss太高（4.5-5.2），模型没有学会生成相关回答
2. **根本原因**：学习率太低（1e-5）+ Warmup太少（100步）+ Batch size太大（192）
3. **解决方案**：增加学习率到5e-5 + 增加warmup到300步 + 减小batch size到96
4. **预期效果**：Loss应该在2-3个epoch内降到2.0以下，5-6个epoch降到1.0-1.5
5. **训练时间**：约5-7小时（10个epoch）
6. **成功标准**：Loss < 1.5，BLEU4 > 0.15，手动测试能生成相关回答

---

## 📞 需要帮助？

如果按照方案A训练后仍然有问题，请提供：

1. 新的训练日志（前100行和最后100行）
2. 第1个epoch的avg_loss
3. 数据样例（前5条）

这样我可以进一步帮你诊断！
