# DPO数据生成质量差的诊断报告

## 🔍 问题描述

你使用BELLE数据训练的SFT模型来生成DPO数据时，生成的`reject`字段完全不相关：

```json
{
    "prompt": "列出5种不同的水果。",
    "chosen": "苹果、香蕉、橘子、菠萝、草莓",
    "reject": "这篇文章主要讨论了人工智能在医疗保健领域的应用..."
}
```

**问题**：reject应该是对prompt的相关但质量稍差的回答，但实际生成的内容完全不相关。

---

## 🎯 根本原因分析

### 1. **训练数据格式与生成格式一致性问题**

#### SFT训练时的数据格式（`MyDataset.__getitem__`）：
```python
def __getitem__(self, index):
    prompt, response = data.iloc[index].prompt, data.iloc[index].response
    # 训练时的格式：prompt和response都添加了[EOS]
    return f"{prompt}[EOS]", f"{response}[EOS]"
```

**训练时模型学到的模式**：
- 输入（encoder）：`"列出5种不同的水果。[EOS]"`
- 输出（decoder）：`"苹果、香蕉、橘子、菠萝、草莓[EOS]"`

#### DPO数据生成时的格式（`dpo_data_process.py`）：
```python
# 当前代码（已修复）
if eos_token:
    batch_prompts.append(f"{prompt}{eos_token}")
else:
    batch_prompts.append(f"{prompt}[EOS]")
```

**生成时的格式**：
- 输入（encoder）：`"列出5种不同的水果。[EOS]"` ✅

**结论**：格式是一致的，这不是问题的根源。

---

### 2. **真正的问题：SFT训练可能不充分**

从你的结果来看，生成的reject都是关于"人工智能在医疗、金融领域的应用"，这说明：

#### 可能的原因：

**A. 模型训练不充分**
- SFT训练的epoch数太少
- 学习率不合适
- 训练数据量不够
- 模型还没有学会根据prompt生成相关回答

**B. 使用了错误的模型**
- 可能加载的是预训练模型而不是SFT模型
- 模型checkpoint路径不对
- 模型权重没有正确加载

**C. 生成策略问题**
- 使用`sampling`策略时，temperature和top_p设置不当
- 生成长度设置不合理
- 模型倾向于生成训练数据中的高频模板

---

## 🔬 诊断步骤

### 步骤1：确认使用的是SFT模型

运行DPO数据生成脚本时，查看输出：

```bash
python utils/dpo_data_process.py
```

**检查输出中的模型路径**：
```
GPU 0: 开始加载模型...
GPU 0: 模型路径: /path/to/model_save/sft/chat_small_t5.best.bin  # 确认这个路径
GPU 0: Tokenizer路径: /path/to/model_save/my_tokenizer_wiki/
GPU 0: EOS token: </s> (id=1)
GPU 0: PAD token: <pad> (id=0)
GPU 0: Vocab size: 40960
```

**关键检查**：
1. ✅ 模型路径是否指向SFT训练后的模型？
2. ✅ 文件是否存在？
3. ✅ 文件大小是否正常（应该约700-800MB）？

### 步骤2：查看生成的样例

脚本会打印前3个样本的生成结果：

```
=== GPU 0 样本 0 ===
Prompt: 列出5种不同的水果。[EOS]
Raw output: 这篇文章主要讨论了人工智能...
```

**如果输出不相关，说明模型有问题。**

### 步骤3：检查SFT训练日志

查看SFT训练时的loss变化：

```bash
grep "training loss" logs/chat_trainer_*.log | tail -50
```

**正常的训练应该看到**：
- loss从高到低逐渐下降
- 最终loss应该在1.0以下
- validation loss也应该下降

**如果loss没有下降或很高，说明训练不充分。**

---

## 💡 解决方案

### 方案1：检查并重新训练SFT模型

#### 1.1 检查SFT训练配置

查看 `config.py` 中的 `TrainConfigSFT`：

```python
@dataclass
class TrainConfigSFT:
    epochs: int = 5                    # 训练轮数，建议至少5轮
    batch_size_per_gpu: int = 24       # batch size
    learn_rate: float = 1e-5           # 学习率
    gradient_accumulation_steps: int = 4
```

**建议配置**：
- `epochs`: 至少5轮，建议10轮
- `learn_rate`: 1e-5（比预训练小10倍）
- `batch_size`: 根据GPU显存调整

#### 1.2 重新训练SFT模型

```bash
# 确保使用正确的数据和配置
accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True
```

**训练时监控**：
```bash
# 实时查看loss
tail -f logs/chat_trainer_*.log | grep "training loss"
```

**期望看到**：
```
epoch 1, step 100, training loss: 2.345
epoch 1, step 200, training loss: 1.876
epoch 1, step 300, training loss: 1.543
...
epoch 5, step 1000, training loss: 0.876
```

#### 1.3 验证SFT模型质量

训练完成后，手动测试模型：

```python
from model.infer import ChatBot
from config import InferConfig

# 配置
config = InferConfig()
config.model_dir = '/path/to/model_save/sft/chat_small_t5.best.bin'
config.tokenizer_dir = '/path/to/model_save/my_tokenizer_wiki/'

# 加载模型
chatbot = ChatBot(config)

# 测试
test_prompts = [
    "列出5种不同的水果。",
    "解释什么是人工智能。",
    "翻译：Hello, how are you?"
]

for prompt in test_prompts:
    response = chatbot.chat(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
```

**期望结果**：
- 回答应该与prompt相关
- 回答应该是合理的中文
- 不应该是完全不相关的内容

---

### 方案2：调整DPO数据生成策略

如果SFT模型质量确认没问题，但生成的reject仍然不好，可以尝试：

#### 2.1 使用greedy而不是sampling

修改 `dpo_data_process.py`：

```python
# 将 search_type 从 "sampling" 改为 "greedy"
outputs = model.my_generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_seq_len=current_max_len,
    search_type="greedy",  # 使用greedy而不是sampling
)
```

**原理**：
- `greedy`：每次选择概率最高的token，生成更稳定
- `sampling`：随机采样，生成更多样但可能不稳定

#### 2.2 调整生成长度

```python
# 减小max_len，避免生成过长的无关内容
generate_alpaca_gpt4_reject_response(
    groups_cnt=500, 
    max_len=200,  # 从320减小到200
    batch_size=256
)
```

#### 2.3 使用beam search

修改生成策略：

```python
outputs = model.my_generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_seq_len=current_max_len,
    search_type="beam",  # 使用beam search
)
```

---

### 方案3：使用不同的reject生成方法

#### 3.1 使用数据增强而不是模型生成

不使用模型生成reject，而是通过数据增强：

```python
def generate_reject_by_augmentation(prompt, chosen):
    """
    通过数据增强生成reject：
    1. 截断chosen的后半部分
    2. 添加一些错误或不完整的内容
    3. 从其他样本中随机选择不太相关的回答
    """
    # 方法1：截断
    reject = chosen[:len(chosen)//2] + "..."
    
    # 方法2：添加无关内容
    reject = chosen + " 此外，人工智能技术..."
    
    # 方法3：随机选择（需要维护一个回答池）
    reject = random.choice(response_pool)
    
    return reject
```

#### 3.2 使用其他模型生成reject

如果你的SFT模型质量不够好，可以：
- 使用更大的预训练模型（如GPT-3.5）生成reject
- 使用其他开源的中文对话模型
- 从其他数据集中采样reject

---

## 📊 预期效果对比

### 当前效果（❌ 不好）

```json
{
    "prompt": "列出5种不同的水果。",
    "chosen": "苹果、香蕉、橘子、菠萝、草莓",
    "reject": "人工智能在医疗保健领域的应用..."  // 完全不相关
}
```

### 期望效果（✅ 好）

```json
{
    "prompt": "列出5种不同的水果。",
    "chosen": "苹果、香蕉、橘子、菠萝、草莓",
    "reject": "苹果、香蕉、橘子"  // 相关但不完整
}
```

或者：

```json
{
    "prompt": "列出5种不同的水果。",
    "chosen": "苹果、香蕉、橘子、菠萝、草莓",
    "reject": "水果有很多种，比如苹果、香蕉等等。"  // 相关但质量稍差
}
```

---

## 🎯 关键要点

1. **SFT训练质量是关键**
   - 必须确保SFT模型训练充分
   - loss应该下降到1.0以下
   - 模型应该能够根据prompt生成相关回答

2. **格式一致性很重要**
   - 训练时和生成时的输入格式必须一致
   - 都要添加`[EOS]` token

3. **生成策略需要调整**
   - `sampling`可能导致生成不稳定
   - 可以尝试`greedy`或`beam search`
   - 调整生成长度避免过长

4. **验证是必须的**
   - 训练后必须手动测试模型质量
   - 生成DPO数据前先验证SFT模型
   - 查看生成的样例确认质量

---

## 🚀 下一步行动

### 立即执行：

1. **运行DPO数据生成脚本**（已添加调试信息）：
   ```bash
   cd /Users/twrong/git/code/ChatLM-mini-Chinese
   python utils/dpo_data_process.py
   ```

2. **查看输出**：
   - 确认模型路径
   - 查看前3个样本的生成结果
   - 判断问题是否是模型质量问题

3. **根据结果决定**：
   - 如果模型路径不对 → 修改配置
   - 如果模型质量不好 → 重新训练SFT
   - 如果生成策略不对 → 调整生成参数

### 如果需要重新训练SFT：

```bash
# 1. 检查数据
python check_sft_data.py

# 2. 检查环境
python check_sft_ready.py

# 3. 开始训练
accelerate launch --multi_gpu --num_processes 2 ./train.py train --is_finetune=True

# 4. 监控训练
tail -f logs/chat_trainer_*.log | grep "training loss"
```

---

## 📞 需要帮助？

如果按照上述步骤仍然无法解决问题，请提供：

1. DPO数据生成脚本的完整输出（特别是模型路径和前3个样本）
2. SFT训练日志中的loss变化
3. 手动测试SFT模型的结果

这样我可以更准确地帮你诊断问题！
