# Tokenizer 训练进度监控指南

## 📋 概述

我为 SentencePiece tokenizer 训练添加了两种进度监控方案：

### **方案 1：增强版原始脚本**（推荐用于集成）
- 文件：[`train_tokenizer.py`](./train_tokenizer.py)
- 特点：在原有脚本基础上增强，显示训练统计和耗时

### **方案 2：独立进度监控脚本**（推荐用于独立使用）
- 文件：[`train_tokenizer_with_progress.py`](./train_tokenizer_with_progress.py)
- 特点：实时显示 SentencePiece 训练日志，更直观

---

## 🚀 快速开始

### **使用方案 1：增强版原始脚本**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 基本用法（和之前一样）
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/my_corpus_clean.txt \
  --output-dir ../model_save/my_tokenizer_sp \
  --vocab-size 40960
```

**新增功能**：
- ✅ 训练前显示数据统计（文件大小、行数）
- ✅ 预估训练时间
- ✅ 显示训练耗时
- ✅ 显示训练总结

**输出示例**：
```
步骤 1: 分析训练数据...
  - 正在统计数据量...
  ✓ 数据文件大小: 7345.23 MB
  ✓ 数据行数: 1,276,893 行
  ✓ 预估训练时间: 30-60 分钟

步骤 2: 开始训练 SentencePiece 模型...
  - 词汇表大小: 40960
  - 模型类型: unigram
  - 字符覆盖率: 0.9995
  - 训练模式: 大语料库模式

  🚀 训练进行中，请耐心等待...
  💡 提示: SentencePiece 会输出详细日志，请关注日志信息
  ==================================================
  
  [SentencePiece 训练日志...]
  
  ==================================================
  ✓ SentencePiece 模型训练完成！
  ⏱  训练耗时: 45 分 23 秒

步骤 3: 加载训练好的模型...
  ✓ 模型已加载，词汇表大小: 40960

步骤 4: 转换为 Hugging Face tokenizer...
  ✓ 已转换为 Hugging Face T5Tokenizer

步骤 5: 保存 tokenizer 到 ../model_save/my_tokenizer_sp...
  ✓ Tokenizer 已保存到 ../model_save/my_tokenizer_sp
    - tokenizer_config.json
    - sentencepiece.model
    - special_tokens_map.json

============================================================
🎉 训练完成！
============================================================
📊 训练统计:
  - 数据量: 1,276,893 行 (7345.23 MB)
  - 词汇表大小: 40960
  - 训练耗时: 45 分 23 秒
  - 输出目录: ../model_save/my_tokenizer_sp

💡 下一步:
  python quick_test_tokenizer.py ../model_save/my_tokenizer_sp
============================================================
```

---

### **使用方案 2：独立进度监控脚本**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 基本用法
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_sp \
  --vocab-size 40960
```

**特点**：
- ✅ 实时显示 SentencePiece 训练日志
- ✅ 5 步清晰的训练流程
- ✅ 详细的数据统计
- ✅ 自动验证训练结果
- ✅ 更友好的输出格式

**输出示例**：
```
======================================================================
🚀 SentencePiece Tokenizer 训练
======================================================================

📊 步骤 1/5: 分析训练数据
----------------------------------------------------------------------
📊 正在统计数据量...
  ✓ 文件路径: ../data/my_corpus_clean.txt
  ✓ 文件大小: 7345.23 MB
  ✓ 数据行数: 1,276,893 行
  ✓ 预估时间: 30-60 分钟

⚙️  步骤 2/5: 准备训练参数
----------------------------------------------------------------------
  • 词汇表大小: 40960
  • 模型类型: unigram
  • 字符覆盖率: 0.9995
  • 训练模式: 大语料库模式
  • 线程数: 16

🔥 步骤 3/5: 训练 SentencePiece 模型
----------------------------------------------------------------------
  💡 提示: SentencePiece 会输出详细日志，请关注下方信息

  trainer_interface.cc(145) LOG(INFO) Loaded 1000000 lines
  trainer_interface.cc(145) LOG(INFO) Loaded 1276893 lines
  trainer_interface.cc(122) LOG(WARNING) Too many sentences are loaded!
  trainer_interface.cc(407) LOG(INFO) Loaded all 1276893 sentences
  trainer_interface.cc(423) LOG(INFO) Adding meta_piece: [PAD]
  trainer_interface.cc(423) LOG(INFO) Adding meta_piece: [UNK]
  trainer_interface.cc(423) LOG(INFO) Adding meta_piece: [BOS]
  trainer_interface.cc(423) LOG(INFO) Adding meta_piece: [EOS]
  trainer_interface.cc(428) LOG(INFO) Normalizing sentences...
  trainer_interface.cc(537) LOG(INFO) all chars count=2841922986
  trainer_interface.cc(548) LOG(INFO) Done: 99.95% characters are covered.
  trainer_interface.cc(558) LOG(INFO) Alphabet size=4931
  trainer_interface.cc(559) LOG(INFO) Final character coverage=0.9995
  trainer_interface.cc(591) LOG(INFO) Done! preprocessed 1276889 sentences.
  
  ✓ 训练完成！

----------------------------------------------------------------------
  ✓ 模型训练完成！
  ⏱  训练耗时: 45 分 23 秒

✅ 步骤 4/5: 验证训练结果
----------------------------------------------------------------------
  ✓ 模型文件: ../model_save/my_tokenizer_sp/sentencepiece.model
  ✓ 词汇表大小: 40960
  ✓ 测试编码: '你好，世界！Hello, World!'
    → ['▁你好', '，', '▁世界', '！', 'Hello', ',', '▁World', '!']

🔄 步骤 5/5: 转换为 Hugging Face Tokenizer
----------------------------------------------------------------------
  ✓ 已转换为 T5Tokenizer
  ✓ 已保存到: ../model_save/my_tokenizer_sp
    - tokenizer_config.json
    - sentencepiece.model
    - special_tokens_map.json

======================================================================
🎉 训练完成！
======================================================================
📊 训练统计:
  • 数据量: 1,276,893 行 (7345.23 MB)
  • 词汇表大小: 40960
  • 训练耗时: 45 分 23 秒
  • 输出目录: ../model_save/my_tokenizer_sp

💡 下一步:
  # 快速测试
  python quick_test_tokenizer.py ../model_save/my_tokenizer_sp

  # 完整评估
  python evaluate_tokenizer.py --tokenizer-dir ../model_save/my_tokenizer_sp
======================================================================
```

---

## 📊 两种方案对比

| 特性 | 方案 1：增强版原始脚本 | 方案 2：独立进度监控脚本 |
|------|---------------------|----------------------|
| **文件** | `train_tokenizer.py` | `train_tokenizer_with_progress.py` |
| **兼容性** | ✅ 完全兼容原有参数 | ⚠️ 新的参数格式 |
| **进度显示** | 基本统计 + 耗时 | 实时日志 + 详细统计 |
| **输出格式** | 简洁 | 详细美观 |
| **训练验证** | ❌ 无 | ✅ 自动验证 |
| **推荐场景** | 集成到现有流程 | 独立使用 |

---

## 🎯 推荐使用方式

### **场景 1：快速训练（采样数据）**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 采样 50 万行
shuf ../data/my_corpus_clean.txt | head -n 500000 > ../data/sampled.txt

# 使用方案 2（更直观）
python train_tokenizer_with_progress.py \
  --input ../data/sampled.txt \
  --output ../model_save/my_tokenizer_sampled \
  --vocab-size 40960

# 预计时间：10-20 分钟
```

---

### **场景 2：完整数据训练**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 使用方案 2（实时监控）
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_full \
  --vocab-size 40960

# 预计时间：30-60 分钟
```

---

### **场景 3：使用 BPE 模型（更快）**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 方案 1
python train_tokenizer.py \
  --method sentencepiece \
  --sp-model-type bpe \
  --wiki-file ../data/my_corpus_clean.txt \
  --output-dir ../model_save/my_tokenizer_bpe \
  --vocab-size 40960

# 或者方案 2
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_bpe \
  --vocab-size 40960 \
  --model-type bpe
```

---

## 💡 进度监控说明

### **SentencePiece 训练日志解读**

训练过程中，SentencePiece 会输出详细日志，主要包括：

#### **1. 数据加载阶段**
```
trainer_interface.cc(145) LOG(INFO) Loaded 1000000 lines
trainer_interface.cc(145) LOG(INFO) Loaded 1276893 lines
```
- 显示已加载的行数
- 每加载 100 万行会输出一次

#### **2. 数据预处理阶段**
```
trainer_interface.cc(407) LOG(INFO) Loaded all 1276893 sentences
trainer_interface.cc(428) LOG(INFO) Normalizing sentences...
```
- 数据加载完成
- 开始归一化处理

#### **3. 字符统计阶段**
```
trainer_interface.cc(537) LOG(INFO) all chars count=2841922986
trainer_interface.cc(548) LOG(INFO) Done: 99.95% characters are covered.
trainer_interface.cc(558) LOG(INFO) Alphabet size=4931
```
- 统计总字符数
- 计算字符覆盖率
- 确定字母表大小

#### **4. 模型训练阶段**
```
trainer_interface.cc(591) LOG(INFO) Done! preprocessed 1276889 sentences.
unigram_model_trainer.cc(xxx) LOG(INFO) EM sub_iter=x size=xxxxx ...
```
- 预处理完成
- EM 算法迭代（unigram 模型）
- 这是最耗时的阶段

#### **5. 训练完成**
```
✓ 训练完成！
```

---

## 🔧 高级用法

### **1. 调整词汇表大小**

```bash
# 小词汇表（训练更快）
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_32k \
  --vocab-size 32000

# 大词汇表（压缩率更高）
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_50k \
  --vocab-size 50000
```

---

### **2. 调整字符覆盖率**

```bash
# 降低覆盖率（过滤低频字符，训练更快）
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_sp \
  --vocab-size 40960 \
  --character-coverage 0.995
```

---

### **3. 使用不同模型类型**

```bash
# Unigram（默认，效果最好）
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_unigram \
  --model-type unigram

# BPE（训练更快）
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_bpe \
  --model-type bpe

# Char（字符级别）
python train_tokenizer_with_progress.py \
  --input ../data/my_corpus_clean.txt \
  --output ../model_save/my_tokenizer_char \
  --model-type char
```

---

## 📈 训练时间参考

基于不同数据量的训练时间参考（16 核 CPU）：

| 数据量 | 行数 | 文件大小 | 训练时间（unigram） | 训练时间（BPE） |
|--------|------|---------|-------------------|----------------|
| 小规模 | 10 万 | 500 MB | 3-5 分钟 | 2-3 分钟 |
| 中规模 | 50 万 | 3 GB | 10-15 分钟 | 7-10 分钟 |
| 大规模 | 100 万 | 6 GB | 20-30 分钟 | 15-20 分钟 |
| 超大规模 | 127 万+ | 7 GB+ | 30-60 分钟 | 20-40 分钟 |

**影响因素**：
- CPU 核心数
- 内存大小
- 磁盘 I/O 速度
- 词汇表大小
- 模型类型

---

## ✅ 训练后验证

训练完成后，立即验证质量：

```bash
# 快速测试
python quick_test_tokenizer.py ../model_save/my_tokenizer_sp

# 完整评估
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_sp \
  --verbose

# 对比评估
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_sp \
  --compare-with ../model_save/my_tokenizer_bpe
```

---

## 🐛 常见问题

### **Q1: 训练过程中没有进度条？**

**A**: SentencePiece 的训练是一个黑盒操作，无法获取内部进度。但我们提供了：
- ✅ 训练前的数据统计和时间预估
- ✅ 实时显示 SentencePiece 的日志输出
- ✅ 训练后的耗时统计

---

### **Q2: 如何知道训练进行到哪一步了？**

**A**: 观察 SentencePiece 的日志输出：
1. `Loaded xxx lines` → 数据加载中
2. `Normalizing sentences` → 数据预处理中
3. `all chars count=xxx` → 字符统计完成
4. `EM sub_iter=x` → 模型训练中（最耗时）
5. `Done!` → 训练完成

---

### **Q3: 训练时间比预估的长很多？**

**A**: 可能的原因：
- 数据量太大（> 100 万行）
- CPU 核心数较少
- 内存不足导致频繁交换
- 使用 unigram 模型（比 BPE 慢）

**解决方案**：
- 使用采样数据（50 万行足够）
- 切换到 BPE 模型（`--model-type bpe`）
- 减少词汇表大小（`--vocab-size 32000`）

---

### **Q4: 训练被中断了怎么办？**

**A**: SentencePiece 不支持断点续训，需要重新开始。建议：
- 先用小数据集测试（10 万行）
- 确认没问题后再用完整数据
- 使用 `screen` 或 `tmux` 防止终端断开

---

## 📚 相关文档

- [SENTENCEPIECE_ERROR_FIX.md](./SENTENCEPIECE_ERROR_FIX.md) - 错误解决方案
- [TOKENIZER_EVALUATION_GUIDE.md](./TOKENIZER_EVALUATION_GUIDE.md) - 评估指南
- [train_tokenizer.py](./train_tokenizer.py) - 原始训练脚本
- [train_tokenizer_with_progress.py](./train_tokenizer_with_progress.py) - 进度监控脚本
- [evaluate_tokenizer.py](./evaluate_tokenizer.py) - 评估工具
- [quick_test_tokenizer.py](./quick_test_tokenizer.py) - 快速测试工具

---

## 🎯 总结

1. **两种方案**：
   - 方案 1：增强版原始脚本（兼容性好）
   - 方案 2：独立进度监控脚本（更直观）

2. **推荐使用**：
   - 快速训练 → 方案 2 + 采样数据
   - 完整训练 → 方案 2 + 完整数据
   - 集成使用 → 方案 1

3. **关键改进**：
   - ✅ 训练前显示数据统计和时间预估
   - ✅ 实时显示 SentencePiece 训练日志
   - ✅ 训练后显示耗时和总结
   - ✅ 自动验证训练结果

4. **最佳实践**：
   - 先用 50 万行采样数据快速训练（15 分钟）
   - 评估质量，如果满意就直接使用
   - 如需最佳效果，再用完整数据训练（60 分钟）

---

**现在开始训练吧！** 🚀
