# 🔧 Tokenizer 训练错误修复指南

## 🔍 问题描述

使用 `train_tokenizer.py` 训练 tokenizer 时出现 Rust panic 错误：

```
thread '<unnamed>' panicked at /home/runner/work/tokenizers/tokenizers/tokenizers/src/models/unigram/trainer.rs:228:53:
called `Result::unwrap()` on an `Err` value: Internal
```

**这是 tokenizers 库的一个已知 Bug！**

---

## 🎯 问题原因

### 可能的原因

1. **数据中有空行或无效字符**
   - 即使使用 `clean_corpus.py` 清洗后，数据可能还有问题
   - tokenizers 库对某些特殊字符非常敏感

2. **数据格式不正确**
   - 包含 `[SEP]` 等特殊标记
   - 行长度不一致
   - 编码问题

3. **数据量问题**
   - 数据量太少（< 1000 行）
   - 数据量太大导致内存不足

4. **tokenizers 库版本问题**
   - 某些版本的 tokenizers 库有 Bug
   - 与 Python 版本不兼容

---

## ✅ 解决方案

### 方案 1：使用预处理脚本（推荐）

我创建了一个专门的修复脚本 `fix_tokenizer_training.py`：

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 步骤 1：预处理数据
python fix_tokenizer_training.py \
  --input ../data/wiki.simple.txt \
  --output ../data/wiki_clean.txt

# 步骤 2：训练 tokenizer
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki_clean.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --batch-size 500
```

**预处理脚本的功能**：
- ✅ 去除空行
- ✅ 去除特殊控制字符
- ✅ 去除多余空格
- ✅ 过滤无效行（太短或不包含有效字符）
- ✅ 统一编码为 UTF-8

---

### 方案 2：直接使用维基百科数据

不要使用清洗后的数据，直接使用原始的维基百科数据：

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --batch-size 500
```

**为什么这样可能有效**：
- 维基百科数据已经是高质量的纯文本
- 不需要额外清洗
- 格式统一，没有特殊标记

---

### 方案 3：调整训练参数

如果上述方法都不行，尝试调整训练参数：

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 32000 \
  --batch-size 1000 \
  --min-frequency 2
```

**参数说明**：
- `--vocab-size 32000`：减小词汇表大小（默认 40960）
- `--batch-size 1000`：增大批次大小（默认 500）
- `--min-frequency 2`：增加最小词频（过滤低频词）

---

### 方案 4：使用 SentencePiece 方法

如果 T5-base 方法不行，尝试使用 SentencePiece：

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_sp \
  --vocab-size 40960
```

---

## 🚀 推荐流程

### 完整的训练流程

#### 步骤 1：预处理数据

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 预处理 wiki.simple.txt
python fix_tokenizer_training.py \
  --input ../data/wiki.simple.txt \
  --output ../data/wiki_clean.txt \
  --min-length 5
```

**预期输出**：
```
📖 读取输入文件: ../data/wiki.simple.txt
📊 文件大小: 1100.00 MB
🧹 预处理文本...
处理进度: 100%|████████████████████| 5234567/5234567
✅ 有效行数: 5,123,456
❌ 无效行数: 111,111
📉 过滤率: 2.12%
✅ 输出文件大小: 1050.00 MB
🎉 预处理完成！
```

#### 步骤 2：训练 Tokenizer

```bash
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki_clean.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --batch-size 500
```

**预期输出**：
```
方法: 基于 T5-base tokenizer 训练
加载维基百科语料: ../data/wiki_clean.txt
总行数: 5123456

步骤 1: 加载 T5-base tokenizer...
步骤 2: 开始训练 tokenizer (词汇表大小: 40960)...
注意: 这是 CPU 密集型任务，可能需要长时间 (约1小时)
处理语料: 100%|████████████████████| 5123456/5123456
T5-base tokenizer 加载完成

步骤 3: 保存 tokenizer...
✅ Tokenizer 已保存到: ../model_save/my_tokenizer_wiki
🎉 训练完成！
```

#### 步骤 3：验证结果

```bash
# 检查输出文件
ls -lh ../model_save/my_tokenizer_wiki/

# 测试 tokenizer
python -c "
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained('../model_save/my_tokenizer_wiki')
text = '中国是一个伟大的国家'
tokens = tokenizer.tokenize(text)
print(f'文本: {text}')
print(f'Tokens: {tokens}')
print(f'Token IDs: {tokenizer.convert_tokens_to_ids(tokens)}')
"
```

---

## 🔧 故障排除

### 问题 1：依然报 Rust panic 错误

**解决方案 A：使用更小的数据集测试**

```bash
# 创建测试数据集（前 10000 行）
head -10000 ../data/wiki.simple.txt > ../data/test_wiki.txt

# 预处理
python fix_tokenizer_training.py \
  --input ../data/test_wiki.txt \
  --output ../data/test_wiki_clean.txt

# 训练
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/test_wiki_clean.txt \
  --output-dir ../model_save/test_tokenizer \
  --vocab-size 10000
```

**解决方案 B：检查数据编码**

```bash
# 检查文件编码
file -I ../data/wiki.simple.txt

# 转换编码（如果需要）
iconv -f GBK -t UTF-8 ../data/wiki.simple.txt > ../data/wiki_utf8.txt
```

**解决方案 C：升级 tokenizers 库**

```bash
pip install --upgrade tokenizers transformers
```

---

### 问题 2：内存不足

**解决方案**：

```bash
# 使用更小的批次大小
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki_clean.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --batch-size 100  # 减小批次大小
```

---

### 问题 3：训练速度太慢

**解决方案**：

```bash
# 增大批次大小
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki_clean.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --batch-size 2000  # 增大批次大小
```

---

### 问题 4：词汇表太大

**解决方案**：

```bash
# 减小词汇表大小
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki_clean.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 32000  # 减小词汇表
```

---

## 📊 不同方法对比

### T5-base vs SentencePiece

| 特性 | T5-base | SentencePiece |
|------|---------|---------------|
| **训练速度** | 慢 | 快 |
| **内存占用** | 高 | 低 |
| **词汇表质量** | 高 | 中 |
| **稳定性** | 中（可能有 Bug） | 高 |
| **推荐场景** | 高质量训练 | 快速测试 |

### 推荐选择

1. **首选**：T5-base + 预处理数据
   ```bash
   python fix_tokenizer_training.py --input ../data/wiki.simple.txt --output ../data/wiki_clean.txt
   python train_tokenizer.py --method t5-base --wiki-file ../data/wiki_clean.txt --output-dir ../model_save/my_tokenizer_wiki
   ```

2. **备选**：SentencePiece + 原始数据
   ```bash
   python train_tokenizer.py --method sentencepiece --wiki-file ../data/wiki.simple.txt --output-dir ../model_save/my_tokenizer_sp
   ```

---

## 💡 最佳实践

### 1. 数据准备

- ✅ 使用高质量的纯文本数据（如维基百科）
- ✅ 预处理数据，去除空行和无效字符
- ✅ 确保数据编码为 UTF-8
- ✅ 数据量至少 100MB（推荐 1GB+）

### 2. 参数选择

- ✅ 词汇表大小：32000-50000（中文推荐 40960）
- ✅ 批次大小：500-2000（根据内存调整）
- ✅ 最小词频：1-3（过滤低频词）

### 3. 训练环境

- ✅ 使用 SSD 硬盘（提升 I/O 速度）
- ✅ 至少 8GB 内存（推荐 16GB+）
- ✅ 多核 CPU（训练是 CPU 密集型）

---

## 🎯 总结

### 核心要点

1. **Rust panic 错误通常是数据问题**
   - 使用 `fix_tokenizer_training.py` 预处理数据
   - 或直接使用高质量的维基百科数据

2. **推荐流程**
   ```bash
   # 预处理
   python fix_tokenizer_training.py --input ../data/wiki.simple.txt --output ../data/wiki_clean.txt
   
   # 训练
   python train_tokenizer.py --method t5-base --wiki-file ../data/wiki_clean.txt --output-dir ../model_save/my_tokenizer_wiki
   ```

3. **如果还是不行**
   - 尝试使用更小的数据集测试
   - 尝试使用 SentencePiece 方法
   - 调整训练参数（词汇表大小、批次大小）
   - 升级 tokenizers 库

---

## 📚 相关文档

- [fix_tokenizer_training.py](/Users/twrong/git/code/ChatLM-mini-Chinese/tokenize/fix_tokenizer_training.py) - 预处理脚本
- [train_tokenizer.py](/Users/twrong/git/code/ChatLM-mini-Chinese/tokenize/train_tokenizer.py) - 训练脚本
- [TOKENIZER_DATA_FORMAT_GUIDE.md](/Users/twrong/git/code/ChatLM-mini-Chinese/tokenize/TOKENIZER_DATA_FORMAT_GUIDE.md) - 数据格式指南

---

**立即开始修复！** 🚀

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 预处理数据
python fix_tokenizer_training.py \
  --input ../data/wiki.simple.txt \
  --output ../data/wiki_clean.txt

# 训练 tokenizer
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki_clean.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --batch-size 500
```
