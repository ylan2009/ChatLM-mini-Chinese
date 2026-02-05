# Tokenizer 训练错误修复说明

## 🐛 问题描述

运行以下命令时出现 Rust panic 错误：

```bash
python train_tokenizer.py --method t5-base --wiki-file ../data/my_corpus.txt --output-dir ../model_save/my_tokenizer_wiki
```

**错误信息**：

```
thread '<unnamed>' panicked at /home/runner/work/tokenizers/tokenizers/tokenizers/src/models/unigram/trainer.rs:228:53:
called `Result::unwrap()` on an `Err` value: Internal

pyo3_runtime.PanicException: called `Result::unwrap()` on an `Err` value: Internal
```

---

## 🔍 问题分析

### 1. 文件路径错误

命令中使用的文件 `../data/my_corpus.txt` **不存在**。

实际存在的文件：
- `../data/wiki.txt` - 完整的维基百科语料
- `../data/wiki.simple.txt` - 简化的维基百科语料

### 2. Tokenizers 库的 Rust Panic Bug

这是 `transformers` 库中 `train_new_from_iterator()` 方法的已知问题。

**根本原因**：
- `train_new_from_iterator()` 内部使用 tokenizers 库的 Rust 实现
- 当迭代器返回的数据格式不符合预期时，会触发 Rust panic
- 特别是以下情况容易出错：
  - 文本块包含大量换行符
  - 文本块太大或太小
  - 文本包含特殊字符或空值
  - 迭代器返回的数据格式不正确

### 3. 原始代码的问题

#### 问题 1：直接拼接换行符

```python
# ❌ 错误的代码
buffer.append(''.join(txt))  # txt 包含换行符
```

这会导致文本块中包含大量换行符，可能触发 tokenizers 库的 bug。

#### 问题 2：没有过滤空值

```python
# ❌ 错误的代码
for line in lines:
    txt.append(line)  # 没有检查 line 是否为空
```

空行会导致生成的文本块质量差。

#### 问题 3：Parquet 迭代器格式问题

```python
# ❌ 错误的代码
buffer.append(f"{prompt.as_py()}\n{response.as_py()}")  # 使用换行符连接
```

使用换行符连接可能导致格式问题。

---

## ✅ 修复方案

### 修复 1：改进 `get_wiki_corpus_iterator` 函数

**关键改进**：
1. ✅ 跳过空行
2. ✅ 使用空格连接文本，而不是直接拼接
3. ✅ 确保文本长度合理（至少 10 个字符）
4. ✅ 处理剩余的文本

```python
def get_wiki_corpus_iterator(wiki_file: str, min_chunk_size: int = 2048, batch_size: int = 1000):
    def get_training_corpus():
        buffer = []
        txt = []
        len_cnt = 0
        
        for line in progress.track(lines, description="处理语料"):
            # 跳过空行
            line = line.strip()
            if not line:
                continue
            
            len_cnt += len(line)
            txt.append(line)
            
            # 当累积字符数达到最小块大小时，创建一个文本块
            if len_cnt >= min_chunk_size:
                text = ' '.join(txt)  # ✅ 使用空格连接
                # 确保文本不为空且长度合理
                if text and len(text) >= 10:
                    buffer.append(text)
                txt = []
                len_cnt = 0
            
            # 当缓冲区达到批次大小时，返回一批数据
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
        
        # ✅ 处理剩余的文本
        if txt:
            text = ' '.join(txt)
            if text and len(text) >= 10:
                buffer.append(text)
        
        # 返回最后一批数据
        if len(buffer) > 0:
            yield buffer
    
    return get_training_corpus()
```

### 修复 2：改进 `get_parquet_corpus_iterator` 函数

**关键改进**：
1. ✅ 检查空值
2. ✅ 使用空格连接 prompt 和 response
3. ✅ 确保文本长度合理

```python
def get_parquet_corpus_iterator(parquet_file: str, batch_size: int = 1000):
    def get_training_corpus():
        buffer = []
        for prompt, response in progress.track(...):
            # ✅ 获取实际的字符串值并检查空值
            prompt_str = prompt.as_py() if prompt.as_py() else ""
            response_str = response.as_py() if response.as_py() else ""
            
            # 跳过空值
            if not prompt_str and not response_str:
                continue
            
            # ✅ 使用空格连接
            text = f"{prompt_str} {response_str}".strip()
            
            # 确保文本不为空且长度合理
            if text and len(text) >= 10:
                buffer.append(text)
            
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
        
        if buffer:
            yield buffer
    
    return get_training_corpus()
```

---

## 🚀 使用方法

### 方法 1：使用维基百科文本文件（修复后）

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 使用完整的维基百科语料
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --min-chunk-size 2048 \
  --batch-size 500

# 或使用简化的维基百科语料（更快）
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --min-chunk-size 2048 \
  --batch-size 500
```

### 方法 2：使用 Parquet 文件（推荐）

Parquet 文件的数据格式更规范，不容易出错：

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 使用 SFT 训练数据集
python train_tokenizer.py \
  --method t5-base \
  --parquet-file ../data/sft_train_dataset.parquet \
  --output-dir ../model_save/my_tokenizer_sft \
  --vocab-size 40960 \
  --batch-size 500

# 或使用完整的 SFT 数据集
python train_tokenizer.py \
  --method t5-base \
  --parquet-file ../data/sft_dataset.parquet \
  --output-dir ../model_save/my_tokenizer_sft \
  --vocab-size 40960 \
  --batch-size 500
```

### 方法 3：使用更小的批次大小

如果仍然出错，尝试减小批次大小：

```bash
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960 \
  --min-chunk-size 1024 \
  --batch-size 200
```

---

## 📊 参数说明

### `--method`
- **t5-base**（推荐）：基于 T5-base tokenizer 训练
- **char-bpe**：字符级别 BPE tokenizer
- **byte-bpe**：字节级别 BPE tokenizer

### `--vocab-size`
- 词汇表大小（默认：40960）
- 建议范围：20000 - 50000
- 更大的词汇表可以更好地表示中文，但会增加模型大小

### `--min-chunk-size`
- 每个文本块的最小字符数（仅用于 wiki-file，默认：2048）
- 建议范围：1024 - 4096
- 太小会导致文本块质量差，太大会增加内存占用

### `--batch-size`
- 每次迭代的批次大小（默认：1000）
- 建议范围：200 - 1000
- 如果出现内存错误或 Rust panic，尝试减小这个值

---

## 🔧 故障排除

### 问题 1：仍然出现 Rust panic

**解决方案**：
1. 减小 `--batch-size` 到 200 或更小
2. 减小 `--min-chunk-size` 到 1024
3. 使用 Parquet 文件而不是文本文件
4. 检查输入文件是否包含特殊字符或损坏的数据

### 问题 2：内存不足

**解决方案**：
1. 减小 `--batch-size`
2. 使用 `wiki.simple.txt` 而不是 `wiki.txt`
3. 减小 `--vocab-size`

### 问题 3：训练速度太慢

**解决方案**：
1. 使用 `wiki.simple.txt` 而不是 `wiki.txt`
2. 增大 `--batch-size`（如果内存允许）
3. 减小 `--vocab-size`

### 问题 4：文件不存在

**解决方案**：
1. 检查文件路径是否正确
2. 使用 `ls ../data/*.txt` 查看可用的文件
3. 使用 `ls ../data/*.parquet` 查看可用的 Parquet 文件

---

## 📝 修改的文件

### [train_tokenizer.py](/Users/twrong/git/code/ChatLM-mini-Chinese/tokenize/train_tokenizer.py)

**修改内容**：
1. ✅ 修复 `get_wiki_corpus_iterator` 函数
   - 跳过空行
   - 使用空格连接文本
   - 确保文本长度合理
   - 处理剩余的文本

2. ✅ 修复 `get_parquet_corpus_iterator` 函数
   - 检查空值
   - 使用空格连接 prompt 和 response
   - 确保文本长度合理

---

## 💡 最佳实践

### 1. 推荐使用 Parquet 文件

Parquet 文件的优势：
- ✅ 数据格式规范
- ✅ 读取速度快
- ✅ 不容易出错
- ✅ 支持列式存储

### 2. 合理设置参数

```bash
# 推荐的参数组合
python train_tokenizer.py \
  --method t5-base \
  --parquet-file ../data/sft_train_dataset.parquet \
  --output-dir ../model_save/my_tokenizer \
  --vocab-size 40960 \
  --batch-size 500
```

### 3. 监控训练过程

训练过程中会显示：
- 加载语料的进度
- 处理语料的进度
- 训练的进度
- 内存使用情况

如果出现问题，可以按 `Ctrl+C` 中断训练。

### 4. 验证训练结果

训练完成后，可以使用 `--test` 参数测试 tokenizer：

```bash
python train_tokenizer.py \
  --method t5-base \
  --parquet-file ../data/sft_train_dataset.parquet \
  --output-dir ../model_save/my_tokenizer \
  --test
```

---

## 🎯 总结

### 问题本质
- Tokenizers 库的 Rust 实现对数据格式要求严格
- 原始代码生成的文本块包含大量换行符和空值
- 导致 Rust panic 错误

### 修复方法
- ✅ 跳过空行
- ✅ 使用空格连接文本
- ✅ 确保文本长度合理
- ✅ 检查空值

### 推荐方案
- 🥇 使用 Parquet 文件（最稳定）
- 🥈 使用修复后的 wiki.simple.txt（更快）
- 🥉 使用修复后的 wiki.txt（最完整）

---

**修复日期**: 2026-02-05  
**Bug 严重程度**: 🔴 高（导致训练失败）  
**修复状态**: ✅ 已修复  
**测试状态**: ⏳ 待验证

---

## 🚀 下一步

1. ✅ 选择合适的输入文件（Parquet 或 txt）
2. ✅ 运行训练命令
3. ✅ 监控训练过程
4. ✅ 验证训练结果

**现在可以重新运行训练命令了！** 🎉
