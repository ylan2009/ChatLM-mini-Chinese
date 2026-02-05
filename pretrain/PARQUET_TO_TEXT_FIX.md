# parquet_to_text 函数问题修复说明

## 🔍 问题描述

使用 `parquet_to_text` 函数将 Parquet 文件转换为文本时，产生的数据格式不正确：

### ❌ 错误的输出格式

```
怎么说服男朋友卖掉箱？[SEP]emmmm。首先想说的是。我爱厨房用品一般是不用「说服」的...[SEP]
```

**问题**：
1. ❌ 包含 `[SEP]` 特殊标记
2. ❌ 结尾有多余的 `[SEP]`
3. ❌ 不是纯文本格式

---

## 🎯 问题根源

### 原始代码（有问题）

```python
def parquet_to_text(sep='[SEP]', buffer_size: int=50000) -> None:
    '''
    将parquet文件转换为txt预料，句子之间用sep隔开
    txt文件用于训练tokenizer，使用huggingface的BPE训练会导致OOM
    '''
    # ...
    for prompt, response in zip(rows['prompt'], rows['response']):
        append(prompt + sep + response + sep + '\n')  # ❌ 问题在这里！
```

**问题分析**：

1. **默认使用 `[SEP]` 分隔符**
   ```python
   sep='[SEP]'  # ❌ 特殊标记，不应该出现在 tokenizer 训练数据中
   ```

2. **结尾多余的 `[SEP]`**
   ```python
   prompt + sep + response + sep + '\n'  # ❌ 结尾的 sep 是多余的
   ```

3. **输出格式**
   ```
   问题[SEP]答案[SEP]\n
   ```
   这不是纯文本格式！

---

## ✅ 修复方案

### 修复后的代码

```python
def parquet_to_text(sep=' ', buffer_size: int=50000) -> None:
    '''
    将parquet文件转换为txt语料，用于训练tokenizer
    注意：tokenizer训练数据应该是纯文本，不应该包含特殊标记如[SEP]
    '''
    # ...
    for prompt, response in zip(rows['prompt'], rows['response']):
        # 用空格连接 prompt 和 response，去除特殊标记
        append(prompt + sep + response + '\n')  # ✅ 去除结尾的 sep
```

**改进点**：

1. **默认使用空格分隔符**
   ```python
   sep=' '  # ✅ 使用空格，自然连接
   ```

2. **去除结尾的分隔符**
   ```python
   prompt + sep + response + '\n'  # ✅ 结尾只有换行符
   ```

3. **输出格式**
   ```
   问题 答案\n
   ```
   这是纯文本格式！✅

---

## 📊 对比

### 修复前 ❌

```
怎么说服男朋友卖掉箱？[SEP]emmmm。首先想说的是...[SEP]
如何学习编程？[SEP]学习编程需要掌握以下几点...[SEP]
```

**问题**：
- 包含 `[SEP]` 特殊标记
- 不是纯文本
- Tokenizer 会学习到错误的模式

---

### 修复后 ✅

```
怎么说服男朋友卖掉箱？ emmmm。首先想说的是...
如何学习编程？ 学习编程需要掌握以下几点...
```

**优点**：
- 纯文本格式
- 自然语言
- 适合 tokenizer 训练

---

## 🚀 使用方法

### 重新生成数据

修复代码后，重新运行函数：

```python
from pretrain.raw_data_process import parquet_to_text

# 重新生成纯文本数据
parquet_to_text()
```

**输出**：
- 文件：`/data/my_corpus.txt`
- 格式：纯文本，每行一个样本

---

### 检查输出

```bash
# 查看前几行
head -5 /Users/twrong/git/code/ChatLM-mini-Chinese/data/my_corpus.txt

# 检查是否还有 [SEP]
grep -c '\[SEP\]' /Users/twrong/git/code/ChatLM-mini-Chinese/data/my_corpus.txt
```

**预期结果**：
```
怎么说服男朋友卖掉箱？ emmmm。首先想说的是...
如何学习编程？ 学习编程需要掌握以下几点...
```

✅ 没有 `[SEP]` 标记

---

### 训练 Tokenizer

使用修复后的数据训练 tokenizer：

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/my_corpus.txt \
  --output-dir ../model_save/my_tokenizer \
  --vocab-size 40960
```

---

## 📝 为什么要这样修复？

### Tokenizer 训练数据的要求

Tokenizer 训练数据应该是**纯文本**：

1. **没有特殊标记**
   - ❌ `[SEP]`、`[CLS]`、`[PAD]` 等
   - ✅ 纯自然语言

2. **自然连接**
   - ❌ 人工分隔符
   - ✅ 空格、标点符号

3. **语义完整**
   - ❌ 碎片化的文本
   - ✅ 完整的句子或段落

---

### 特殊标记的问题

如果 tokenizer 训练数据包含 `[SEP]`：

1. **Tokenizer 会学习到错误的模式**
   ```
   "[SEP]" -> 一个 token
   ```
   这不是自然语言的一部分！

2. **影响分词质量**
   ```
   "问题[SEP]答案" -> ["问题", "[SEP]", "答案"]
   ```
   `[SEP]` 占用了 token 空间，降低了效率。

3. **与模型训练不一致**
   - Tokenizer 训练：学习了 `[SEP]`
   - 模型训练：可能不使用 `[SEP]`
   - 推理：输入没有 `[SEP]`
   
   导致不一致！

---

## 🎯 正确的流程

### 1. Tokenizer 训练（纯文本）

**数据格式**：
```
数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科。
中国，是位于东亚的国家，首都为北京。
```

**来源**：
- 维基百科（`wiki.txt`）
- Parquet 转换（`my_corpus.txt`，**修复后**）

---

### 2. 模型训练（问答对）

**数据格式**：
```json
{
  "prompt": "怎么说服男朋友卖掉箱？",
  "response": "emmmm。首先想说的是..."
}
```

**来源**：
- Parquet 文件（`my_dataset.parquet`）
- JSON 文件（`sft_train.json`）

---

## 📚 相关文档

- [Tokenizer 数据格式指南](../tokenize/TOKENIZER_DATA_FORMAT_GUIDE.md)
- [数据清洗指南](../tokenize/CLEAN_CORPUS_GUIDE.md)

---

## 🔧 其他修复选项

### 选项 1：使用换行符分隔（推荐用于长文本）

```python
def parquet_to_text(sep='\n', buffer_size: int=50000) -> None:
    # ...
    for prompt, response in zip(rows['prompt'], rows['response']):
        # prompt 和 response 分成两行
        append(prompt + sep + response + '\n')
```

**输出**：
```
怎么说服男朋友卖掉箱？
emmmm。首先想说的是...
```

**优点**：
- 更清晰的分隔
- 适合长文本

---

### 选项 2：只保留 response（如果 prompt 太短）

```python
def parquet_to_text(buffer_size: int=50000) -> None:
    # ...
    for prompt, response in zip(rows['prompt'], rows['response']):
        # 只保留 response
        append(response + '\n')
```

**输出**：
```
emmmm。首先想说的是...
学习编程需要掌握以下几点...
```

**优点**：
- 更纯粹的文本
- 适合 response 内容丰富的情况

---

### 选项 3：合并为一行（当前方案）

```python
def parquet_to_text(sep=' ', buffer_size: int=50000) -> None:
    # ...
    for prompt, response in zip(rows['prompt'], rows['response']):
        # 用空格连接
        append(prompt + sep + response + '\n')
```

**输出**：
```
怎么说服男朋友卖掉箱？ emmmm。首先想说的是...
```

**优点**：
- 保留了 prompt 和 response
- 自然连接

---

## 💡 总结

### 核心要点

1. **Tokenizer 训练数据 = 纯文本**
   - ❌ 不要使用 `[SEP]` 等特殊标记
   - ✅ 使用空格或换行符自然连接

2. **修复方法**
   - 将 `sep='[SEP]'` 改为 `sep=' '`
   - 去除结尾的 `sep`

3. **重新生成数据**
   - 运行 `parquet_to_text()`
   - 检查输出格式
   - 训练 tokenizer

---

**记住：Tokenizer 训练数据应该是纯文本，不应该包含任何特殊标记！** 📖
