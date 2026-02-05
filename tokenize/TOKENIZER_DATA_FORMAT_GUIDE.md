# Tokenizer 训练数据格式说明

## 📋 问题：为什么训练数据是这个样子？

如果你看到训练数据是这样的格式：

```
[10] 黄金和白银的投资，有何区别？[SEP]黄金与白银价格研究。 黄金。1、影响因素。影响黄金价格最主要的因素是供需关系...
```

**这个格式是错误的！** ❌

---

## 🎯 正确的 Tokenizer 训练数据格式

### ✅ 应该是什么样子

Tokenizer 训练数据应该是**纯文本**，每行一个文本块：

```
数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科，从某种角度看属于形式科学的一种。数学透过抽象化和逻辑推理的使用，由计数、计算、量度和对物体形状及运动的观察而产生。
中国，是位于东亚的国家，首都为北京。中国是世界上人口最多的国家，拥有悠久的历史和灿烂的文化。
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
```

**特点**：
- ✅ **纯文本**：没有特殊标记、索引、分隔符
- ✅ **自然语言**：连贯的句子和段落
- ✅ **适中长度**：每行 1000-4000 字符
- ✅ **语义完整**：每行是一个完整的语义单元

---

## ❌ 错误的格式

### 问题 1：包含索引编号

```
❌ [10] 黄金和白银的投资，有何区别？...
```

**问题**：
- `[10]` 是索引编号，不是自然语言
- Tokenizer 会学习到这种模式，导致生成的 token 不自然

### 问题 2：使用 [SEP] 分隔符

```
❌ 问题？[SEP]答案...
```

**问题**：
- `[SEP]` 是特殊标记，用于模型训练，不是用于 tokenizer 训练
- Tokenizer 训练应该使用纯文本，不需要任何特殊标记

### 问题 3：问答对格式

```
❌ 黄金和白银的投资，有何区别？[SEP]黄金与白银价格研究...
```

**问题**：
- 这是**模型训练数据**的格式（SFT/指令微调）
- **不是** tokenizer 训练数据的格式
- Tokenizer 训练只需要纯文本，不需要问答结构

---

## 🔍 两种数据的区别

### 1. Tokenizer 训练数据（纯文本）

**用途**：训练 tokenizer，学习如何将文本分词

**格式**：
```
数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科。
中国，是位于东亚的国家，首都为北京。
人工智能是计算机科学的一个分支。
```

**来源**：
- 维基百科
- 新闻文章
- 书籍
- 网页文本

**文件示例**：
- `wiki.txt`
- `wiki.simple.txt`

---

### 2. 模型训练数据（问答对）

**用途**：训练语言模型，学习如何回答问题

**格式**：
```json
{
  "prompt": "黄金和白银的投资，有何区别？",
  "response": "黄金与白银价格研究。黄金。1、影响因素..."
}
```

或者：
```
[10] 黄金和白银的投资，有何区别？[SEP]黄金与白银价格研究...
```

**来源**：
- 问答数据集
- 对话数据集
- 指令数据集

**文件示例**：
- `sft_train_dataset.parquet`
- `sft_dataset.parquet`

---

## 📊 对比表格

| 特性 | Tokenizer 训练数据 | 模型训练数据 |
|------|-------------------|-------------|
| **格式** | 纯文本 | 问答对/对话 |
| **结构** | 无结构 | 有结构（prompt/response） |
| **特殊标记** | 无 | 有（[SEP]、[CLS] 等） |
| **用途** | 学习分词 | 学习回答问题 |
| **来源** | 维基百科、新闻 | 问答数据集 |
| **文件类型** | .txt | .json、.parquet |

---

## 🚀 正确的使用方法

### 步骤 1：训练 Tokenizer（使用纯文本）

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 使用维基百科纯文本
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_wiki \
  --vocab-size 40960
```

**数据格式**（wiki.simple.txt）：
```
数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科。
中国，是位于东亚的国家，首都为北京。
```

---

### 步骤 2：训练模型（使用问答对）

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese

# 使用问答对数据集
python train.py \
  --model-name my_model \
  --tokenizer-path ./model_save/my_tokenizer_wiki \
  --train-data ./data/sft_train_dataset.parquet
```

**数据格式**（sft_train_dataset.parquet）：
```json
{
  "prompt": "黄金和白银的投资，有何区别？",
  "response": "黄金与白银价格研究..."
}
```

---

## 💡 为什么要区分？

### Tokenizer 训练的目标

Tokenizer 的作用是**将文本分解为 token**：

```
输入: "中国是一个国家"
输出: ["中国", "是", "一个", "国家"]
```

**需要学习的内容**：
- 中文字符的组合规律
- 常见词汇的边界
- 标点符号的处理
- 数字和英文的处理

**不需要学习的内容**：
- 问答的结构
- 特殊标记的含义
- 对话的格式

---

### 模型训练的目标

模型的作用是**理解语义并生成回答**：

```
输入: "黄金和白银的投资，有何区别？"
输出: "黄金与白银在投资上有以下区别：1. 价格波动..."
```

**需要学习的内容**：
- 问题和答案的关系
- 语义理解
- 逻辑推理
- 知识记忆

---

## 🔧 如何获取正确的 Tokenizer 训练数据

### 方法 1：使用现有的维基百科数据（推荐）

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 直接使用 wiki.simple.txt
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_wiki
```

**优点**：
- ✅ 数据已经是纯文本格式
- ✅ 内容丰富，覆盖面广
- ✅ 质量高，语法正确

---

### 方法 2：从 Parquet 提取纯文本

如果你只有问答对数据（Parquet），可以提取纯文本：

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 使用 Parquet 文件（会自动提取 prompt 和 response）
python train_tokenizer.py \
  --method t5-base \
  --parquet-file ../data/sft_train_dataset.parquet \
  --output-dir ../model_save/my_tokenizer_sft
```

**注意**：
- 脚本会自动提取 `prompt` 和 `response` 字段
- 会去除特殊标记和索引
- 会合并为纯文本

---

### 方法 3：清洗自定义数据

如果你有自己的数据，使用清洗脚本：

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 清洗数据
python clean_corpus.py \
  --input /path/to/your/data.txt \
  --output ../data/my_corpus_clean.txt \
  --preview

# 训练 tokenizer
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/my_corpus_clean.txt \
  --output-dir ../model_save/my_tokenizer
```

---

## 📝 检查数据格式

### 检查你的数据是否正确

```bash
# 查看前几行
head -10 /path/to/your/data.txt

# 检查是否包含特殊标记
grep -E '\[SEP\]|\[CLS\]|\[.*\]' /path/to/your/data.txt | head -5
```

### 正确的数据示例

```bash
$ head -5 ../data/wiki.simple.txt
数学:
数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科...
基础数学的知识与运用总是个人与团体生活中不可或缺的一环...
今日，数学使用在不同的领域中，包括科学、工程、医学、经济学和金融学等...
词源.
```

**特点**：
- ✅ 纯文本
- ✅ 没有索引
- ✅ 没有特殊标记
- ✅ 自然语言

---

### 错误的数据示例

```bash
$ head -5 wrong_data.txt
[10] 黄金和白银的投资，有何区别？[SEP]黄金与白银价格研究...
[11] 如何学习编程？[SEP]学习编程需要掌握以下几点...
[12] 什么是人工智能？[SEP]人工智能是计算机科学的一个分支...
```

**问题**：
- ❌ 包含索引 `[10]`
- ❌ 包含分隔符 `[SEP]`
- ❌ 问答对格式

---

## 🎯 总结

### 核心要点

1. **Tokenizer 训练数据 = 纯文本**
   - 维基百科、新闻、书籍等
   - 没有特殊标记和结构

2. **模型训练数据 = 问答对**
   - 问答数据集、对话数据集
   - 有结构（prompt/response）

3. **不要混淆两者**
   - Tokenizer 训练用纯文本
   - 模型训练用问答对

### 推荐做法

```bash
# 1. 训练 Tokenizer（使用纯文本）
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_wiki

# 2. 训练模型（使用问答对）
python train.py \
  --tokenizer-path ./model_save/my_tokenizer_wiki \
  --train-data ./data/sft_train_dataset.parquet
```

---

## 📚 相关文档

- [train_tokenizer.py 使用指南](./train_tokenizer.py)
- [clean_corpus.py 使用指南](./CLEAN_CORPUS_GUIDE.md)
- [Tokenizer 训练错误修复](./BUGFIX_TOKENIZER_TRAINING.md)

---

**记住：Tokenizer 训练数据应该是纯文本，不是问答对！** 📖
