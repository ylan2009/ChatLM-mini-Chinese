# SentencePiece 训练错误解决方案

## ❌ 错误信息

```
unigram_model_trainer.cc(208) [array.size() <= (static_cast<size_t>(std::numeric_limits<node_int_type>::max()))] 
Input corpus too large, try with train_extremely_large_corpus=true
Program terminated with an unrecoverable error.
```

---

## 🔍 问题原因

**错误原因**：训练数据太大，超过了 SentencePiece 默认模式的处理能力

**触发条件**：
- 训练数据 > 5GB
- 或者句子数量 > 100 万行
- 或者总字符数 > 10 亿

**你的数据情况**：
- 文件大小：7.3GB
- 行数：1,276,893 行
- 总字符数：约 28 亿（从日志看到 `all chars count=2841922986`）

---

## ✅ 解决方案

### **方案 1：启用大语料库训练模式（已修复）**

**修改内容**：
在 `train_tokenizer.py` 的第 148 行，将：
```python
'--train_extremely_large_corpus=false',
```

改为：
```python
'--train_extremely_large_corpus=true',  # 启用大语料库训练模式
```

**优点**：
- ✅ 可以处理任意大小的语料库
- ✅ 不需要修改数据
- ✅ 训练质量不受影响

**缺点**：
- ⚠️ 训练时间会更长（可能需要 30-60 分钟）
- ⚠️ 内存占用会更高（建议 16GB+ 内存）

**使用方法**：
```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 现在可以直接训练大语料库了
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/my_corpus_clean.txt \
  --output-dir ../model_save/my_tokenizer_sp \
  --vocab-size 40960
```

---

### **方案 2：减少训练数据（快速方案）**

如果你想快速训练，可以使用部分数据：

```bash
# 使用前 50 万行（约 3GB）
head -n 500000 /Users/twrong/git/code/ChatLM-mini-Chinese/data/my_corpus_clean.txt > \
  /Users/twrong/git/code/ChatLM-mini-Chinese/data/my_corpus_500k.txt

# 训练（10-20 分钟）
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/my_corpus_500k.txt \
  --output-dir ../model_save/my_tokenizer_sp_500k \
  --vocab-size 40960
```

**优点**：
- ✅ 训练速度快（10-20 分钟）
- ✅ 内存占用低（< 8GB）
- ✅ 对于 tokenizer 训练，50 万行已经足够

**缺点**：
- ⚠️ 词汇覆盖可能略低（但影响不大）

---

### **方案 3：使用采样数据**

随机采样，保证数据多样性：

```bash
# 随机采样 50 万行
shuf /Users/twrong/git/code/ChatLM-mini-Chinese/data/my_corpus_clean.txt | \
  head -n 500000 > \
  /Users/twrong/git/code/ChatLM-mini-Chinese/data/my_corpus_sampled.txt

# 训练
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/my_corpus_sampled.txt \
  --output-dir ../model_save/my_tokenizer_sp_sampled \
  --vocab-size 40960
```

**优点**：
- ✅ 数据多样性更好
- ✅ 训练速度快
- ✅ 词汇覆盖更全面

---

### **方案 4：调整 SentencePiece 参数**

如果内存不足，可以调整参数：

修改 `train_tokenizer.py`，添加以下参数：

```python
train_args = [
    # ... 其他参数 ...
    '--train_extremely_large_corpus=true',
    '--input_sentence_size=2000000',  # 限制输入句子数量
    '--shuffle_input_sentence=true',  # 随机打乱输入
    '--num_threads=8',  # 减少线程数（如果内存不足）
]
```

---

## 📊 不同方案对比

| 方案 | 训练时间 | 内存需求 | 词汇覆盖 | 推荐度 |
|------|---------|---------|---------|--------|
| 方案 1：大语料库模式 | 30-60 分钟 | 16GB+ | 最好 | ⭐⭐⭐⭐⭐ |
| 方案 2：前 50 万行 | 10-20 分钟 | 8GB | 很好 | ⭐⭐⭐⭐ |
| 方案 3：随机采样 | 10-20 分钟 | 8GB | 很好 | ⭐⭐⭐⭐⭐ |
| 方案 4：调整参数 | 20-40 分钟 | 12GB | 好 | ⭐⭐⭐ |

---

## 🚀 推荐执行流程

### **如果你的内存 >= 16GB（推荐）**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 直接使用方案 1：大语料库模式
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/my_corpus_clean.txt \
  --output-dir ../model_save/my_tokenizer_sp \
  --vocab-size 40960

# 预计时间：30-60 分钟
# 内存占用：16-24GB
```

---

### **如果你的内存 < 16GB 或想快速训练**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 方案 3：随机采样（推荐）
shuf ../data/my_corpus_clean.txt | head -n 500000 > ../data/my_corpus_sampled.txt

python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/my_corpus_sampled.txt \
  --output-dir ../model_save/my_tokenizer_sp_sampled \
  --vocab-size 40960

# 预计时间：10-20 分钟
# 内存占用：6-8GB
```

---

## 💡 训练数据量建议

对于 tokenizer 训练，**不是数据越多越好**！

### **推荐数据量**

| 数据量 | 行数 | 文件大小 | 训练时间 | 效果 |
|--------|------|---------|---------|------|
| 小规模 | 10 万 | 500MB | 5 分钟 | 一般 |
| 中规模 | 50 万 | 3GB | 15 分钟 | 很好 ✅ |
| 大规模 | 100 万 | 6GB | 30 分钟 | 很好 ✅ |
| 超大规模 | 200 万+ | 12GB+ | 60 分钟+ | 略好 |

**结论**：
- ✅ **50 万行（3GB）是性价比最高的选择**
- ✅ 100 万行已经足够覆盖绝大部分词汇
- ⚠️ 超过 100 万行，收益递减

---

## 🔧 其他优化建议

### 1. **使用 BPE 模型（更快）**

```bash
python train_tokenizer.py \
  --method sentencepiece \
  --sp-model-type bpe \  # 使用 BPE 而不是 unigram
  --wiki-file ../data/my_corpus_sampled.txt \
  --output-dir ../model_save/my_tokenizer_sp_bpe \
  --vocab-size 40960
```

**优点**：
- ✅ 训练速度更快（快 30-50%）
- ✅ 内存占用更低

**缺点**：
- ⚠️ 效果可能略差于 unigram

---

### 2. **减少词汇表大小**

```bash
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/my_corpus_clean.txt \
  --output-dir ../model_save/my_tokenizer_sp_32k \
  --vocab-size 32000  # 减少到 32000
```

**优点**：
- ✅ 训练速度更快
- ✅ 模型参数更少

**缺点**：
- ⚠️ 压缩率可能略低

---

### 3. **调整字符覆盖率**

```bash
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/my_corpus_clean.txt \
  --output-dir ../model_save/my_tokenizer_sp \
  --vocab-size 40960 \
  --sp-character-coverage 0.995  # 降低到 0.995（默认 0.9995）
```

**优点**：
- ✅ 训练速度更快
- ✅ 过滤掉极低频字符

**缺点**：
- ⚠️ 可能增加 UNK token

---

## 📋 完整训练命令

### **推荐命令（方案 1 + 优化）**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 如果内存充足（>= 16GB），使用完整数据
python train_tokenizer.py \
  --method sentencepiece \
  --sp-model-type unigram \
  --wiki-file ../data/my_corpus_clean.txt \
  --output-dir ../model_save/my_tokenizer_sp_full \
  --vocab-size 40960 \
  --sp-character-coverage 0.9995

# 预计时间：30-60 分钟
```

### **推荐命令（方案 3 + 优化）**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 如果内存有限或想快速训练，使用采样数据
shuf ../data/my_corpus_clean.txt | head -n 500000 > ../data/my_corpus_sampled.txt

python train_tokenizer.py \
  --method sentencepiece \
  --sp-model-type unigram \
  --wiki-file ../data/my_corpus_sampled.txt \
  --output-dir ../model_save/my_tokenizer_sp_sampled \
  --vocab-size 40960 \
  --sp-character-coverage 0.9995

# 预计时间：10-20 分钟
```

---

## ✅ 验证训练结果

训练完成后，使用评估工具验证：

```bash
# 快速测试
python quick_test_tokenizer.py ../model_save/my_tokenizer_sp_sampled

# 完整评估
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_sp_sampled \
  --verbose
```

---

## 🎯 总结

1. **已修复**：`train_tokenizer.py` 已启用 `train_extremely_large_corpus=true`
2. **推荐方案**：
   - 内存充足（>= 16GB）→ 使用完整数据（方案 1）
   - 内存有限或快速训练 → 使用采样数据（方案 3）
3. **最佳实践**：50 万行采样数据是性价比最高的选择
4. **训练后**：使用评估工具验证质量

---

## 📚 相关文档

- [TOKENIZER_EVALUATION_GUIDE.md](./TOKENIZER_EVALUATION_GUIDE.md) - 评估指南
- [train_tokenizer.py](./train_tokenizer.py) - 训练脚本
- [evaluate_tokenizer.py](./evaluate_tokenizer.py) - 评估工具

---

**现在可以重新运行训练命令了！** 🚀
