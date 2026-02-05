# Tokenizer 训练质量评估指南

## 📋 目录

1. [评估指标](#评估指标)
2. [评估工具使用](#评估工具使用)
3. [评分标准](#评分标准)
4. [常见问题](#常见问题)
5. [优化建议](#优化建议)

---

## 📊 评估指标

### 1. **词汇表大小（Vocabulary Size）**

**定义**：Tokenizer 包含的 token 总数

**标准**：
- ✅ **推荐**：30,000 - 50,000（中文为主）
- ✅ **推荐**：40,000 - 60,000（中英混合）
- ⚠️ **过小**：< 20,000（可能导致压缩率低）
- ⚠️ **过大**：> 100,000（增加模型参数，训练慢）

**影响**：
- 词汇表太小 → 分词过细 → 序列过长 → 训练慢
- 词汇表太大 → 模型参数多 → 内存占用高

---

### 2. **压缩率（Compression Ratio）**

**定义**：平均每个 token 代表多少个字符

```
压缩率 = 总字符数 / 总 token 数
```

**标准**：
- 🌟 **优秀**：2.0 - 3.0（中文）
- 🌟 **优秀**：1.5 - 2.5（英文）
- ✅ **良好**：1.5 - 2.0 或 3.0 - 3.5
- ⚠️ **一般**：< 1.5 或 > 4.0

**示例**：
```python
文本: "人工智能是计算机科学的一个分支"  # 16个字符
Tokens: ['人工', '智能', '是', '计算机', '科学', '的', '一个', '分支']  # 8个tokens
压缩率 = 16 / 8 = 2.0  # ✅ 优秀
```

**影响**：
- 压缩率高 → 序列短 → 训练快 → 但可能损失细节
- 压缩率低 → 序列长 → 训练慢 → 但保留更多信息

---

### 3. **未知词比例（UNK Ratio）**

**定义**：分词结果中 `[UNK]` token 的比例

```
未知词比例 = UNK token 数 / 总 token 数
```

**标准**：
- 🌟 **优秀**：< 1%
- ✅ **良好**：1% - 5%
- ⚠️ **一般**：5% - 10%
- ❌ **较差**：> 10%

**示例**：
```python
# 好的 tokenizer
文本: "深度学习使用神经网络"
Tokens: ['深度', '学习', '使用', '神经', '网络']
UNK 比例 = 0%  # ✅ 优秀

# 差的 tokenizer
文本: "深度学习使用神经网络"
Tokens: ['深度', '[UNK]', '使用', '[UNK]', '网络']
UNK 比例 = 40%  # ❌ 很差
```

**影响**：
- 未知词多 → 信息丢失 → 模型效果差

---

### 4. **可逆性（Reversibility）**

**定义**：编码后再解码，能否还原原文

```python
原文 → encode → decode → 还原文本
```

**标准**：
- ✅ **必须**：100% 可逆（忽略空格差异）

**示例**：
```python
# 可逆（✅）
原文: "人工智能"
编码: [123, 456]
解码: "人工智能"  # ✅ 完全一致

# 不可逆（❌）
原文: "人工智能"
编码: [123, 1, 456]  # 1 是 [UNK]
解码: "人工[UNK]能"  # ❌ 信息丢失
```

---

### 5. **特殊字符处理**

**测试内容**：
- 标点符号：`，。！？；：""''`
- 英文标点：`, . ! ? ; : " '`
- 数字：`0-9`
- 特殊符号：`@ # $ % & * ( ) [ ] { }`
- URL：`https://www.example.com`
- Email：`test@example.com`

**标准**：
- ✅ 能正确分词
- ✅ 不产生过多 UNK
- ✅ 保持语义完整

---

### 6. **中英文混合处理**

**测试内容**：
```
"使用 Python 进行 AI 开发"
"Transformer 模型在 NLP 领域很流行"
```

**标准**：
- ✅ 中英文边界清晰
- ✅ 不会把中英文粘在一起
- ✅ 压缩率合理

---

### 7. **长短文本处理**

**测试内容**：
- 短文本：1-5 个字符
- 中等文本：20-50 个字符
- 长文本：100+ 个字符

**标准**：
- ✅ 各种长度都能正确处理
- ✅ 压缩率相对稳定

---

## 🛠️ 评估工具使用

### 安装依赖

```bash
pip install transformers
```

### 基本使用

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese/tokenize

# 评估单个 tokenizer
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_wiki
```

### 对比多个 tokenizer

```bash
# 对比 3 个 tokenizer
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_wiki \
  --compare-with ../model_save/my_tokenizer_sp \
  --compare-with ../model_save/my_tokenizer_char
```

### 使用自定义测试文件

```bash
# 使用你自己的测试数据
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_wiki \
  --test-file ../data/test_corpus.txt \
  --max-samples 200
```

### 详细模式

```bash
# 显示每个样本的详细分词结果
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_wiki \
  --verbose
```

### 保存评估结果

```bash
# 保存为 JSON 文件
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_wiki \
  --output evaluation_results.json
```

---

## 📈 评分标准

### 综合评分计算

```
总分 = 压缩率评分（60分） + 未知词评分（40分）
```

#### 压缩率评分（满分 60）

| 压缩率范围 | 评分 | 等级 |
|-----------|------|------|
| 2.0 - 3.0 | 60 | 🌟 优秀 |
| 1.5 - 2.0 或 3.0 - 3.5 | 50 | ✅ 良好 |
| 1.0 - 1.5 或 3.5 - 4.0 | 40 | ⚠️ 一般 |
| < 1.0 或 > 4.0 | 30 | ❌ 较差 |

#### 未知词评分（满分 40）

| 未知词比例 | 评分 | 等级 |
|-----------|------|------|
| < 1% | 40 | 🌟 优秀 |
| 1% - 5% | 30 | ✅ 良好 |
| 5% - 10% | 20 | ⚠️ 一般 |
| > 10% | 10 | ❌ 较差 |

#### 总分等级

| 总分范围 | 等级 | 说明 |
|---------|------|------|
| 90 - 100 | 🎉 优秀 | Tokenizer 训练质量非常好 |
| 75 - 89 | ✅ 良好 | Tokenizer 训练质量不错 |
| 60 - 74 | ⚠️ 一般 | Tokenizer 可能需要改进 |
| < 60 | ❌ 较差 | 建议重新训练 Tokenizer |

---

## 🔍 评估报告示例

```
================================================================================
📊 my_tokenizer_wiki 评估报告
================================================================================

📚 基本信息:
  词汇表大小: 40,960

🔖 特殊 Token:
  pad_token: [PAD]
  unk_token: [UNK]
  bos_token: [BOS]
  eos_token: [EOS]
  cls_token: [CLS]
  sep_token: [SEP]
  mask_token: [MASK]

📈 总体统计:
  测试样本数: 20
  总字符数: 1,234
  总 token 数: 456
  平均压缩率: 2.71 字符/token
  未知词比例: 0.22%

⭐ 综合评分: 95.0/100
  🎉 优秀！Tokenizer 训练质量非常好

📊 分类统计:
类别          样本数    平均压缩率    未知词比例
--------------------------------------------------
纯中文        3        2.85         0.00%
纯英文        2        1.92         0.00%
中英混合      3        2.45         0.50%
包含数字      2        2.67         0.00%
包含标点      2        2.89         0.00%
长文本        1        2.95         0.00%
短文本        3        1.50         1.00%
专业术语      2        2.34         0.80%
```

---

## ❓ 常见问题

### Q1: 压缩率太低（< 1.5）怎么办？

**原因**：
- 词汇表太小
- 训练数据不足
- 分词过细

**解决方案**：
```bash
# 增加词汇表大小
python train_tokenizer.py \
  --method t5-base \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_large \
  --vocab-size 60000  # 增加到 60000
```

---

### Q2: 未知词比例太高（> 5%）怎么办？

**原因**：
- 训练数据覆盖不足
- 词汇表太小
- 测试数据与训练数据差异大

**解决方案**：
1. **增加训练数据**：
   ```bash
   # 合并多个数据源
   cat wiki.txt my_corpus.txt > combined.txt
   
   python train_tokenizer.py \
     --method t5-base \
     --wiki-file combined.txt \
     --output-dir ../model_save/my_tokenizer
   ```

2. **增加词汇表大小**：
   ```bash
   python train_tokenizer.py \
     --vocab-size 50000  # 增加词汇表
   ```

3. **使用更好的训练方法**：
   ```bash
   # 使用 SentencePiece unigram 模型
   python train_tokenizer.py \
     --method sentencepiece \
     --sp-model-type unigram \
     --sp-character-coverage 0.9995
   ```

---

### Q3: 中英文混合文本处理不好怎么办？

**解决方案**：
```bash
# 使用字节级别 tokenizer
python train_tokenizer.py \
  --method byte-bpe \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_byte
```

---

### Q4: 如何选择最佳的训练方法？

**推荐顺序**：

1. **SentencePiece (unigram)** - 首选
   - ✅ 训练快（10-20分钟）
   - ✅ 稳定性高
   - ✅ 效果好
   ```bash
   python train_tokenizer.py --method sentencepiece
   ```

2. **T5-base** - 备选
   - ✅ 效果最好
   - ⚠️ 训练慢（1-2小时）
   - ⚠️ 可能有 Rust panic
   ```bash
   python train_tokenizer.py --method t5-base
   ```

3. **Byte-BPE** - 特殊场景
   - ✅ 处理任意字符
   - ⚠️ 压缩率可能较低
   ```bash
   python train_tokenizer.py --method byte-bpe
   ```

---

## 💡 优化建议

### 1. 数据准备

**✅ 好的训练数据**：
- 纯文本，无特殊标记
- 覆盖目标领域
- 数据量充足（> 100MB）
- 质量高，无乱码

**❌ 差的训练数据**：
- 包含 `[SEP]`、`[10]` 等标记
- 数据量太小（< 10MB）
- 包含大量重复内容
- 有乱码或格式错误

### 2. 参数调优

**词汇表大小**：
```bash
# 中文为主
--vocab-size 40960

# 中英混合
--vocab-size 50000

# 多语言
--vocab-size 60000
```

**SentencePiece 参数**：
```bash
# 中文优化
--sp-model-type unigram \
--sp-character-coverage 0.9995

# 英文优化
--sp-model-type bpe \
--sp-character-coverage 0.995
```

### 3. 评估流程

```bash
# 步骤 1: 训练 tokenizer
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_v1

# 步骤 2: 评估
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_v1 \
  --verbose

# 步骤 3: 如果评分 < 75，调整参数重新训练
python train_tokenizer.py \
  --method sentencepiece \
  --wiki-file ../data/wiki.simple.txt \
  --output-dir ../model_save/my_tokenizer_v2 \
  --vocab-size 50000  # 调整参数

# 步骤 4: 对比评估
python evaluate_tokenizer.py \
  --tokenizer-dir ../model_save/my_tokenizer_v1 \
  --compare-with ../model_save/my_tokenizer_v2
```

### 4. 最佳实践

1. **多次训练，选择最佳**：
   ```bash
   # 训练 3 个版本
   python train_tokenizer.py --method sentencepiece --vocab-size 40000 --output-dir v1
   python train_tokenizer.py --method sentencepiece --vocab-size 50000 --output-dir v2
   python train_tokenizer.py --method t5-base --vocab-size 40000 --output-dir v3
   
   # 对比选择
   python evaluate_tokenizer.py --tokenizer-dir v1 --compare-with v2 --compare-with v3
   ```

2. **在真实数据上测试**：
   ```bash
   # 使用你的实际应用数据测试
   python evaluate_tokenizer.py \
     --tokenizer-dir ../model_save/my_tokenizer \
     --test-file ../data/my_real_data.txt
   ```

3. **持续监控**：
   - 训练模型时监控 token 长度
   - 检查生成文本的质量
   - 定期重新评估

---

## 📚 参考资料

- [Hugging Face Tokenizers 文档](https://huggingface.co/docs/tokenizers/)
- [SentencePiece 文档](https://github.com/google/sentencepiece)
- [BPE 算法原理](https://arxiv.org/abs/1508.07909)

---

## 🎯 快速检查清单

训练完 tokenizer 后，检查以下项目：

- [ ] 词汇表大小在 30,000 - 60,000 之间
- [ ] 压缩率在 2.0 - 3.0 之间（中文）
- [ ] 未知词比例 < 5%
- [ ] 所有测试样本都可逆
- [ ] 中英文混合文本处理正常
- [ ] 特殊字符处理正常
- [ ] 综合评分 > 75

如果以上都满足，恭喜你！你的 tokenizer 训练得很好！🎉
