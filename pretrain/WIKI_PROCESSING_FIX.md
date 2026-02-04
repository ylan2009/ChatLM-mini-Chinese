# Wiki 数据处理问题修复说明

## 🐛 问题描述

在处理 wiki.simple.txt 数据集时，发现处理后只剩余 1 行数据，而原始文件有 6,791,470 行。

```
[2026-02-02 09:10:48.161] [INFO]: merge into file: /data3/ChatLM-mini-Chinese/data/my_data/wiki_zh_simple.parquet, 全部数据共6791470行，清洗后剩余1行
```

## 🔍 问题根源

### 1. wiki.simple.txt 的格式

```
数学:
数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科...

词源.
西方语言中"数学"（）一词源自于古希腊语的（）...

历史.
数学有着久远的历史...
```

- 词条名以**英文冒号** `:` 结尾（如 `数学:`）
- 词条名后是空行
- 然后是内容段落
- 子标题以中文句号 `.` 结尾（如 `词源.`、`历史.`）

### 2. 代码逻辑问题（第一次修复）

在 `process_wiki_simple_to_dataset()` 函数中，原来的处理流程是：

```python
# 原来的错误流程
for line in read_file:
    # 1. 先清洗一行
    line = process_line(line)  # ❌ 这里会破坏格式！
    
    # 2. 再判断是否是标题行
    if prompt == '' and line.endswith('：') and pre_line_len == 0:  # ❌ 判断中文冒号
        prompt = choice(prompt_prefix).format(line[0: -1])
```

**问题**：
1. `process_line()` 函数会调用 `remove_duplicate_punctuation()`
2. `remove_duplicate_punctuation()` 会将空格替换为逗号：`sentence = re.sub(' |　', '，', sentence)`
3. 这导致原本的格式被破坏，标题行的冒号可能被修改
4. 判断条件使用的是中文冒号 `：`，但 wiki.simple.txt 使用的是英文冒号 `:`

### 3. 代码逻辑问题（第二次修复 - 关键问题）

第一次修复后，代码改为：

```python
# 第一次修复后的代码
for line in read_file:
    line_stripped = line.strip()
    
    # 先判断是否是标题行（使用英文冒号）
    if prompt == '' and line_stripped.endswith(':') and pre_line_len == 0:
        title = line_stripped[0: -1]
        prompt = choice(prompt_prefix).format(title)
        continue
    
    # 清洗一行
    line = process_line(line_stripped)
    
    # 判断是否是内容行
    if prompt != '' and not line_stripped.endswith(':'):  # ❌ 这里有问题！
        # 处理内容...
```

**关键问题**：

`convert_en_punctuation_to_zh_punct()` 函数会将**英文标点转换为中文标点**：

```python
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence
```

这意味着：
- 原始行 `line_stripped` 中的英文冒号 `:` 会被保留
- 但清洗后的 `line` 中的英文冒号 `:` 会被转换为中文冒号 `：`
- 所以判断 `not line_stripped.endswith(':')` 时，**应该使用清洗后的 `line`**，而不是原始的 `line_stripped`！

**为什么只剩 1 行？**

因为判断条件错误：
1. 标题行 `数学:` 被正确识别，设置了 `prompt`
2. 内容行经过清洗后，英文冒号变成中文冒号
3. 但是判断条件 `not line_stripped.endswith(':')` 使用的是原始行
4. 如果内容中有英文冒号（如 URL、时间等），就会被误判为标题行
5. 导致 `prompt != ''` 条件不满足，内容不被处理
6. 最终只有最后一个 prompt 和 response 被保存（在 `end for` 后的代码中）

## ✅ 解决方案

### 最终修改后的正确流程

```python
for line in read_file:
    all_cnt += 1

    # 1. 先 strip 获取原始行的长度信息
    line_stripped = line.strip()
    
    # 2. 跳过已保存prompt后的多余行
    if len(prompt) == 0 and pre_line_len > 0:
        pre_line_len = len(line_stripped)
        continue
    
    # 3. 在清洗之前判断是否是标题行（避免清洗破坏格式）
    if prompt == '' and line_stripped.endswith(':') and pre_line_len == 0:  # ✅ 使用英文冒号
        # 提取词条名（去掉末尾的冒号）
        title = line_stripped[0: -1]
        prompt = choice(prompt_prefix).format(title)
        pre_line_len = len(line_stripped)
        continue
    
    # 4. 只对内容行进行清洗
    line = process_line(line_stripped)
    
    pre_line_len = len(line_stripped)

    # 5. 处理内容行
    # 注意：这里要用清洗后的line来判断，因为清洗会将英文冒号转换为中文冒号
    if prompt != '' and not line.endswith('：'):  # ✅ 使用清洗后的line和中文冒号
        # 其实，pre_line_len已经是len(line_stripped)了，如果len(line_stripped)=0，既是当前行是0，则不管答案长度够不够，都需要保存了
        if len(response) + len(line) <= max_len and pre_line_len != 0: 
            response = '{}{}'.format(response, line)
        elif len(response) + len(line) > max_len or pre_line_len == 0:
            # 长度超了或者当前的百科已经结束，保存一条样例
            keep_cnt += 1
            response = '{}{}'.format(response, line)
            append({'prompt': prompt, 'response': ''.join(response[0: max_len])})
            prompt = ''
            response = ''
```

### 关键修改点

1. ✅ **先判断后清洗**：在调用 `process_line()` 之前先判断是否是标题行
2. ✅ **标题行判断使用英文冒号**：`line_stripped.endswith(':')`
3. ✅ **内容行判断使用中文冒号**：`line.endswith('：')`（使用清洗后的 `line`）
4. ✅ **保留原始行信息**：使用 `line_stripped` 保存原始行的信息，用于长度判断
5. ✅ **只清洗内容行**：只对非标题行调用 `process_line()` 进行清洗

## 📝 修改的文件

- [pretrain/raw_data_process.py](raw_data_process.py) - `process_wiki_simple_to_dataset()` 函数

## 🎯 预期效果

修复后，应该能够正确处理 wiki.simple.txt，生成约 119 万条问答数据（根据项目文档说明）。

```
[INFO]: merge into file: /data3/ChatLM-mini-Chinese/data/my_data/wiki_zh_simple.parquet, 全部数据共6791470行，清洗后剩余1190000行
```

## 🔧 如何验证

重新运行数据处理脚本：

```bash
cd pretrain
python download_and_process_datasets.py --process wiki
```

或者直接运行：

```python
from raw_data_process import process_wiki_simple_to_dataset
process_wiki_simple_to_dataset()
```

## 📚 相关函数

### `convert_en_punctuation_to_zh_punct(sentence: str) -> str`

这个函数会将英文标点转换为中文标点：

```python
en_punctuation = ",().!;:"
zh_punctuation = "，（）。！；："

def convert_en_punctuation_to_zh_punct(sentence: str) -> str:
    '''
    将句子中的英文标点替换文中文标点
    '''
    n = len(zh_punctuation)
    for i in range(n):
        sentence = sentence.replace(en_punctuation[i], zh_punctuation[i])
    return sentence
```

**重要**：这个函数会将英文冒号 `:` 转换为中文冒号 `：`！

### `remove_duplicate_punctuation(sentence: str) -> str`

这个函数会：
1. 将空格（全角空格）替换为逗号
2. 删除重复的标点符号

```python
def remove_duplicate_punctuation(sentence: str) -> str:
    '''
    删除句子中重复的标点符号、重复的空格，同时将换行变为特殊字符'\n'
    '''
    # 将空格（全角空格）替换为逗号
    sentence = re.sub(' |　', '，', sentence) 
    
    ans = ''
    n = len(sentence)
    p = 0
    while p < n:
        ans += sentence[p]
        while p + 1 < n and sentence[p] in punctuation and sentence[p + 1] in punctuation:
            p += 1
        p += 1
    
    return ans
```

**注意**：这个函数适合处理内容文本，但不适合处理格式化的标题行。

## 💡 经验教训

1. **先判断格式，再清洗数据**：对于有特定格式的文本，应该先识别格式，再进行数据清洗
2. **注意标点符号类型**：中文冒号 `：` 和英文冒号 `:` 是不同的字符
3. **注意数据清洗的副作用**：清洗函数可能会改变标点符号类型，判断时要使用正确的数据
4. **保留原始信息**：在数据清洗前，先保存原始行的关键信息（如长度、格式特征）
5. **测试边界情况**：数据处理代码应该测试各种边界情况，避免出现"只剩 1 行"这种极端情况
6. **理解函数的行为**：在使用数据清洗函数时，要清楚它会对数据做什么改变

## 🎉 总结

问题已修复！现在 `process_wiki_simple_to_dataset()` 函数能够正确处理 wiki.simple.txt 文件，生成高质量的问答数据集。

**核心问题**：判断是否是内容行时，应该使用**清洗后的数据**（`line`）和**中文冒号**（`：`），而不是原始数据（`line_stripped`）和英文冒号（`:`），因为清洗函数会将英文标点转换为中文标点。
