# 数据管道诊断工具使用说明

## 📋 工具简介

`check_data_pipeline.py` 是一个全面的数据质量诊断工具，用于检查数据处理流程中每一步的数据质量，特别是检查 prompt 为空的问题。

## 🚀 使用方法

### 1. 检查整个数据处理管道（推荐）

```bash
cd /data3/ChatLM-mini-Chinese/pretrain

# 检查整个管道
python check_data_pipeline.py --pipeline

# 或者简写（默认行为）
python check_data_pipeline.py
```

**功能**：
- 自动检查数据处理流程中的所有关键文件
- 对比每一步的数据质量变化
- 找出问题出现在哪一步

**检查的文件**：
1. 原始 Belle 数据：`data/raw_data/belle/Belle_open_source_0.5M.parquet`
2. 去重后的数据：`data/my_dataset_no_dulpticates.parquet`
3. 处理后的微调数据：`data/my_finetune_data_zh.parquet`
4. Shuffle 后的数据：`data/my_finetune_data_zh_shuffled.parquet`

---

### 2. 检查单个文件

```bash
# 检查特定文件
python check_data_pipeline.py --file /data3/ChatLM-mini-Chinese/data/my_finetune_data_zh.parquet

# 显示更多样例
python check_data_pipeline.py --file /path/to/file.parquet --samples 10
```

**功能**：
- 详细分析单个文件的数据质量
- 显示空值样例和有效数据样例
- 统计长度分布

---

### 3. 指定数据目录

```bash
# 如果数据在其他目录
python check_data_pipeline.py --pipeline --data-dir /path/to/data
```

---

## 📊 输出报告说明

### 1. 基本统计

```
📈 基本统计:
  总行数: 1,761,447
  有效数据 (prompt 和 response 都不为空): 1,700,000 (96.51%)
```

- **总行数**：文件中的总数据条数
- **有效数据**：prompt 和 response 都不为空的数据

---

### 2. 空值统计

```
🔍 空值统计:
  空 prompt: 61,447 (3.49%)
    - None 值: 50,000
    - 仅空白字符: 11,447
  空 response: 0 (0.00%)
    - None 值: 0
    - 仅空白字符: 0
  两者都为空: 0 (0.00%)
```

- **空 prompt/response**：为空或仅包含空白字符的数据
- **None 值**：字段值为 None 的数据
- **仅空白字符**：只包含空格、制表符、换行符的数据

---

### 3. 长度统计

```
📏 Prompt 长度统计:
  平均长度: 45.23
  最小长度: 0
  最大长度: 320
```

---

### 4. 数据质量评级

```
⭐ 数据质量评级:
  ✅ 优秀 (96.51% 有效数据)
```

**评级标准**：
- ✅ 优秀：≥95% 有效数据
- ⚠️ 良好：80-95% 有效数据
- ⚠️ 一般：50-80% 有效数据
- ❌ 较差：<50% 有效数据

---

### 5. 管道对比报告

```
步骤                           总行数          有效数据         空Prompt       空Response
----------------------------------------------------------------------------------------------------
原始 Belle 数据                500,000    480,000 (96.0%)   20,000 (4.0%)      0 (0.0%)
去重后的数据                   450,000    432,000 (96.0%)   18,000 (4.0%)      0 (0.0%)
处理后的微调数据             1,761,447          0 (0.0%) 1,761,447 (100.0%)   0 (0.0%)  ❌
Shuffle 后的数据             1,761,447          0 (0.0%) 1,761,447 (100.0%)   0 (0.0%)
```

**如何解读**：
- 如果某一步的空 prompt 比例突然增加，说明这一步有问题
- 上面的例子中，"处理后的微调数据" 这一步出现了问题（100% 空 prompt）

---

### 6. 问题分析

```
🔍 问题分析:

❌ 严重问题: '处理后的微调数据' 中有 100.0% 的数据 prompt 为空！
   文件路径: /data3/ChatLM-mini-Chinese/data/my_finetune_data_zh.parquet
   这一步可能存在问题！
```

工具会自动分析并指出问题所在。

---

## 🔍 典型使用场景

### 场景 1: 发现 JSON 转换失败（0 条数据）

```bash
# 1. 检查整个管道
python check_data_pipeline.py --pipeline

# 2. 查看对比报告，找出哪一步开始出现空 prompt

# 3. 检查具体的问题文件
python check_data_pipeline.py --file /data3/ChatLM-mini-Chinese/data/my_finetune_data_zh.parquet --samples 10
```

---

### 场景 2: 验证修复效果

```bash
# 修复代码后，重新运行数据处理
cd /data3/ChatLM-mini-Chinese/pretrain
python download_and_process_datasets.py --process

# 检查修复效果
python check_data_pipeline.py --pipeline
```

**预期结果**：
- 所有步骤的有效数据比例应该 >95%
- 不应该有突然增加的空 prompt

---

### 场景 3: 检查原始数据质量

```bash
# 检查从 Hugging Face 下载的原始数据
python check_data_pipeline.py --file /data3/ChatLM-mini-Chinese/data/raw_data/belle/Belle_open_source_0.5M.parquet
```

---

## 📝 样例输出

### 空 Prompt 样例

```
🔍 空 Prompt 样例 (前 3 条):

第 7 行:
  Prompt (is_none=False): ''
  Response: 以下是给出该文本中所有名词的列表：

第 8 行:
  Prompt (is_none=False): ''
  Response: 人名：李华

第 10 行:
  Prompt (is_none=True): 'None'
  Response: 这些物品可以被分为以下几个类别：
```

**解读**：
- `is_none=True`：字段值为 None
- `is_none=False`：字段值为空字符串 `''`

---

### 有效数据样例

```
✅ 有效数据样例 (前 3 条):

第 1 行:
  Prompt: 解释什么是人工智能
  Response: 人工智能（Artificial Intelligence, AI）是计算机科学的一个分支...

第 2 行:
  Prompt: 如何学习 Python？
  Response: 学习 Python 可以按照以下步骤进行：1. 安装 Python...
```

---

## 🎯 常见问题诊断

### 问题 1: 所有 prompt 都是空的

**症状**：
```
空 prompt: 1,761,447 (100.0%)
```

**可能原因**：
1. 列名映射错误（读取了错误的列）
2. 数据处理函数有 bug（没有正确写入 prompt）
3. 过滤函数没有检查空值

**诊断方法**：
```bash
# 检查整个管道，找出哪一步开始出现问题
python check_data_pipeline.py --pipeline
```

---

### 问题 2: 部分 prompt 为空

**症状**：
```
空 prompt: 61,447 (3.49%)
```

**可能原因**：
1. 源数据质量问题（原始数据就有空值）
2. 过滤函数没有检查空值

**诊断方法**：
```bash
# 检查原始数据
python check_data_pipeline.py --file /data3/ChatLM-mini-Chinese/data/raw_data/belle/Belle_open_source_0.5M.parquet

# 如果原始数据就有空值，说明是源数据问题
# 如果原始数据没有空值，说明是处理过程引入的问题
```

---

### 问题 3: 某一步突然增加空值

**症状**：
```
⚠️  警告: '处理后的微调数据' 中空 prompt 比例增加了 96.0%
   从 4.0% 增加到 100.0%
```

**可能原因**：
- 这一步的处理函数有 bug

**诊断方法**：
1. 检查这一步的代码逻辑
2. 检查列名映射是否正确
3. 检查是否正确读取和写入数据

---

## 🛠️ 高级用法

### 自定义检查文件列表

如果你想检查其他文件，可以修改脚本中的 `pipeline_files` 列表：

```python
pipeline_files = [
    {
        'name': '你的数据名称',
        'path': f'{data_dir}/your_file.parquet',
        'description': '数据说明'
    },
    # ... 添加更多文件
]
```

---

### 集成到自动化流程

```bash
#!/bin/bash

# 数据处理脚本
python download_and_process_datasets.py --process

# 自动检查数据质量
python check_data_pipeline.py --pipeline > data_quality_report.txt

# 检查是否有严重问题
if grep -q "严重问题" data_quality_report.txt; then
    echo "数据质量检查失败！"
    exit 1
fi

echo "数据质量检查通过！"
```

---

## 📌 注意事项

1. **大文件处理**：对于大文件（>1GB），扫描可能需要几分钟时间
2. **内存使用**：工具使用流式读取，内存占用很小
3. **列名识别**：工具会自动识别常见的列名（prompt/instruction/input 等）
4. **None vs 空字符串**：工具会区分 None 值和空字符串

---

## 🎯 总结

这个工具可以帮助你：
- ✅ 快速定位数据质量问题
- ✅ 找出问题出现在哪一步
- ✅ 验证修复效果
- ✅ 监控数据处理管道的健康状态

**推荐工作流程**：
1. 运行 `python check_data_pipeline.py --pipeline` 检查整个管道
2. 查看对比报告，找出问题步骤
3. 检查该步骤的代码逻辑
4. 修复问题后重新运行
5. 再次检查验证修复效果
