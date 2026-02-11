# 🚨 数据集列名不匹配错误修复

## ❌ 错误信息

```python
KeyError: 'input'

File "/home/rongtw/anaconda3/envs/chatlm/lib/python3.10/site-packages/llmtuner/data/aligner.py", line 34, in convert_alpaca
    for i in range(len(examples[dataset_attr.prompt])):
KeyError: 'input'
```

---

## 🔍 问题分析

### 错误原因

**数据集列名不匹配！**

LLaMA-Factory 根据 `dataset_info.json` 的配置去查找列名，但是 parquet 文件中没有对应的列！

### 当前配置

```json
// dataset_info.json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "columns": {
      "prompt": "input",    ← LLaMA-Factory 去找 "input" 列
      "response": "target"  ← LLaMA-Factory 去找 "target" 列
    }
  }
}
```

### 问题

**parquet 文件中可能没有 "input" 和 "target" 列！**

可能的情况：
1. 列名不是 "input" 和 "target"
2. 列名大小写不同（例如 "Input" vs "input"）
3. 列名是其他名称（例如 "text", "prompt", "question" 等）

---

## 🔧 解决步骤

### 步骤1: 检查 parquet 文件的实际列名

**在服务器上执行：**

```bash
cd /data3/ChatLM-mini-Chinese

# 运行检查脚本
python check_parquet_columns.py data/my_train_dataset.parquet
```

**这个脚本会显示：**
- ✅ 文件的所有列名
- ✅ 每列的数据类型
- ✅ 前3行示例数据
- ✅ 空值统计
- ✅ 自动生成正确的 dataset_info.json 配置

---

### 步骤2: 根据实际列名修改 dataset_info.json

#### 情况1: 列名是 "text" 和 "summary"

如果 parquet 文件的列名是：
```
text, summary
```

则修改 `dataset_info.json` 为：

```json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "text",      ← 改为实际的列名
      "response": "summary"  ← 改为实际的列名
    }
  }
}
```

---

#### 情况2: 列名是 "question" 和 "answer"

如果 parquet 文件的列名是：
```
question, answer
```

则修改 `dataset_info.json` 为：

```json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "question",  ← 改为实际的列名
      "response": "answer"   ← 改为实际的列名
    }
  }
}
```

---

#### 情况3: 列名就是 "input" 和 "target"（但大小写不同）

如果 parquet 文件的列名是：
```
Input, Target  （注意大写）
```

则修改 `dataset_info.json` 为：

```json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "Input",   ← 注意大小写
      "response": "Target" ← 注意大小写
    }
  }
}
```

---

#### 情况4: 只有一列（纯文本预训练）

如果 parquet 文件只有一列：
```
text
```

则修改 `dataset_info.json` 为：

```json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "text"  ← 只指定 prompt，不指定 response
    }
  }
}
```

**并且修改配置文件：**

```yaml
# llamafactory_config_3x3080.yaml
stage: pt  # 预训练模式
template: default
```

---

### 步骤3: 重新运行训练

```bash
cd /data3/ChatLM-mini-Chinese

# 重新运行
bash run_llamafactory_3x3080.sh

# 选择方式 3
```

---

## 📊 LLaMA-Factory 列名映射规则

### columns 配置说明

```json
{
  "columns": {
    "prompt": "实际列名1",    ← LLaMA-Factory 的字段名 → parquet 的实际列名
    "response": "实际列名2",  ← LLaMA-Factory 的字段名 → parquet 的实际列名
    "query": "实际列名3",     ← 可选：对话历史
    "history": "实际列名4"    ← 可选：多轮对话
  }
}
```

### LLaMA-Factory 支持的字段

| LLaMA-Factory 字段 | 说明 | 必需 | 示例 |
|-------------------|------|------|------|
| `prompt` | 输入文本/问题 | ✅ 是 | "请介绍一下北京" |
| `response` | 输出文本/答案 | ⚠️ 监督学习必需 | "北京是中国的首都..." |
| `query` | 当前问题 | ❌ 否 | "天气怎么样？" |
| `history` | 对话历史 | ❌ 否 | [["你好", "你好！"]] |
| `system` | 系统提示 | ❌ 否 | "你是一个助手" |

### 不同训练阶段的要求

| 训练阶段 | stage | 必需字段 | 说明 |
|---------|-------|---------|------|
| 预训练 | `pt` | `prompt` | 只需要文本，不需要 response |
| 监督微调 | `sft` | `prompt`, `response` | 需要输入和输出 |
| 奖励模型 | `rm` | `prompt`, `response` | 需要输入和输出 |
| 强化学习 | `ppo` | `prompt` | 只需要输入 |

---

## 🔍 常见列名对应关系

### 常见的输入列名

| 实际列名 | 对应配置 |
|---------|---------|
| `input` | `"prompt": "input"` |
| `text` | `"prompt": "text"` |
| `prompt` | `"prompt": "prompt"` |
| `question` | `"prompt": "question"` |
| `instruction` | `"prompt": "instruction"` |
| `query` | `"prompt": "query"` |
| `content` | `"prompt": "content"` |

### 常见的输出列名

| 实际列名 | 对应配置 |
|---------|---------|
| `target` | `"response": "target"` |
| `output` | `"response": "output"` |
| `response` | `"response": "response"` |
| `answer` | `"response": "answer"` |
| `completion` | `"response": "completion"` |
| `summary` | `"response": "summary"` |
| `label` | `"response": "label"` |

---

## 🛠️ 手动检查 parquet 文件

### 方法1: 使用 Python

```bash
cd /data3/ChatLM-mini-Chinese

python -c "
import pandas as pd

# 读取文件
df = pd.read_parquet('data/my_train_dataset.parquet')

# 显示列名
print('列名:', df.columns.tolist())

# 显示前3行
print('\n前3行:')
print(df.head(3))
"
```

---

### 方法2: 使用 check_parquet_columns.py（推荐）

```bash
cd /data3/ChatLM-mini-Chinese

# 运行检查脚本
python check_parquet_columns.py data/my_train_dataset.parquet
```

**输出示例：**

```
正在检查文件: data/my_train_dataset.parquet
================================================================================

✓ 文件读取成功！
  总行数: 8,813,083
  总列数: 2

📋 列名列表:
  1. text
  2. summary

📊 数据类型:
  text: object
  summary: object

📝 前3行数据:

第 1 行:
  text: 这是一段输入文本...
  summary: 这是对应的摘要...

...

💡 dataset_info.json 配置建议:
--------------------------------------------------------------------------------

{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "text",
      "response": "summary"
    }
  }
}

✓ 自动识别到:
  - prompt 列: text
  - response 列: summary
================================================================================
```

---

## 📝 修复示例

### 示例1: 列名是 "text" 和 "summary"

#### 检查结果

```bash
$ python check_parquet_columns.py data/my_train_dataset.parquet

列名: ['text', 'summary']
```

#### 修复 dataset_info.json

```json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "text",      ← 改为 "text"
      "response": "summary"  ← 改为 "summary"
    }
  }
}
```

#### 修复命令

```bash
cd /data3/ChatLM-mini-Chinese

# 备份原文件
cp dataset_info.json dataset_info.json.bak

# 修改文件
cat > dataset_info.json << 'EOF'
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "text",
      "response": "summary"
    }
  }
}
EOF

# 验证
cat dataset_info.json
```

---

### 示例2: 只有一列 "text"（预训练）

#### 检查结果

```bash
$ python check_parquet_columns.py data/my_train_dataset.parquet

列名: ['text']
```

#### 修复 dataset_info.json

```json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "text"  ← 只有一列
    }
  }
}
```

#### 修复命令

```bash
cd /data3/ChatLM-mini-Chinese

# 修改 dataset_info.json
cat > dataset_info.json << 'EOF'
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "text"
    }
  }
}
EOF

# 确认配置文件是预训练模式
grep "stage:" llamafactory_config_3x3080.yaml
# 应该输出: stage: pt
```

---

## 🎯 快速修复流程

### 1️⃣ 检查列名

```bash
cd /data3/ChatLM-mini-Chinese
python check_parquet_columns.py data/my_train_dataset.parquet
```

### 2️⃣ 记录实际列名

假设输出是：
```
列名: ['text', 'summary']
```

### 3️⃣ 修改 dataset_info.json

```bash
cat > dataset_info.json << 'EOF'
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {
      "prompt": "text",
      "response": "summary"
    }
  }
}
EOF
```

### 4️⃣ 验证修改

```bash
cat dataset_info.json
```

### 5️⃣ 重新运行训练

```bash
bash run_llamafactory_3x3080.sh
# 选择方式 3
```

---

## 💡 深入理解

### LLaMA-Factory 数据加载流程

```python
# 1. 读取 dataset_info.json
dataset_info = {
    "custom_t5_dataset": {
        "columns": {
            "prompt": "input",    # LLaMA-Factory 字段 → parquet 列名
            "response": "target"
        }
    }
}

# 2. 加载 parquet 文件
df = pd.read_parquet("data/my_train_dataset.parquet")
# df.columns = ['text', 'summary']  ← 实际列名

# 3. 尝试访问列（这里会出错！）
prompt_column = dataset_info["columns"]["prompt"]  # "input"
examples[prompt_column]  # 尝试访问 examples["input"]
# ❌ KeyError: 'input'  因为实际列名是 'text'，不是 'input'

# 4. 正确的配置应该是
dataset_info = {
    "custom_t5_dataset": {
        "columns": {
            "prompt": "text",     # ✅ 使用实际列名
            "response": "summary"
        }
    }
}
```

### 列名映射原理

```
LLaMA-Factory 内部字段 → dataset_info.json 映射 → parquet 实际列名

prompt                  → "prompt": "text"      → df["text"]
response                → "response": "summary" → df["summary"]
```

**关键点：**
- `"prompt": "text"` 表示：LLaMA-Factory 的 `prompt` 字段对应 parquet 的 `text` 列
- `"response": "summary"` 表示：LLaMA-Factory 的 `response` 字段对应 parquet 的 `summary` 列

---

## 🔍 调试技巧

### 1. 查看详细错误信息

```bash
# 运行时添加调试信息
export PYTHONPATH=/data3/ChatLM-mini-Chinese:$PYTHONPATH
export DATASETS_VERBOSITY=debug

bash run_llamafactory_3x3080.sh
```

### 2. 测试数据加载

```bash
cd /data3/ChatLM-mini-Chinese

python -c "
import json
import pandas as pd

# 读取配置
with open('dataset_info.json') as f:
    config = json.load(f)

dataset_config = config['custom_t5_dataset']
file_name = dataset_config['file_name']
columns = dataset_config['columns']

print(f'配置的列名映射:')
for k, v in columns.items():
    print(f'  {k} → {v}')

# 读取数据
df = pd.read_parquet(file_name)
print(f'\n实际的列名:')
for col in df.columns:
    print(f'  {col}')

# 检查映射是否正确
print(f'\n映射检查:')
for k, v in columns.items():
    if v in df.columns:
        print(f'  ✓ {k} → {v} (存在)')
    else:
        print(f'  ✗ {k} → {v} (不存在！)')
        print(f'    可用列名: {list(df.columns)}')
"
```

---

## 📚 相关文档

- [check_parquet_columns.py](check_parquet_columns.py) - 列名检查脚本
- [FIX_DATASET_PATH.md](FIX_DATASET_PATH.md) - 数据集路径错误
- [FIX_NCCL_SHM_ERROR.md](FIX_NCCL_SHM_ERROR.md) - NCCL 错误
- [llamafactory_config_3x3080.yaml](llamafactory_config_3x3080.yaml) - 训练配置

---

## 🎉 总结

**问题：** LLaMA-Factory 找不到 "input" 列

**原因：** dataset_info.json 配置的列名与 parquet 文件实际列名不匹配

**解决方案：**
1. 运行 `python check_parquet_columns.py data/my_train_dataset.parquet`
2. 查看实际列名
3. 修改 `dataset_info.json` 中的 `columns` 配置
4. 重新运行训练

**关键命令：**
```bash
cd /data3/ChatLM-mini-Chinese
python check_parquet_columns.py data/my_train_dataset.parquet
# 根据输出修改 dataset_info.json
bash run_llamafactory_3x3080.sh
```
