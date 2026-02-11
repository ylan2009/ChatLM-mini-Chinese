# 🚨 dataset_info.json 文件路径错误修复

## ❌ 错误信息

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/dataset_info.json'
ValueError: Cannot open data/dataset_info.json
```

---

## 🔍 问题分析

### 错误原因

**LLaMA-Factory 在错误的位置查找 dataset_info.json！**

```
项目结构:
/data3/ChatLM-mini-Chinese/
├── dataset_info.json          ← 文件在这里
├── llamafactory_config_3x3080.yaml
└── data/
    └── my_train_dataset.parquet

LLaMA-Factory 查找:
/data3/ChatLM-mini-Chinese/data/dataset_info.json  ← 找不到！
```

### 为什么会这样？

LLaMA-Factory 默认配置：
- `dataset_dir` 默认值为 `"data"`
- 会在 `data/` 目录下查找 `dataset_info.json`
- 但你的文件在项目根目录

---

## ✅ 解决方案

### 🎯 方案1: 修改配置文件（推荐）⭐⭐⭐⭐⭐

**在配置文件中指定 dataset_dir 为当前目录**

#### 修改内容

```yaml
# llamafactory_config_3x3080.yaml

# ========== 数据配置 ==========
dataset: custom_t5_dataset
dataset_dir: .  # ← 添加这一行！指定当前目录
template: default
```

#### 为什么推荐？

- ✅ 不需要移动文件
- ✅ 保持项目结构清晰
- ✅ 配置文件和数据定义在同一目录
- ✅ 易于管理和版本控制

---

### 🎯 方案2: 移动文件到 data/ 目录⭐⭐⭐⭐

**将 dataset_info.json 移动到 data/ 目录**

#### 在服务器上执行

```bash
cd /data3/ChatLM-mini-Chinese

# 确保 data 目录存在
mkdir -p data

# 移动文件
mv dataset_info.json data/

# 或者复制（保留原文件）
cp dataset_info.json data/
```

#### 为什么可行？

- ✅ 符合 LLaMA-Factory 默认配置
- ✅ 数据文件集中管理
- ⚠️ 需要移动文件

#### 移动后的结构

```
/data3/ChatLM-mini-Chinese/
├── llamafactory_config_3x3080.yaml
└── data/
    ├── dataset_info.json          ← 移动到这里
    └── my_train_dataset.parquet
```

---

### 🎯 方案3: 使用绝对路径⭐⭐⭐

**在配置文件中使用绝对路径**

```yaml
# llamafactory_config_3x3080.yaml

# ========== 数据配置 ==========
dataset: custom_t5_dataset
dataset_dir: /data3/ChatLM-mini-Chinese  # 绝对路径
template: default
```

#### 为什么不太推荐？

- ⚠️ 不便于移植（路径硬编码）
- ⚠️ 不同机器需要修改配置
- ✅ 但是最明确

---

## 🔧 已修复的配置文件

我已经修复了 `llamafactory_config_3x3080.yaml`，添加了 `dataset_dir: .`：

### 修复内容

```yaml
# 修复前（错误）
# ========== 数据配置 ==========
dataset: custom_t5_dataset
template: default
# ❌ 没有指定 dataset_dir，默认使用 "data"

# 修复后（正确）
# ========== 数据配置 ==========
dataset: custom_t5_dataset
dataset_dir: .  # ✅ 指定当前目录
template: default
```

---

## 🚀 现在可以运行了

### 在服务器上执行

```bash
cd /data3/ChatLM-mini-Chinese

# 重新运行启动脚本
bash run_llamafactory_3x3080.sh

# 选择方式 3（deepspeed）
# 现在应该可以找到 dataset_info.json 了！
```

---

## 📊 LLaMA-Factory 文件查找规则

### dataset_info.json 查找顺序

| 配置 | 查找路径 | 说明 |
|------|---------|------|
| `dataset_dir: .` | `./dataset_info.json` | 当前目录 ✅ |
| `dataset_dir: data` | `data/dataset_info.json` | data 目录（默认）|
| `dataset_dir: /path/to/dir` | `/path/to/dir/dataset_info.json` | 绝对路径 |
| 未指定 | `data/dataset_info.json` | 默认值 |

### 数据文件查找规则

```yaml
# dataset_info.json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet"  # 相对于 dataset_dir
  }
}
```

**完整路径计算：**
```
完整路径 = dataset_dir + "/" + file_name
         = "." + "/" + "data/my_train_dataset.parquet"
         = "./data/my_train_dataset.parquet"
```

---

## 🎯 推荐的项目结构

### 方式1: 配置文件和数据定义在根目录（推荐）⭐⭐⭐⭐⭐

```
/data3/ChatLM-mini-Chinese/
├── llamafactory_config_3x3080.yaml  ← 配置文件
├── dataset_info.json                ← 数据定义
├── ds_config_zero2.json             ← DeepSpeed配置
├── run_llamafactory_3x3080.sh       ← 启动脚本
│
├── data/                            ← 数据目录
│   └── my_train_dataset.parquet
│
├── model_save/                      ← 模型目录
│   └── ChatLM-mini-Chinese/
│
└── logs/                            ← 日志目录
    └── llamafactory_3x3080/
```

**配置：**
```yaml
dataset_dir: .  # 当前目录
```

**优点：**
- ✅ 配置文件集中管理
- ✅ 易于查看和修改
- ✅ 版本控制友好

---

### 方式2: 所有数据文件在 data/ 目录⭐⭐⭐⭐

```
/data3/ChatLM-mini-Chinese/
├── llamafactory_config_3x3080.yaml
├── ds_config_zero2.json
├── run_llamafactory_3x3080.sh
│
├── data/                            ← 所有数据文件
│   ├── dataset_info.json            ← 移动到这里
│   └── my_train_dataset.parquet
│
├── model_save/
│   └── ChatLM-mini-Chinese/
│
└── logs/
    └── llamafactory_3x3080/
```

**配置：**
```yaml
dataset_dir: data  # 或者不指定（默认值）
```

**优点：**
- ✅ 数据文件集中管理
- ✅ 符合 LLaMA-Factory 默认配置
- ⚠️ 需要移动文件

---

## 💡 深入理解

### LLaMA-Factory 数据加载流程

```python
# 1. 读取配置文件
config = yaml.load("llamafactory_config_3x3080.yaml")
dataset_dir = config.get("dataset_dir", "data")  # 默认 "data"

# 2. 查找 dataset_info.json
dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")
# 例如: "." + "/" + "dataset_info.json" = "./dataset_info.json"

# 3. 读取数据集定义
with open(dataset_info_path) as f:
    dataset_info = json.load(f)

# 4. 获取数据文件路径
dataset_name = config["dataset"]  # "custom_t5_dataset"
file_name = dataset_info[dataset_name]["file_name"]  # "data/my_train_dataset.parquet"

# 5. 构建完整路径
data_file_path = os.path.join(dataset_dir, file_name)
# 例如: "." + "/" + "data/my_train_dataset.parquet" = "./data/my_train_dataset.parquet"

# 6. 加载数据
dataset = load_dataset("parquet", data_files=data_file_path)
```

### 路径解析示例

#### 示例1: dataset_dir = "."

```yaml
# 配置文件
dataset_dir: .
dataset: custom_t5_dataset
```

```json
// dataset_info.json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet"
  }
}
```

**路径计算：**
```
dataset_info.json: . + / + dataset_info.json = ./dataset_info.json ✅
数据文件: . + / + data/my_train_dataset.parquet = ./data/my_train_dataset.parquet ✅
```

---

#### 示例2: dataset_dir = "data"

```yaml
# 配置文件
dataset_dir: data
dataset: custom_t5_dataset
```

```json
// dataset_info.json（需要在 data/ 目录下）
{
  "custom_t5_dataset": {
    "file_name": "my_train_dataset.parquet"  # 注意：不需要 "data/" 前缀
  }
}
```

**路径计算：**
```
dataset_info.json: data + / + dataset_info.json = data/dataset_info.json ✅
数据文件: data + / + my_train_dataset.parquet = data/my_train_dataset.parquet ✅
```

---

#### 示例3: 绝对路径

```yaml
# 配置文件
dataset_dir: /data3/ChatLM-mini-Chinese
dataset: custom_t5_dataset
```

```json
// dataset_info.json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet"
  }
}
```

**路径计算：**
```
dataset_info.json: /data3/ChatLM-mini-Chinese/dataset_info.json ✅
数据文件: /data3/ChatLM-mini-Chinese/data/my_train_dataset.parquet ✅
```

---

## 🔍 验证配置

### 检查文件是否存在

```bash
cd /data3/ChatLM-mini-Chinese

# 检查 dataset_info.json
ls -lh dataset_info.json
# 应该输出: -rw-r--r-- 1 rongtw rongtw 188 Feb 10 20:11 dataset_info.json

# 检查数据文件
ls -lh data/my_train_dataset.parquet
# 应该输出: -rw-r--r-- 1 rongtw rongtw XXX Feb XX XX:XX data/my_train_dataset.parquet

# 检查配置文件
grep "dataset_dir" llamafactory_config_3x3080.yaml
# 应该输出: dataset_dir: .
```

### 测试配置

```bash
# 测试 Python 路径解析
python -c "
import os
import json
import yaml

# 读取配置
with open('llamafactory_config_3x3080.yaml') as f:
    config = yaml.safe_load(f)

dataset_dir = config.get('dataset_dir', 'data')
print(f'dataset_dir: {dataset_dir}')

# 检查 dataset_info.json
dataset_info_path = os.path.join(dataset_dir, 'dataset_info.json')
print(f'dataset_info.json 路径: {dataset_info_path}')
print(f'文件存在: {os.path.exists(dataset_info_path)}')

# 读取数据集定义
with open(dataset_info_path) as f:
    dataset_info = json.load(f)

dataset_name = config['dataset']
file_name = dataset_info[dataset_name]['file_name']
data_file_path = os.path.join(dataset_dir, file_name)
print(f'数据文件路径: {data_file_path}')
print(f'文件存在: {os.path.exists(data_file_path)}')
"
```

**预期输出：**
```
dataset_dir: .
dataset_info.json 路径: ./dataset_info.json
文件存在: True
数据文件路径: ./data/my_train_dataset.parquet
文件存在: True
```

---

## 📝 快速参考

### 诊断命令

```bash
# 1. 检查当前目录
pwd
# 应该输出: /data3/ChatLM-mini-Chinese

# 2. 检查文件结构
ls -lh dataset_info.json
ls -lh data/my_train_dataset.parquet

# 3. 检查配置
grep "dataset_dir" llamafactory_config_3x3080.yaml
grep "dataset:" llamafactory_config_3x3080.yaml

# 4. 查看 dataset_info.json 内容
cat dataset_info.json
```

### 修复命令

```bash
# 方案1: 修改配置文件（已完成）
# dataset_dir: . 已添加到配置文件

# 方案2: 移动文件（备选）
mkdir -p data
cp dataset_info.json data/

# 方案3: 使用绝对路径（备选）
# 修改配置文件: dataset_dir: /data3/ChatLM-mini-Chinese
```

---

## 🎉 总结

**问题：** LLaMA-Factory 在 `data/dataset_info.json` 查找文件，但文件在项目根目录

**原因：** 配置文件中未指定 `dataset_dir`，使用了默认值 `"data"`

**解决方案：** 在配置文件中添加 `dataset_dir: .`

**执行命令：**
```bash
cd /data3/ChatLM-mini-Chinese
bash run_llamafactory_3x3080.sh
# 选择方式 3
```

**现在应该可以找到 dataset_info.json 了！** 🎉

---

## 🔗 相关文档

- [FILE_ORGANIZATION.md](FILE_ORGANIZATION.md) - 文件组织说明
- [FIX_NCCL_SHM_ERROR.md](FIX_NCCL_SHM_ERROR.md) - NCCL 共享内存错误
- [FIX_LOCAL_RANK_ARGS.md](FIX_LOCAL_RANK_ARGS.md) - 参数解析错误
- [FIX_RELATIVE_IMPORT.md](FIX_RELATIVE_IMPORT.md) - 相对导入错误
