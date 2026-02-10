# 📁 LLaMA-Factory 文件组织说明

## ✅ 当前文件组织（推荐方式）

**好消息：所有文件都已经在正确的位置，可以直接使用！**

```
/Users/twrong/git/code/ChatLM-mini-Chinese/  ← 你的项目根目录
│
├── 📄 llamafactory_config_3x3080.yaml    ← 主配置文件（在这里！）
├── 📄 ds_config_zero2.json               ← DeepSpeed配置（在这里！）
├── 📄 dataset_info.json                  ← 数据集定义（在这里！）
├── 📄 run_llamafactory_3x3080.sh         ← 启动脚本（在这里！）
│
├── 📂 data/                              ← 数据目录
│   ├── my_train_dataset.parquet          ← 训练数据
│   └── my_valid_dataset.parquet          ← 验证数据（可选）
│
├── 📂 model_save/                        ← 模型目录
│   ├── ChatLM-mini-Chinese/              ← 原始模型
│   └── llamafactory_3x3080_output/       ← 训练输出（自动创建）
│
└── 📂 logs/                              ← 日志目录
    └── llamafactory_3x3080/              ← TensorBoard日志（自动创建）
```

---

## 🎯 为什么不需要放到 LLaMA-Factory 目录？

### LLaMA-Factory 的两种使用方式

#### 方式1: pip 安装（推荐，你正在使用）✅

```bash
# 安装到系统
pip install llmtuner

# 文件组织
~/.local/lib/python3.x/site-packages/llmtuner/  ← LLaMA-Factory安装位置（不用管）
/Users/twrong/git/code/ChatLM-mini-Chinese/     ← 你的项目目录（配置文件在这里）
```

**特点：**
- ✅ 配置文件放在**你的项目目录**
- ✅ 通过命令行参数指定配置文件路径
- ✅ 不需要修改 LLaMA-Factory 的代码
- ✅ 多个项目可以共用一个 LLaMA-Factory 安装

#### 方式2: 源码安装（不推荐，除非要修改源码）

```bash
# 克隆源码
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装
pip install -e .

# 文件组织
LLaMA-Factory/
├── examples/          ← 官方示例配置
├── data/             ← 官方数据集定义
└── src/llmtuner/     ← 源码
```

**特点：**
- ⚠️ 配置文件可以放在 `LLaMA-Factory/examples/` 下
- ⚠️ 数据集定义可以放在 `LLaMA-Factory/data/` 下
- ⚠️ 但这样会污染源码目录，不推荐

---

## 📋 文件说明

### 1. llamafactory_config_3x3080.yaml（主配置文件）

**位置：** 项目根目录  
**作用：** 定义所有训练参数  
**路径引用：**

```yaml
# 使用相对路径（相对于配置文件所在目录）
model_name_or_path: ./model_save/ChatLM-mini-Chinese/
output_dir: ./model_save/llamafactory_3x3080_output
logging_dir: ./logs/llamafactory_3x3080
deepspeed: ds_config_zero2.json  # 相对路径
```

**启动时：**
```bash
# 在项目根目录执行
cd /Users/twrong/git/code/ChatLM-mini-Chinese
llamafactory-cli train llamafactory_config_3x3080.yaml
```

---

### 2. ds_config_zero2.json（DeepSpeed配置）

**位置：** 项目根目录（与主配置文件同目录）  
**作用：** DeepSpeed ZeRO-2 优化配置  
**引用方式：**

```yaml
# 在 llamafactory_config_3x3080.yaml 中引用
deepspeed: ds_config_zero2.json  # 相对路径
```

**也可以使用绝对路径：**
```yaml
deepspeed: /Users/twrong/git/code/ChatLM-mini-Chinese/ds_config_zero2.json
```

---

### 3. dataset_info.json（数据集定义）

**位置：** 项目根目录  
**作用：** 定义数据集的格式和位置  
**查找顺序：**

LLaMA-Factory 会按以下顺序查找 `dataset_info.json`：

1. **当前工作目录**（优先级最高）
2. `~/.cache/huggingface/datasets/`
3. LLaMA-Factory 安装目录的 `data/` 目录

**内容：**
```json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet",  // 相对路径
    "file_format": "parquet",
    "columns": {
      "prompt": "input",
      "response": "target"
    }
  }
}
```

**也可以使用绝对路径：**
```json
{
  "custom_t5_dataset": {
    "file_name": "/Users/twrong/git/code/ChatLM-mini-Chinese/data/my_train_dataset.parquet"
  }
}
```

---

### 4. run_llamafactory_3x3080.sh（启动脚本）

**位置：** 项目根目录  
**作用：** 自动化启动训练  
**使用：**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese
bash run_llamafactory_3x3080.sh
```

---

## 🚀 启动方式对比

### 方式1: 在项目目录启动（推荐）✅

```bash
# 进入项目目录
cd /Users/twrong/git/code/ChatLM-mini-Chinese

# 方式A: 使用脚本
bash run_llamafactory_3x3080.sh

# 方式B: 使用命令行
export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed --num_gpus=3 -m llmtuner.cli train llamafactory_config_3x3080.yaml
```

**优点：**
- ✅ 所有路径都是相对路径，清晰明了
- ✅ 配置文件和数据在一起，便于管理
- ✅ 可以版本控制（git）
- ✅ 多个项目互不干扰

---

### 方式2: 在任意目录启动（使用绝对路径）

```bash
# 在任意目录
cd ~

# 使用绝对路径
export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed --num_gpus=3 -m llmtuner.cli train \
    /Users/twrong/git/code/ChatLM-mini-Chinese/llamafactory_config_3x3080.yaml
```

**注意：** 如果使用绝对路径启动，配置文件中的相对路径会相对于**当前工作目录**，可能导致找不到文件。

**解决方案：** 配置文件中也使用绝对路径

---

## 🔧 路径配置最佳实践

### 推荐配置（相对路径）✅

```yaml
# llamafactory_config_3x3080.yaml
model_name_or_path: ./model_save/ChatLM-mini-Chinese/
output_dir: ./model_save/llamafactory_3x3080_output
logging_dir: ./logs/llamafactory_3x3080
deepspeed: ds_config_zero2.json
```

```json
// dataset_info.json
{
  "custom_t5_dataset": {
    "file_name": "data/my_train_dataset.parquet"
  }
}
```

**启动：**
```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese
llamafactory-cli train llamafactory_config_3x3080.yaml
```

---

### 备选配置（绝对路径）

```yaml
# llamafactory_config_3x3080.yaml
model_name_or_path: /Users/twrong/git/code/ChatLM-mini-Chinese/model_save/ChatLM-mini-Chinese/
output_dir: /Users/twrong/git/code/ChatLM-mini-Chinese/model_save/llamafactory_3x3080_output
logging_dir: /Users/twrong/git/code/ChatLM-mini-Chinese/logs/llamafactory_3x3080
deepspeed: /Users/twrong/git/code/ChatLM-mini-Chinese/ds_config_zero2.json
```

```json
// dataset_info.json
{
  "custom_t5_dataset": {
    "file_name": "/Users/twrong/git/code/ChatLM-mini-Chinese/data/my_train_dataset.parquet"
  }
}
```

**启动：**
```bash
# 可以在任意目录
llamafactory-cli train /Users/twrong/git/code/ChatLM-mini-Chinese/llamafactory_config_3x3080.yaml
```

---

## 📝 常见问题

### Q1: 必须把配置文件放到 LLaMA-Factory 目录吗？

**答：不需要！** 

LLaMA-Factory 是一个 Python 包，安装后可以在任何地方使用。配置文件放在你的项目目录即可。

---

### Q2: dataset_info.json 必须在特定位置吗？

**答：不是必须，但有查找顺序。**

LLaMA-Factory 会按以下顺序查找：
1. **当前工作目录**（推荐放这里）
2. `~/.cache/huggingface/datasets/`
3. LLaMA-Factory 安装目录

**最佳实践：** 放在项目根目录，启动时在项目根目录执行命令。

---

### Q3: 可以把配置文件放到子目录吗？

**答：可以，但要注意相对路径。**

```bash
# 目录结构
/Users/twrong/git/code/ChatLM-mini-Chinese/
├── configs/
│   └── llamafactory_config_3x3080.yaml
├── data/
└── model_save/

# 启动时指定完整路径
cd /Users/twrong/git/code/ChatLM-mini-Chinese
llamafactory-cli train configs/llamafactory_config_3x3080.yaml

# 配置文件中的相对路径要调整
model_name_or_path: ../model_save/ChatLM-mini-Chinese/  # 注意 ../
```

---

### Q4: 如何验证路径配置是否正确？

**方法1: 检查文件是否存在**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese

# 检查配置文件
ls -lh llamafactory_config_3x3080.yaml
ls -lh ds_config_zero2.json
ls -lh dataset_info.json

# 检查数据文件
ls -lh data/my_train_dataset.parquet

# 检查模型文件
ls -lh model_save/ChatLM-mini-Chinese/
```

**方法2: 试运行（dry-run）**

```bash
# 使用 --help 检查配置是否能被正确解析
llamafactory-cli train llamafactory_config_3x3080.yaml --help
```

---

## 💡 总结

### ✅ 当前配置（完全正确）

```
你的项目目录: /Users/twrong/git/code/ChatLM-mini-Chinese/
├── llamafactory_config_3x3080.yaml  ✅ 在这里
├── ds_config_zero2.json             ✅ 在这里
├── dataset_info.json                ✅ 在这里
├── run_llamafactory_3x3080.sh       ✅ 在这里
├── data/                            ✅ 在这里
└── model_save/                      ✅ 在这里
```

### 🚀 启动命令（在项目根目录执行）

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese
bash run_llamafactory_3x3080.sh
```

**或者：**

```bash
cd /Users/twrong/git/code/ChatLM-mini-Chinese
export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed --num_gpus=3 -m llmtuner.cli train llamafactory_config_3x3080.yaml
```

---

**结论：文件已经在正确的位置，不需要移动！直接使用即可！** 🎉
