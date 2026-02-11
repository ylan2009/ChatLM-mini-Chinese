# 🚨 DeepSpeed 启动错误修复

## ❌ 错误信息

```bash
deepspeed: error: unrecognized arguments: -m
```

---

## 🔍 问题分析

### 错误原因

**deepspeed 和 torchrun 不支持 `-m` 参数！**

| 启动器 | 支持 `-m` | 正确用法 |
|--------|----------|---------|
| `python` | ✅ | `python -m llmtuner.cli` |
| `accelerate launch` | ✅ | `accelerate launch -m llmtuner.cli` |
| `deepspeed` | ❌ | `deepspeed script.py` |
| `torchrun` | ❌ | `torchrun script.py` |

### 原因说明

- `python -m module` 是 Python 的模块运行方式
- `deepspeed` 和 `torchrun` 是**启动器**，不是 Python 解释器
- 它们需要**直接的 Python 脚本路径**，而不是模块名

---

## ✅ 解决方案

### 🎯 方案1: 使用 llamafactory-cli（推荐）⭐⭐⭐

**最简单的方式：**

```bash
# 直接使用 llamafactory-cli
llamafactory-cli train llamafactory_config_3x3080.yaml
```

**优点：**
- ✅ 自动处理 DeepSpeed 配置
- ✅ 不需要关心脚本路径
- ✅ 最简单，最不容易出错

---

### 🎯 方案2: 找到脚本路径并使用 deepspeed

**正确的 deepspeed 启动方式：**

```bash
# 1. 找到 llmtuner 的 cli.py 路径
LLMTUNER_CLI=$(python -c "import llmtuner; import os; print(os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py'))")

# 2. 使用 deepspeed 启动（不要用 -m）
deepspeed \
    --num_gpus=3 \
    --master_port=29500 \
    "$LLMTUNER_CLI" train llamafactory_config_3x3080.yaml
```

**注意：**
- ❌ 错误：`deepspeed -m llmtuner.cli`
- ✅ 正确：`deepspeed /path/to/cli.py`

---

### 🎯 方案3: 使用 accelerate launch（支持 -m）

**accelerate 支持 `-m` 参数：**

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=3 \
    --main_process_port=29500 \
    -m llmtuner.cli train llamafactory_config_3x3080.yaml
```

---

### 🎯 方案4: 使用 torchrun（需要脚本路径）

**正确的 torchrun 启动方式：**

```bash
# 1. 找到脚本路径
LLMTUNER_CLI=$(python -c "import llmtuner; import os; print(os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py'))")

# 2. 使用 torchrun 启动（不要用 -m）
torchrun \
    --nproc_per_node=3 \
    --master_port=29500 \
    "$LLMTUNER_CLI" train llamafactory_config_3x3080.yaml
```

---

## 🔧 已修复的启动脚本

我已经修复了 `run_llamafactory_3x3080.sh`，现在可以正确使用所有启动方式：

### 修复内容

#### 1. DeepSpeed 启动（方式3）

```bash
# 修复前（错误）
deepspeed --num_gpus=3 -m llmtuner.cli train config.yaml

# 修复后（正确）
LLMTUNER_CLI=$(python -c "import llmtuner; import os; print(os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py'))")
deepspeed --num_gpus=3 "$LLMTUNER_CLI" train config.yaml
```

#### 2. Torchrun 启动（方式4）

```bash
# 修复前（错误）
torchrun --nproc_per_node=3 -m llmtuner.cli train config.yaml

# 修复后（正确）
LLMTUNER_CLI=$(python -c "import llmtuner; import os; print(os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py'))")
torchrun --nproc_per_node=3 "$LLMTUNER_CLI" train config.yaml
```

#### 3. 环境变量修复

```bash
# 修复前（已弃用）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 修复后（新版本）
export PYTORCH_ALLOC_CONF=max_split_size_mb:128
```

---

## 🚀 现在可以运行了

### Step 1: 重新运行启动脚本

```bash
cd /data3/ChatLM-mini-Chinese
bash run_llamafactory_3x3080.sh
```

### Step 2: 选择训练方式

```
请选择训练方式:
  1) 使用 llamafactory-cli (推荐，最简单)
  2) 使用 accelerate launch (更灵活)
  3) 使用 deepspeed (最优显存利用)      ← 现在可以正常工作了！
  4) 使用 torchrun (标准DDP)            ← 现在可以正常工作了！

请输入选项 [1-4]: 3
```

### Step 3: 开始训练

脚本会自动：
1. ✅ 找到正确的脚本路径
2. ✅ 使用正确的启动命令
3. ✅ 开始训练

---

## 📊 各种启动方式对比

| 启动方式 | 命令格式 | 优点 | 缺点 | 推荐度 |
|---------|---------|------|------|--------|
| **llamafactory-cli** | `llamafactory-cli train config.yaml` | 最简单，自动处理一切 | 灵活性较低 | ⭐⭐⭐⭐⭐ |
| **accelerate** | `accelerate launch -m llmtuner.cli` | 灵活，支持多种配置 | 需要配置 | ⭐⭐⭐⭐ |
| **deepspeed** | `deepspeed script.py` | 最优显存利用 | 需要脚本路径 | ⭐⭐⭐⭐ |
| **torchrun** | `torchrun script.py` | 标准DDP | 需要脚本路径 | ⭐⭐⭐ |

---

## 💡 为什么会出现这个问题？

### 原因1: 混淆了 Python 和启动器

```bash
# Python 解释器（支持 -m）
python -m module_name  # ✅ 正确

# 启动器（不支持 -m）
deepspeed -m module_name  # ❌ 错误
torchrun -m module_name   # ❌ 错误
```

### 原因2: 文档示例不统一

不同的文档可能使用不同的启动方式，容易混淆：

```bash
# 有些文档这样写（适用于 accelerate）
accelerate launch -m llmtuner.cli

# 有些文档这样写（适用于 deepspeed）
deepspeed /path/to/script.py

# 导致用户混淆
```

---

## 🔒 防止再次出现

### 方法1: 优先使用 llamafactory-cli

```bash
# 最简单，不会出错
llamafactory-cli train config.yaml
```

### 方法2: 记住启动器的特性

| 启动器 | 支持 `-m` | 需要脚本路径 |
|--------|----------|-------------|
| `python` | ✅ | ❌ |
| `accelerate launch` | ✅ | ❌ |
| `deepspeed` | ❌ | ✅ |
| `torchrun` | ❌ | ✅ |

### 方法3: 使用封装脚本

使用 `run_llamafactory_3x3080.sh`，它已经处理了所有细节。

---

## 📝 快速参考

### 找到 llmtuner 脚本路径

```bash
python -c "import llmtuner; import os; print(os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py'))"
```

### 正确的启动命令

```bash
# 方式1: llamafactory-cli（推荐）
llamafactory-cli train config.yaml

# 方式2: accelerate（支持 -m）
accelerate launch -m llmtuner.cli train config.yaml

# 方式3: deepspeed（需要脚本路径）
SCRIPT=$(python -c "import llmtuner; import os; print(os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py'))")
deepspeed --num_gpus=3 "$SCRIPT" train config.yaml

# 方式4: torchrun（需要脚本路径）
SCRIPT=$(python -c "import llmtuner; import os; print(os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py'))")
torchrun --nproc_per_node=3 "$SCRIPT" train config.yaml
```

---

## 🎯 总结

**问题：** deepspeed 不支持 `-m` 参数

**解决方案：**
1. 使用 `llamafactory-cli`（最简单）
2. 或者找到脚本路径，使用 `deepspeed script.py`

**执行命令：**
```bash
cd /data3/ChatLM-mini-Chinese
bash run_llamafactory_3x3080.sh
# 选择方式 3（deepspeed）
```

**现在应该可以正常工作了！** 🎉

---

## 📞 还有问题？

如果还有错误，请提供：
1. 完整的错误信息
2. 使用的启动方式（1-4）
3. Python 和 PyTorch 版本

```bash
# 收集诊断信息
python -c "
import sys
import torch
import transformers
import llmtuner
import os

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('transformers:', transformers.__version__)
print('CUDA:', torch.version.cuda)
print()
print('llmtuner 路径:', llmtuner.__file__)
print('cli.py 路径:', os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py'))
print('cli.py 存在:', os.path.exists(os.path.join(os.path.dirname(llmtuner.__file__), 'cli.py')))
"
```
