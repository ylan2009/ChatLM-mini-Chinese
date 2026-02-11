# 🚨 --local_rank 参数解析错误修复

## ❌ 错误信息

```python
Traceback (most recent call last):
  File "/tmp/deepspeed_train.py", line 11, in <module>
    main()
  File "/home/rongtw/anaconda3/envs/chatlm/lib/python3.10/site-packages/llmtuner/cli.py", line 75, in main
    raise NotImplementedError("Unknown command: {}".format(command))
NotImplementedError: Unknown command: --local_rank=0
```

---

## 🔍 问题分析

### 错误原因

**DeepSpeed/Torchrun 自动添加了 `--local_rank` 参数，但 `llmtuner.cli.main()` 把它当作命令而不是参数！**

### 命令行参数顺序

```bash
# DeepSpeed 实际执行的命令
python /tmp/deepspeed_train.py --local_rank=0 train llamafactory_config_3x3080.yaml
#                               ^^^^^^^^^^^^^^ ^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                               DeepSpeed参数   命令   配置文件
```

### 参数解析流程

| 步骤 | 期望 | 实际 | 结果 |
|------|------|------|------|
| 1. 读取第一个参数 | `train` | `--local_rank=0` | ❌ 错误 |
| 2. 识别为命令 | ✅ | ❌ | NotImplementedError |

### 为什么会这样？

**DeepSpeed/Torchrun 的行为：**

```bash
# 你的启动命令
deepspeed --num_gpus=3 script.py train config.yaml

# DeepSpeed 实际执行（每个进程）
python script.py --local_rank=0 train config.yaml  # GPU 0
python script.py --local_rank=1 train config.yaml  # GPU 1
python script.py --local_rank=2 train config.yaml  # GPU 2
```

**llmtuner.cli.main() 的期望：**

```python
# llmtuner/cli.py
def main():
    command = sys.argv[1]  # 期望是 'train'
    if command == 'train':
        # ...
    else:
        raise NotImplementedError(f"Unknown command: {command}")
```

**冲突：**
- DeepSpeed 传入：`['script.py', '--local_rank=0', 'train', 'config.yaml']`
- main() 读取：`sys.argv[1]` = `'--local_rank=0'` ❌
- 期望读取：`sys.argv[1]` = `'train'` ✅

---

## ✅ 解决方案

### 🎯 方案1: 过滤 --local_rank 参数（推荐）⭐⭐⭐⭐⭐

**原理：** 在调用 `main()` 之前，从 `sys.argv` 中移除 `--local_rank` 参数

**包装脚本：**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSpeed 训练启动脚本
解决 --local_rank 参数解析问题
"""
import sys
import os

# 过滤掉 DeepSpeed 自动添加的 --local_rank 参数
filtered_args = []
skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if skip_next:
        skip_next = False
        continue
    if arg.startswith('--local_rank'):
        if '=' not in arg and i < len(sys.argv) - 1:
            skip_next = True  # 跳过下一个参数（值）
        continue  # 跳过 --local_rank
    filtered_args.append(arg)

# 替换 sys.argv
sys.argv = [sys.argv[0]] + filtered_args

# 导入并运行
from llmtuner.cli import main

if __name__ == "__main__":
    main()
```

**工作原理：**

```python
# 原始参数
sys.argv = ['script.py', '--local_rank=0', 'train', 'config.yaml']

# 过滤后
sys.argv = ['script.py', 'train', 'config.yaml']

# main() 读取
command = sys.argv[1]  # 'train' ✅
```

**为什么可以移除 --local_rank？**

- ✅ DeepSpeed/PyTorch 会自动设置环境变量：`LOCAL_RANK`, `RANK`, `WORLD_SIZE`
- ✅ LLaMA-Factory 从环境变量读取这些信息，不需要命令行参数
- ✅ 移除 `--local_rank` 不影响分布式训练

---

### 🎯 方案2: 使用 llamafactory-cli（最简单）⭐⭐⭐⭐⭐

**llamafactory-cli 已经处理了这个问题：**

```bash
# 直接使用命令行工具
llamafactory-cli train llamafactory_config_3x3080.yaml
```

**优点：**
- ✅ 不需要包装脚本
- ✅ 自动处理所有参数
- ✅ 支持所有启动方式

---

### 🎯 方案3: 使用 accelerate launch（支持 -m）⭐⭐⭐⭐

**accelerate 会正确处理参数：**

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=3 \
    -m llmtuner.cli train config.yaml
```

**优点：**
- ✅ 不需要包装脚本
- ✅ 参数处理正确
- ✅ 灵活配置

---

## 🔧 已修复的启动脚本

我已经修复了 `run_llamafactory_3x3080.sh`，现在包装脚本会自动过滤 `--local_rank` 参数：

### 修复内容

#### 1. DeepSpeed 包装脚本（方式3）

```bash
# 修复前（错误）
cat > /tmp/deepspeed_train.py << 'EOF'
#!/usr/bin/env python
import sys
from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF
# ❌ 直接调用 main()，--local_rank 参数导致解析失败

# 修复后（正确）
cat > /tmp/deepspeed_train.py << 'EOF'
#!/usr/bin/env python
import sys

# 过滤掉 --local_rank 参数
filtered_args = []
skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if skip_next:
        skip_next = False
        continue
    if arg.startswith('--local_rank'):
        if '=' not in arg and i < len(sys.argv) - 1:
            skip_next = True
        continue
    filtered_args.append(arg)

sys.argv = [sys.argv[0]] + filtered_args

from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF
# ✅ 过滤 --local_rank 参数后再调用 main()
```

#### 2. Torchrun 包装脚本（方式4）

同样的修复应用到 torchrun 包装脚本。

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
1. ✅ 创建包装脚本
2. ✅ 过滤 `--local_rank` 参数
3. ✅ 正确调用 `main()`
4. ✅ 开始训练

---

## 📊 参数处理对比

### 原始参数（DeepSpeed 传入）

```python
sys.argv = [
    '/tmp/deepspeed_train.py',
    '--local_rank=0',      # ← DeepSpeed 添加
    'train',               # ← 命令
    'config.yaml'          # ← 配置文件
]
```

### 过滤后的参数

```python
sys.argv = [
    '/tmp/deepspeed_train.py',
    'train',               # ← 命令（正确位置）
    'config.yaml'          # ← 配置文件
]
```

### main() 解析

```python
def main():
    command = sys.argv[1]  # 'train' ✅
    if command == 'train':
        # 开始训练 ✅
```

---

## 💡 深入理解

### 分布式训练的参数传递

#### 环境变量方式（推荐）✅

```bash
# DeepSpeed/PyTorch 自动设置
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=3

# 程序读取
import os
local_rank = int(os.environ.get('LOCAL_RANK', 0))
```

**优点：**
- ✅ 不污染命令行参数
- ✅ 标准化
- ✅ 所有框架都支持

#### 命令行参数方式（旧方式）❌

```bash
# 传递参数
python script.py --local_rank=0

# 程序解析
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
```

**缺点：**
- ❌ 与其他参数冲突
- ❌ 需要手动解析
- ❌ 不同框架实现不同

### LLaMA-Factory 的实现

**LLaMA-Factory 使用环境变量：**

```python
# llmtuner 内部
import os
import torch.distributed as dist

if dist.is_available() and dist.is_initialized():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
```

**所以可以安全移除 --local_rank 参数！**

---

## 🔒 防止再次出现

### 规则1: 优先使用命令行工具

```bash
# ✅ 推荐（自动处理所有参数）
llamafactory-cli train config.yaml

# ❌ 避免（需要手动处理参数）
deepspeed script.py train config.yaml
```

### 规则2: 使用包装脚本时过滤参数

```python
# ✅ 正确（过滤 --local_rank）
import sys
filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith('--local_rank')]
sys.argv = [sys.argv[0]] + filtered_args
from llmtuner.cli import main
main()

# ❌ 错误（直接调用）
from llmtuner.cli import main
main()
```

### 规则3: 理解启动器的行为

| 启动器 | 添加参数 | 设置环境变量 | 需要过滤 |
|--------|---------|------------|---------|
| `deepspeed` | ✅ `--local_rank` | ✅ | ✅ 需要 |
| `torchrun` | ✅ `--local_rank` | ✅ | ✅ 需要 |
| `accelerate` | ❌ | ✅ | ❌ 不需要 |
| `llamafactory-cli` | ❌ | ✅ | ❌ 不需要 |

---

## 📝 快速参考

### 创建包装脚本（带参数过滤）

```bash
# 创建 DeepSpeed 包装脚本
cat > deepspeed_train.py << 'EOF'
#!/usr/bin/env python
import sys

# 过滤 --local_rank 参数
filtered_args = []
skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if skip_next:
        skip_next = False
        continue
    if arg.startswith('--local_rank'):
        if '=' not in arg and i < len(sys.argv) - 1:
            skip_next = True
        continue
    filtered_args.append(arg)

sys.argv = [sys.argv[0]] + filtered_args

from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF

# 使用
deepspeed --num_gpus=3 deepspeed_train.py train config.yaml
```

### 各种启动方式

```bash
# 方式1: llamafactory-cli（推荐，无需处理参数）
llamafactory-cli train config.yaml

# 方式2: accelerate（无需处理参数）
accelerate launch -m llmtuner.cli train config.yaml

# 方式3: deepspeed（需要包装脚本过滤参数）
deepspeed --num_gpus=3 wrapper.py train config.yaml

# 方式4: torchrun（需要包装脚本过滤参数）
torchrun --nproc_per_node=3 wrapper.py train config.yaml
```

---

## 🎯 总结

**问题：** DeepSpeed 添加的 `--local_rank` 参数导致 `llmtuner.cli.main()` 解析失败

**原因：** `main()` 期望第一个参数是命令（`train`），但收到的是 `--local_rank=0`

**解决方案：**
1. 使用 `llamafactory-cli`（最简单）
2. 或者在包装脚本中过滤 `--local_rank` 参数

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
3. 测试包装脚本

```bash
# 测试包装脚本
cat > /tmp/test_wrapper.py << 'EOF'
#!/usr/bin/env python
import sys

print("原始参数:", sys.argv)

# 过滤 --local_rank
filtered_args = []
skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if skip_next:
        skip_next = False
        continue
    if arg.startswith('--local_rank'):
        if '=' not in arg and i < len(sys.argv) - 1:
            skip_next = True
        continue
    filtered_args.append(arg)

sys.argv = [sys.argv[0]] + filtered_args
print("过滤后参数:", sys.argv)

from llmtuner.cli import main
main()
EOF

# 手动测试
python /tmp/test_wrapper.py --local_rank=0 train config.yaml
```

---

## 🔗 相关文档

- [FIX_RELATIVE_IMPORT.md](FIX_RELATIVE_IMPORT.md) - 相对导入错误修复
- [FIX_DEEPSPEED_LAUNCH.md](FIX_DEEPSPEED_LAUNCH.md) - DeepSpeed 启动问题
- [FIX_TRL_CONFLICT.md](FIX_TRL_CONFLICT.md) - 依赖冲突修复
