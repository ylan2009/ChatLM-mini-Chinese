# 🚨 相对导入错误修复

## ❌ 错误信息

```python
Traceback (most recent call last):
  File "/path/to/llmtuner/cli.py", line 4, in <module>
    from .api.app import run_api
ImportError: attempted relative import with no known parent package
```

---

## 🔍 问题分析

### 错误原因

**当直接运行 Python 脚本时，相对导入会失败！**

```python
# cli.py 中的代码
from .api.app import run_api  # ❌ 直接运行脚本时会失败
```

### 为什么会失败？

| 运行方式 | Python 包识别 | 相对导入 | 结果 |
|---------|-------------|---------|------|
| `python cli.py` | ❌ 不识别包结构 | ❌ 失败 | ImportError |
| `python -m llmtuner.cli` | ✅ 识别包结构 | ✅ 成功 | 正常运行 |
| `deepspeed cli.py` | ❌ 不识别包结构 | ❌ 失败 | ImportError |
| `deepspeed wrapper.py` | ✅ wrapper导入模块 | ✅ 成功 | 正常运行 |

### 技术细节

1. **直接运行脚本：**
   ```bash
   python /path/to/llmtuner/cli.py
   # Python 认为这是一个独立脚本，不知道它属于 llmtuner 包
   # 相对导入 from .api.app 失败
   ```

2. **模块方式运行：**
   ```bash
   python -m llmtuner.cli
   # Python 知道这是 llmtuner 包的一部分
   # 相对导入 from .api.app 成功
   ```

3. **DeepSpeed 的问题：**
   ```bash
   deepspeed -m llmtuner.cli  # ❌ deepspeed 不支持 -m 参数
   deepspeed cli.py           # ❌ 直接运行导致相对导入失败
   ```

---

## ✅ 解决方案

### 🎯 方案1: 使用 llamafactory-cli（最简单）⭐⭐⭐⭐⭐

**推荐方式：**

```bash
# 直接使用命令行工具
llamafactory-cli train llamafactory_config_3x3080.yaml
```

**优点：**
- ✅ 自动处理所有导入问题
- ✅ 支持所有启动方式（包括 DeepSpeed）
- ✅ 最简单，不会出错

---

### 🎯 方案2: 创建包装脚本（适用于 deepspeed/torchrun）⭐⭐⭐⭐

**原理：** 创建一个脚本，使用 `import` 导入模块而不是直接运行

**包装脚本：**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSpeed/Torchrun 训练启动脚本
解决直接运行 cli.py 时的相对导入问题
"""
import sys
from llmtuner.cli import main

if __name__ == "__main__":
    main()
```

**使用方式：**

```bash
# 1. 创建包装脚本
cat > /tmp/deepspeed_train.py << 'EOF'
#!/usr/bin/env python
import sys
from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF

# 2. 使用 deepspeed 启动
deepspeed --num_gpus=3 /tmp/deepspeed_train.py train config.yaml
```

**为什么这样可以？**
- ✅ `from llmtuner.cli import main` 是**绝对导入**，不是相对导入
- ✅ Python 能正确识别 `llmtuner` 包结构
- ✅ 包内的相对导入（`from .api.app`）也能正常工作

---

### 🎯 方案3: 使用 accelerate launch（支持 -m）⭐⭐⭐⭐

**accelerate 支持模块方式运行：**

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=3 \
    -m llmtuner.cli train config.yaml
```

**优点：**
- ✅ 支持 `-m` 参数，可以使用模块方式
- ✅ 不需要包装脚本
- ✅ 灵活配置

---

### 🎯 方案4: 使用 Python 模块方式（单GPU）⭐⭐⭐

**适用于单GPU或调试：**

```bash
python -m llmtuner.cli train config.yaml
```

**优点：**
- ✅ 最直接的方式
- ✅ 适合调试

**缺点：**
- ❌ 不支持多GPU（需要配合其他启动器）

---

## 🔧 已修复的启动脚本

我已经修复了 `run_llamafactory_3x3080.sh`，现在使用包装脚本方式：

### 修复内容

#### 1. DeepSpeed 启动（方式3）

```bash
# 修复前（错误）
deepspeed --num_gpus=3 /path/to/cli.py train config.yaml
# ❌ 直接运行 cli.py 导致相对导入失败

# 修复后（正确）
cat > /tmp/deepspeed_train.py << 'EOF'
#!/usr/bin/env python
import sys
from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF

deepspeed --num_gpus=3 /tmp/deepspeed_train.py train config.yaml
# ✅ 使用包装脚本，通过绝对导入解决问题
```

#### 2. Torchrun 启动（方式4）

```bash
# 修复前（错误）
torchrun --nproc_per_node=3 /path/to/cli.py train config.yaml
# ❌ 直接运行 cli.py 导致相对导入失败

# 修复后（正确）
cat > /tmp/torchrun_train.py << 'EOF'
#!/usr/bin/env python
import sys
from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF

torchrun --nproc_per_node=3 /tmp/torchrun_train.py train config.yaml
# ✅ 使用包装脚本，通过绝对导入解决问题
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
1. ✅ 创建包装脚本
2. ✅ 使用正确的启动命令
3. ✅ 开始训练

---

## 📊 各种方案对比

| 方案 | 复杂度 | 兼容性 | 推荐度 | 适用场景 |
|------|--------|--------|--------|---------|
| **llamafactory-cli** | ⭐ 最简单 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 所有场景 |
| **包装脚本** | ⭐⭐ 简单 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | deepspeed/torchrun |
| **accelerate -m** | ⭐⭐ 简单 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 灵活配置 |
| **python -m** | ⭐ 最简单 | ⭐⭐⭐ | ⭐⭐⭐ | 单GPU/调试 |

---

## 💡 深入理解

### Python 导入机制

#### 相对导入（Relative Import）

```python
# 在 llmtuner/cli.py 中
from .api.app import run_api      # 相对导入
from ..utils import helper         # 相对导入
```

**要求：**
- ✅ 必须在包内使用
- ✅ Python 必须知道包结构
- ❌ 不能在直接运行的脚本中使用

#### 绝对导入（Absolute Import）

```python
# 在任何地方都可以使用
from llmtuner.api.app import run_api  # 绝对导入
from llmtuner.utils import helper      # 绝对导入
```

**优点：**
- ✅ 可以在任何地方使用
- ✅ 不依赖当前位置
- ✅ 更清晰明确

### 为什么包装脚本有效？

```python
# wrapper.py（包装脚本）
from llmtuner.cli import main  # ← 绝对导入，总是有效

if __name__ == "__main__":
    main()  # ← 调用 main() 时，已经在正确的包上下文中
```

**执行流程：**

1. **运行包装脚本：**
   ```bash
   deepspeed wrapper.py
   ```

2. **Python 执行：**
   ```python
   from llmtuner.cli import main  # 绝对导入成功
   ```

3. **进入 llmtuner 包：**
   ```python
   # 现在在 llmtuner.cli 模块中
   from .api.app import run_api  # 相对导入成功！
   ```

---

## 🔒 防止再次出现

### 规则1: 优先使用命令行工具

```bash
# ✅ 推荐
llamafactory-cli train config.yaml

# ❌ 避免
python /path/to/cli.py train config.yaml
```

### 规则2: 使用模块方式运行

```bash
# ✅ 正确
python -m llmtuner.cli train config.yaml

# ❌ 错误
python /path/to/llmtuner/cli.py train config.yaml
```

### 规则3: 对于不支持 -m 的启动器，使用包装脚本

```bash
# ✅ 正确（使用包装脚本）
deepspeed wrapper.py train config.yaml

# ❌ 错误（直接运行）
deepspeed cli.py train config.yaml
```

---

## 📝 快速参考

### 创建包装脚本

```bash
# 创建 DeepSpeed 包装脚本
cat > deepspeed_train.py << 'EOF'
#!/usr/bin/env python
import sys
from llmtuner.cli import main

if __name__ == "__main__":
    main()
EOF

# 使用
deepspeed --num_gpus=3 deepspeed_train.py train config.yaml
```

### 各种启动方式

```bash
# 方式1: llamafactory-cli（推荐）
llamafactory-cli train config.yaml

# 方式2: accelerate（支持 -m）
accelerate launch -m llmtuner.cli train config.yaml

# 方式3: deepspeed（需要包装脚本）
deepspeed --num_gpus=3 wrapper.py train config.yaml

# 方式4: torchrun（需要包装脚本）
torchrun --nproc_per_node=3 wrapper.py train config.yaml

# 方式5: python（单GPU）
python -m llmtuner.cli train config.yaml
```

---

## 🎯 总结

**问题：** 直接运行 `cli.py` 导致相对导入失败

**原因：** Python 不知道脚本的包结构

**解决方案：**
1. 使用 `llamafactory-cli`（最简单）
2. 或者创建包装脚本使用绝对导入

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
3. Python 版本和包版本

```bash
# 收集诊断信息
python -c "
import sys
import torch
import transformers
import llmtuner

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('transformers:', transformers.__version__)
print('llmtuner:', llmtuner.__file__)
print()

# 测试导入
try:
    from llmtuner.cli import main
    print('✓ 绝对导入成功')
except Exception as e:
    print('✗ 绝对导入失败:', e)
"
```
