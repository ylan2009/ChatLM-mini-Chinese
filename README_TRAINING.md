# 项目训练脚本说明

## 📁 文件清单

本项目包含以下训练相关文件：

### 训练脚本

1. **`train_low_mem.py`** - 原始手动训练循环实现
   - 使用 `accelerate` 手动实现训练循环
   - 自定义低内存优化
   - 代码量：~850行

2. **`train_with_transformers_trainer.py`** - Transformers Trainer API实现
   - 使用 Transformers 原生 `Trainer` API
   - 自动处理训练、评估、保存
   - 代码量：~200行
   - ⚠️ **之前误命名为 `train_with_llamafactory.py`**

3. **`train_with_real_llamafactory.py`** - 真正的 LLaMA-Factory实现
   - 使用 `llmtuner` 库（LLaMA-Factory的包名）
   - 配置驱动（YAML）
   - 代码量：~100行

### 启动脚本

4. **`run_multi_gpu_examples.sh`** - 多GPU训练启动示例
   - 展示6种不同的多GPU启动方式
   - 交互式菜单选择

### 文档

5. **`TRAINING_METHODS_COMPARISON.md`** - 训练方式对比
   - 详细对比三种训练实现
   - 使用场景推荐

6. **`MULTI_GPU_TRAINING_GUIDE.md`** - 多GPU训练指南
   - 多GPU训练方式详解
   - 性能对比和推荐

7. **`README_TRAINING.md`** - 本文件
   - 项目训练脚本总览

---

## 🎯 快速开始

### 推荐方式：使用 Transformers Trainer

```bash
# 单GPU
python train_with_transformers_trainer.py

# 多GPU（推荐torchrun）
torchrun --nproc_per_node=2 train_with_transformers_trainer.py

# 多GPU（自动检测，最简单）
CUDA_VISIBLE_DEVICES=0,1 python train_with_transformers_trainer.py
```

### 使用真正的 LLaMA-Factory

```bash
# 首先安装
pip install llmtuner

# 运行
python train_with_real_llamafactory.py

# 或使用命令行
llamafactory-cli train llamafactory_config.yaml
```

### 使用原始手动训练循环

```bash
# 需要accelerate
accelerate launch --multi_gpu --num_processes=2 train_low_mem.py
```

---

## ❓ 常见问题

### Q1: 哪个脚本最好用？

**A**: 取决于你的需求：
- **快速开发**: `train_with_real_llamafactory.py`（最简单）
- **标准训练**: `train_with_transformers_trainer.py`（推荐，平衡）
- **研究实验**: `train_low_mem.py`（最灵活）

### Q2: 为什么有个文件叫 `train_with_llamafactory.py`？

**A**: 这是一个**命名错误**！
- 它实际上使用的是 **Transformers Trainer**，不是 LLaMA-Factory
- 已重命名为 `train_with_transformers_trainer.py`
- 真正的 LLaMA-Factory 实现在 `train_with_real_llamafactory.py`

### Q3: 多GPU训练一定要用 accelerate 吗？

**A**: **不需要！** Transformers Trainer 支持多种方式：
- `torchrun`（推荐，无需额外依赖）
- 自动检测（最简单）
- `accelerate`（最灵活）
- `DeepSpeed`（大模型）

详见 [MULTI_GPU_TRAINING_GUIDE.md](MULTI_GPU_TRAINING_GUIDE.md)

### Q4: 三种训练方式性能有差异吗？

**A**: 性能差异很小（<5%），主要区别在于：
- **代码复杂度**: 手动循环 > Trainer > LLaMA-Factory
- **灵活性**: 手动循环 > Trainer > LLaMA-Factory
- **易用性**: LLaMA-Factory > Trainer > 手动循环

---

## 📚 详细文档

- [训练方式对比](TRAINING_METHODS_COMPARISON.md) - 三种训练实现的详细对比
- [多GPU训练指南](MULTI_GPU_TRAINING_GUIDE.md) - 多GPU训练方式详解

---

## 🔧 依赖安装

### 基础依赖（所有脚本）
```bash
pip install torch transformers datasets
```

### Transformers Trainer
```bash
pip install torch_optimizer  # 可选，用于Adafactor优化器
```

### 手动训练循环
```bash
pip install accelerate
```

### 真正的 LLaMA-Factory
```bash
pip install llmtuner
```

---

## 📝 总结

| 脚本 | 使用的技术 | 推荐场景 |
|------|-----------|---------|
| `train_low_mem.py` | Accelerator + 手动循环 | 研究实验 |
| `train_with_transformers_trainer.py` | Transformers Trainer | 标准训练（推荐） |
| `train_with_real_llamafactory.py` | LLaMA-Factory (llmtuner) | 快速开发 |

**新手推荐**: 从 `train_with_transformers_trainer.py` 开始！
