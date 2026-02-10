# 🚀 快速开始 - 3×RTX 3080 20GB

## 📦 安装依赖

```bash
pip install llmtuner deepspeed
```

## ⚡ 最快启动方式

### 方式1: 交互式脚本（推荐新手）

```bash
bash run_llamafactory_3x3080.sh
```

### 方式2: 命令行（推荐高手）

```bash
# 使用 DeepSpeed（最优显存利用）
export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed --num_gpus=3 -m llmtuner.cli train llamafactory_config_3x3080.yaml
```

## 📊 配置说明

| 参数 | 值 | 说明 |
|------|---|------|
| **GPU数量** | 3 | 3张RTX 3080 |
| **单卡batch** | 8 | 每张卡处理8个样本 |
| **梯度累积** | 16 | 累积16步 |
| **有效batch** | 384 | 8×3×16=384 |
| **混合精度** | BF16 | 节省显存 |
| **显存占用** | ~15GB/卡 | 使用DeepSpeed ZeRO-2 |

## 🔧 如果遇到问题

### 显存不足（OOM）

编辑 `llamafactory_config_3x3080.yaml`：

```yaml
per_device_train_batch_size: 4  # 8 -> 4
gradient_accumulation_steps: 32  # 16 -> 32
```

### 内存不足（RAM）

编辑 `llamafactory_config_3x3080.yaml`：

```yaml
preprocessing_num_workers: 1  # 2 -> 1
max_samples: 1000000  # 限制样本数
```

## 📈 监控训练

```bash
# 查看GPU
watch -n 1 nvidia-smi

# 查看日志
tensorboard --logdir=./logs/llamafactory_3x3080
```

## 📚 详细文档

- [完整使用指南](LLAMAFACTORY_GUIDE_3x3080.md)
- [快速命令参考](QUICK_START_3x3080.sh)
- [训练方式对比](TRAINING_METHODS_COMPARISON.md)

---

**就这么简单！🎉**
