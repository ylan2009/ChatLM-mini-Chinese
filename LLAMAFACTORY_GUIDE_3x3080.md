# LLaMA-Factory 训练指南（3×RTX 3080 20GB）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 LLaMA-Factory
pip install llmtuner

# 安装 DeepSpeed（推荐，用于优化显存）
pip install deepspeed

# 安装其他依赖
pip install transformers datasets torch pyyaml tensorboard
```

### 2. 准备数据

确保你的数据文件存在：
- `data/my_train_dataset.parquet` - 训练数据
- `data/my_valid_dataset.parquet` - 验证数据（可选）

数据格式要求：
- 必须包含 `input` 列（输入文本）
- 必须包含 `target` 列（目标文本）

### 3. 启动训练

**最简单的方式（推荐）：**

```bash
bash run_llamafactory_3x3080.sh
```

然后选择训练方式（推荐选择 1 或 3）。

---

## 📋 命令行方式

### 方式1: llamafactory-cli（最简单）

```bash
# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2

# 启动训练
llamafactory-cli train llamafactory_config_3x3080.yaml
```

### 方式2: accelerate launch（更灵活）

```bash
# 首次使用需要配置
accelerate config

# 启动训练
export CUDA_VISIBLE_DEVICES=0,1,2
accelerate launch \
    --multi_gpu \
    --num_processes=3 \
    -m llmtuner.cli train llamafactory_config_3x3080.yaml
```

### 方式3: deepspeed（最优显存利用，推荐！）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed \
    --num_gpus=3 \
    -m llmtuner.cli train llamafactory_config_3x3080.yaml
```

### 方式4: torchrun（标准DDP）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
torchrun \
    --nproc_per_node=3 \
    -m llmtuner.cli train llamafactory_config_3x3080.yaml
```

---

## ⚙️ 配置说明

### 核心配置（`llamafactory_config_3x3080.yaml`）

```yaml
# 批次大小配置（针对3×20GB GPU优化）
per_device_train_batch_size: 8      # 每张卡batch=8
gradient_accumulation_steps: 16     # 梯度累积16步
# 有效batch size = 8 × 3 × 16 = 384

# 显存优化
bf16: true                          # 使用BF16混合精度
gradient_checkpointing: true        # 梯度检查点
deepspeed: ds_config_zero2.json     # DeepSpeed ZeRO-2

# 内存优化（针对12GB内存）
preprocessing_num_workers: 2        # 降低内存占用
dataloader_num_workers: 0           # 禁用多进程加载
dataloader_pin_memory: false        # 禁用pin memory
```

### 如果显存不足，可以调整：

```yaml
# 方案1: 减小batch size
per_device_train_batch_size: 4      # 8 -> 4
gradient_accumulation_steps: 32     # 16 -> 32（保持有效batch不变）

# 方案2: 减小序列长度
cutoff_len: 256                     # 512 -> 256

# 方案3: 使用LoRA微调（显存占用更小）
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
```

---

## 📊 预期显存占用

### 使用当前配置（batch_size=8, bf16, gradient_checkpointing）

| 配置 | 单卡显存占用 | 总显存占用 |
|------|------------|-----------|
| **不使用DeepSpeed** | ~18GB | ~54GB |
| **使用DeepSpeed ZeRO-2** | ~15GB | ~45GB |
| **使用LoRA** | ~10GB | ~30GB |

你的硬件（3×20GB）完全够用！✅

---

## 🔧 常见问题

### Q1: 显存不足（OOM）

**解决方案**：

```bash
# 方案1: 减小batch size
# 编辑 llamafactory_config_3x3080.yaml
per_device_train_batch_size: 4  # 改为4

# 方案2: 使用DeepSpeed ZeRO-3（更激进的显存优化）
# 创建 ds_config_zero3.json，然后修改配置
deepspeed: ds_config_zero3.json

# 方案3: 使用LoRA微调
finetuning_type: lora
```

### Q2: 内存不足（RAM）

**解决方案**：

```bash
# 减少数据预处理进程
preprocessing_num_workers: 1  # 改为1

# 减少最大样本数
max_samples: 1000000  # 限制为100万

# 启用数据流式加载
streaming: true
```

### Q3: NCCL 初始化失败

**解决方案**：

```bash
# 添加环境变量
export NCCL_SHM_DISABLE=1
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1

# 或使用Gloo后端
export ACCELERATE_USE_GLOO=1
```

### Q4: 训练速度慢

**优化方案**：

```yaml
# 1. 增大batch size（如果显存允许）
per_device_train_batch_size: 12

# 2. 禁用评估（训练时）
evaluation_strategy: "no"

# 3. 减少日志频率
logging_steps: 100
save_steps: 10000

# 4. 使用更快的优化器
optim: adamw_torch  # 比adafactor快，但占用更多显存
```

---

## 📈 监控训练

### 使用 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs/llamafactory_3x3080

# 在浏览器打开
# http://localhost:6006
```

### 查看GPU使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

---

## 🎯 性能优化建议

### 针对你的硬件（3×3080 20GB + 12GB RAM）

1. **推荐配置**（当前配置）：
   - `per_device_train_batch_size: 8`
   - `gradient_accumulation_steps: 16`
   - 使用 DeepSpeed ZeRO-2
   - 预期速度：~1000 samples/s

2. **激进配置**（最大化GPU利用）：
   - `per_device_train_batch_size: 12`
   - `gradient_accumulation_steps: 10`
   - 使用 DeepSpeed ZeRO-2
   - 预期速度：~1500 samples/s

3. **保守配置**（最稳定）：
   - `per_device_train_batch_size: 4`
   - `gradient_accumulation_steps: 32`
   - 不使用 DeepSpeed
   - 预期速度：~600 samples/s

---

## 📝 完整训练流程

```bash
# 1. 安装依赖
pip install llmtuner deepspeed

# 2. 检查数据
ls -lh data/my_train_dataset.parquet

# 3. 启动训练（推荐使用DeepSpeed）
export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed --num_gpus=3 -m llmtuner.cli train llamafactory_config_3x3080.yaml

# 4. 监控训练（另开一个终端）
tensorboard --logdir=./logs/llamafactory_3x3080

# 5. 训练完成后，模型保存在
ls -lh ./model_save/llamafactory_3x3080_output/
```

---

## 🆚 与其他方式对比

| 特性 | LLaMA-Factory | Transformers Trainer | 手动训练循环 |
|------|--------------|---------------------|-------------|
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **配置方式** | YAML文件 | Python代码 | Python代码 |
| **代码量** | ~100行 | ~200行 | ~850行 |
| **显存优化** | 自动优化 | 需手动配置 | 需手动实现 |
| **学习曲线** | 最平缓 | 平缓 | 陡峭 |
| **推荐度** | ✅ 强烈推荐 | ✅ 推荐 | 仅研究用 |

---

## 💡 总结

**对于你的硬件配置（3×3080 20GB + 12GB RAM），推荐使用：**

```bash
# 最简单的启动方式
bash run_llamafactory_3x3080.sh
# 然后选择 3 (使用 deepspeed)
```

**或者直接命令行：**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
deepspeed --num_gpus=3 -m llmtuner.cli train llamafactory_config_3x3080.yaml
```

这个配置已经针对你的硬件优化过，可以直接使用！🚀
