# SFT训练配置对比说明

## 📊 三种配置对比

### 1. TrainConfigSFTSmall（低内存模式）
**适用场景**：内存紧张（<10GB可用）

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size_per_gpu | 1 | 极致低内存 |
| gradient_accumulation_steps | 8 | 通过梯度累积补偿小batch |
| 实际有效batch_size | 16 | 1 × 2(GPU) × 8 |
| GPU显存占用 | 2-3GB/GPU | 显存利用率低 |
| 内存占用 | 6-8GB | 适合16GB内存 |
| 训练速度 | 基准速度 | 较慢 |

**启动命令**：
```bash
./quick_start_sft_gloo.sh
# 或
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

---

### 2. TrainConfigSFTFast（高性能模式）⭐ 推荐
**适用场景**：GPU显存充足（20GB），内存可用（>7GB）

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size_per_gpu | 8 | 充分利用GPU显存 |
| gradient_accumulation_steps | 2 | 减少内存占用 |
| 实际有效batch_size | 32 | 8 × 2(GPU) × 2 |
| GPU显存占用 | 8-12GB/GPU | 显存利用率提升4-5倍 |
| 内存占用 | 8-12GB | 适合16GB内存 |
| 训练速度 | **3-4倍** | 大幅提升 |

**启动命令**：
```bash
./quick_start_sft_fast.sh
# 或
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True
```

---

### 3. TrainConfigSFT（标准模式）
**适用场景**：大数据集（>10k样本），内存充足（>20GB）

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size_per_gpu | 20 | 标准配置 |
| gradient_accumulation_steps | 6 | 平衡性能 |
| 实际有效batch_size | 240 | 20 × 2(GPU) × 6 |
| GPU显存占用 | 15-18GB/GPU | 接近显存上限 |
| 内存占用 | 15-20GB | 需要更大内存 |
| 训练速度 | 最快 | 但需要更多资源 |

**启动命令**：
```bash
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True
```

---

## 🎯 如何选择配置？

### 根据你的硬件情况：
- **GPU显存**: 20GB × 2 ✅
- **内存**: 16GB（可用7GB）✅
- **数据集**: 5,000样本（小数据集）

### 推荐方案：
**使用 TrainConfigSFTFast（高性能模式）**

**理由**：
1. ✅ GPU显存充足（20GB），目前只用了2.5GB（12.5%），浪费严重
2. ✅ 内存可用7GB，足够支持batch_size=8
3. ✅ 小数据集不需要超大batch_size，32已经足够
4. ✅ 训练速度提升3-4倍，大幅缩短训练时间

---

## 📈 性能对比（5000样本，3个epoch）

| 配置 | 每epoch步数 | 单epoch时长 | 总训练时长 | GPU利用率 |
|------|------------|------------|-----------|----------|
| Small | 312步 | ~2小时 | ~6小时 | 12% |
| **Fast** | **156步** | **~30分钟** | **~1.5小时** | **50-60%** |
| Standard | 21步 | ~10分钟 | ~30分钟 | 90% |

**注意**：Standard模式虽然最快，但需要更大内存（>20GB），你的16GB内存可能不够。

---

## 🚀 立即开始

### 方法1：使用快速启动脚本（推荐）
```bash
cd /data3/ChatLM-mini-Chinese
chmod +x quick_start_sft_fast.sh
./quick_start_sft_fast.sh
```

### 方法2：手动启动
```bash
cd /data3/ChatLM-mini-Chinese
export ACCELERATE_USE_GLOO=1

accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True
```

---

## 🔧 进一步优化（可选）

如果训练过程中发现：

### 1. 内存还有余量（>3GB可用）
可以尝试进一步增大batch_size：
```bash
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True \
    --batch_size_per_gpu=12
```

### 2. GPU显存还有余量（>8GB可用）
可以尝试更大的batch_size：
```bash
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True \
    --batch_size_per_gpu=16
```

### 3. 内存不够（OOM）
降低batch_size：
```bash
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True \
    --batch_size_per_gpu=4
```

---

## 📝 监控训练

训练过程中，使用以下命令监控资源使用：

```bash
# 监控GPU
watch -n 1 nvidia-smi

# 监控内存
watch -n 1 free -h

# 查看训练日志
tail -f logs/chat_trainer_*.log
```

---

## ⚠️ 注意事项

1. **首次使用Fast模式**：建议先运行1个epoch，观察内存和GPU显存占用
2. **如果出现OOM**：降低batch_size_per_gpu（从8降到4或6）
3. **训练稳定后**：可以尝试进一步增大batch_size以提升速度
4. **保存位置**：模型保存在 `./model_save/sft_fast/`

---

## 🎉 预期效果

使用 **TrainConfigSFTFast** 后：
- ✅ GPU显存利用率：从12% → 50-60%（提升4-5倍）
- ✅ 训练速度：从6小时 → 1.5小时（提升4倍）
- ✅ 内存占用：保持在10GB左右（安全范围）
- ✅ 训练效果：与Small模式相同（有效batch_size更大，可能更好）
