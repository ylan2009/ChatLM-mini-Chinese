# 🚀 GPU显存优化指南 - 进阶版

## 📊 当前优化状态

### 已完成的优化
✅ **batch_size提升**：从1 → 16（提升16倍）  
✅ **梯度累积优化**：从8 → 2（减少内存占用）  
✅ **有效batch_size**：从16 → 64（提升4倍）  
✅ **预期GPU显存利用率**：从12% → 60-80%（提升5-7倍）

---

## 🎯 三种使用方式

### 方式1：使用默认配置（batch_size=16）⭐ 推荐

```bash
cd /data3/ChatLM-mini-Chinese
./quick_start_sft_fast.sh
```

**预期效果**：
- GPU显存：12-16GB/GPU（60-80%利用率）
- 内存：10-14GB
- 训练速度：比Small模式快5-6倍

---

### 方式2：自定义batch_size（灵活调整）

```bash
cd /data3/ChatLM-mini-Chinese
chmod +x train_sft_custom.sh

# 尝试不同的batch_size
./train_sft_custom.sh 12   # 保守：GPU显存10-14GB
./train_sft_custom.sh 16   # 推荐：GPU显存12-16GB
./train_sft_custom.sh 20   # 激进：GPU显存14-18GB
./train_sft_custom.sh 24   # 极限：GPU显存16-19GB
```

**如何选择batch_size？**

| batch_size | GPU显存占用 | 内存占用 | 速度提升 | 推荐场景 |
|-----------|-----------|---------|---------|---------|
| 8 | 8-12GB | 8-12GB | 3-4x | 保守，确保稳定 |
| 12 | 10-14GB | 9-13GB | 4-5x | 平衡性能和稳定性 |
| **16** | **12-16GB** | **10-14GB** | **5-6x** | **推荐，性价比最高** |
| 20 | 14-18GB | 11-15GB | 6-7x | 激进，追求速度 |
| 24 | 16-19GB | 12-15GB | 7-8x | 极限，接近硬件上限 |
| 28+ | 18-20GB | 13-16GB | 8-10x | 危险，可能OOM |

---

### 方式3：手动指定参数（完全控制）

```bash
cd /data3/ChatLM-mini-Chinese
export ACCELERATE_USE_GLOO=1

accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True \
    --batch_size_per_gpu=20
```

---

## 📈 渐进式优化策略

### 第1步：从batch_size=16开始（推荐）

```bash
./quick_start_sft_fast.sh
```

**观察指标**：
- GPU显存占用（nvidia-smi）
- 内存占用（free -h）
- 训练速度（每步耗时）

---

### 第2步：根据资源使用情况调整

#### 情况A：GPU显存还有余量（<14GB）

**说明**：GPU显存利用率不足70%，可以继续提升

**操作**：
```bash
# 停止当前训练（Ctrl+C）
# 尝试更大的batch_size
./train_sft_custom.sh 20
```

---

#### 情况B：GPU显存接近上限（>16GB）

**说明**：GPU显存利用率超过80%，已经很好

**操作**：保持当前配置，继续训练

---

#### 情况C：内存不足（OOM）

**说明**：内存占用超过可用内存，需要降低batch_size

**操作**：
```bash
# 停止当前训练（Ctrl+C）
# 降低batch_size
./train_sft_custom.sh 12
# 或更保守
./train_sft_custom.sh 8
```

---

### 第3步：找到最优配置

通过多次尝试，找到：
- GPU显存利用率：70-85%
- 内存占用：<14GB（留2GB余量）
- 训练稳定，无OOM

**记录最优batch_size**，后续训练使用该配置。

---

## 🔍 实时监控

### 监控GPU（必须）

```bash
watch -n 1 nvidia-smi
```

**关注指标**：
- GPU Memory-Usage：目标70-85%
- GPU-Util：目标50-90%
- Temperature：<85°C

---

### 监控内存（必须）

```bash
watch -n 1 free -h
```

**关注指标**：
- used：应<14GB（留2GB余量）
- available：应>2GB

---

### 监控训练日志

```bash
tail -f logs/chat_trainer_*.log
```

**关注指标**：
- loss下降趋势
- 每步耗时
- 是否有错误信息

---

## 💡 优化技巧

### 技巧1：动态调整batch_size

训练过程中发现资源使用不理想？

```bash
# 停止训练（Ctrl+C）
# 调整batch_size后重新启动
./train_sft_custom.sh 20

# 训练会从上次保存的checkpoint继续
```

---

### 技巧2：使用更大的评估batch_size

代码已自动优化：评估时使用 `batch_size * 3`

**原因**：评估不需要梯度，可以用更大batch_size提升速度

---

### 技巧3：减少保存频率（可选）

如果磁盘空间紧张，可以减少保存频率：

编辑 `config.py`：
```python
save_steps: int = 156  # 改为 312（每2个epoch保存1次）
```

---

## 📊 性能对比表

### 5000样本，3个epoch训练

| 配置 | batch_size | 有效batch | 每epoch步数 | 单epoch时长 | 总时长 | GPU利用率 |
|------|-----------|----------|-----------|-----------|--------|----------|
| Small | 1 | 16 | 312步 | ~2小时 | ~6小时 | 12% |
| Fast-8 | 8 | 32 | 156步 | ~40分钟 | ~2小时 | 40-50% |
| **Fast-16** | **16** | **64** | **78步** | **~20分钟** | **~1小时** | **60-80%** |
| Fast-20 | 20 | 80 | 62步 | ~15分钟 | ~45分钟 | 70-85% |
| Fast-24 | 24 | 96 | 52步 | ~12分钟 | ~36分钟 | 80-95% |

---

## ⚠️ 常见问题

### Q1：训练启动后立即OOM

**原因**：batch_size太大，超过内存或GPU显存

**解决**：
```bash
# 降低batch_size
./train_sft_custom.sh 8
```

---

### Q2：训练一段时间后OOM

**原因**：内存泄漏或梯度累积导致内存增长

**解决**：
1. 降低batch_size
2. 检查是否有其他程序占用内存
3. 重启训练

---

### Q3：GPU显存利用率还是很低（<50%）

**原因**：batch_size还可以继续增大

**解决**：
```bash
# 逐步增大batch_size
./train_sft_custom.sh 20
./train_sft_custom.sh 24
./train_sft_custom.sh 28
```

**注意**：每次增大4-8，观察稳定后再继续增大

---

### Q4：训练速度没有明显提升

**可能原因**：
1. 数据加载成为瓶颈（检查CPU使用率）
2. 磁盘I/O成为瓶颈（检查磁盘使用率）
3. GPU之间通信成为瓶颈（NCCL/Gloo）

**解决**：
1. 增加num_workers（已自动优化）
2. 使用SSD存储数据
3. 使用更快的网络（如果是多机训练）

---

### Q5：如何找到最优batch_size？

**二分查找法**：

```bash
# 第1步：尝试较大值
./train_sft_custom.sh 24
# 如果OOM，说明太大

# 第2步：尝试中间值
./train_sft_custom.sh 16
# 如果成功，说明可以继续增大

# 第3步：尝试更大值
./train_sft_custom.sh 20
# 如果成功且GPU利用率>70%，找到最优值

# 第4步：微调
./train_sft_custom.sh 22
# 继续尝试，直到找到最大可用值
```

---

## 🎯 推荐配置总结

### 你的硬件配置
- GPU：RTX 3080 20GB × 2
- 内存：16GB（可用7GB）
- 数据集：5000样本

### 推荐配置
```bash
# 保守配置（确保稳定）
./train_sft_custom.sh 12

# 推荐配置（性价比最高）⭐
./train_sft_custom.sh 16

# 激进配置（追求速度）
./train_sft_custom.sh 20

# 极限配置（榨干硬件）
./train_sft_custom.sh 24
```

### 建议
1. **首次训练**：使用 `batch_size=16`（推荐配置）
2. **观察资源**：运行10-20步后，查看GPU和内存使用
3. **动态调整**：根据实际情况增大或减小batch_size
4. **找到最优**：记录最优配置，后续训练使用

---

## 📝 快速参考

### 启动命令
```bash
# 默认配置（batch_size=16）
./quick_start_sft_fast.sh

# 自定义batch_size
./train_sft_custom.sh 20

# 手动指定
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_fast_config=True \
    --batch_size_per_gpu=20
```

### 监控命令
```bash
# GPU监控
watch -n 1 nvidia-smi

# 内存监控
watch -n 1 free -h

# 日志监控
tail -f logs/chat_trainer_*.log
```

---

## 🎉 预期效果

使用 **batch_size=16** 后：

✅ GPU显存利用率：12% → **60-80%**（提升5-7倍）  
✅ 训练速度：6小时 → **1小时**（提升6倍）  
✅ 内存占用：保持在10-14GB（安全范围）  
✅ 训练效果：有效batch_size更大，可能更好

---

**立即开始优化训练！** 🚀
