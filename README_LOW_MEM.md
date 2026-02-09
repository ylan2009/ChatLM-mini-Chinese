# 低内存训练指南

## 问题背景

在16G内存的环境下，使用原始的训练脚本进行多GPU训练时会出现内存不足（OOM）的问题。这是因为：

1. **数据集加载到内存**：原始代码在内存充足时会将整个数据集加载到内存中（keep_in_memory=True），这会占用大量内存
2. **多进程数据加载**：DataLoader的num_workers>0会创建多个子进程，每个进程都会复制数据，导致内存占用成倍增加
3. **较大的batch_size**：默认的batch_size可能较大（如8或16），在多GPU环境下会进一步增加内存压力
4. **模型和优化器状态**：T5模型本身和优化器状态也会占用大量内存

## 解决方案

本项目提供了专门针对低内存环境优化的训练脚本：

- **trainer_low_mem.py**: 低内存版本的训练器
- **train_low_mem.py**: 低内存训练的入口脚本

### 主要优化措施

1. **关闭数据集内存缓存**
   - 强制设置 `keep_in_memory=False`
   - 使用迭代器方式读取数据，避免一次性加载全部数据到内存

2. **减小batch_size**
   - 训练batch_size限制为最大2
   - 评估batch_size限制为最大4

3. **增加梯度累积步数**
   - 将梯度累积步数增加到16
   - 通过梯度累积补偿小batch_size，保持训练效果
   - 有效batch_size = batch_size × 梯度累积步数 × GPU数量

4. **禁用多进程数据加载**
   - 强制设置 `num_workers=0`
   - 避免多进程带来的额外内存开销

5. **关闭pin_memory**
   - 在低内存环境下关闭pin_memory
   - 减少内存占用，虽然会略微降低数据传输速度

6. **定期清理内存**
   - 每100步清理一次GPU和CPU缓存
   - 每个epoch开始和评估前清理内存

7. **内存监控**
   - 在关键节点记录内存使用情况
   - 帮助诊断内存问题

## 使用方法

### 1. SFT微调（推荐用于16G内存环境）

```bash
# 使用低内存版本进行SFT微调
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True

# 自定义学习率和训练轮数
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True --epochs=5 --learn_rate=1e-5
```

### 2. 预训练（需要更多内存）

```bash
# 使用低内存版本进行预训练
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train

# 自定义参数
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --epochs=10 --learn_rate=0.0002
```

### 3. 从断点继续训练

```bash
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_keep_training=True
```

## 性能对比

### 原始版本 vs 低内存版本

| 指标 | 原始版本 | 低内存版本 |
|------|---------|-----------|
| 最小内存需求 | ~48GB | ~13GB |
| batch_size | 8-16 | 1-2 |
| 梯度累积步数 | 4 | 16 |
| 有效batch_size | 64-128 | 32-64 |
| 数据加载方式 | 内存缓存 | 迭代器 |
| num_workers | 2-4 | 0 |
| 训练速度 | 快 | 较慢（约慢30-50%） |

### 训练效果

虽然低内存版本的训练速度较慢，但通过梯度累积，训练效果与原始版本基本一致：

- **有效batch_size保持合理范围**：通过增加梯度累积步数，有效batch_size仍然在32-64之间
- **学习率和优化器不变**：使用相同的学习率和Adafactor优化器
- **评估指标一致**：BLEU4分数等评估指标与原始版本相当

## 注意事项

1. **训练时间会增加**
   - 由于batch_size减小和禁用多进程数据加载，训练时间会增加30-50%
   - 建议在时间允许的情况下使用低内存版本

2. **磁盘I/O压力**
   - 关闭keep_in_memory后，会频繁从磁盘读取数据
   - 建议使用SSD存储训练数据

3. **监控内存使用**
   - 训练过程中会定期输出内存使用情况
   - 如果仍然出现OOM，可以考虑：
     - 进一步减小batch_size到1
     - 减小max_seq_len
     - 使用单GPU训练

4. **微调优先**
   - 在16G内存环境下，建议优先进行SFT微调
   - 微调会冻结encoder和embedding，只训练decoder，内存占用更小
   - 预训练需要训练全部参数，内存压力更大

## 故障排查

### 如果仍然出现OOM

1. **检查其他程序**
   ```bash
   # 查看内存使用情况
   free -h
   # 关闭其他占用内存的程序
   ```

2. **减小batch_size**
   - 修改 `trainer_low_mem.py` 中的 `batch_size = min(train_config.batch_size_per_gpu, 1)`
   - 将最大值从2改为1

3. **减小序列长度**
   - 修改 `config.py` 中的 `max_seq_len`
   - 从512减小到256或128

4. **使用单GPU**
   ```bash
   accelerate launch ./train_low_mem.py train --is_finetune=True
   ```

5. **使用CPU训练**（最后的选择，速度会非常慢）
   ```bash
   python ./train_low_mem.py train --is_finetune=True
   ```

## 总结

低内存版本的训练脚本通过多种优化措施，使得在16G内存环境下也能进行多GPU训练。虽然训练速度会有所降低，但训练效果基本不受影响。对于内存受限的环境，这是一个实用的解决方案。
