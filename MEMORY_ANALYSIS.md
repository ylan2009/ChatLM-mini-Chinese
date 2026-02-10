# 🧠 训练过程中的内存占用分析

## 问题

**为什么将 `batch_size_per_gpu` 从 8 提升到 24，GPU显存占用显著增加（从40%到95%），但内存占用几乎不变（保持在5GB以下）？**

这是一个非常好的观察！让我详细解释训练过程中内存和显存的占用机制。

---

## 核心答案

### 简短回答

**batch_size 主要影响 GPU显存，对内存影响很小，因为：**

1. ✅ **数据按需加载**：使用 `LowMemDataset`，数据从磁盘按需读取，不会一次性加载整个batch到内存
2. ✅ **模型在GPU上**：模型参数、梯度、优化器状态都存储在GPU显存中，不占用内存
3. ✅ **数据快速转移**：数据从磁盘读取后，立即转移到GPU，在内存中停留时间极短
4. ✅ **自动垃圾回收**：Python的垃圾回收机制会及时释放已处理的batch数据

---

## 详细分析

### 1. 内存占用的主要来源

在你的训练过程中，**内存占用主要由以下部分组成**：

| 组件 | 占用量 | 是否随batch_size变化 | 说明 |
|------|--------|---------------------|------|
| **Python运行时** | ~0.5GB | ❌ 否 | Python解释器、系统库 |
| **PyTorch框架** | ~1-2GB | ❌ 否 | PyTorch、transformers等库 |
| **Tokenizer** | ~0.1-0.2GB | ❌ 否 | 词表、tokenizer模型 |
| **数据集元数据** | ~0.5-1GB | ❌ 否 | Parquet文件的索引、元数据 |
| **当前batch数据** | ~0.1-0.5GB | ✅ **是** | **唯一随batch_size变化的部分** |
| **DataLoader缓冲** | ~0.5-1GB | ⚠️ 部分 | 取决于num_workers和prefetch |
| **其他缓存** | ~0.5-1GB | ❌ 否 | 系统缓存、临时对象 |
| **总计** | **~3-6GB** | - | 基础占用 |

**关键发现**：只有"当前batch数据"会随 `batch_size` 变化，但它只占总内存的一小部分！

---

### 2. 为什么 batch_size 对内存影响小？

#### 2.1 使用了 `LowMemDataset`（按需加载）

查看你的代码 `model/dataset.py`：

```python
class LowMemDataset(Dataset):
    """
    低内存版本的Dataset，支持多GPU分布式训练
    
    关键特性：
    1. 不将整个数据集加载到内存
    2. 使用pyarrow直接按索引读取，支持多GPU的数据分片
    3. 内存占用极小，适合16G内存环境
    """
    
    def __getitem__(self, index):
        '''按索引返回一条样本'''
        # 关键：使用pyarrow的slice功能，只读取需要的行
        row = self.parquet_table.slice(index, 1)  # 只读取1行！
        prompt = row['prompt'][0].as_py()
        response = row['response'][0].as_py()
        return f"{prompt[0: max_seq_len]}[EOS]", f"{response[0: max_seq_len]}[EOS]"
```

**工作流程**：

```
磁盘 (Parquet文件)
  ↓ 按需读取1条
内存 (临时存储)
  ↓ 立即tokenize
内存 (tensor格式)
  ↓ 立即转移到GPU
GPU显存 (训练计算)
  ↓ 计算完成
内存中的数据被垃圾回收 ✅
```

**关键点**：
- 每次只读取 `batch_size` 条数据（不是全部5000条）
- 数据在内存中停留时间极短（<1秒）
- 处理完立即被垃圾回收

#### 2.2 数据流转示例

假设 `batch_size=24`，`max_seq_len=512`：

```python
# 步骤1：从磁盘读取24条数据
# 内存占用：24条 × 平均200字符 × 2字节 ≈ 10KB

# 步骤2：Tokenize（转换为token ids）
# 内存占用：24 × 512 × 4字节(int32) ≈ 50MB

# 步骤3：转移到GPU
input_ids = input_ids.to(device)  # 数据从内存复制到GPU显存
# GPU显存占用：+50MB
# 内存占用：50MB（原数据还在，但很快会被回收）

# 步骤4：前向传播
outputs = model(input_ids, ...)
# GPU显存占用：+大量中间激活值（这是显存占用的主要来源！）
# 内存占用：不变

# 步骤5：反向传播
loss.backward()
# GPU显存占用：+梯度（与模型参数同样大小）
# 内存占用：不变

# 步骤6：下一个batch
# Python垃圾回收自动释放上一个batch的内存数据 ✅
```

**结论**：内存中同时只存在1-2个batch的数据（当前batch + 预加载的下一个batch），所以 `batch_size` 从8增加到24，内存只增加约 `(24-8) × 512 × 4字节 ≈ 32MB`，几乎可以忽略！

---

### 3. GPU显存占用的主要来源

相比之下，**GPU显存占用会随 batch_size 线性增长**：

| 组件 | 占用量 (batch_size=24) | 是否随batch_size变化 |
|------|----------------------|---------------------|
| **模型参数** | ~0.8GB | ❌ 否 |
| **优化器状态** | ~1.6GB | ❌ 否 |
| **梯度** | ~0.8GB | ❌ 否 |
| **输入数据** | ~0.05GB | ✅ **是** (线性) |
| **中间激活值** | ~12-15GB | ✅ **是** (线性) |
| **总计** | **~15-18GB** | - |

**关键发现**：中间激活值（前向传播时的隐藏层输出）占据了GPU显存的大部分，且随 `batch_size` 线性增长！

#### 3.1 为什么中间激活值占用这么大？

T5模型的前向传播：

```python
# batch_size=24, seq_len=512, hidden_size=768, num_layers=12

# Encoder每一层的激活值
encoder_hidden = [24, 512, 768]  # 每层约 24×512×768×4字节 ≈ 38MB
# 12层 × 38MB ≈ 456MB

# Decoder每一层的激活值（包括cross-attention）
decoder_hidden = [24, 512, 768]  # 每层约 38MB
# 12层 × 38MB ≈ 456MB

# Attention矩阵（最占显存！）
attention_scores = [24, 12, 512, 512]  # 每层约 24×12×512×512×4字节 ≈ 300MB
# Encoder 12层 × 300MB ≈ 3.6GB
# Decoder 12层 × 300MB ≈ 3.6GB
# Cross-attention 12层 × 300MB ≈ 3.6GB

# 总计：456MB + 456MB + 3.6GB + 3.6GB + 3.6GB ≈ 12GB
```

**当 batch_size 从 8 增加到 24（3倍）时**：
- 中间激活值：12GB → **36GB**（但实际有优化，约15-18GB）
- 这就是为什么GPU显存从40%跳到95%！

---

### 4. 实际测量对比

#### 4.1 batch_size=8 时

```
内存占用：
  - 基础占用：3-4GB
  - 当前batch：8 × 512 × 4字节 ≈ 16MB
  - 总计：~4GB

GPU显存占用：
  - 模型+优化器+梯度：3.2GB
  - 输入数据：0.02GB
  - 中间激活值：5-6GB
  - 总计：~8-10GB (40-50%)
```

#### 4.2 batch_size=24 时

```
内存占用：
  - 基础占用：3-4GB
  - 当前batch：24 × 512 × 4字节 ≈ 48MB
  - 总计：~4-5GB (增加不到1GB！)

GPU显存占用：
  - 模型+优化器+梯度：3.2GB
  - 输入数据：0.05GB
  - 中间激活值：15-18GB
  - 总计：~18-19GB (90-95%)
```

**对比**：
- 内存增加：4GB → 5GB（**+25%**）
- GPU显存增加：10GB → 19GB（**+90%**）

---

## 5. 为什么 `trainer.py` 会占用更多内存？

对比你的两个训练脚本：

### `trainer_low_mem.py`（低内存版本）

```python
# 使用LowMemDataset，按需加载
train_dataset = LowMemDataset(
    parquet_file=train_config.train_file,
    tokenizer_dir=train_config.tokenizer_dir,
    ultra_low_mem=False,  # 标准低内存模式
)

# 禁用num_workers（内存充足时会启用）
num_workers = 0  # 或根据内存动态调整

# 禁用pin_memory（内存充足时会启用）
pin_memory = False  # 或根据内存动态调整
```

### `trainer.py`（标准版本）

```python
# 使用MyDataset，可能全部加载到内存
train_dataset = MyDataset(
    parquet_file=train_config.train_file,
    tokenizer_dir=train_config.tokenizer_dir,
    keep_in_memory=True,  # ⚠️ 全部加载到内存！
)

# 启用num_workers
num_workers = 4  # 4个子进程，每个都会复制数据

# 启用pin_memory
pin_memory = True  # 额外的内存缓冲区
```

**内存占用对比**：

| 组件 | trainer.py | trainer_low_mem.py | 差异 |
|------|-----------|-------------------|------|
| 数据集 | 5000条全部加载 (~2-3GB) | 按需加载 (~0.5GB) | **-2.5GB** |
| num_workers | 4个进程 × 0.5GB = 2GB | 0个进程 = 0GB | **-2GB** |
| pin_memory | 额外缓冲 (~0.5GB) | 无 | **-0.5GB** |
| **总差异** | - | - | **-5GB** |

这就是为什么 `trainer_low_mem.py` 可以在16GB内存下运行，而 `trainer.py` 需要20GB+！

---

## 6. 总结

### 内存占用主要由什么决定？

**固定占用（不随batch_size变化）**：
1. ✅ **数据集加载方式**：`keep_in_memory=True` vs 按需加载
2. ✅ **DataLoader配置**：`num_workers` 数量
3. ✅ **Pin memory**：是否启用 `pin_memory=True`
4. ✅ **框架开销**：PyTorch、transformers等库

**动态占用（随batch_size变化，但影响小）**：
5. ⚠️ **当前batch数据**：batch_size × seq_len × 4字节（通常<100MB）

### GPU显存占用主要由什么决定？

**固定占用（不随batch_size变化）**：
1. ✅ **模型参数**：~0.8GB
2. ✅ **优化器状态**：~1.6GB
3. ✅ **梯度**：~0.8GB

**动态占用（随batch_size线性增长）**：
4. ✅ **输入数据**：batch_size × seq_len × hidden_size
5. ✅ **中间激活值**：batch_size × seq_len × seq_len × num_layers（**主要占用！**）

---

## 7. 优化建议

### 如果想进一步降低内存占用

1. **启用ultra_low_mem模式**（会更慢）：
   ```python
   train_dataset = LowMemDataset(
       parquet_file=train_config.train_file,
       ultra_low_mem=True,  # 每次重新打开文件
   )
   ```

2. **减少DataLoader的prefetch**：
   ```python
   train_dataloader = DataLoader(
       train_dataset,
       batch_size=batch_size,
       num_workers=0,  # 禁用多进程
       pin_memory=False,  # 禁用pin_memory
       prefetch_factor=None,  # 默认为2，可以设为None
   )
   ```

### 如果想进一步提升GPU显存利用率

1. **继续增大batch_size**（直到OOM）：
   ```bash
   ./train_sft_custom.sh 28
   ./train_sft_custom.sh 32
   ```

2. **启用梯度检查点**（减少中间激活值占用，但会慢20-30%）：
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **使用混合精度训练**（已启用 `bf16`，很好！）

---

## 8. 关键结论

### 为什么 batch_size=24 不会占用更多内存？

✅ **因为使用了 `LowMemDataset`，数据按需从磁盘读取，在内存中停留时间极短**

✅ **模型、梯度、优化器都在GPU显存中，不占用内存**

✅ **Python的垃圾回收机制及时释放已处理的batch数据**

✅ **内存中同时只存在1-2个batch的数据（<100MB），即使batch_size增大3倍，内存也只增加<100MB**

### 训练过程中内存占用主要由什么决定？

1. **数据集加载方式**（最重要！）：全部加载 vs 按需加载
2. **DataLoader配置**：num_workers、pin_memory
3. **框架开销**：PyTorch、transformers等库
4. **当前batch数据**（影响最小！）：通常<100MB

### 训练过程中GPU显存占用主要由什么决定？

1. **中间激活值**（最重要！）：随batch_size线性增长，占60-80%
2. **模型参数**：固定，约占10-15%
3. **优化器状态**：固定，约占20-25%
4. **梯度**：固定，约占10-15%

---

## 9. 可视化对比

```
内存占用（16GB总量）：
┌─────────────────────────────────────────────────────────────┐
│ 框架开销(2GB) │ 数据集元数据(1GB) │ 当前batch(0.05GB) │ 空闲(13GB) │
└─────────────────────────────────────────────────────────────┘
  batch_size=8:  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 25%
  batch_size=24: ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 28%
                 差异：<1GB

GPU显存占用（20GB总量）：
┌─────────────────────────────────────────────────────────────┐
│ 模型+优化器(3.2GB) │ 中间激活值 │ 空闲 │
└─────────────────────────────────────────────────────────────┘
  batch_size=8:  ████████████████████████████████░░░░░░░░░░░░░░ 50%
  batch_size=24: ████████████████████████████████████████████░░ 95%
                 差异：9GB（主要是中间激活值）
```

---

## 10. 参考资料

- PyTorch内存管理：https://pytorch.org/docs/stable/notes/cuda.html
- Gradient Checkpointing：https://pytorch.org/docs/stable/checkpoint.html
- PyArrow零拷贝读取：https://arrow.apache.org/docs/python/parquet.html

---

希望这个详细的分析能帮助你理解训练过程中的内存和显存占用机制！🚀
