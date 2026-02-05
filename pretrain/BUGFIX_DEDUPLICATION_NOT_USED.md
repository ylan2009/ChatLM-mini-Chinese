# 数据处理流程 Bug 修复说明

## 🐛 发现的问题

### 问题描述
在数据处理流程中发现了一个严重的 bug：**去重后的数据没有被后续步骤使用**。

### 问题详情

#### 原始流程（有问题）：
```
1. merge_dataset_as_single_file()
   ↓ 生成: my_dataset.parquet

2. remove_dataset_duplicate_rows()
   ↓ 读取: my_dataset.parquet
   ↓ 生成: my_dataset_no_dulpticates.parquet  ✅ 去重后的数据

3. shuffle_parquet_dataset()
   ↓ 读取: my_dataset.parquet  ❌ 错误！使用了原始数据
   ↓ 生成: my_dataset.shuffle.parquet

4. split_train_valid_test_datasets()
   ↓ 读取: my_dataset.shuffle.parquet  ❌ 包含重复数据
   ↓ 生成: train/valid/test 数据集
```

### 问题影响

1. **❌ 去重操作白做了**
   - `remove_dataset_duplicate_rows` 花费了大量时间（10-15小时）
   - 生成的去重文件 `my_dataset_no_dulpticates.parquet` 没有被使用
   - 浪费了计算资源和时间

2. **❌ 训练数据包含重复**
   - 最终的训练集、验证集、测试集都包含重复数据
   - 可能导致模型过拟合
   - 影响模型训练效果

3. **❌ 数据统计不准确**
   - 报告的去重率是正确的
   - 但实际使用的数据没有去重
   - 统计信息与实际不符

### 问题原因

在 `download_and_process_datasets.py` 的 `process_all_datasets()` 函数中：

```python
# 第 6 步：去重
remove_dataset_duplicate_rows(groups_cnt=50000)
# 生成: my_dataset_no_dulpticates.parquet

# 第 7 步：打乱（错误的代码）
shuffle_parquet_dataset(
    parquet_file=PROJECT_ROOT + '/data/my_dataset.parquet',  # ❌ 使用了原始文件
    shuffle_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',
    seed=23333
)
```

---

## ✅ 修复方案

### 修复后的流程

```
1. merge_dataset_as_single_file()
   ↓ 生成: my_dataset.parquet

2. remove_dataset_duplicate_rows()
   ↓ 读取: my_dataset.parquet
   ↓ 生成: my_dataset_no_dulpticates.parquet  ✅ 去重后的数据

3. shuffle_parquet_dataset()
   ↓ 读取: my_dataset_no_dulpticates.parquet  ✅ 正确！使用去重后的数据
   ↓ 生成: my_dataset.shuffle.parquet

4. split_train_valid_test_datasets()
   ↓ 读取: my_dataset.shuffle.parquet  ✅ 不包含重复数据
   ↓ 生成: train/valid/test 数据集
```

### 修复代码

修改 `download_and_process_datasets.py` 中的第 7 步：

```python
# 7. 打乱数据（使用去重后的数据集）
log.info("打乱数据集...", save_to_file=True)
shuffle_parquet_dataset(
    parquet_file=PROJECT_ROOT + '/data/my_dataset_no_dulpticates.parquet',  # ✅ 使用去重后的文件
    shuffle_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',
    seed=23333
)
```

---

## 📊 修复效果

### 修复前 vs 修复后

| 指标 | 修复前（错误） | 修复后（正确） |
|------|--------------|--------------|
| **去重操作** | 执行但未使用 | 执行并使用 ✅ |
| **训练数据** | 包含重复 ❌ | 不包含重复 ✅ |
| **数据质量** | 低 | 高 ✅ |
| **模型效果** | 可能过拟合 | 正常 ✅ |
| **时间浪费** | 10-15小时白费 | 无浪费 ✅ |

### 数据量变化示例

假设原始数据有 500 万条，去重率为 15%：

```
修复前（错误流程）：
├─ my_dataset.parquet: 500万条
├─ my_dataset_no_dulpticates.parquet: 425万条（未使用）
├─ my_dataset.shuffle.parquet: 500万条（包含重复）
└─ 训练集: 500万条（包含重复）❌

修复后（正确流程）：
├─ my_dataset.parquet: 500万条
├─ my_dataset_no_dulpticates.parquet: 425万条（已使用）✅
├─ my_dataset.shuffle.parquet: 425万条（不包含重复）
└─ 训练集: 425万条（不包含重复）✅
```

---

## 🔍 如何验证修复

### 1. 检查文件大小

修复后，`my_dataset.shuffle.parquet` 的大小应该与 `my_dataset_no_dulpticates.parquet` 相同：

```bash
ls -lh /path/to/data/*.parquet
```

**预期结果**：
```
my_dataset.parquet                  # 较大（包含重复）
my_dataset_no_dulpticates.parquet   # 较小（去重后）
my_dataset.shuffle.parquet          # 与 no_dulpticates 大小相同 ✅
```

### 2. 检查行数

```python
import pyarrow.parquet as pq

# 读取文件行数
original = pq.read_table('my_dataset.parquet').num_rows
dedup = pq.read_table('my_dataset_no_dulpticates.parquet').num_rows
shuffle = pq.read_table('my_dataset.shuffle.parquet').num_rows

print(f"原始数据: {original:,} 行")
print(f"去重后: {dedup:,} 行")
print(f"打乱后: {shuffle:,} 行")

# 验证
assert dedup == shuffle, "打乱后的数据应该与去重后的数据行数相同！"
print("✅ 验证通过！")
```

**预期输出**：
```
原始数据: 5,000,000 行
去重后: 4,250,000 行
打乱后: 4,250,000 行
✅ 验证通过！
```

### 3. 检查日志

查看处理日志，确认去重率：

```bash
grep "去重率" logs/download_datasets.log
```

**预期输出**：
```
去重率: 15.00%
```

---

## ⚠️ 重要提醒

### 如果你已经运行过原始代码

如果你已经使用有 bug 的代码处理过数据，需要：

1. **删除错误的文件**：
   ```bash
   rm /path/to/data/my_dataset.shuffle.parquet
   rm /path/to/data/my_train_data.parquet
   rm /path/to/data/my_valid_data.parquet
   rm /path/to/data/my_test_data.parquet
   ```

2. **重新运行后续步骤**：
   ```bash
   # 从打乱数据开始重新运行
   python download_and_process_datasets.py --process
   ```
   
   或者手动运行：
   ```python
   from raw_data_process import (
       shuffle_parquet_dataset,
       split_train_valid_test_datasets,
   )
   
   # 7. 打乱数据（使用去重后的数据）
   shuffle_parquet_dataset(
       parquet_file=PROJECT_ROOT + '/data/my_dataset_no_dulpticates.parquet',
       shuffle_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',
       seed=23333
   )
   
   # 8. 划分数据集
   split_train_valid_test_datasets(
       source_parquet_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',
       max_len=320,
       groups_cnt=50000
   )
   ```

3. **验证修复**：
   使用上面的验证方法确认数据正确

### 如果你还没有运行过

直接使用修复后的代码即可，无需额外操作。

---

## 📝 相关文件

### 修改的文件

1. **[download_and_process_datasets.py](/Users/twrong/git/code/ChatLM-mini-Chinese/pretrain/download_and_process_datasets.py)**
   - 修改了第 7 步的 `shuffle_parquet_dataset` 调用
   - 从使用 `my_dataset.parquet` 改为 `my_dataset_no_dulpticates.parquet`

### 涉及的函数

1. **`remove_dataset_duplicate_rows`** (raw_data_process.py)
   - 输入: `my_dataset.parquet`
   - 输出: `my_dataset_no_dulpticates.parquet`

2. **`shuffle_parquet_dataset`** (raw_data_process.py)
   - 输入: `my_dataset_no_dulpticates.parquet` ✅（修复后）
   - 输出: `my_dataset.shuffle.parquet`

3. **`split_train_valid_test_datasets`** (raw_data_process.py)
   - 输入: `my_dataset.shuffle.parquet`
   - 输出: `my_train_data.parquet`, `my_valid_data.parquet`, `my_test_data.parquet`

---

## 🎯 总结

### Bug 本质
- 数据处理流程中的文件路径错误
- 去重后的数据没有被后续步骤使用

### 修复方法
- 修改 `shuffle_parquet_dataset` 的输入文件路径
- 从 `my_dataset.parquet` 改为 `my_dataset_no_dulpticates.parquet`

### 修复效果
- ✅ 去重操作不再白费
- ✅ 训练数据不包含重复
- ✅ 数据质量提升
- ✅ 模型训练效果更好

### 验证方法
- 检查文件大小
- 检查行数
- 检查日志

---

**修复日期**: 2026-02-05  
**Bug 严重程度**: 🔴 高（影响数据质量和训练效果）  
**修复状态**: ✅ 已修复  
**测试状态**: ⏳ 待验证