# 快速开始示例

## 方式一：使用一键脚本（推荐）

### Linux/Mac 用户

```bash
cd pretrain
chmod +x setup_datasets.sh
./setup_datasets.sh
```

### Windows 用户

```cmd
cd pretrain
setup_datasets.bat
```

## 方式二：手动执行

### 1. 安装依赖

```bash
pip install requests tqdm ujson pandas pyarrow fastparquet datasets opencc-python-reimplemented rich matplotlib
```

### 2. 下载所有数据集（包括维基百科）

```bash
cd pretrain
python download_and_process_datasets.py --download-all
```

### 3. 下载数据集（不包括维基百科）

```bash
python download_and_process_datasets.py --download webtext2019zh baike_qa chinese_medical belle zhihu_kol
```

### 4. 处理数据集

```bash
python download_and_process_datasets.py --process
```

### 5. 一键下载并处理

```bash
python download_and_process_datasets.py --download-all --process
```

## 方式三：分步执行

### 只下载特定数据集

```bash
# 下载webtext2019zh
python download_and_process_datasets.py --download webtext2019zh

# 下载baike_qa
python download_and_process_datasets.py --download baike_qa

# 下载多个数据集
python download_and_process_datasets.py --download webtext2019zh baike_qa belle
```

### 手动下载后处理

如果网络不好，可以手动下载数据集到 `data/raw_data/` 目录，然后运行：

```bash
python download_and_process_datasets.py --process
```

## 预期输出

处理完成后，会在 `data/` 目录下生成以下文件：

```
data/
├── my_train_dataset.parquet      # 训练集（约930万条）
├── my_test_dataset.parquet       # 测试集（约89万条）
├── my_valid_dataset.parquet      # 验证集（约2.5万条）
├── my_corpus.txt                 # 文本格式语料库
├── my_dataset.parquet            # 原始合并数据集
├── my_dataset.shuffle.parquet    # 打乱后的数据集
└── my_finetune_data_zh.parquet   # 微调数据集
```

## 常见问题

### Q: 下载速度很慢

A: 某些数据集托管在HuggingFace，国内访问较慢。可以：
1. 使用代理
2. 设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`
3. 手动下载后放到 `data/raw_data/` 目录

### Q: 内存不足

A: 脚本使用流式处理，内存占用较小。如果仍不足，可以修改 `groups_cnt` 参数。

### Q: 某个数据集下载失败

A: 可以跳过该数据集，单独下载：
```bash
python download_and_process_datasets.py --download <dataset_name>
```

### Q: 如何验证数据集

A: 查看日志文件：
```bash
cat ../logs/download_datasets.log
cat ../logs/raw_data_process.log
```

## 下一步

数据集准备完成后，可以：

1. **训练 Tokenizer**
   ```bash
   cd ../tokenize
   python train_tokenizer.py
   ```

2. **开始预训练**
   ```bash
   cd ../pretrain
   python pretrain.py
   ```

3. **微调模型**
   ```bash
   cd ../sft
   python sft_train.py
   ```

## 数据集说明

| 数据集 | 数量 | 用途 |
|--------|------|------|
| my_train_dataset.parquet | 930万 | 预训练 |
| my_test_dataset.parquet | 89万 | 测试 |
| my_valid_dataset.parquet | 2.5万 | 验证 |
| my_corpus.txt | 全部 | 训练tokenizer |
| my_finetune_data_zh.parquet | 约100万 | 微调 |

## 时间估算

- 下载时间：1-3小时（取决于网络）
- 处理时间：2-4小时（取决于CPU）
- 总计：3-7小时

## 磁盘空间

- 原始数据：约10GB
- 处理后：约5GB
- 总计：约15-20GB

## 更多信息

详细说明请查看 [README_DOWNLOAD.md](README_DOWNLOAD.md)
