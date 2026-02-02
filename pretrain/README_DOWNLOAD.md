# 预训练数据集下载和处理指南

本指南说明如何使用 `download_and_process_datasets.py` 脚本从远端下载并处理预训练所需的数据集。

## 📋 数据集概览

本项目使用以下公开数据集进行预训练：

| 数据集 | 描述 | 原始数量 | 清洗后数量 | 来源 |
|--------|------|----------|------------|------|
| webtext2019zh | 社区问答数据集 | 410万 | 260万 | [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus) |
| baike_qa2019 | 百科类问答 | 140万 | 130万 | [百度AI Studio](https://aistudio.baidu.com/datasetdetail/107726) |
| chinese_medical | 医药领域问答 | 79万 | 79万 | [Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data) |
| zhihu_kol | 知乎问答数据 | 100万 | 97万 | [Zhihu-KOL](https://huggingface.co/datasets/wangrui6/Zhihu-KOL) |
| belle | BELLE指令训练数据 | 370万 | 338万 | [BelleGroup](https://huggingface.co/BelleGroup) |
| wiki | 维基百科词条 | - | 119万 | [zhwiki](https://dumps.wikimedia.org/zhwiki/) |

**总计**: 约1023万条数据（预训练集930万 + 评估集2.5万）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install requests tqdm ujson pandas pyarrow fastparquet datasets opencc-python-reimplemented colorlog rich matplotlib
```

### 2. 下载所有数据集

```bash
cd pretrain
python download_and_process_datasets.py --download-all
```

### 3. 处理数据集

```bash
python download_and_process_datasets.py --process
```

### 4. 一键下载并处理

```bash
python download_and_process_datasets.py --download-all --process
```

## 📖 详细使用说明

### 下载特定数据集

如果只想下载部分数据集，可以使用 `--download` 参数：

```bash
# 只下载webtext2019zh和baike_qa
python download_and_process_datasets.py --download webtext2019zh baike_qa

# 下载除wiki外的所有数据集
python download_and_process_datasets.py --download webtext2019zh baike_qa chinese_medical belle zhihu_kol
```

### 跳过维基百科数据集

维基百科数据文件较大（约2.7GB），如果不需要可以跳过：

```bash
python download_and_process_datasets.py --download-all --skip-wiki
```

### 只处理已下载的数据集

如果已经手动下载了数据集，可以直接处理：

```bash
python download_and_process_datasets.py --process
```

## 📁 目录结构

下载和处理后的目录结构如下：

```
ChatLM-mini-Chinese/
├── data/
│   ├── raw_data/                          # 原始数据集
│   │   ├── web_text_zh_train.json
│   │   ├── web_text_zh_valid.json
│   │   ├── web_text_zh_test.json
│   │   ├── baike_qa_train.json
│   │   ├── baike_qa_valid.json
│   │   ├── chinese_medical_dialogue_datasets/
│   │   ├── bell_open_source/
│   │   │   ├── Belle_open_source_1M.json
│   │   │   ├── train_2M_CN.json
│   │   │   └── train_3.5M_CN.json
│   │   ├── zhihu-kol/
│   │   └── zhwiki-latest-pages-articles-multistream.xml.bz2
│   │
│   ├── my_data/                           # 处理后的数据集
│   │   ├── my_web_text_zh.parquet
│   │   ├── my_baike_qa.parquet
│   │   ├── my_chinese_medical_dialogue.parquet
│   │   ├── zhihu_kol.parquet
│   │   ├── my_belll_3M_cn.parquet
│   │   └── wiki_zh_simple.parquet
│   │
│   ├── my_dataset.parquet                 # 合并后的数据集
│   ├── my_dataset.shuffle.parquet         # 打乱后的数据集
│   ├── my_train_dataset.parquet           # 训练集
│   ├── my_test_dataset.parquet            # 测试集
│   ├── my_valid_dataset.parquet           # 验证集
│   ├── my_corpus.txt                      # 文本格式（用于训练tokenizer）
│   └── my_finetune_data_zh.parquet        # 微调数据集
│
└── logs/
    ├── download_datasets.log              # 下载日志
    └── raw_data_process.log               # 处理日志
```

## 🔧 数据处理流程

脚本会自动执行以下处理步骤：

1. **数据清洗**: 
   - 删除重复的标点符号
   - 将英文标点转换为中文标点
   - 删除过短的问答对
   - 过滤低质量数据

2. **格式统一**: 
   - 统一转换为 `{prompt, response}` 格式
   - 保存为 parquet 格式（高效压缩）

3. **数据合并**: 
   - 合并所有处理后的数据集
   - 限制最大长度（默认512字符）

4. **去重**: 
   - 使用MinHash算法去除重复文档
   - 相似度阈值：0.85

5. **数据打乱**: 
   - 随机打乱数据顺序
   - 固定随机种子：23333

6. **数据划分**: 
   - 训练集：91%
   - 测试集：8.75%
   - 验证集：0.25%

7. **格式转换**: 
   - 生成文本格式（用于训练tokenizer）
   - 生成JSON格式（用于其他用途）

## ⚠️ 注意事项

### 1. 网络问题

某些数据集托管在HuggingFace上，国内访问可能较慢。建议：

- 使用代理或镜像站
- 设置HuggingFace镜像：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

### 2. 磁盘空间

确保有足够的磁盘空间：

- 原始数据：约10GB
- 处理后数据：约5GB
- 总计需要：约15-20GB

### 3. 处理时间

完整处理所有数据集可能需要：

- 下载时间：1-3小时（取决于网络速度）
- 处理时间：2-4小时（取决于CPU性能）

### 4. 维基百科数据

维基百科数据需要额外处理：

1. 下载 bz2 文件
2. 使用 WikiExtractor 提取文本
3. 转换为简体中文

详细步骤请参考 `tokenize/process_zhwiki.py`

### 5. 手动下载

如果自动下载失败，可以手动下载数据集并放置到对应目录：

- webtext2019zh: 从 [HuggingFace](https://huggingface.co/datasets/silver/webtext2019zh) 下载
- baike_qa: 从 [百度AI Studio](https://aistudio.baidu.com/datasetdetail/107726) 下载
- chinese_medical: 从 [GitHub](https://github.com/Toyhom/Chinese-medical-dialogue-data) 下载
- belle: 从 [HuggingFace BelleGroup](https://huggingface.co/BelleGroup) 下载
- zhihu_kol: 从 [HuggingFace](https://huggingface.co/datasets/wangrui6/Zhihu-KOL) 下载
- wiki: 从 [Wikimedia](https://dumps.wikimedia.org/zhwiki/) 下载

## 🐛 常见问题

### Q1: 下载速度很慢怎么办？

A: 可以使用代理或者手动下载后放到对应目录。

### Q2: 内存不足怎么办？

A: 脚本使用流式处理，内存占用较小。如果仍然不足，可以：
- 减小 `groups_cnt` 参数（默认50000）
- 分批处理数据集

### Q3: 某个数据集下载失败怎么办？

A: 可以单独下载该数据集：
```bash
python download_and_process_datasets.py --download <dataset_name>
```

### Q4: 如何验证数据集是否正确？

A: 脚本会在处理完成后自动统计数据量，也可以手动运行：
```python
from raw_data_process import count_my_parquet_data
count_my_parquet_data(PROJECT_ROOT + '/data/')
```

### Q5: 出现 "cannot import name 'Logger'" 错误怎么办？

A: 这是因为系统中安装了第三方 `logger` 包导致命名冲突。解决方案：
```bash
# 方案1: 卸载冲突的包（如果不需要）
pip uninstall logger

# 方案2: 已经修复，确保使用最新代码
git pull origin main
```

详细说明请查看 [IMPORT_FIX.md](IMPORT_FIX.md)

## 📊 数据统计

处理完成后，可以查看数据统计信息：

```bash
# 查看日志
cat ../logs/download_datasets.log
cat ../logs/raw_data_process.log

# 查看生成的图表
# 数据长度分布图会保存在 img/sentence_length.png
```

## 🔗 相关链接

- [原始数据处理脚本](raw_data_process.py)
- [维基百科处理脚本](../tokenize/process_zhwiki.py)
- [项目主页](https://github.com/your-repo/ChatLM-mini-Chinese)

## 📝 许可证

各数据集遵循其原始许可证，请在使用前查看相应的许可证信息。

## 🤝 贡献

如果发现问题或有改进建议，欢迎提交Issue或Pull Request。
