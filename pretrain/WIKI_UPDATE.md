# Wiki数据处理方式更新说明

## 📝 更新内容

根据用户需求，已将wiki数据的处理方式从"下载远程数据"改为"使用项目中已有的 `data/wiki.simple.txt` 文件"。

## 🔄 主要修改

### 1. 修改 `download_and_process_datasets.py`

#### 移除wiki下载配置
- ❌ 删除了 `DATASETS_CONFIG` 中的 `wiki` 下载配置
- ✅ 添加了 `check_wiki_simple_file()` 函数来检查文件是否存在

#### 更新下载流程
```python
# 修改前：下载wiki数据
results['wiki'] = download_wiki()

# 修改后：检查wiki.simple.txt文件
results['wiki'] = check_wiki_simple_file()
```

#### 更新处理流程
```python
# 修改前：需要先用WikiExtractor提取
wiki_file = PROJECT_ROOT + '/data/raw_data/zhwiki-latest-pages-articles-multistream.xml.bz2'
if os.path.exists(wiki_file):
    # 需要WikiExtractor处理...

# 修改后：直接使用wiki.simple.txt
wiki_simple_file = PROJECT_ROOT + '/data/wiki.simple.txt'
if os.path.exists(wiki_simple_file):
    process_wiki_simple_to_dataset(groups_cnt=10000, max_len=512)
```

#### 更新命令行参数
- 移除了 `--download` 参数中的 `wiki` 选项
- 更新了 `--skip-wiki` 参数的说明

### 2. 新增 `raw_data_process.py` 中的函数

添加了新函数 `process_wiki_simple_to_dataset()`：

```python
def process_wiki_simple_to_dataset(groups_cnt: int=10000, max_len: int=512, seed: int=23333) -> None:
    '''
    将wiki.simple.txt转换为问答数据集
    注意：wiki.simple.txt已经是简体中文，不需要再转换
    '''
```

**与原函数 `process_zh_wiki_data_to_datset()` 的区别**：
- ✅ 直接读取 `data/wiki.simple.txt`（已经是简体中文）
- ✅ 不需要繁体转简体转换（`OpenCC('t2s')`）
- ✅ 只做基本的标点符号清洗
- ✅ 处理逻辑相同：提取词条标题作为问题，内容作为答案

### 3. 更新文档

#### README_DOWNLOAD.md
- 更新了数据集表格，说明wiki使用已有文件
- 更新了跳过wiki的说明
- 添加了如何生成 `wiki.simple.txt` 的说明

#### QUICKSTART.md
- 更新了下载步骤说明
- 添加了wiki数据的注意事项

## 📂 文件位置说明

### Wiki数据文件位置
```
ChatLM-mini-Chinese/
├── data/
│   └── wiki.simple.txt          # Wiki简体中文数据（必需）
└── tokenize/
    └── process_zhwiki.py        # 生成wiki.simple.txt的脚本
```

### 处理后的输出
```
ChatLM-mini-Chinese/
└── data/
    └── my_data/
        └── wiki_zh_simple.parquet   # 处理后的wiki问答数据集
```

## 🚀 使用方法

### 方式一：如果已有 `data/wiki.simple.txt`

直接运行处理脚本：

```bash
cd pretrain
python download_and_process_datasets.py --download-all --process
```

脚本会自动检测 `data/wiki.simple.txt` 并处理。

### 方式二：如果没有 `data/wiki.simple.txt`

先生成该文件：

```bash
cd tokenize
python process_zhwiki.py
```

然后再运行处理脚本：

```bash
cd ../pretrain
python download_and_process_datasets.py --download-all --process
```

### 方式三：跳过wiki数据

如果不需要wiki数据：

```bash
cd pretrain
python download_and_process_datasets.py --download-all --process --skip-wiki
```

## ✅ 优势

相比之前的方式，新方式有以下优势：

1. **不需要下载大文件**：避免下载2.7GB的wiki原始数据
2. **不需要WikiExtractor**：不需要额外的提取步骤
3. **处理更快**：`wiki.simple.txt` 已经是清洗后的简体中文
4. **更灵活**：可以使用项目中已有的文件，不依赖网络

## 🔍 检查wiki文件

运行脚本时会自动检查：

```bash
python download_and_process_datasets.py --download-all
```

输出示例：

```
============================================================
检查 wiki.simple.txt 文件
============================================================
✓ 找到 wiki.simple.txt 文件: /path/to/data/wiki.simple.txt
  文件大小: 1234.56 MB
```

如果文件不存在：

```
============================================================
检查 wiki.simple.txt 文件
============================================================
✗ 未找到 wiki.simple.txt 文件: /path/to/data/wiki.simple.txt
  请先使用 tokenize/process_zhwiki.py 处理维基百科数据
```

## 📊 数据处理流程

```mermaid
graph LR
    A[data/wiki.simple.txt] --> B[process_wiki_simple_to_dataset]
    B --> C[标点符号清洗]
    C --> D[提取词条标题]
    D --> E[构造问答对]
    E --> F[data/my_data/wiki_zh_simple.parquet]
```

## 🔧 技术细节

### 数据格式

**输入** (`wiki.simple.txt`)：
```
词条标题：
词条内容第一段
词条内容第二段

另一个词条标题：
内容...
```

**输出** (`wiki_zh_simple.parquet`)：
```json
{
  "prompt": "什么是词条标题？",
  "response": "词条内容第一段词条内容第二段..."
}
```

### 问题模板

脚本会随机选择以下模板之一来构造问题：

- 什么是{}？
- 介绍一下{}
- 介绍一下什么是{}
- 写一篇关于{}的介绍
- {}是什么？
- 你知道{}吗？
- 生成关于{}的介绍
- 我想知道关于{}的详细信息
- 你了解{}吗？
- 请解释一下{}
- 对于{}，你有什么了解或看法吗？
- 请告诉我关于{}的信息
- 请简要描述一下{}
- 请提供有关{}的一些详细信息
- 能否解释一下{}是什么?
- 请分享一些关于{}的背景知识
- 请简要概括一下{}
- 能给我一些关于{}的背景资料吗?
- 有关{}的信息可以分享一下吗？
- 你能告诉我{}是什么吗？

## 📝 总结

✅ **已完成的修改**：
1. 移除wiki数据的远程下载功能
2. 添加wiki.simple.txt文件检查功能
3. 新增专门处理wiki.simple.txt的函数
4. 更新所有相关文档和说明
5. 更新命令行参数和帮助信息

✅ **向后兼容**：
- 保留了原有的 `process_zh_wiki_data_to_datset()` 函数
- 不影响其他数据集的下载和处理
- 可以通过 `--skip-wiki` 跳过wiki处理

✅ **用户体验**：
- 自动检测文件是否存在
- 提供清晰的错误提示
- 给出解决方案的指引

现在可以直接使用项目中已有的 `data/wiki.simple.txt` 文件进行数据处理了！🎉
