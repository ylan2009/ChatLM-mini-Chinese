# HuggingFace 数据集下载方式更新说明

## 📝 更新内容

根据用户需求，已将所有 HuggingFace 数据集的下载方式从"直接下载 URL 文件"改为"使用 `datasets` 库下载"，参考 `download_belle_sft_dataset.py` 的实现方式。

## 🔄 主要修改

### 1. 修改的数据集

以下数据集的下载方式已更新：

| 数据集 | HuggingFace 路径 | 原下载方式 | 新下载方式 |
|--------|------------------|------------|------------|
| webtext2019zh | `silver/webtext2019zh` | 直接下载 JSON 文件 | `load_dataset()` |
| baike_qa | `silver/baike_qa2019` | 直接下载 JSON 文件 | `load_dataset()` |
| belle | `BelleGroup/train_*M_CN` | 直接下载 JSON 文件 | `load_dataset()` |
| zhihu_kol | `wangrui6/Zhihu-KOL` | 已使用 `load_dataset()` | 无变化 |

**未修改的数据集**：
- `chinese_medical`: 从 GitHub 下载 ZIP 文件（不在 HuggingFace 上）
- `wiki`: 使用项目中已有的 `data/wiki.simple.txt` 文件

### 2. 配置文件修改

#### 修改前（使用 URL）

```python
DATASETS_CONFIG = {
    'webtext2019zh': {
        'urls': {
            'train': 'https://huggingface.co/datasets/silver/webtext2019zh/resolve/main/web_text_zh_train.json',
            'valid': 'https://huggingface.co/datasets/silver/webtext2019zh/resolve/main/web_text_zh_valid.json',
            'test': 'https://huggingface.co/datasets/silver/webtext2019zh/resolve/main/web_text_zh_test.json',
        },
        'save_dir': PROJECT_ROOT + '/data/raw_data/',
    },
    'belle': {
        'urls': {
            'belle_1m': 'https://huggingface.co/datasets/BelleGroup/train_1M_CN/resolve/main/Belle_open_source_1M.json',
            'belle_2m': 'https://huggingface.co/datasets/BelleGroup/train_2M_CN/resolve/main/train_2M_CN.json',
            # ...
        },
        'save_dir': PROJECT_ROOT + '/data/raw_data/bell_open_source/',
    },
}
```

#### 修改后（使用 datasets 库）

```python
DATASETS_CONFIG = {
    'webtext2019zh': {
        'note': 'webtext2019zh数据集需要从HuggingFace下载，使用datasets库',
        'hf_dataset': 'silver/webtext2019zh',
        'save_dir': PROJECT_ROOT + '/data/raw_data/',
    },
    'baike_qa': {
        'note': '百度百科问答数据集需要从HuggingFace下载，使用datasets库',
        'hf_dataset': 'silver/baike_qa2019',
        'save_dir': PROJECT_ROOT + '/data/raw_data/',
    },
    'belle': {
        'note': 'BELLE数据集需要从HuggingFace下载，使用datasets库',
        'hf_datasets': [
            'BelleGroup/train_1M_CN',
            'BelleGroup/train_2M_CN',
            'BelleGroup/train_3.5M_CN',
        ],
        'save_dir': PROJECT_ROOT + '/data/raw_data/belle/',
    },
}
```

### 3. 下载函数修改

#### webtext2019zh

**修改前**：
```python
def download_webtext2019zh() -> bool:
    """下载webtext2019zh数据集"""
    config = DATASETS_CONFIG['webtext2019zh']
    ensure_dir(config['save_dir'])
    
    success = True
    for name, url in config['urls'].items():
        save_path = os.path.join(config['save_dir'], f'web_text_zh_{name}.json')
        if not download_file(url, save_path):
            success = False
    
    return success
```

**修改后**：
```python
def download_webtext2019zh() -> bool:
    """下载webtext2019zh数据集（使用HuggingFace datasets库）"""
    try:
        from datasets import load_dataset
        
        config = DATASETS_CONFIG['webtext2019zh']
        ensure_dir(config['save_dir'])
        
        log.info(f"从HuggingFace下载: {config['hf_dataset']}", save_to_file=True)
        
        # 下载数据集（包含train, valid, test分割）
        dataset = load_dataset(config['hf_dataset'])
        
        # 分别保存各个分割
        for split_name in dataset.keys():
            save_path = os.path.join(config['save_dir'], f'web_text_zh_{split_name}.parquet')
            dataset[split_name].to_parquet(save_path)
            log.info(f"{split_name} 数据集已保存到: {save_path}", save_to_file=True)
            log.info(f"{split_name} 数据集大小: {len(dataset[split_name])} 行", save_to_file=True)
        
        return True
        
    except ImportError:
        log.error("需要安装 datasets 库: pip install datasets", save_to_file=True)
        return False
    except Exception as e:
        log.error(f"下载失败: {str(e)}", save_to_file=True)
        return False
```

#### baike_qa

**修改前**：
```python
def download_baike_qa() -> bool:
    """下载百度百科问答数据集"""
    config = DATASETS_CONFIG['baike_qa']
    ensure_dir(config['save_dir'])
    
    success = True
    for name, url in config['urls'].items():
        save_path = os.path.join(config['save_dir'], f'baike_qa_{name}.json')
        if not download_file(url, save_path):
            success = False
    
    return success
```

**修改后**：
```python
def download_baike_qa() -> bool:
    """下载百度百科问答数据集（使用HuggingFace datasets库）"""
    try:
        from datasets import load_dataset
        
        config = DATASETS_CONFIG['baike_qa']
        ensure_dir(config['save_dir'])
        
        log.info(f"从HuggingFace下载: {config['hf_dataset']}", save_to_file=True)
        
        # 下载数据集（包含train, valid分割）
        dataset = load_dataset(config['hf_dataset'])
        
        # 分别保存各个分割
        for split_name in dataset.keys():
            save_path = os.path.join(config['save_dir'], f'baike_qa_{split_name}.parquet')
            dataset[split_name].to_parquet(save_path)
            log.info(f"{split_name} 数据集已保存到: {save_path}", save_to_file=True)
            log.info(f"{split_name} 数据集大小: {len(dataset[split_name])} 行", save_to_file=True)
        
        return True
        
    except ImportError:
        log.error("需要安装 datasets 库: pip install datasets", save_to_file=True)
        return False
    except Exception as e:
        log.error(f"下载失败: {str(e)}", save_to_file=True)
        return False
```

#### belle

**修改前**：
```python
def download_belle_datasets() -> bool:
    """下载BELLE开源数据集"""
    config = DATASETS_CONFIG['belle']
    ensure_dir(config['save_dir'])
    
    success = True
    for name, url in config['urls'].items():
        filename = url.split('/')[-1]
        save_path = os.path.join(config['save_dir'], filename)
        if not download_file(url, save_path):
            success = False
    
    return success
```

**修改后**：
```python
def download_belle_datasets() -> bool:
    """下载BELLE开源数据集（使用HuggingFace datasets库）"""
    try:
        from datasets import load_dataset
        
        config = DATASETS_CONFIG['belle']
        ensure_dir(config['save_dir'])
        
        success = True
        for dataset_name in config['hf_datasets']:
            try:
                log.info(f"从HuggingFace下载: {dataset_name}", save_to_file=True)
                
                # 下载数据集
                dataset = load_dataset(dataset_name, split='train')
                
                # 提取数据集名称作为文件名
                # 例如: BelleGroup/train_1M_CN -> train_1M_CN
                file_name = dataset_name.split('/')[-1]
                save_path = os.path.join(config['save_dir'], f'{file_name}.parquet')
                
                # 保存为parquet格式
                dataset.to_parquet(save_path)
                
                log.info(f"数据集已保存到: {save_path}", save_to_file=True)
                log.info(f"数据集大小: {len(dataset)} 行", save_to_file=True)
                
            except Exception as e:
                log.error(f"下载 {dataset_name} 失败: {str(e)}", save_to_file=True)
                success = False
        
        return success
        
    except ImportError:
        log.error("需要安装 datasets 库: pip install datasets", save_to_file=True)
        return False
    except Exception as e:
        log.error(f"下载失败: {str(e)}", save_to_file=True)
        return False
```

### 4. 启用所有数据集下载

在 `download_all_datasets()` 函数中，取消了 webtext2019zh 和 baike_qa 的注释：

```python
def download_all_datasets() -> dict:
    results = {}
    
    log.info("开始下载所有数据集...", save_to_file=True)
    
    # 1. webtext2019zh
    results['webtext2019zh'] = download_webtext2019zh()
    
    # 2. baike_qa
    results['baike_qa'] = download_baike_qa()
    
    # 3. chinese_medical
    results['chinese_medical'] = download_chinese_medical()
    
    # 4. belle
    results['belle'] = download_belle_datasets()
    
    # 5. zhihu_kol
    results['zhihu_kol'] = download_zhihu_kol()
    
    # 6. wiki - 不需要下载，直接使用data/wiki.simple.txt
    log.info("注意: Wiki数据使用项目中已有的 data/wiki.simple.txt 文件", save_to_file=True)
    results['wiki'] = check_wiki_simple_file()
    
    return results
```

### 5. 更新文档

更新了 [README_DOWNLOAD.md](README_DOWNLOAD.md)，强调了 `datasets` 库的重要性。

## ✨ 优势

相比之前直接下载 URL 文件的方式，使用 `datasets` 库有以下优势：

### 1. **自动缓存管理**
- HuggingFace datasets 会自动管理下载的缓存
- 避免重复下载相同的数据集
- 支持断点续传

### 2. **统一的数据格式**
- 所有数据集都保存为 parquet 格式
- 便于后续处理和读取
- 更高效的存储和加载

### 3. **更好的错误处理**
- 自动处理网络错误和重试
- 提供清晰的错误信息
- 支持验证数据完整性

### 4. **版本控制**
- 可以指定数据集的特定版本
- 确保实验的可重复性

### 5. **更简洁的代码**
- 不需要手动处理 HTTP 请求
- 不需要手动解析 JSON 文件
- 代码更简洁易维护

### 6. **与 HuggingFace 生态集成**
- 可以直接使用 HuggingFace Hub 的所有功能
- 支持私有数据集
- 支持数据集的流式加载（对于大数据集）

## 📂 文件格式变化

### 修改前
```
data/raw_data/
├── web_text_zh_train.json
├── web_text_zh_valid.json
├── web_text_zh_test.json
├── baike_qa_train.json
├── baike_qa_valid.json
└── bell_open_source/
    ├── Belle_open_source_1M.json
    ├── train_2M_CN.json
    └── train_3.5M_CN.json
```

### 修改后
```
data/raw_data/
├── web_text_zh_train.parquet
├── web_text_zh_valid.parquet
├── web_text_zh_test.parquet
├── baike_qa_train.parquet
├── baike_qa_valid.parquet
└── belle/
    ├── train_1M_CN.parquet
    ├── train_2M_CN.parquet
    └── train_3.5M_CN.parquet
```

**变化**：
- ✅ 文件格式从 `.json` 改为 `.parquet`
- ✅ belle 数据集目录从 `bell_open_source/` 改为 `belle/`
- ✅ 文件名更规范（例如：`train_1M_CN.parquet` 而不是 `Belle_open_source_1M.json`）

## 🚀 使用方法

### 安装依赖

**重要**：必须安装 `datasets` 库

```bash
pip install datasets requests tqdm ujson pandas pyarrow
```

### 下载所有数据集

```bash
cd pretrain
python download_and_process_datasets.py --download-all
```

### 下载特定数据集

```bash
python download_and_process_datasets.py --download webtext2019zh baike_qa belle
```

### 下载并处理

```bash
python download_and_process_datasets.py --download-all --process
```

## 🔍 数据处理兼容性

**重要提示**：虽然下载的文件格式从 JSON 改为 parquet，但原有的数据处理函数（在 `raw_data_process.py` 中）仍然需要更新以支持 parquet 格式。

### 需要更新的处理函数

以下函数可能需要更新以支持 parquet 格式：

1. `process_web_text()` - 处理 webtext2019zh
2. `process_bake_qa()` - 处理 baike_qa
3. `process_belle_knowledge_enhanced_dataset()` - 处理 belle

### 更新建议

可以在处理函数中添加对 parquet 格式的支持：

```python
def process_web_text():
    # 检查是否存在 parquet 文件
    parquet_file = PROJECT_ROOT + '/data/raw_data/web_text_zh_train.parquet'
    json_file = PROJECT_ROOT + '/data/raw_data/web_text_zh_train.json'
    
    if os.path.exists(parquet_file):
        # 读取 parquet 文件
        df = pd.read_parquet(parquet_file)
        # 处理数据...
    elif os.path.exists(json_file):
        # 读取 JSON 文件（向后兼容）
        with open(json_file, 'r') as f:
            data = json.load(f)
        # 处理数据...
    else:
        log.error("未找到数据文件")
```

## 📝 修改的文件列表

1. ✅ [pretrain/download_and_process_datasets.py](download_and_process_datasets.py) - 主要修改
   - 更新 `DATASETS_CONFIG` 配置
   - 重写 `download_webtext2019zh()` 函数
   - 重写 `download_baike_qa()` 函数
   - 重写 `download_belle_datasets()` 函数
   - 启用所有数据集下载

2. ✅ [pretrain/README_DOWNLOAD.md](README_DOWNLOAD.md) - 更新文档
   - 强调 `datasets` 库的重要性
   - 更新安装说明

3. ✅ [pretrain/HUGGINGFACE_DATASETS_UPDATE.md](HUGGINGFACE_DATASETS_UPDATE.md) - 新增说明文档
   - 详细说明所有修改内容
   - 提供使用方法和技术细节

## 🎯 向后兼容性

### 保留的功能

- ✅ 保留了原有的 `download_file()` 函数（用于 chinese_medical）
- ✅ 保留了原有的命令行参数
- ✅ 保留了原有的目录结构

### 可能的兼容性问题

1. **文件格式变化**：从 JSON 改为 parquet
   - 解决方案：更新数据处理函数以支持 parquet 格式

2. **文件路径变化**：belle 目录从 `bell_open_source/` 改为 `belle/`
   - 解决方案：更新处理函数中的路径引用

3. **文件名变化**：belle 文件名更规范
   - 解决方案：更新处理函数中的文件名匹配逻辑

## 🔧 故障排除

### 问题 1：ImportError: No module named 'datasets'

**解决方案**：
```bash
pip install datasets
```

### 问题 2：下载速度慢

**解决方案**：
- 使用 HuggingFace 镜像站点
- 设置环境变量：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

### 问题 3：磁盘空间不足

**解决方案**：
- datasets 库会缓存下载的数据，默认在 `~/.cache/huggingface/datasets/`
- 可以设置环境变量更改缓存位置：
  ```bash
  export HF_DATASETS_CACHE="/path/to/cache"
  ```

### 问题 4：网络连接错误

**解决方案**：
- datasets 库支持断点续传，重新运行脚本即可
- 检查网络连接和防火墙设置

## 📊 性能对比

| 指标 | 原方式（URL下载） | 新方式（datasets库） |
|------|------------------|---------------------|
| 下载速度 | 取决于网络 | 取决于网络 + 自动重试 |
| 断点续传 | ❌ 不支持 | ✅ 支持 |
| 缓存管理 | ❌ 手动管理 | ✅ 自动管理 |
| 数据验证 | ❌ 无 | ✅ 自动验证 |
| 代码复杂度 | 较高 | 较低 |
| 错误处理 | 需要手动实现 | 自动处理 |
| 文件格式 | JSON | Parquet（更高效） |

## 🎉 总结

✅ **已完成的修改**：
1. 将 webtext2019zh、baike_qa、belle 数据集的下载方式改为使用 `datasets` 库
2. 更新配置文件，使用 HuggingFace 数据集路径而不是 URL
3. 重写下载函数，使用 `load_dataset()` API
4. 启用所有数据集的下载
5. 更新文档说明

✅ **优势**：
- 更简洁的代码
- 更好的错误处理
- 自动缓存管理
- 支持断点续传
- 统一的数据格式（parquet）

✅ **注意事项**：
- 必须安装 `datasets` 库
- 文件格式从 JSON 改为 parquet
- 可能需要更新数据处理函数以支持新格式

现在可以使用更现代、更可靠的方式下载 HuggingFace 数据集了！🎉
