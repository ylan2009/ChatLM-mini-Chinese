# 数据集下载功能改造说明

## 改造目标

将原有的 `raw_data_process.py` 改造为可以从远端自动下载数据集的模式，解决用户没有原始数据集的问题。

## 新增文件

### 1. download_and_process_datasets.py
**位置**: `pretrain/download_and_process_datasets.py`

**功能**:
- 从远端下载所有预训练所需的数据集
- 支持选择性下载特定数据集
- 自动调用原有的数据处理函数
- 完整的错误处理和日志记录

**主要特性**:
- ✅ 支持断点续传（已下载的文件会跳过）
- ✅ 显示下载进度条
- ✅ 支持命令行参数控制
- ✅ 完整的日志记录
- ✅ 自动创建必要的目录

**使用方法**:
```bash
# 下载所有数据集
python download_and_process_datasets.py --download-all

# 下载特定数据集
python download_and_process_datasets.py --download webtext2019zh baike_qa

# 处理已下载的数据集
python download_and_process_datasets.py --process

# 一键下载并处理
python download_and_process_datasets.py --download-all --process
```

### 2. setup_datasets.sh
**位置**: `pretrain/setup_datasets.sh`

**功能**:
- Linux/Mac 一键脚本
- 自动检查Python环境
- 自动安装依赖
- 交互式选择是否下载维基百科数据
- 显示友好的进度提示

**使用方法**:
```bash
chmod +x setup_datasets.sh
./setup_datasets.sh
```

### 3. setup_datasets.bat
**位置**: `pretrain/setup_datasets.bat`

**功能**:
- Windows 一键脚本
- 功能与 shell 脚本相同
- 适配 Windows 命令行环境

**使用方法**:
```cmd
setup_datasets.bat
```

### 4. README_DOWNLOAD.md
**位置**: `pretrain/README_DOWNLOAD.md`

**内容**:
- 📋 数据集概览表格
- 🚀 快速开始指南
- 📖 详细使用说明
- 📁 目录结构说明
- 🔧 数据处理流程
- ⚠️ 注意事项
- 🐛 常见问题解答
- 📊 数据统计信息

### 5. QUICKSTART.md
**位置**: `pretrain/QUICKSTART.md`

**内容**:
- 三种使用方式（一键脚本、手动执行、分步执行）
- 预期输出说明
- 常见问题快速解答
- 下一步操作指引
- 时间和空间估算

## 数据集来源配置

脚本支持以下数据集的自动下载：

| 数据集 | 来源 | 下载方式 |
|--------|------|----------|
| webtext2019zh | HuggingFace | HTTP下载 |
| baike_qa | HuggingFace | HTTP下载 |
| chinese_medical | GitHub | HTTP下载+解压 |
| belle | HuggingFace | HTTP下载 |
| zhihu_kol | HuggingFace | datasets库 |
| wiki | Wikimedia | HTTP下载 |

## 技术实现

### 下载功能
- 使用 `requests` 库进行HTTP下载
- 使用 `tqdm` 显示进度条
- 支持流式下载，避免内存溢出
- 自动检测文件是否已存在

### 数据处理
- 复用原有的 `raw_data_process.py` 中的处理函数
- 保持原有的数据清洗逻辑不变
- 使用相同的输出格式（parquet）

### 错误处理
- 完整的异常捕获
- 详细的日志记录
- 友好的错误提示

## 与原代码的关系

### 保留的功能
- ✅ 所有数据处理函数保持不变
- ✅ 数据清洗逻辑保持不变
- ✅ 输出格式保持不变
- ✅ 目录结构保持不变

### 新增的功能
- ✅ 自动下载数据集
- ✅ 命令行参数控制
- ✅ 进度显示
- ✅ 一键脚本

### 不需要修改的文件
- `raw_data_process.py` - 保持原样，仍可独立使用
- 其他所有文件 - 无需修改

## 使用流程

### 完整流程
```
1. 运行一键脚本
   ↓
2. 自动下载数据集
   ↓
3. 自动处理数据集
   ↓
4. 生成训练数据
```

### 分步流程
```
1. 下载数据集
   python download_and_process_datasets.py --download-all
   ↓
2. 处理数据集
   python download_and_process_datasets.py --process
```

## 优势

### 1. 用户友好
- 无需手动下载数据集
- 一键完成所有操作
- 清晰的进度提示
- 详细的文档说明

### 2. 灵活性
- 支持选择性下载
- 支持断点续传
- 支持手动下载后处理
- 支持跳过特定数据集

### 3. 可维护性
- 代码结构清晰
- 配置集中管理
- 完整的日志记录
- 详细的注释

### 4. 兼容性
- 支持 Linux/Mac/Windows
- 不影响原有代码
- 可独立使用
- 向后兼容

## 注意事项

### 网络问题
- HuggingFace 在国内访问较慢
- 建议使用镜像或代理
- 支持手动下载后处理

### 磁盘空间
- 原始数据约10GB
- 处理后约5GB
- 总计需要15-20GB

### 处理时间
- 下载：1-3小时
- 处理：2-4小时
- 总计：3-7小时

### 维基百科数据
- 文件较大（2.7GB）
- 需要额外处理步骤
- 可选择跳过

## 测试建议

### 基本测试
```bash
# 测试下载单个数据集
python download_and_process_datasets.py --download webtext2019zh

# 测试处理功能
python download_and_process_datasets.py --process

# 测试一键脚本
./setup_datasets.sh
```

### 完整测试
```bash
# 测试完整流程
python download_and_process_datasets.py --download-all --process
```

## 后续改进建议

### 短期
1. 添加下载速度显示
2. 支持多线程下载
3. 添加数据集校验功能

### 长期
1. 支持更多数据源
2. 添加数据集版本管理
3. 支持增量更新
4. 添加数据质量检查

## 总结

本次改造成功实现了：
- ✅ 从远端自动下载数据集
- ✅ 保持原有代码不变
- ✅ 提供多种使用方式
- ✅ 完善的文档说明
- ✅ 跨平台支持

用户现在可以：
- 无需手动下载数据集
- 一键完成所有准备工作
- 灵活选择需要的数据集
- 轻松开始模型训练

## 文件清单

新增文件：
- ✅ `pretrain/download_and_process_datasets.py` - 主脚本
- ✅ `pretrain/setup_datasets.sh` - Linux/Mac一键脚本
- ✅ `pretrain/setup_datasets.bat` - Windows一键脚本
- ✅ `pretrain/README_DOWNLOAD.md` - 详细文档
- ✅ `pretrain/QUICKSTART.md` - 快速开始
- ✅ `pretrain/CHANGES.md` - 本文档

保持不变：
- ✅ `pretrain/raw_data_process.py` - 原始处理脚本
- ✅ 其他所有文件

## 联系方式

如有问题或建议，请：
- 查看文档：README_DOWNLOAD.md
- 查看示例：QUICKSTART.md
- 提交Issue
- 发起Pull Request
