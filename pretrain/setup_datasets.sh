#!/bin/bash
# 预训练数据集一键下载和处理脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  ChatLM-mini-Chinese 预训练数据集准备"
echo "=========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3"
    exit 1
fi

echo "✓ Python 环境检查通过"
echo ""

# 检查并安装依赖
echo "检查依赖库..."
pip3 install -q requests tqdm ujson pandas pyarrow fastparquet datasets opencc-python-reimplemented colorlog rich matplotlib

echo "✓ 依赖库安装完成"
echo ""

# 询问是否下载维基百科数据
echo "是否下载维基百科数据？(文件约2.7GB，下载和处理时间较长)"
read -p "输入 y 下载，输入 n 跳过 [y/N]: " download_wiki

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 下载数据集
echo ""
echo "=========================================="
echo "  步骤 1/2: 下载数据集"
echo "=========================================="
echo ""

if [[ "$download_wiki" =~ ^[Yy]$ ]]; then
    python3 download_and_process_datasets.py --download-all
else
    python3 download_and_process_datasets.py --download webtext2019zh baike_qa chinese_medical belle zhihu_kol
fi

# 处理数据集
echo ""
echo "=========================================="
echo "  步骤 2/2: 处理数据集"
echo "=========================================="
echo ""

python3 download_and_process_datasets.py --process

echo ""
echo "=========================================="
echo "  ✓ 所有任务完成！"
echo "=========================================="
echo ""
echo "生成的文件："
echo "  - 训练集: ../data/my_train_dataset.parquet"
echo "  - 测试集: ../data/my_test_dataset.parquet"
echo "  - 验证集: ../data/my_valid_dataset.parquet"
echo "  - 语料库: ../data/my_corpus.txt"
echo ""
echo "日志文件："
echo "  - 下载日志: ../logs/download_datasets.log"
echo "  - 处理日志: ../logs/raw_data_process.log"
echo ""
echo "下一步："
echo "  1. 训练 tokenizer: cd ../tokenize && python train_tokenizer.py"
echo "  2. 开始预训练: cd ../pretrain && python pretrain.py"
echo ""
