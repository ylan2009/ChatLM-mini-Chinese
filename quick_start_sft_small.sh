#!/bin/bash
# 一键准备小数据集并开始SFT训练 - 16G内存优化版

set -e  # 遇到错误立即退出

echo "=========================================="
echo "小数据集SFT训练 - 16G内存优化版"
echo "=========================================="
echo ""

# 默认配置
INPUT_FILE="data/sft_train_dataset.parquet"
NUM_SAMPLES=5000
VALID_RATIO=0.1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --valid-ratio)
            VALID_RATIO="$2"
            shift 2
            ;;
        --help)
            echo "使用方法："
            echo "  $0 [选项]"
            echo ""
            echo "选项："
            echo "  --input FILE        输入数据文件（默认：data/sft_train_dataset.parquet）"
            echo "  --samples N         采样数量（默认：5000）"
            echo "  --valid-ratio R     验证集比例（默认：0.1）"
            echo "  --help              显示此帮助信息"
            echo ""
            echo "示例："
            echo "  # 使用默认配置（5000样本）"
            echo "  $0"
            echo ""
            echo "  # 使用3000样本"
            echo "  $0 --samples 3000"
            echo ""
            echo "  # 从JSON文件采样"
            echo "  $0 --input data/alpaca_gpt4_data_zh.json --samples 5000"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误：输入文件不存在: $INPUT_FILE"
    echo ""
    echo "可用的数据文件："
    ls -lh data/*.parquet data/*.json 2>/dev/null || echo "  未找到数据文件"
    exit 1
fi

echo "配置信息："
echo "  输入文件: $INPUT_FILE"
echo "  采样数量: $NUM_SAMPLES"
echo "  验证集比例: $VALID_RATIO"
echo ""

# 步骤1：准备小数据集
echo "=========================================="
echo "步骤1/3：准备小数据集"
echo "=========================================="
python prepare_small_sft_data.py \
    --input "$INPUT_FILE" \
    --output "data/sft_${NUM_SAMPLES}.parquet" \
    --num_samples $NUM_SAMPLES \
    --valid_ratio $VALID_RATIO

if [ $? -ne 0 ]; then
    echo "❌ 数据准备失败"
    exit 1
fi

echo ""
echo "✅ 数据准备完成"
echo ""

# 步骤2：更新配置文件
echo "=========================================="
echo "步骤2/3：更新配置文件"
echo "=========================================="

TRAIN_FILE="data/sft_${NUM_SAMPLES}_train.parquet"
VALID_FILE="data/sft_${NUM_SAMPLES}_valid.parquet"

echo "更新 config.py 中的 TrainConfigSFTSmall 配置..."
echo "  train_file: $TRAIN_FILE"
echo "  validation_file: $VALID_FILE"

# 使用Python更新配置
python -c "
import re

config_file = 'config.py'
with open(config_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 更新TrainConfigSFTSmall中的train_file和validation_file
content = re.sub(
    r\"(class TrainConfigSFTSmall:.*?train_file: str = PROJECT_ROOT \+ ')([^']+)('\",
    r\"\1/$TRAIN_FILE\3\",
    content,
    flags=re.DOTALL
)
content = re.sub(
    r\"(class TrainConfigSFTSmall:.*?validation_file: str = PROJECT_ROOT \+ ')([^']+)('\",
    r\"\1/$VALID_FILE\3\",
    content,
    flags=re.DOTALL
)

with open(config_file, 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ 配置文件已更新')
"

if [ $? -ne 0 ]; then
    echo "⚠️  自动更新配置失败，请手动修改 config.py"
    echo "   将 TrainConfigSFTSmall 中的："
    echo "   train_file = PROJECT_ROOT + '/$TRAIN_FILE'"
    echo "   validation_file = PROJECT_ROOT + '/$VALID_FILE'"
    echo ""
    read -p "是否继续训练？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# 步骤3：开始训练
echo "=========================================="
echo "步骤3/3：开始训练"
echo "=========================================="
echo ""
echo "训练配置："
echo "  - 训练样本: $NUM_SAMPLES"
echo "  - 验证样本: $(echo "$NUM_SAMPLES * $VALID_RATIO" | bc | cut -d. -f1)"
echo "  - Batch size: 1 per GPU"
echo "  - 梯度累积: 8 steps"
echo "  - Epochs: 3"
echo "  - 预期内存: 8-10GB (双GPU)"
echo ""
echo "开始训练（按Ctrl+C可中断）..."
echo ""

# 等待3秒让用户看清信息
sleep 3

# 启动训练
accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "模型保存位置："
echo "  - 最佳模型: model_save/sft_small/chat_small_t5.best.bin"
echo "  - 训练状态: model_save/sft_small/train_latest_state_sft_small/"
echo ""
echo "查看训练日志："
echo "  tail -f logs/*.log"
echo ""
