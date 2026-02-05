#!/bin/bash

# 数据管道诊断快速脚本
# 用于快速检查数据处理流程中的问题

echo "=================================="
echo "数据管道诊断工具"
echo "=================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "check_data_pipeline.py" ]; then
    echo "❌ 错误: 请在 pretrain 目录下运行此脚本"
    echo "   cd /data3/ChatLM-mini-Chinese/pretrain"
    exit 1
fi

# 显示菜单
echo "请选择操作:"
echo "1. 检查整个数据处理管道（推荐）"
echo "2. 检查单个文件"
echo "3. 快速检查最终输出文件"
echo "4. 检查原始 Belle 数据"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🔍 检查整个数据处理管道..."
        echo ""
        python check_data_pipeline.py --pipeline
        ;;
    2)
        echo ""
        read -p "请输入文件路径: " filepath
        echo ""
        echo "🔍 检查文件: $filepath"
        echo ""
        python check_data_pipeline.py --file "$filepath" --samples 10
        ;;
    3)
        echo ""
        echo "🔍 检查最终输出文件..."
        echo ""
        
        # 检查可能的输出文件
        files=(
            "../data/my_finetune_data_zh.parquet"
            "../data/my_finetune_data_zh_shuffled.parquet"
            "../data/sft_train.json"
        )
        
        for file in "${files[@]}"; do
            if [ -f "$file" ]; then
                echo "检查: $file"
                python check_data_pipeline.py --file "$file" --samples 5
                echo ""
            fi
        done
        ;;
    4)
        echo ""
        echo "🔍 检查原始 Belle 数据..."
        echo ""
        python check_data_pipeline.py --file "../data/raw_data/belle/Belle_open_source_0.5M.parquet" --samples 5
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "诊断完成！"
echo "=================================="
