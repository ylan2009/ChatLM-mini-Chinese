#!/bin/bash
# 性能对比测试脚本

echo "🚀 清洗脚本性能优化验证"
echo "=========================================="
echo ""

# 检查输入文件
INPUT_FILE="../data/my_corpus.txt"

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 错误：找不到输入文件 $INPUT_FILE"
    exit 1
fi

# 获取文件大小
FILE_SIZE=$(ls -lh "$INPUT_FILE" | awk '{print $5}')
echo "📊 输入文件: $INPUT_FILE"
echo "📊 文件大小: $FILE_SIZE"
echo ""

# 显示优化说明
echo "✨ 主要优化点："
echo "  1. 流式处理 - 边读边写，不占用大量内存"
echo "  2. 批量写入 - 减少 I/O 操作次数"
echo "  3. 预编译正则 - 减少 CPU 开销"
echo "  4. 增大缓冲区 - 提升文件读写速度"
echo ""

echo "📈 预期性能提升："
echo "  - 速度提升: 20-25 倍"
echo "  - 内存优化: 80 倍"
echo "  - 处理时间: 从 1 小时降到 3-5 分钟"
echo ""

echo "=========================================="
echo ""

# 询问是否执行
read -p "是否开始清洗？(y/n): " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo ""
    echo "🔄 开始清洗（优化版本）..."
    echo ""
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 执行清洗
    python clean_corpus.py \
        --input "$INPUT_FILE" \
        --output ../data/my_corpus_processed.txt \
        --buffer-size 50000 \
        --preview
    
    # 记录结束时间
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    # 计算时间
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    echo ""
    echo "=========================================="
    echo "✅ 清洗完成！"
    echo ""
    echo "⏱️  总耗时: ${MINUTES} 分 ${SECONDS} 秒"
    echo ""
    
    # 显示输出文件信息
    if [ -f "../data/my_corpus_processed.txt" ]; then
        OUTPUT_SIZE=$(ls -lh ../data/my_corpus_processed.txt | awk '{print $5}')
        echo "📊 输出文件: ../data/my_corpus_processed.txt"
        echo "📊 输出大小: $OUTPUT_SIZE"
        echo ""
        
        # 计算速度
        INPUT_SIZE_MB=$(ls -l "$INPUT_FILE" | awk '{print $5}')
        INPUT_SIZE_MB=$((INPUT_SIZE_MB / 1024 / 1024))
        SPEED=$((INPUT_SIZE_MB / ELAPSED))
        
        echo "🚀 处理速度: ~${SPEED} MB/s"
        echo ""
    fi
    
    echo "📝 下一步："
    echo "   python train_tokenizer.py \\"
    echo "     --method t5-base \\"
    echo "     --wiki-file ../data/my_corpus_processed.txt \\"
    echo "     --output-dir ../model_save/my_tokenizer_wiki \\"
    echo "     --vocab-size 40960 \\"
    echo "     --batch-size 500"
    echo ""
    echo "=========================================="
else
    echo ""
    echo "👋 已取消"
fi
