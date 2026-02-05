#!/bin/bash
# 文本语料清洗快速使用脚本

echo "🚀 文本语料清洗脚本 - 快速使用指南"
echo "=========================================="
echo ""

# 检查输入文件
echo "📋 检查可用的输入文件..."
echo ""
echo "可用的文本文件："
ls -lh ../data/*.txt 2>/dev/null || echo "  ❌ 没有找到 .txt 文件"
echo ""

# 显示使用示例
echo "📝 使用示例："
echo ""
echo "1️⃣  基本用法（使用 wiki.simple.txt）："
echo "   python clean_corpus.py \\"
echo "     --input ../data/wiki.simple.txt \\"
echo "     --output ../data/my_corpus.txt"
echo ""

echo "2️⃣  自定义参数 + 预览："
echo "   python clean_corpus.py \\"
echo "     --input ../data/wiki.simple.txt \\"
echo "     --output ../data/my_corpus.txt \\"
echo "     --target-length 2048 \\"
echo "     --min-length 10 \\"
echo "     --max-length 50000 \\"
echo "     --preview"
echo ""

echo "3️⃣  快速测试（小文件）："
echo "   head -10000 ../data/wiki.simple.txt > ../data/test_input.txt"
echo "   python clean_corpus.py \\"
echo "     --input ../data/test_input.txt \\"
echo "     --output ../data/test_output.txt \\"
echo "     --preview"
echo ""

echo "=========================================="
echo ""

# 询问是否执行
read -p "是否执行基本清洗命令？(y/n): " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo ""
    echo "🔄 开始清洗..."
    echo ""
    
    # 检查输入文件是否存在
    if [ -f "../data/wiki.simple.txt" ]; then
        python clean_corpus.py \
            --input ../data/wiki.simple.txt \
            --output ../data/my_corpus.txt \
            --preview
    elif [ -f "../data/wiki.txt" ]; then
        echo "⚠️  wiki.simple.txt 不存在，使用 wiki.txt（可能较慢）"
        python clean_corpus.py \
            --input ../data/wiki.txt \
            --output ../data/my_corpus.txt \
            --preview
    else
        echo "❌ 错误：找不到输入文件"
        echo "   请确保 ../data/wiki.txt 或 ../data/wiki.simple.txt 存在"
        exit 1
    fi
    
    echo ""
    echo "✅ 清洗完成！"
    echo ""
    echo "📊 输出文件信息："
    ls -lh ../data/my_corpus.txt
    echo ""
    echo "📝 下一步："
    echo "   python train_tokenizer.py \\"
    echo "     --method t5-base \\"
    echo "     --wiki-file ../data/my_corpus.txt \\"
    echo "     --output-dir ../model_save/my_tokenizer_wiki \\"
    echo "     --vocab-size 40960 \\"
    echo "     --batch-size 500"
else
    echo ""
    echo "👋 已取消"
fi
