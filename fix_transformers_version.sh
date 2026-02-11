#!/bin/bash
# -*- coding: utf-8 -*-
#
# 修复 transformers 版本问题
# 错误: ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
#

set -e

echo "=========================================="
echo "修复 transformers 版本问题"
echo "=========================================="
echo ""

# 检查当前版本
echo "1. 检查当前版本..."
CURRENT_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "未安装")
echo "   当前 transformers 版本: $CURRENT_VERSION"

LLMTUNER_VERSION=$(python -c "import llmtuner; print(llmtuner.__version__)" 2>/dev/null || echo "未安装")
echo "   当前 llmtuner 版本: $LLMTUNER_VERSION"
echo ""

# 显示修复选项
echo "请选择修复方式:"
echo "  1) 仅升级 transformers（快速，推荐）"
echo "  2) 重新安装 llmtuner（会自动安装正确的依赖）"
echo "  3) 从源码安装 LLaMA-Factory（最稳定）"
echo "  4) 显示详细诊断信息"
echo ""
read -p "请输入选项 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "=========================================="
        echo "方式1: 升级 transformers"
        echo "=========================================="
        echo ""
        
        echo "升级 transformers 到最新版本..."
        pip install --upgrade transformers
        
        echo ""
        echo "验证安装..."
        NEW_VERSION=$(python -c "import transformers; print(transformers.__version__)")
        echo "✓ 新版本: $NEW_VERSION"
        
        echo ""
        echo "测试导入..."
        python -c "from transformers import AutoModelForVision2Seq; print('✓ AutoModelForVision2Seq 导入成功')"
        python -c "import llmtuner; print('✓ llmtuner 导入成功')"
        
        echo ""
        echo "✓ 修复完成！"
        ;;
    
    2)
        echo ""
        echo "=========================================="
        echo "方式2: 重新安装 llmtuner"
        echo "=========================================="
        echo ""
        
        echo "卸载旧版本..."
        pip uninstall llmtuner -y
        
        echo ""
        echo "安装新版本..."
        pip install "llmtuner[torch,metrics]"
        
        echo ""
        echo "验证安装..."
        NEW_TRANSFORMERS=$(python -c "import transformers; print(transformers.__version__)")
        NEW_LLMTUNER=$(python -c "import llmtuner; print(llmtuner.__version__)")
        echo "✓ transformers 版本: $NEW_TRANSFORMERS"
        echo "✓ llmtuner 版本: $NEW_LLMTUNER"
        
        echo ""
        echo "测试导入..."
        python -c "from transformers import AutoModelForVision2Seq; print('✓ AutoModelForVision2Seq 导入成功')"
        python -c "import llmtuner; print('✓ llmtuner 导入成功')"
        
        echo ""
        echo "✓ 修复完成！"
        ;;
    
    3)
        echo ""
        echo "=========================================="
        echo "方式3: 从源码安装 LLaMA-Factory"
        echo "=========================================="
        echo ""
        
        # 创建临时目录
        TEMP_DIR=$(mktemp -d)
        echo "临时目录: $TEMP_DIR"
        
        echo ""
        echo "克隆 LLaMA-Factory 源码..."
        git clone https://github.com/hiyouga/LLaMA-Factory.git "$TEMP_DIR/LLaMA-Factory"
        
        echo ""
        echo "卸载旧版本..."
        pip uninstall llmtuner -y || true
        
        echo ""
        echo "从源码安装..."
        cd "$TEMP_DIR/LLaMA-Factory"
        pip install -e ".[torch,metrics]"
        
        echo ""
        echo "验证安装..."
        NEW_TRANSFORMERS=$(python -c "import transformers; print(transformers.__version__)")
        NEW_LLMTUNER=$(python -c "import llmtuner; print(llmtuner.__version__)")
        echo "✓ transformers 版本: $NEW_TRANSFORMERS"
        echo "✓ llmtuner 版本: $NEW_LLMTUNER"
        
        echo ""
        echo "测试导入..."
        python -c "from transformers import AutoModelForVision2Seq; print('✓ AutoModelForVision2Seq 导入成功')"
        python -c "import llmtuner; print('✓ llmtuner 导入成功')"
        
        echo ""
        echo "清理临时文件..."
        rm -rf "$TEMP_DIR"
        
        echo ""
        echo "✓ 修复完成！"
        ;;
    
    4)
        echo ""
        echo "=========================================="
        echo "详细诊断信息"
        echo "=========================================="
        echo ""
        
        echo "Python 环境:"
        python --version
        echo ""
        
        echo "已安装的包版本:"
        pip show transformers || echo "  transformers: 未安装"
        echo ""
        pip show llmtuner || echo "  llmtuner: 未安装"
        echo ""
        pip show torch || echo "  torch: 未安装"
        echo ""
        pip show deepspeed || echo "  deepspeed: 未安装"
        echo ""
        
        echo "测试导入:"
        python -c "
import sys
print('Python路径:', sys.executable)
print()

try:
    import transformers
    print(f'✓ transformers {transformers.__version__}')
except Exception as e:
    print(f'✗ transformers: {e}')

try:
    from transformers import AutoModelForVision2Seq
    print('✓ AutoModelForVision2Seq 可导入')
except Exception as e:
    print(f'✗ AutoModelForVision2Seq: {e}')

try:
    import llmtuner
    print(f'✓ llmtuner {llmtuner.__version__}')
except Exception as e:
    print(f'✗ llmtuner: {e}')

try:
    import torch
    print(f'✓ torch {torch.__version__}')
except Exception as e:
    print(f'✗ torch: {e}')

try:
    import deepspeed
    print(f'✓ deepspeed {deepspeed.__version__}')
except Exception as e:
    print(f'✗ deepspeed: {e}')
"
        echo ""
        echo "=========================================="
        echo "修复建议:"
        echo "=========================================="
        echo ""
        echo "根据上述诊断信息，推荐的修复方式:"
        echo ""
        echo "如果 transformers 版本 < 4.37.0:"
        echo "  → 运行: pip install --upgrade transformers"
        echo ""
        echo "如果 llmtuner 导入失败:"
        echo "  → 运行: pip install --upgrade llmtuner"
        echo ""
        echo "如果问题依然存在:"
        echo "  → 重新运行此脚本，选择方式2或3"
        echo ""
        ;;
    
    *)
        echo "❌ 无效选项: $choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "现在可以运行训练脚本了:"
echo "  bash run_llamafactory_3x3080.sh"
echo ""
