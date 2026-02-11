#!/bin/bash
# -*- coding: utf-8 -*-
#
# 一键修复 transformers 和 trl 依赖冲突
#

set -e

echo "=========================================="
echo "依赖冲突修复脚本"
echo "=========================================="
echo ""

# 检查当前版本
echo "检查当前版本..."
python -c "
try:
    import transformers
    print(f'  transformers: {transformers.__version__}')
except:
    print('  transformers: 未安装')

try:
    import trl
    print(f'  trl: {trl.__version__}')
except:
    print('  trl: 未安装')

try:
    import llmtuner
    print(f'  llmtuner: 已安装')
except:
    print('  llmtuner: 未安装')
"

echo ""
echo "=========================================="
echo "问题诊断"
echo "=========================================="
echo ""

# 检查是否有冲突
HAS_CONFLICT=0

# 检查 transformers 版本
if python -c "
import transformers
from packaging import version
v = transformers.__version__
exit(0 if (version.parse(v) >= version.parse('4.37.0') and version.parse(v) < version.parse('5.0.0')) else 1)
" 2>/dev/null; then
    echo "✓ transformers 版本正常"
else
    echo "✗ transformers 版本不兼容"
    HAS_CONFLICT=1
fi

# 检查 trl 版本
if python -c "import trl" 2>/dev/null; then
    if python -c "
import trl
from packaging import version
v = trl.__version__
exit(0 if version.parse(v) < version.parse('0.9.0') else 1)
" 2>/dev/null; then
        echo "✓ trl 版本正常"
    else
        echo "✗ trl 版本不兼容"
        HAS_CONFLICT=1
    fi
fi

# 检查 AutoModelForVision2Seq
if python -c "from transformers import AutoModelForVision2Seq" 2>/dev/null; then
    echo "✓ AutoModelForVision2Seq 可导入"
else
    echo "✗ AutoModelForVision2Seq 无法导入"
    HAS_CONFLICT=1
fi

# 检查 llmtuner
if python -c "import llmtuner" 2>/dev/null; then
    echo "✓ llmtuner 可导入"
else
    echo "✗ llmtuner 无法导入"
    HAS_CONFLICT=1
fi

echo ""

if [ $HAS_CONFLICT -eq 0 ]; then
    echo "=========================================="
    echo "✓ 没有发现依赖冲突"
    echo "=========================================="
    echo ""
    echo "你的环境已经正常，可以开始训练了！"
    echo ""
    exit 0
fi

echo "=========================================="
echo "修复方案"
echo "=========================================="
echo ""
echo "将安装以下兼容版本:"
echo "  - transformers==4.44.0"
echo "  - trl==0.8.6"
echo ""

read -p "是否立即修复? [y/N] " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "取消修复。"
    echo ""
    echo "手动修复命令:"
    echo "  pip uninstall trl transformers -y"
    echo "  pip install transformers==4.44.0 trl==0.8.6"
    echo ""
    exit 0
fi

echo ""
echo "=========================================="
echo "开始修复"
echo "=========================================="
echo ""

# Step 1: 卸载冲突的包
echo "Step 1: 卸载冲突的包..."
pip uninstall trl transformers -y

echo ""

# Step 2: 安装兼容版本
echo "Step 2: 安装 transformers 4.44.0..."
pip install transformers==4.44.0

echo ""

echo "Step 3: 安装 trl 0.8.6..."
pip install trl==0.8.6

echo ""

# Step 3: 验证修复
echo "=========================================="
echo "验证修复"
echo "=========================================="
echo ""

python -c "
import sys

# 检查版本
import transformers
import trl
print(f'✓ transformers: {transformers.__version__}')
print(f'✓ trl: {trl.__version__}')
print()

# 测试导入
try:
    from transformers import AutoModelForVision2Seq
    print('✓ AutoModelForVision2Seq 导入成功')
except Exception as e:
    print(f'✗ AutoModelForVision2Seq 导入失败: {e}')
    sys.exit(1)

try:
    import llmtuner
    print('✓ llmtuner 导入成功')
except Exception as e:
    print(f'✗ llmtuner 导入失败: {e}')
    sys.exit(1)

try:
    from llmtuner.chat import ChatModel
    print('✓ ChatModel 导入成功')
except Exception as e:
    print(f'✗ ChatModel 导入失败: {e}')
    sys.exit(1)

print()
print('✓ 所有测试通过！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 修复成功！"
    echo "=========================================="
    echo ""
    echo "现在可以开始训练了:"
    echo "  bash run_llamafactory_3x3080.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ 修复失败"
    echo "=========================================="
    echo ""
    echo "请查看错误信息，或尝试完全重装:"
    echo "  pip uninstall trl transformers llmtuner -y"
    echo "  pip install transformers==4.44.0"
    echo "  pip install trl==0.8.6"
    echo "  pip install llmtuner"
    echo ""
    exit 1
fi

# 检查依赖冲突
echo "检查依赖冲突..."
pip check

echo ""
echo "详细文档:"
echo "  - FIX_TRANSFORMERS_5X.md (transformers 版本问题)"
echo "  - FIX_TRL_CONFLICT.md (trl 依赖冲突)"
echo ""
