#!/bin/bash
# -*- coding: utf-8 -*-
#
# 检查 transformers 和 llmtuner 版本兼容性
#

echo "=========================================="
echo "检查 transformers 版本兼容性"
echo "=========================================="
echo ""

# 检查当前版本
echo "当前安装的版本:"
python -c "
try:
    import transformers
    print(f'  transformers: {transformers.__version__}')
except:
    print('  transformers: 未安装')

try:
    import llmtuner
    print(f'  llmtuner: {llmtuner.__version__}')
except:
    print('  llmtuner: 未安装')
"

echo ""
echo "=========================================="
echo "问题分析"
echo "=========================================="
echo ""
echo "transformers 5.1.0 移除了 AutoModelForVision2Seq 类"
echo "这是一个 API 变更，导致 llmtuner 无法正常工作"
echo ""
echo "解决方案: 降级到兼容版本"
echo ""

echo "=========================================="
echo "推荐的版本组合"
echo "=========================================="
echo ""
echo "方案1: 使用 transformers 4.40.0 (稳定版)"
echo "  pip install transformers==4.40.0"
echo ""
echo "方案2: 使用 transformers 4.37.0 (最低要求)"
echo "  pip install transformers==4.37.0"
echo ""
echo "方案3: 使用 transformers 4.44.0 (推荐)"
echo "  pip install transformers==4.44.0"
echo ""

read -p "是否立即修复? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "开始修复..."
    echo ""
    
    # 卸载当前版本
    echo "1. 卸载 transformers 5.1.0..."
    pip uninstall transformers -y
    
    # 安装兼容版本
    echo ""
    echo "2. 安装 transformers 4.44.0..."
    pip install transformers==4.44.0
    
    # 验证
    echo ""
    echo "3. 验证安装..."
    python -c "
import transformers
print(f'✓ transformers 版本: {transformers.__version__}')

from transformers import AutoModelForVision2Seq
print('✓ AutoModelForVision2Seq 导入成功')

import llmtuner
print('✓ llmtuner 导入成功')
"
    
    echo ""
    echo "✓ 修复完成！"
else
    echo ""
    echo "请手动执行以下命令修复:"
    echo "  pip uninstall transformers -y"
    echo "  pip install transformers==4.44.0"
fi
