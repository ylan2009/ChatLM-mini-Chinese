#!/bin/bash
# NCCL 共享内存错误快速修复脚本 v2
# 针对 "Error while attaching to shared memory segment" 错误

echo "=========================================="
echo "NCCL 共享内存错误修复脚本 v2"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查系统信息
echo ""
echo "1. 系统信息:"
echo "  操作系统: $(uname -s)"
echo "  内核版本: $(uname -r)"

# 2. 检查 /dev/shm 状态
echo ""
echo "2. 检查 /dev/shm 状态:"
df -h /dev/shm

# 3. 检查 NCCL 文件
echo ""
echo "3. 检查 NCCL 残留文件:"
if ls /dev/shm/nccl-* 2>/dev/null; then
    echo -e "${YELLOW}  发现 NCCL 文件，正在清理...${NC}"
    sudo rm -f /dev/shm/nccl-* 2>/dev/null && echo -e "${GREEN}  ✓ 已清理${NC}" || echo -e "${RED}  ✗ 清理失败（需要sudo权限）${NC}"
else
    echo -e "${GREEN}  ✓ 没有 NCCL 残留文件${NC}"
fi

# 4. 检查残留进程
echo ""
echo "4. 检查残留进程:"
TRAIN_PIDS=$(ps aux | grep train_low_mem | grep -v grep | awk '{print $2}')
if [ -n "$TRAIN_PIDS" ]; then
    echo -e "${YELLOW}  发现残留进程: $TRAIN_PIDS${NC}"
    echo "  正在清理..."
    echo "$TRAIN_PIDS" | xargs kill -9 2>/dev/null && echo -e "${GREEN}  ✓ 已清理${NC}" || echo -e "${RED}  ✗ 清理失败${NC}"
else
    echo -e "${GREEN}  ✓ 没有残留进程${NC}"
fi

# 5. 检查 GPU 状态
echo ""
echo "5. GPU 状态:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s (已用: %sMB / 总计: %sMB)\n", $1, $2, $3, $4}'
else
    echo -e "${RED}  ✗ nvidia-smi 不可用${NC}"
fi

# 6. 运行诊断脚本
echo ""
echo "6. 运行 NCCL 诊断:"
if [ -f "diagnose_nccl.py" ]; then
    echo "  单进程诊断..."
    python diagnose_nccl.py
    
    echo ""
    echo "  多进程诊断（模拟实际训练环境）..."
    accelerate launch --multi_gpu --num_processes 2 ./diagnose_nccl.py
else
    echo -e "${YELLOW}  ✗ 诊断脚本不存在，跳过${NC}"
fi

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "现在可以尝试以下方案："
echo ""
echo -e "${GREEN}方案1: 使用修复后的 train_low_mem.py（推荐）${NC}"
echo "  accelerate launch --multi_gpu --num_processes 2 \\"
echo "      ./train_low_mem.py train \\"
echo "      --is_finetune=True \\"
echo "      --use_small_config=True"
echo ""
echo -e "${GREEN}方案2: 禁用 NCCL 共享内存${NC}"
echo "  export NCCL_SHM_DISABLE=1"
echo "  accelerate launch --multi_gpu --num_processes 2 \\"
echo "      ./train_low_mem.py train \\"
echo "      --is_finetune=True \\"
echo "      --use_small_config=True"
echo ""
echo -e "${GREEN}方案3: 使用 Gloo 后端（兼容性最好）${NC}"
echo "  export ACCELERATE_USE_GLOO=1"
echo "  accelerate launch --multi_gpu --num_processes 2 \\"
echo "      ./train_low_mem.py train \\"
echo "      --is_finetune=True \\"
echo "      --use_small_config=True"
echo ""
echo -e "${GREEN}方案4: 使用单 GPU 训练${NC}"
echo "  python train_low_mem.py train \\"
echo "      --is_finetune=True \\"
echo "      --use_small_config=True"
echo ""
echo "详细说明请查看: cat FIX_NCCL_ERROR.md"
echo "=========================================="
