# NCCL 共享内存错误修复指南

## 🔴 错误信息

```
torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:3690
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. 
Last error:
Error while attaching to shared memory segment /dev/shm/nccl-Ќ (size 0), error: No such file or directory (2)
```

## 🎯 问题原因

这个错误是由于 `/dev/shm` (共享内存) 空间不足或存在残留的NCCL共享内存文件导致的。

NCCL在多GPU通信时需要使用共享内存，如果：
1. `/dev/shm` 空间不足
2. 存在旧的NCCL进程残留文件
3. 权限问题

就会导致这个错误。

---

## ✅ 解决方案

### 方案1：清理共享内存（最简单，推荐）

在**服务器上**执行以下命令：

```bash
# 1. 检查 /dev/shm 使用情况
df -h /dev/shm

# 2. 清理旧的NCCL共享内存文件
sudo rm -f /dev/shm/nccl-*

# 3. 清理其他临时文件（可选）
sudo rm -f /dev/shm/sem.*

# 4. 再次检查空间
df -h /dev/shm
```

**然后重新运行训练命令**：

```bash
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

---

### 方案2：设置环境变量（如果方案1无效）

在训练命令前添加环境变量：

```bash
# 禁用NCCL使用共享内存，改用socket通信
export NCCL_SHM_DISABLE=1

# 或者指定NCCL使用其他临时目录
export NCCL_SHM_DIR=/tmp

# 然后运行训练
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

---

### 方案3：增加 /dev/shm 大小（如果空间确实不足）

如果 `/dev/shm` 空间太小（通常默认是内存的50%），可以临时增加：

```bash
# 查看当前大小
df -h /dev/shm

# 临时增加到8GB（需要root权限）
sudo mount -o remount,size=8G /dev/shm

# 验证
df -h /dev/shm
```

**永久修改**（需要root权限）：

编辑 `/etc/fstab`，添加或修改：

```
tmpfs /dev/shm tmpfs defaults,size=8G 0 0
```

然后重启系统。

---

### 方案4：使用单GPU训练（临时方案）

如果上述方案都不行，可以先用单GPU训练：

```bash
# 单GPU不需要NCCL通信，不会有这个问题
python train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

**注意**：单GPU训练速度会慢一些，但可以正常工作。

---

## 🔧 快速修复脚本

创建一个修复脚本 `fix_nccl.sh`：

```bash
#!/bin/bash
# 在服务器上运行此脚本

echo "=========================================="
echo "修复NCCL共享内存问题"
echo "=========================================="

# 1. 检查空间
echo ""
echo "1. 当前 /dev/shm 使用情况:"
df -h /dev/shm

# 2. 清理NCCL文件
echo ""
echo "2. 清理旧的NCCL文件..."
sudo rm -f /dev/shm/nccl-* 2>/dev/null
sudo rm -f /dev/shm/sem.* 2>/dev/null
echo "清理完成"

# 3. 检查清理后的空间
echo ""
echo "3. 清理后的 /dev/shm 使用情况:"
df -h /dev/shm

# 4. 杀死可能残留的训练进程
echo ""
echo "4. 检查并清理残留的训练进程..."
pkill -f "train_low_mem.py" 2>/dev/null
pkill -f "accelerate" 2>/dev/null
echo "进程清理完成"

echo ""
echo "=========================================="
echo "修复完成！现在可以重新运行训练命令："
echo "accelerate launch --multi_gpu --num_processes 2 \\"
echo "    ./train_low_mem.py train \\"
echo "    --is_finetune=True \\"
echo "    --use_small_config=True"
echo "=========================================="
```

**使用方法**：

```bash
# 在服务器上
chmod +x fix_nccl.sh
./fix_nccl.sh
```

---

## 📋 推荐的完整修复流程

### 步骤1：在服务器上清理环境

```bash
# SSH到服务器
ssh rongtw@rongtw

# 进入项目目录
cd /data3/ChatLM-mini-Chinese

# 清理共享内存
sudo rm -f /dev/shm/nccl-*

# 清理残留进程
pkill -f "train_low_mem.py"
pkill -f "accelerate"

# 检查空间
df -h /dev/shm
```

### 步骤2：设置环境变量（可选）

```bash
# 添加到 ~/.bashrc 或临时设置
export NCCL_SHM_DISABLE=0  # 0=启用, 1=禁用
export NCCL_DEBUG=INFO     # 启用调试信息（可选）
```

### 步骤3：重新运行训练

```bash
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

---

## 🔍 诊断命令

如果问题仍然存在，运行以下诊断命令：

```bash
# 1. 检查 /dev/shm 空间
df -h /dev/shm

# 2. 查看 /dev/shm 中的文件
ls -lh /dev/shm/

# 3. 检查NCCL版本
python -c "import torch; print(torch.cuda.nccl.version())"

# 4. 检查GPU状态
nvidia-smi

# 5. 检查是否有残留进程
ps aux | grep train_low_mem
ps aux | grep accelerate

# 6. 测试NCCL通信
python -c "import torch; import torch.distributed as dist; print('NCCL available:', torch.cuda.nccl.is_available(['cuda:0', 'cuda:1']))"
```

---

## ⚠️ 常见问题

### Q1: 为什么会出现这个错误？

**A**: 通常是因为：
1. 之前的训练进程异常退出，留下了残留的共享内存文件
2. `/dev/shm` 空间不足（默认是内存的50%）
3. 多个训练任务同时运行，占用了共享内存

### Q2: 清理 /dev/shm 会影响其他程序吗？

**A**: 清理 `nccl-*` 文件是安全的，这些是NCCL的临时文件。但不要删除其他程序的共享内存文件。

### Q3: 如果没有sudo权限怎么办？

**A**: 使用方案2，设置环境变量：
```bash
export NCCL_SHM_DISABLE=1
```
或者使用单GPU训练（方案4）。

### Q4: 为什么单GPU训练不会有这个问题？

**A**: 单GPU训练不需要NCCL进行GPU间通信，所以不会使用共享内存。

---

## 🎯 最快的解决方法（3步）

```bash
# 1. 清理（在服务器上）
sudo rm -f /dev/shm/nccl-*

# 2. 杀死残留进程
pkill -f train_low_mem

# 3. 重新运行
accelerate launch --multi_gpu --num_processes 2 \
    ./train_low_mem.py train \
    --is_finetune=True \
    --use_small_config=True
```

---

## 📚 参考资料

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [PyTorch Distributed Troubleshooting](https://pytorch.org/docs/stable/distributed.html#troubleshooting)
- [Accelerate Multi-GPU Training](https://huggingface.co/docs/accelerate/usage_guides/distributed)

---

## ✅ 验证修复成功

如果看到以下输出，说明修复成功：

```
================================================================================
使用 TrainConfigSFTSmall 配置（小数据集 - 适合16G内存）
================================================================================
[2026-02-09 11:25:21.049] [INFO]: 低内存模式训练 - 针对16G内存优化
[2026-02-09 11:25:21.049] [INFO]: cpu memory available: 13.15 GB, disk space available: 44.79 GB
[2026-02-09 11:25:21.049] [INFO]: 使用LowMemDataset: 支持多GPU + 低内存模式，按需从磁盘读取数据
...
[2026-02-09 11:25:26.228] [INFO]: train dataset size: 5000, steps per epoch:2500
```

然后训练会正常开始，不会再报NCCL错误。

---

**祝训练顺利！🚀**
