# 🚨 NCCL 共享内存错误修复

## ❌ 错误信息

```
torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:3690, 
unhandled system error (run with NCCL_DEBUG=INFO for details), NCCL version 2.27.7
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. 
Last error:
Error while attaching to shared memory segment /dev/shm/nccl-Ќ (size 0), 
error: No such file or directory (2)
```

---

## 🔍 问题分析

### 错误原因

**NCCL 无法访问共享内存 `/dev/shm/`！**

NCCL（NVIDIA Collective Communications Library）是 PyTorch 分布式训练使用的通信库，默认会使用共享内存来加速 GPU 间通信。

### 可能的原因

| 原因 | 说明 | 检查方法 |
|------|------|---------|
| **1. 共享内存空间不足** | `/dev/shm` 空间太小 | `df -h /dev/shm` |
| **2. 权限问题** | 用户无权访问 `/dev/shm` | `ls -la /dev/shm` |
| **3. 共享内存文件残留** | 之前训练的文件未清理 | `ls /dev/shm/nccl-*` |
| **4. Docker 容器限制** | 容器内 `/dev/shm` 太小 | `df -h /dev/shm` |

### 为什么会失败？

```bash
# NCCL 尝试创建共享内存文件
/dev/shm/nccl-XXXXXXXX

# 但是失败了：
# - 空间不足
# - 权限不够
# - 或者文件系统问题
```

---

## ✅ 解决方案

### 🎯 方案1: 禁用 NCCL 共享内存（推荐）⭐⭐⭐⭐⭐

**最简单、最可靠的方法：禁用共享内存，使用 P2P 通信**

#### 修改环境变量

```bash
# 禁用共享内存
export NCCL_SHM_DISABLE=1

# 启用 P2P 通信（GPU 间直接通信）
export NCCL_P2P_DISABLE=0

# 其他配置
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO  # 可选，用于调试
```

#### 为什么可以禁用共享内存？

- ✅ **3×RTX 3080 在同一台机器上**：可以使用 P2P（PCIe 直接通信）
- ✅ **P2P 性能很好**：通过 PCIe 直接通信，速度足够快
- ✅ **避免共享内存问题**：不依赖 `/dev/shm`
- ✅ **更稳定**：减少系统依赖

#### 性能对比

| 通信方式 | 带宽 | 延迟 | 稳定性 | 推荐度 |
|---------|------|------|--------|--------|
| **共享内存（SHM）** | 最高 | 最低 | ⚠️ 依赖系统 | ⭐⭐⭐ |
| **P2P（PCIe）** | 高 | 低 | ✅ 稳定 | ⭐⭐⭐⭐⭐ |
| **Socket** | 中 | 中 | ✅ 稳定 | ⭐⭐⭐⭐ |

**对于 3×RTX 3080，P2P 性能完全够用！**

---

### 🎯 方案2: 增加共享内存空间（如果需要使用 SHM）⭐⭐⭐

#### 检查当前空间

```bash
# 查看 /dev/shm 大小
df -h /dev/shm

# 输出示例
# Filesystem      Size  Used Avail Use% Mounted on
# tmpfs           6.0G  1.2M  6.0G   1% /dev/shm
```

#### 增加空间（需要 root 权限）

```bash
# 临时增加（重启后失效）
sudo mount -o remount,size=8G /dev/shm

# 永久增加（编辑 /etc/fstab）
sudo vim /etc/fstab
# 添加或修改：
# tmpfs /dev/shm tmpfs defaults,size=8G 0 0

# 重新挂载
sudo mount -o remount /dev/shm
```

#### 推荐大小

| 系统内存 | `/dev/shm` 大小 | 说明 |
|---------|----------------|------|
| 12GB | 4-6GB | 你的配置 |
| 32GB | 8-16GB | 标准配置 |
| 64GB+ | 16-32GB | 高端配置 |

---

### 🎯 方案3: 清理共享内存文件⭐⭐⭐⭐

#### 检查残留文件

```bash
# 查看 NCCL 文件
ls -lh /dev/shm/nccl-*

# 查看所有共享内存文件
ls -lh /dev/shm/
```

#### 清理文件

```bash
# 清理 NCCL 文件
rm -f /dev/shm/nccl-*

# 清理所有共享内存文件（谨慎！）
# sudo rm -rf /dev/shm/*
```

#### 自动清理脚本

```bash
# 添加到训练脚本开头
echo "清理共享内存..."
rm -f /dev/shm/nccl-* 2>/dev/null || true
echo "✓ 清理完成"
```

---

### 🎯 方案4: Docker 容器配置（如果使用 Docker）⭐⭐⭐⭐

#### 启动容器时增加共享内存

```bash
# 使用 --shm-size 参数
docker run --gpus all --shm-size=8g ...

# 或者使用 --ipc=host（共享主机 IPC）
docker run --gpus all --ipc=host ...
```

#### docker-compose.yml

```yaml
services:
  training:
    image: your-image
    shm_size: '8gb'
    # 或者
    ipc: host
```

---

## 🔧 已修复的启动脚本

我已经修复了 `run_llamafactory_3x3080.sh`，现在会自动禁用 NCCL 共享内存：

### 修复内容

```bash
# 修复前（错误）
export NCCL_SHM_DISABLE=0  # 启用共享内存
# ❌ 导致 /dev/shm 错误

# 修复后（正确）
export NCCL_SHM_DISABLE=1  # 禁用共享内存
export NCCL_P2P_DISABLE=0  # 启用 P2P 通信
export NCCL_DEBUG=INFO     # 启用调试信息
# ✅ 使用 P2P 通信，避免共享内存问题
```

---

## 🚀 现在可以运行了

### Step 1: 重新运行启动脚本

```bash
cd /data3/ChatLM-mini-Chinese
bash run_llamafactory_3x3080.sh
```

### Step 2: 选择训练方式

```
请选择训练方式:
  1) 使用 llamafactory-cli (推荐，最简单)
  2) 使用 accelerate launch (更灵活)
  3) 使用 deepspeed (最优显存利用)      ← 现在可以正常工作了！
  4) 使用 torchrun (标准DDP)

请输入选项 [1-4]: 3
```

### Step 3: 开始训练

脚本会自动：
1. ✅ 设置 `NCCL_SHM_DISABLE=1`
2. ✅ 启用 P2P 通信
3. ✅ 避免共享内存错误
4. ✅ 开始训练

---

## 📊 NCCL 通信方式对比

### 通信方式说明

#### 1. 共享内存（SHM）

```
GPU 0 ←→ /dev/shm ←→ GPU 1
         (共享内存)
```

**优点：**
- ✅ 带宽最高
- ✅ 延迟最低

**缺点：**
- ❌ 依赖 `/dev/shm`
- ❌ 可能空间不足
- ❌ 权限问题

#### 2. P2P（PCIe）

```
GPU 0 ←→ PCIe ←→ GPU 1
       (直接通信)
```

**优点：**
- ✅ 不依赖共享内存
- ✅ 性能很好
- ✅ 稳定可靠

**缺点：**
- ⚠️ 需要 GPU 支持 P2P（RTX 3080 支持）

#### 3. Socket

```
GPU 0 → CPU → 网络 → CPU → GPU 1
```

**优点：**
- ✅ 最通用
- ✅ 跨机器支持

**缺点：**
- ⚠️ 性能较低

### 性能测试

```bash
# 测试 P2P 是否可用
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        for j in range(i+1, torch.cuda.device_count()):
            can_access = torch.cuda.can_device_access_peer(i, j)
            print(f'GPU {i} ↔ GPU {j}: P2P = {can_access}')
"

# 输出示例（RTX 3080）
# GPU 0 ↔ GPU 1: P2P = True
# GPU 0 ↔ GPU 2: P2P = True
# GPU 1 ↔ GPU 2: P2P = True
```

---

## 💡 深入理解

### NCCL 通信流程

#### 正常流程（使用共享内存）

```
1. GPU 0 计算梯度
2. GPU 0 写入 /dev/shm/nccl-XXXX
3. GPU 1 从 /dev/shm/nccl-XXXX 读取
4. GPU 1 更新参数
```

#### 失败原因

```
1. GPU 0 计算梯度
2. GPU 0 尝试写入 /dev/shm/nccl-XXXX
   ❌ 错误：No such file or directory
3. 训练失败
```

#### 修复后（使用 P2P）

```
1. GPU 0 计算梯度
2. GPU 0 通过 PCIe 直接发送给 GPU 1
3. GPU 1 接收并更新参数
4. ✅ 成功
```

### 环境变量详解

| 环境变量 | 值 | 说明 |
|---------|---|------|
| `NCCL_SHM_DISABLE` | `1` | 禁用共享内存 |
| `NCCL_P2P_DISABLE` | `0` | 启用 P2P 通信 |
| `NCCL_IB_DISABLE` | `1` | 禁用 InfiniBand |
| `NCCL_TIMEOUT` | `3600` | 超时时间（秒）|
| `NCCL_DEBUG` | `INFO` | 调试级别 |
| `NCCL_DEBUG_SUBSYS` | `ALL` | 调试子系统 |

### 调试命令

```bash
# 启用详细调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 运行训练
bash run_llamafactory_3x3080.sh

# 查看 NCCL 日志
# 会输出详细的通信信息
```

---

## 🔒 防止再次出现

### 规则1: 优先使用 P2P 通信

```bash
# ✅ 推荐（稳定可靠）
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=0

# ❌ 避免（可能失败）
export NCCL_SHM_DISABLE=0  # 依赖共享内存
```

### 规则2: 定期清理共享内存

```bash
# 添加到训练脚本
rm -f /dev/shm/nccl-* 2>/dev/null || true
```

### 规则3: 监控共享内存使用

```bash
# 检查空间
df -h /dev/shm

# 检查文件
ls -lh /dev/shm/
```

### 规则4: Docker 容器配置

```bash
# 启动容器时
docker run --gpus all --shm-size=8g ...
# 或者
docker run --gpus all --ipc=host ...
```

---

## 📝 快速参考

### 诊断命令

```bash
# 1. 检查共享内存空间
df -h /dev/shm

# 2. 检查 NCCL 文件
ls -lh /dev/shm/nccl-*

# 3. 测试 P2P 支持
python -c "
import torch
for i in range(torch.cuda.device_count()):
    for j in range(i+1, torch.cuda.device_count()):
        print(f'GPU {i} ↔ GPU {j}: {torch.cuda.can_device_access_peer(i, j)}')
"

# 4. 查看 NCCL 版本
python -c "import torch; print(torch.cuda.nccl.version())"
```

### 修复命令

```bash
# 方案1: 禁用共享内存（推荐）
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=0

# 方案2: 清理共享内存
rm -f /dev/shm/nccl-*

# 方案3: 增加共享内存空间
sudo mount -o remount,size=8G /dev/shm

# 方案4: 启用调试
export NCCL_DEBUG=INFO
```

---

## 🎯 总结

**问题：** NCCL 无法访问共享内存 `/dev/shm/`

**原因：** 共享内存空间不足、权限问题或文件系统问题

**解决方案：**
1. **禁用共享内存，使用 P2P 通信**（推荐）✅
2. 或者增加共享内存空间
3. 或者清理残留文件

**执行命令：**
```bash
cd /data3/ChatLM-mini-Chinese
bash run_llamafactory_3x3080.sh
# 选择方式 3（deepspeed）
```

**现在应该可以正常工作了！** 🎉

---

## 📞 还有问题？

### 如果还有 NCCL 错误

```bash
# 启用详细调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 运行训练
bash run_llamafactory_3x3080.sh

# 查看日志，找到具体错误
```

### 常见 NCCL 错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `ncclSystemError` | 共享内存问题 | `NCCL_SHM_DISABLE=1` |
| `ncclInternalError` | 通信超时 | 增加 `NCCL_TIMEOUT` |
| `ncclInvalidUsage` | 参数错误 | 检查配置文件 |
| `ncclRemoteError` | 网络问题 | 检查防火墙 |

---

## 🔗 相关文档

- [FIX_LOCAL_RANK_ARGS.md](FIX_LOCAL_RANK_ARGS.md) - 参数解析错误修复
- [FIX_RELATIVE_IMPORT.md](FIX_RELATIVE_IMPORT.md) - 相对导入错误修复
- [FIX_DEEPSPEED_LAUNCH.md](FIX_DEEPSPEED_LAUNCH.md) - DeepSpeed 启动问题
- [FIX_TRL_CONFLICT.md](FIX_TRL_CONFLICT.md) - 依赖冲突修复
