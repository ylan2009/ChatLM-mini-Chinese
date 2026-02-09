#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NCCL 和分布式训练环境诊断脚本

使用方法：
    # 单进程测试
    python diagnose_nccl.py
    
    # 多进程测试（模拟实际训练环境）
    accelerate launch --multi_gpu --num_processes 2 ./diagnose_nccl.py
"""

import os
import sys
import torch
import torch.distributed as dist
from psutil import virtual_memory
import platform

def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def check_system_info():
    """检查系统信息"""
    print_section("系统信息")
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    
    mem = virtual_memory()
    print(f"系统内存: {mem.total / (1024**3):.2f} GB")
    print(f"可用内存: {mem.available / (1024**3):.2f} GB ({100 - mem.percent:.1f}%)")

def check_nccl_info():
    """检查 NCCL 信息"""
    print_section("NCCL 信息")
    
    if torch.cuda.is_available() and hasattr(torch.cuda, 'nccl'):
        try:
            nccl_version = torch.cuda.nccl.version()
            print(f"NCCL版本: {nccl_version}")
        except Exception as e:
            print(f"无法获取NCCL版本: {e}")
    else:
        print("NCCL不可用")

def check_env_vars():
    """检查环境变量"""
    print_section("环境变量")
    
    important_vars = [
        'NCCL_DEBUG',
        'NCCL_SHM_DISABLE',
        'NCCL_TIMEOUT',
        'NCCL_SOCKET_IFNAME',
        'NCCL_IB_DISABLE',
        'MASTER_ADDR',
        'MASTER_PORT',
        'RANK',
        'LOCAL_RANK',
        'WORLD_SIZE',
    ]
    
    for var in important_vars:
        value = os.environ.get(var, 'not set')
        print(f"  {var}: {value}")

def check_shm():
    """检查共享内存"""
    print_section("共享内存 (/dev/shm)")
    
    try:
        import shutil
        shm_stat = shutil.disk_usage('/dev/shm')
        total_gb = shm_stat.total / (1024**3)
        used_gb = shm_stat.used / (1024**3)
        free_gb = shm_stat.free / (1024**3)
        percent = (used_gb / total_gb) * 100
        
        print(f"总大小: {total_gb:.2f} GB")
        print(f"已使用: {used_gb:.2f} GB")
        print(f"可用: {free_gb:.2f} GB")
        print(f"使用率: {percent:.1f}%")
        
        # 检查是否有 NCCL 文件
        try:
            nccl_files = [f for f in os.listdir('/dev/shm') if 'nccl' in f.lower()]
            if nccl_files:
                print(f"\nNCCL文件: {len(nccl_files)} 个")
                for f in nccl_files[:5]:  # 只显示前5个
                    print(f"  - {f}")
            else:
                print("\n✓ 没有NCCL残留文件")
        except Exception as e:
            print(f"\n无法列出/dev/shm文件: {e}")
            
    except Exception as e:
        print(f"无法检查/dev/shm: {e}")

def test_nccl_init():
    """测试 NCCL 初始化"""
    print_section("NCCL 初始化测试")
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过NCCL测试")
        return
    
    # 检查是否在分布式环境中
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("检测到分布式环境")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"  RANK: {rank}")
        print(f"  LOCAL_RANK: {local_rank}")
        print(f"  WORLD_SIZE: {world_size}")
        
        try:
            # 设置设备
            torch.cuda.set_device(local_rank)
            print(f"  设置设备: cuda:{local_rank}")
            
            # 初始化进程组
            if not dist.is_initialized():
                print("  初始化分布式进程组...")
                dist.init_process_group(backend='nccl')
                print("  ✓ 分布式进程组初始化成功")
            else:
                print("  ✓ 分布式进程组已初始化")
            
            # 测试简单的通信
            print("  测试 all_reduce 操作...")
            tensor = torch.ones(1).cuda()
            dist.all_reduce(tensor)
            expected = float(world_size)
            if tensor.item() == expected:
                print(f"  ✓ all_reduce 测试成功 (结果: {tensor.item()}, 期望: {expected})")
            else:
                print(f"  ✗ all_reduce 测试失败 (结果: {tensor.item()}, 期望: {expected})")
            
            # 测试 barrier
            print("  测试 barrier 操作...")
            dist.barrier()
            print("  ✓ barrier 测试成功")
            
        except Exception as e:
            print(f"  ✗ NCCL初始化或通信失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("非分布式环境，跳过NCCL初始化测试")
        print("提示: 使用 'accelerate launch --multi_gpu --num_processes 2 ./diagnose_nccl.py' 进行完整测试")

def test_model_init():
    """测试模型初始化"""
    print_section("模型初始化测试")
    
    try:
        # 创建一个简单的模型
        model = torch.nn.Linear(10, 10)
        print("✓ 模型创建成功")
        
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model = model.to(device)
            print(f"✓ 模型移动到 {device} 成功")
            
            # 测试前向传播
            x = torch.randn(2, 10).to(device)
            y = model(x)
            print(f"✓ 前向传播成功 (输出shape: {y.shape})")
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("  NCCL 和分布式训练环境诊断")
    print("=" * 80)
    
    check_system_info()
    check_nccl_info()
    check_env_vars()
    check_shm()
    test_nccl_init()
    test_model_init()
    
    print("\n" + "=" * 80)
    print("  诊断完成")
    print("=" * 80)
    
    # 如果在分布式环境中，清理
    if dist.is_initialized():
        dist.destroy_process_group()
        print("\n✓ 分布式进程组已清理")

if __name__ == '__main__':
    main()
