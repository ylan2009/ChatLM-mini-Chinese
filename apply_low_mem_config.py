#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速配置脚本 - 自动优化config.py以适应低内存环境

使用方法：
    python apply_low_mem_config.py --mode ultra  # 极致低内存模式（~8-10GB）
    python apply_low_mem_config.py --mode balanced  # 平衡模式（~10-12GB）
    python apply_low_mem_config.py --mode restore  # 恢复默认配置
"""

import os
import sys
import argparse
import shutil
from datetime import datetime

CONFIG_FILE = './config.py'
BACKUP_DIR = './config_backups'

# 配置模板
CONFIGS = {
    'ultra': {
        'name': '极致低内存模式',
        'memory': '~8-10GB',
        'settings': {
            'batch_size_per_gpu': 1,
            'gradient_accumulation_steps': 8,
            'max_seq_len': 256,
            'mixed_precision': "'no'",
            'd_model': 384,
            'd_ff': 1536,
            'num_layers': 4,
            'num_heads': 6,
        }
    },
    'balanced': {
        'name': '平衡模式',
        'memory': '~10-12GB',
        'settings': {
            'batch_size_per_gpu': 2,
            'gradient_accumulation_steps': 16,
            'max_seq_len': 384,
            'mixed_precision': "'fp16'",
            'd_model': 512,
            'd_ff': 2048,
            'num_layers': 6,
            'num_heads': 8,
        }
    }
}

def backup_config():
    """备份当前配置"""
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 配置文件不存在: {CONFIG_FILE}")
        sys.exit(1)
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = os.path.join(BACKUP_DIR, f'config_{timestamp}.py')
    shutil.copy2(CONFIG_FILE, backup_file)
    print(f"✅ 已备份配置到: {backup_file}")
    return backup_file

def read_config():
    """读取配置文件"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def write_config(content):
    """写入配置文件"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

def apply_settings(content, settings):
    """应用配置设置"""
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        # 跳过注释行
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        
        # 匹配并替换配置项
        for key, value in settings.items():
            if f'{key} =' in line or f'{key}=' in line:
                # 保留缩进
                indent = len(line) - len(line.lstrip())
                lines[i] = ' ' * indent + f'{key} = {value}'
                modified = True
                print(f"  ✓ {key} = {value}")
    
    if not modified:
        print("⚠️  警告：未找到任何配置项进行修改")
    
    return '\n'.join(lines)

def apply_mode(mode):
    """应用指定模式的配置"""
    if mode not in CONFIGS:
        print(f"❌ 未知模式: {mode}")
        print(f"可用模式: {', '.join(CONFIGS.keys())}, restore")
        sys.exit(1)
    
    config_info = CONFIGS[mode]
    print(f"\n🔧 应用配置: {config_info['name']}")
    print(f"📊 预期内存占用: {config_info['memory']}")
    print(f"\n正在修改配置项:")
    
    # 备份
    backup_file = backup_config()
    
    # 读取并修改配置
    content = read_config()
    content = apply_settings(content, config_info['settings'])
    
    # 写入
    write_config(content)
    print(f"\n✅ 配置已应用！")
    print(f"\n💡 提示:")
    print(f"  - 备份文件: {backup_file}")
    print(f"  - 如需恢复: python {sys.argv[0]} --mode restore --backup {backup_file}")
    print(f"\n🚀 现在可以运行训练:")
    print(f"  accelerate launch --multi_gpu --num_processes 2 ./train_low_mem.py train --is_finetune=True")

def restore_config(backup_file):
    """恢复配置"""
    if not backup_file:
        # 查找最新的备份
        if not os.path.exists(BACKUP_DIR):
            print("❌ 没有找到备份文件")
            sys.exit(1)
        
        backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.py')])
        if not backups:
            print("❌ 没有找到备份文件")
            sys.exit(1)
        
        backup_file = os.path.join(BACKUP_DIR, backups[-1])
        print(f"📂 使用最新备份: {backup_file}")
    
    if not os.path.exists(backup_file):
        print(f"❌ 备份文件不存在: {backup_file}")
        sys.exit(1)
    
    # 先备份当前配置
    current_backup = backup_config()
    
    # 恢复
    shutil.copy2(backup_file, CONFIG_FILE)
    print(f"✅ 已恢复配置从: {backup_file}")
    print(f"💡 当前配置已备份到: {current_backup}")

def show_current_config():
    """显示当前配置"""
    content = read_config()
    
    print("\n📋 当前配置:")
    print("-" * 50)
    
    # 提取关键配置
    key_settings = [
        'batch_size_per_gpu',
        'gradient_accumulation_steps',
        'max_seq_len',
        'mixed_precision',
        'd_model',
        'd_ff',
        'num_layers',
        'num_heads',
    ]
    
    for line in content.split('\n'):
        stripped = line.strip()
        if any(f'{key} =' in line or f'{key}=' in line for key in key_settings):
            if not stripped.startswith('#'):
                print(f"  {stripped}")
    
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(
        description='快速配置脚本 - 自动优化config.py以适应低内存环境',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 应用极致低内存模式
  python %(prog)s --mode ultra
  
  # 应用平衡模式
  python %(prog)s --mode balanced
  
  # 恢复到备份
  python %(prog)s --mode restore
  
  # 查看当前配置
  python %(prog)s --show
        """
    )
    
    parser.add_argument('--mode', choices=['ultra', 'balanced', 'restore'],
                        help='配置模式: ultra(极致低内存), balanced(平衡), restore(恢复)')
    parser.add_argument('--backup', help='恢复时指定备份文件路径')
    parser.add_argument('--show', action='store_true', help='显示当前配置')
    
    args = parser.parse_args()
    
    if args.show:
        show_current_config()
        return
    
    if not args.mode:
        parser.print_help()
        print("\n💡 提示: 使用 --show 查看当前配置")
        return
    
    if args.mode == 'restore':
        restore_config(args.backup)
    else:
        apply_mode(args.mode)

if __name__ == '__main__':
    main()
