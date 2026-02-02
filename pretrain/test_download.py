#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试下载功能的简单脚本
用于验证各个数据集的下载链接是否有效
"""

import sys
sys.path.extend(['.', '..'])

from download_and_process_datasets import (
    DATASETS_CONFIG,
    download_file,
    ensure_dir,
    log
)
import os
from config import PROJECT_ROOT

def test_download_links():
    """测试所有下载链接的可用性"""
    
    log.info("=" * 60, save_to_file=True)
    log.info("测试数据集下载链接", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    results = {}
    
    for dataset_name, config in DATASETS_CONFIG.items():
        log.info(f"\n测试数据集: {dataset_name}", save_to_file=True)
        
        if 'urls' in config:
            dataset_results = {}
            for name, url in config['urls'].items():
                log.info(f"  测试链接: {name}", save_to_file=True)
                log.info(f"  URL: {url}", save_to_file=True)
                
                # 只测试链接是否可访问，不实际下载
                try:
                    import requests
                    response = requests.head(url, timeout=10, allow_redirects=True)
                    if response.status_code == 200:
                        log.info(f"  ✓ 链接有效", save_to_file=True)
                        dataset_results[name] = True
                    else:
                        log.info(f"  ✗ 链接无效 (状态码: {response.status_code})", save_to_file=True)
                        dataset_results[name] = False
                except Exception as e:
                    log.error(f"  ✗ 测试失败: {str(e)}", save_to_file=True)
                    dataset_results[name] = False
            
            results[dataset_name] = dataset_results
        elif 'hf_dataset' in config:
            log.info(f"  HuggingFace数据集: {config['hf_dataset']}", save_to_file=True)
            log.info(f"  需要使用 datasets 库下载", save_to_file=True)
            results[dataset_name] = {'hf': True}
    
    # 打印测试结果
    log.info("\n" + "=" * 60, save_to_file=True)
    log.info("测试结果汇总", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    for dataset_name, dataset_results in results.items():
        log.info(f"\n{dataset_name}:", save_to_file=True)
        for name, success in dataset_results.items():
            status = "✓" if success else "✗"
            log.info(f"  {status} {name}", save_to_file=True)
    
    return results

def test_directory_structure():
    """测试目录结构是否正确"""
    
    log.info("\n" + "=" * 60, save_to_file=True)
    log.info("测试目录结构", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    required_dirs = [
        PROJECT_ROOT + '/data',
        PROJECT_ROOT + '/data/raw_data',
        PROJECT_ROOT + '/data/my_data',
        PROJECT_ROOT + '/logs',
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            log.info(f"✓ {dir_path}", save_to_file=True)
        else:
            log.info(f"✗ {dir_path} (不存在，将自动创建)", save_to_file=True)
            ensure_dir(dir_path)
            log.info(f"  ✓ 已创建", save_to_file=True)

def test_dependencies():
    """测试依赖库是否已安装"""
    
    log.info("\n" + "=" * 60, save_to_file=True)
    log.info("测试依赖库", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    required_packages = [
        'requests',
        'tqdm',
        'ujson',
        'pandas',
        'pyarrow',
        'fastparquet',
        'datasets',
        'opencc',
        'rich',
        'matplotlib',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            log.info(f"✓ {package}", save_to_file=True)
        except ImportError:
            log.info(f"✗ {package} (未安装)", save_to_file=True)
            missing_packages.append(package)
    
    if missing_packages:
        log.info("\n缺少以下依赖库:", save_to_file=True)
        log.info(f"pip install {' '.join(missing_packages)}", save_to_file=True)
        return False
    else:
        log.info("\n✓ 所有依赖库已安装", save_to_file=True)
        return True

def main():
    """主函数"""
    
    log.info("开始测试...\n", save_to_file=True)
    
    # 1. 测试依赖库
    deps_ok = test_dependencies()
    
    # 2. 测试目录结构
    test_directory_structure()
    
    # 3. 测试下载链接（如果依赖库都安装了）
    if deps_ok:
        test_download_links()
    else:
        log.info("\n请先安装缺失的依赖库，然后重新运行测试", save_to_file=True)
    
    log.info("\n" + "=" * 60, save_to_file=True)
    log.info("测试完成", save_to_file=True)
    log.info("=" * 60, save_to_file=True)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
