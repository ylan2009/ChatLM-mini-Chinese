#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练数据集下载和处理脚本
功能：
1. 从远端下载所有需要的数据集
2. 调用原有的处理函数进行数据清洗和格式化
3. 生成最终的训练数据集

数据集来源：
1. webtext2019zh - 社区问答数据集
2. baike_qa2019 - 百科类问答
3. Chinese-medical-dialogue-data - 医药领域问答
4. Zhihu-KOL - 知乎问答数据
5. BELLE - 指令训练数据
6. Wikipedia - 维基百科词条

使用方法：
    python download_and_process_datasets.py
"""

import os
import sys
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
import ujson

# 添加项目路径
sys.path.extend(['.', '..'])

from config import PROJECT_ROOT
from utils.logger import Logger

# 初始化日志
log = Logger('download_datasets', save2file=True, file_name=PROJECT_ROOT + '/logs/download_datasets.log')

# 数据集下载配置
DATASETS_CONFIG = {
    'webtext2019zh': {
        'note': 'webtext2019zh数据集需要从HuggingFace下载，使用datasets库',
        'hf_dataset': 'silver/webtext2019zh',
        'save_dir': PROJECT_ROOT + '/data/raw_data/',
    },
    'baike_qa': {
        'note': '百度百科问答数据集需要从HuggingFace下载，使用datasets库',
        'hf_dataset': 'silver/baike_qa2019',
        'save_dir': PROJECT_ROOT + '/data/raw_data/',
    },
    'chinese_medical': {
        'urls': {
            'main': 'https://github.com/Toyhom/Chinese-medical-dialogue-data/archive/refs/heads/master.zip',
        },
        'save_dir': PROJECT_ROOT + '/data/raw_data/',
        'extract': True,
    },
    'belle': {
        'note': 'BELLE数据集需要从HuggingFace下载，使用datasets库',
        'hf_datasets': [
            'BelleGroup/train_1M_CN',
            'BelleGroup/train_2M_CN',
            'BelleGroup/train_3.5M_CN',

            # 'BelleGroup/generated_chat_0.4M',
            # 'BelleGroup/train_0.5M_CN'
            
        ],
        'save_dir': PROJECT_ROOT + '/data/raw_data/belle/',
    },
    'zhihu_kol': {
        'note': '知乎KOL数据集需要从HuggingFace下载，使用datasets库',
        'hf_dataset': 'wangrui6/Zhihu-KOL',
    },
}


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def download_file(url: str, save_path: str, chunk_size: int = 8192) -> bool:
    """
    下载文件，显示进度条
    
    Args:
        url: 下载链接
        save_path: 保存路径
        chunk_size: 下载块大小
        
    Returns:
        是否下载成功
    """
    try:
        log.info(f"开始下载: {url}", save_to_file=True)
        log.info(f"保存到: {save_path}", save_to_file=True)
        
        # 如果文件已存在，询问是否覆盖
        if os.path.exists(save_path):
            log.info(f"文件已存在: {save_path}，跳过下载", save_to_file=True)
            return True
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        log.info(f"下载完成: {save_path}", save_to_file=True)
        return True
        
    except Exception as e:
        log.error(f"下载失败: {url}, 错误: {str(e)}", save_to_file=True)
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """解压zip文件"""
    try:
        log.info(f"解压文件: {zip_path}", save_to_file=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        log.info(f"解压完成: {extract_to}", save_to_file=True)
        return True
    except Exception as e:
        log.error(f"解压失败: {str(e)}", save_to_file=True)
        return False


def download_webtext2019zh() -> bool:
    """下载webtext2019zh数据集（使用HuggingFace datasets库）"""
    log.info("=" * 60, save_to_file=True)
    log.info("下载 webtext2019zh 数据集", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    try:
        from datasets import load_dataset
        
        config = DATASETS_CONFIG['webtext2019zh']
        ensure_dir(config['save_dir'])
        
        log.info(f"从HuggingFace下载: {config['hf_dataset']}", save_to_file=True)
        
        # 下载数据集（包含train, valid, test分割）
        dataset = load_dataset(config['hf_dataset'])
        
        # 分别保存各个分割
        for split_name in dataset.keys():
            save_path = os.path.join(config['save_dir'], f'web_text_zh_{split_name}.parquet')
            dataset[split_name].to_parquet(save_path)
            log.info(f"{split_name} 数据集已保存到: {save_path}", save_to_file=True)
            log.info(f"{split_name} 数据集大小: {len(dataset[split_name])} 行", save_to_file=True)
        
        return True
        
    except ImportError:
        log.error("需要安装 datasets 库: pip install datasets", save_to_file=True)
        return False
    except Exception as e:
        log.error(f"下载失败: {str(e)}", save_to_file=True)
        return False


def download_baike_qa() -> bool:
    """下载百度百科问答数据集（使用HuggingFace datasets库）"""
    log.info("=" * 60, save_to_file=True)
    log.info("下载 baike_qa 数据集", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    try:
        from datasets import load_dataset
        
        config = DATASETS_CONFIG['baike_qa']
        ensure_dir(config['save_dir'])
        
        log.info(f"从HuggingFace下载: {config['hf_dataset']}", save_to_file=True)
        
        # 下载数据集（包含train, valid分割）
        dataset = load_dataset(config['hf_dataset'])
        
        # 分别保存各个分割
        for split_name in dataset.keys():
            save_path = os.path.join(config['save_dir'], f'baike_qa_{split_name}.parquet')
            dataset[split_name].to_parquet(save_path)
            log.info(f"{split_name} 数据集已保存到: {save_path}", save_to_file=True)
            log.info(f"{split_name} 数据集大小: {len(dataset[split_name])} 行", save_to_file=True)
        
        return True
        
    except ImportError:
        log.error("需要安装 datasets 库: pip install datasets", save_to_file=True)
        return False
    except Exception as e:
        log.error(f"下载失败: {str(e)}", save_to_file=True)
        return False


def download_chinese_medical() -> bool:
    """下载中国医药领域问答数据集"""
    log.info("=" * 60, save_to_file=True)
    log.info("下载 Chinese-medical-dialogue-data 数据集", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    config = DATASETS_CONFIG['chinese_medical']
    ensure_dir(config['save_dir'])
    
    url = config['urls']['main']
    save_path = os.path.join(config['save_dir'], 'chinese_medical_dialogue.zip')
    
    if not download_file(url, save_path):
        return False
    
    # 解压文件
    extract_to = os.path.join(config['save_dir'], 'chinese_medical_dialogue_datasets')
    if config.get('extract', False):
        return extract_zip(save_path, extract_to)
    
    return True


def download_belle_datasets() -> bool:
    """下载BELLE开源数据集（使用HuggingFace datasets库）"""
    log.info("=" * 60, save_to_file=True)
    log.info("下载 BELLE 数据集", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    try:
        from datasets import load_dataset
        
        config = DATASETS_CONFIG['belle']
        ensure_dir(config['save_dir'])
        
        success = True
        for dataset_name in config['hf_datasets']:
            try:
                log.info(f"从HuggingFace下载: {dataset_name}", save_to_file=True)
                
                # 下载数据集
                dataset = load_dataset(dataset_name, split='train')
                
                # 提取数据集名称作为文件名
                # 例如: BelleGroup/train_1M_CN -> train_1M_CN
                file_name = dataset_name.split('/')[-1]
                save_path = os.path.join(config['save_dir'], f'{file_name}.parquet')
                
                # 保存为parquet格式
                dataset.to_parquet(save_path)
                
                log.info(f"数据集已保存到: {save_path}", save_to_file=True)
                log.info(f"数据集大小: {len(dataset)} 行", save_to_file=True)
                
            except Exception as e:
                log.error(f"下载 {dataset_name} 失败: {str(e)}", save_to_file=True)
                success = False
        
        return success
        
    except ImportError:
        log.error("需要安装 datasets 库: pip install datasets", save_to_file=True)
        return False
    except Exception as e:
        log.error(f"下载失败: {str(e)}", save_to_file=True)
        return False


def download_zhihu_kol() -> bool:
    """下载知乎KOL数据集（使用HuggingFace datasets库）"""
    log.info("=" * 60, save_to_file=True)
    log.info("下载 Zhihu-KOL 数据集", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    try:
        from datasets import load_dataset
        
        config = DATASETS_CONFIG['zhihu_kol']
        save_dir = PROJECT_ROOT + '/data/raw_data/zhihu-kol'
        ensure_dir(save_dir)
        
        log.info(f"从HuggingFace下载: {config['hf_dataset']}", save_to_file=True)
        
        # 下载数据集
        dataset = load_dataset(config['hf_dataset'], split='train')
        
        # 保存为parquet格式
        save_path = os.path.join(save_dir, 'zhihu_kol.parquet')
        dataset.to_parquet(save_path)
        
        log.info(f"数据集已保存到: {save_path}", save_to_file=True)
        return True
        
    except ImportError:
        log.error("需要安装 datasets 库: pip install datasets", save_to_file=True)
        return False
    except Exception as e:
        log.error(f"下载失败: {str(e)}", save_to_file=True)
        return False


def check_wiki_simple_file() -> bool:
    """检查wiki.simple.txt文件是否存在"""
    log.info("=" * 60, save_to_file=True)
    log.info("检查 wiki.simple.txt 文件", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    wiki_simple_file = PROJECT_ROOT + '/data/wiki.simple.txt'
    
    if os.path.exists(wiki_simple_file):
        file_size = os.path.getsize(wiki_simple_file) / (1024 * 1024)  # MB
        log.info(f"✓ 找到 wiki.simple.txt 文件: {wiki_simple_file}", save_to_file=True)
        log.info(f"  文件大小: {file_size:.2f} MB", save_to_file=True)
        return True
    else:
        log.warning(f"✗ 未找到 wiki.simple.txt 文件: {wiki_simple_file}", save_to_file=True)
        log.warning("  请先使用 tokenize/process_zhwiki.py 处理维基百科数据", save_to_file=True)
        return False


def download_all_datasets() -> dict:
    """
    下载所有数据集
    
    Returns:
        下载结果字典
    """
    results = {}
    
    log.info("开始下载所有数据集...", save_to_file=True)
    
    # 1. webtext2019zh
    # results['webtext2019zh'] = download_webtext2019zh()
    
    # 2. baike_qa
    # results['baike_qa'] = download_baike_qa()
    
    # 3. chinese_medical
    results['chinese_medical'] = download_chinese_medical()
    
    # 4. belle
    results['belle'] = download_belle_datasets()
    
    # 5. zhihu_kol
    results['zhihu_kol'] = download_zhihu_kol()
    
    # 6. wiki - 不需要下载，直接使用data/wiki.simple.txt
    log.info("注意: Wiki数据使用项目中已有的 data/wiki.simple.txt 文件", save_to_file=True)
    results['wiki'] = check_wiki_simple_file()
    
    return results


def process_all_datasets() -> None:
    """处理所有已下载的数据集"""
    log.info("=" * 60, save_to_file=True)
    log.info("开始处理数据集", save_to_file=True)
    log.info("=" * 60, save_to_file=True)
    
    # 导入原有的处理函数
    from raw_data_process import (
        process_web_text,
        process_belle,
        process_chinese_medical_datasets,
        process_zhihu_kol_dataset,
        process_belle_knowledge_enhanced_dataset,
        process_wiki_simple_to_dataset,
        merge_dataset_as_single_file,
        remove_dataset_duplicate_rows,
        shuffle_parquet_dataset,
        split_train_valid_test_datasets,
        parquet_to_text,
        count_my_parquet_data,
        dataset_length_cnt,
        process_belle_knowledge_enhanced_dataset_for_finetune,
        parquet_to_json,
    )
    
    # 确保输出目录存在
    processed_file_dir = PROJECT_ROOT + '/data/my_data'
    if not os.path.exists(processed_file_dir):
        os.makedirs(processed_file_dir)
    
    try:


        # 1. 处理chinese_medical
        log.info("处理 chinese_medical 数据集...", save_to_file=True)
        # process_chinese_medical_datasets(response_less_word=15)
        
        # 2. 处理zhihu_kol
        log.info("处理 zhihu_kol 数据集...", save_to_file=True)
        # process_zhihu_kol_dataset(prompt_less_word=4, response_less_word=10)
        
        # 3. belle
        log.info("处理 belle 数据集...", save_to_file=True)
        # process_belle(response_less_word=15)
        
        # 4. 处理wiki（使用已有的wiki.simple.txt）
        wiki_simple_file = PROJECT_ROOT + '/data/wiki.simple.txt'
        if os.path.exists(wiki_simple_file):
            log.info("处理 wiki 数据集...", save_to_file=True)
            # process_wiki_simple_to_dataset(groups_cnt=10000, max_len=512)
        else:
            log.warning("未找到 wiki.simple.txt 文件，跳过wiki数据处理", save_to_file=True)
            log.warning("如需处理wiki数据，请先运行: python tokenize/process_zhwiki.py", save_to_file=True)
        
        # 5. 合并所有数据集
        log.info("合并所有数据集...", save_to_file=True)
        # merge_dataset_as_single_file(groups_cnt=100000, min_len=3, max_len=512, cut_max_len=True)
        
        # 6. 去重（优化版：分批处理，减少内存占用）
        log.info("去除重复数据...", save_to_file=True)
        # groups_cnt: 每次写入文件的行数
        # batch_size: 每批处理的数据量（默认100000），可根据内存大小调整
        # remove_dataset_duplicate_rows(groups_cnt=100000)
        
        # 7. 打乱数据（使用去重后的数据集）
        log.info("打乱数据集...", save_to_file=True)
        # shuffle_parquet_dataset(
        #     parquet_file=PROJECT_ROOT + '/data/my_dataset_no_dulpticates.parquet',  # 使用去重后的文件
        #     shuffle_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',
        #     seed=23333
        # )
        
        # 8. 划分训练集、验证集、测试集
        log.info("划分训练集、验证集、测试集...", save_to_file=True)
        # split_train_valid_test_datasets(
        #     source_parquet_file=PROJECT_ROOT + '/data/my_dataset.shuffle.parquet',
        #     max_len=320,
        #     groups_cnt=100000
        # )
        
        # 9. 转换为文本格式（用于训练tokenizer）
        log.info("转换为文本格式...", save_to_file=True)
        # parquet_to_text()
        
        # 9. 统计数据
        log.info("统计数据集信息...", save_to_file=True)
        # count_my_parquet_data(PROJECT_ROOT + '/data/')
        
        # 10. 统计长度分布
        log.info("统计长度分布...", save_to_file=True)
        # dataset_length_cnt()
        
        # 11. 处理微调数据集
        log.info("处理微调数据集...", save_to_file=True)
        process_belle_knowledge_enhanced_dataset_for_finetune(max_len=320, group_cnt=100000)
        
        # 12. 转换为JSON格式
        log.info("转换为JSON格式...", save_to_file=True)
        parquet_to_json()
        
        log.info("=" * 60, save_to_file=True)
        log.info("所有数据集处理完成！", save_to_file=True)
        log.info("=" * 60, save_to_file=True)
        
    except Exception as e:
        log.error(f"处理数据集时出错: {str(e)}", save_to_file=True)
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='下载和处理预训练数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载所有数据集
  python download_and_process_datasets.py --download-all
  
  # 只下载特定数据集
  python download_and_process_datasets.py --download webtext2019zh baike_qa
  
  # 处理已下载的数据集
  python download_and_process_datasets.py --process
  
  # 下载并处理
  python download_and_process_datasets.py --download-all --process

数据集说明:
  1. webtext2019zh - 社区问答数据集（约410万条）
  2. baike_qa - 百科类问答（约140万条）
  3. chinese_medical - 医药领域问答（约79万条）
  4. belle - BELLE指令训练数据（约370万条）
  5. zhihu_kol - 知乎问答数据（约100万条）
  6. wiki - 维基百科词条（约119万条，使用已有的data/wiki.simple.txt）
        """
    )
    
    parser.add_argument(
        '--download-all',
        action='store_true',
        help='下载所有数据集'
    )
    
    parser.add_argument(
        '--download',
        nargs='+',
        choices=['webtext2019zh', 'baike_qa', 'chinese_medical', 'belle', 'zhihu_kol'],
        help='下载指定的数据集'
    )
    
    parser.add_argument(
        '--process',
        action='store_true',
        help='处理已下载的数据集'
    )
    
    parser.add_argument(
        '--skip-wiki',
        action='store_true',
        help='跳过维基百科数据集处理'
    )
    
    args = parser.parse_args()
    
    # 确保必要的目录存在
    ensure_dir(PROJECT_ROOT + '/data/raw_data')
    ensure_dir(PROJECT_ROOT + '/data/my_data')
    ensure_dir(PROJECT_ROOT + '/logs')
    
    try:
        # 下载数据集
        if args.download_all:
            results = download_all_datasets()
            
            # 打印下载结果
            log.info("=" * 60, save_to_file=True)
            log.info("下载结果汇总:", save_to_file=True)
            for dataset, success in results.items():
                status = "✓ 成功" if success else "✗ 失败"
                log.info(f"  {dataset}: {status}", save_to_file=True)
            log.info("=" * 60, save_to_file=True)
            
        elif args.download:
            for dataset in args.download:
                if dataset == 'webtext2019zh':
                    download_webtext2019zh()
                elif dataset == 'baike_qa':
                    download_baike_qa()
                elif dataset == 'chinese_medical':
                    download_chinese_medical()
                elif dataset == 'belle':
                    download_belle_datasets()
                elif dataset == 'zhihu_kol':
                    download_zhihu_kol()
        
        # 处理数据集
        if args.process:
            process_all_datasets()
        
        if not args.download_all and not args.download and not args.process:
            parser.print_help()
            
    except KeyboardInterrupt:
        log.info("\n用户中断操作", save_to_file=True)
        sys.exit(1)
    except Exception as e:
        log.error(f"错误: {str(e)}", save_to_file=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
