#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文维基百科数据处理脚本
功能：
1. 下载中文维基百科数据（可选，如果文件已存在则跳过）
2. 使用WikiExtractor提取文本
3. 使用OpenCC转换为简体中文
4. 保存到项目根目录的data目录

使用方法：
    python process_zhwiki.py --wiki-url <下载地址>
    或
    python process_zhwiki.py --wiki-file <本地bz2文件路径>
    
依赖库：
    pip install requests tqdm opencc-python-reimplemented
"""

import argparse
import os
import sys
import re
import subprocess
from pathlib import Path

def check_libraries():
    """检查必要的Python库"""
    missing = []
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")
    
    try:
        from opencc import OpenCC
    except ImportError:
        missing.append("opencc-python-reimplemented")
    
    if missing:
        print("错误: 缺少以下必要的库:")
        for lib in missing:
            print(f"  - {lib}")
        print("\n请安装缺失的库:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    
    # 导入库
    global requests, tqdm, OpenCC
    import requests
    from tqdm import tqdm
    from opencc import OpenCC


# 延迟导入，只在需要时检查
requests = None
tqdm = None
OpenCC = None

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR =  PROJECT_ROOT / 'data'
WIKI_EXTRACTOR = PROJECT_ROOT / 'WikiExtractor.py'

# 默认中文维基百科下载地址（需要根据实际日期更新）
DEFAULT_WIKI_URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles-multistream.xml.bz2"


def ensure_dir(path):
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, output_path, chunk_size=8192):
    """下载文件，显示进度条"""
    check_libraries()  # 确保库已导入
    print(f"开始下载: {url}")
    print(f"保存到: {output_path}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    print(f"下载完成: {output_path}")
    return output_path


def extract_wiki_text(bz2_file, output_file):
    """使用WikiExtractor提取维基百科文本"""
    print(f"\n开始提取文本: {bz2_file}")
    print(f"输出文件: {output_file}")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # WikiExtractor会输出到当前目录的wiki.txt
    default_output = PROJECT_ROOT / 'wiki.txt'
    
    # 如果已存在，询问是否删除
    if default_output.exists():
        response = input(f"发现已存在的wiki.txt，是否删除？(y/n): ").lower()
        if response == 'y':
            default_output.unlink()
        else:
            # 如果不想删除，询问是否使用现有文件
            response = input("是否使用现有文件？(y/n): ").lower()
            if response == 'y':
                import shutil
                shutil.move(str(default_output), str(output_file))
                print(f"使用现有文件: {output_file}")
                return output_file
    
    # 调用WikiExtractor（会在当前目录生成wiki.txt）
    cmd = [
        sys.executable,
        str(WIKI_EXTRACTOR),
        '--infn', str(bz2_file),
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("注意：提取过程可能需要较长时间，请耐心等待...")
    
    # 实时输出进度信息
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1,
        universal_newlines=True
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"WikiExtractor执行失败，返回码: {process.returncode}")
    
    # WikiExtractor默认输出到wiki.txt，我们需要将其移动到目标位置
    if default_output.exists():
        # 移动文件到目标位置
        import shutil
        shutil.move(str(default_output), str(output_file))
        print(f"\n文本提取完成: {output_file}")
        # 获取文件大小
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"文件大小: {file_size:.2f} MB")
    else:
        raise FileNotFoundError(f"未找到输出文件: {default_output}")
    
    return output_file


def convert_to_simplified(input_file, output_file, buffer_size=10000):
    """将繁体中文转换为简体中文"""
    check_libraries()  # 确保库已导入
    print(f"\n开始转换繁体为简体: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化OpenCC转换器（繁体转简体）
    cc = OpenCC('t2s')
    
    # 获取文件大小用于进度条
    input_size = os.path.getsize(input_file)
    
    cur_rows = []
    processed_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as read_f:
        with open(output_file, 'w', encoding='utf-8') as write_f:
            with tqdm(total=input_size, unit='B', unit_scale=True, desc='转换中') as pbar:
                for line in read_f:
                    # 将繁体转换为简体
                    line = cc.convert(line)
                    
                    # 清理空行
                    if len(line.strip()) == 0:
                        continue
                    
                    # 添加换行符（如果没有）
                    if not line.endswith('\n'):
                        line = line + '\n'
                    
                    cur_rows.append(line)
                    processed_lines += 1
                    pbar.update(len(line.encode('utf-8')))
                    
                    # 批量写入
                    if len(cur_rows) >= buffer_size:
                        write_f.writelines(cur_rows)
                        cur_rows = []
                
                # 写入剩余内容
                if len(cur_rows) > 0:
                    write_f.writelines(cur_rows)
    
    print(f"转换完成: {output_file}")
    print(f"共处理 {processed_lines} 行")
    return output_file


def process_zhwiki(wiki_file=None, wiki_url=None, skip_download=False):
    """处理中文维基百科的完整流程"""
    
    # 确保data目录存在
    ensure_dir(DATA_DIR)
    
    # 步骤1: 确定输入的bz2文件
    if wiki_file:
        bz2_file = Path(wiki_file)
        if not bz2_file.exists():
            raise FileNotFoundError(f"文件不存在: {bz2_file}")
        print(f"使用本地文件: {bz2_file}")
    elif wiki_url and not skip_download:
        # 下载文件
        bz2_file = DATA_DIR / 'zhwiki-pages-articles-multistream.xml.bz2'
        if bz2_file.exists():
            print(f"文件已存在: {bz2_file}")
            response = input("是否重新下载？(y/n): ").lower()
            if response != 'y':
                skip_download = True
            else:
                bz2_file.unlink()
        
        if not skip_download:
            download_file(wiki_url, bz2_file)
    else:
        # 尝试查找已存在的bz2文件
        bz2_files = list(DATA_DIR.glob('*.bz2'))
        if bz2_files:
            bz2_file = bz2_files[0]
            print(f"使用已存在的文件: {bz2_file}")
        else:
            raise ValueError("请提供wiki文件路径(--wiki-file)或下载地址(--wiki-url)")
    
    # 步骤2: 提取文本
    wiki_txt = DATA_DIR / 'wiki.txt'
    if wiki_txt.exists():
        print(f"提取的文本文件已存在: {wiki_txt}")
        response = input("是否重新提取？(y/n): ").lower()
        if response == 'y':
            wiki_txt.unlink()
            extract_wiki_text(bz2_file, wiki_txt)
    else:
        extract_wiki_text(bz2_file, wiki_txt)
    
    # 步骤3: 转换为简体中文
    wiki_simple_txt = DATA_DIR / 'wiki.simple.txt'
    if wiki_simple_txt.exists():
        print(f"简体文本文件已存在: {wiki_simple_txt}")
        response = input("是否重新转换？(y/n): ").lower()
        if response == 'y':
            wiki_simple_txt.unlink()
            convert_to_simplified(wiki_txt, wiki_simple_txt)
    else:
        convert_to_simplified(wiki_txt, wiki_simple_txt)
    
    print("\n" + "="*60)
    print("处理完成！")
    print(f"最终文件: {wiki_simple_txt}")
    print("="*60)
    
    return wiki_simple_txt


def get_latest_wiki_url():
    """获取最新的中文维基百科下载地址"""
    check_libraries()  # 确保库已导入
    base_url = "https://dumps.wikimedia.org/zhwiki/latest/"
    
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            # 查找multistream文件
            pattern = r'href="(zhwiki-latest-pages-articles-multistream\.xml\.bz2)"'
            match = re.search(pattern, response.text)
            if match:
                return base_url + match.group(1)
    except Exception as e:
        print(f"获取最新地址失败: {e}")
    
    return DEFAULT_WIKI_URL


def check_dependencies():
    """检查必要的依赖文件是否存在"""
    if not WIKI_EXTRACTOR.exists():
        print(f"错误: 未找到 WikiExtractor.py 文件: {WIKI_EXTRACTOR}")
        print("请确保 WikiExtractor.py 文件在项目根目录下")
        sys.exit(1)
    
    print(f"✓ WikiExtractor.py 存在: {WIKI_EXTRACTOR}")
    print(f"✓ 数据目录: {DATA_DIR}")
    print()


def main():
    # 检查依赖
    check_dependencies()
    
    parser = argparse.ArgumentParser(
        description='处理中文维基百科数据：下载、提取、转换为简体中文',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从URL下载并处理（推荐）
  python process_zhwiki.py --wiki-url https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles-multistream.xml.bz2
  
  # 使用本地bz2文件
  python process_zhwiki.py --wiki-file /path/to/zhwiki-20240101-pages-articles-multistream.xml.bz2
  
  # 使用默认URL下载（自动获取最新版本）
  python process_zhwiki.py
  
  # 如果data目录已有bz2文件，直接处理
  python process_zhwiki.py --skip-download

中文维基百科下载地址：
  https://dumps.wikimedia.org/zhwiki/
  
  推荐下载文件: zhwiki-[日期]-pages-articles-multistream.xml.bz2 (约2.7GB)

注意事项：
  1. 下载文件较大（约2.7GB），请确保网络连接稳定
  2. 提取过程可能需要较长时间（取决于CPU性能）
  3. 转换过程需要一定时间（取决于文件大小）
  4. 最终输出的wiki.simple.txt文件会保存在 data/ 目录下
        """
    )
    
    parser.add_argument(
        '--wiki-url',
        type=str,
        help='中文维基百科bz2文件的下载地址（如果不提供，将使用默认地址）'
    )
    
    parser.add_argument(
        '--wiki-file',
        type=str,
        help='本地中文维基百科bz2文件路径（如果提供，将不会下载）'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='跳过下载步骤（如果文件已存在）'
    )
    
    parser.add_argument(
        '--auto',
        action='store_true',
        help='自动模式：如果中间文件已存在则跳过对应步骤'
    )
    
    args = parser.parse_args()
    
    # 确定wiki_url
    wiki_url = args.wiki_url
    if not wiki_url and not args.wiki_file and not args.skip_download:
        print("未指定wiki文件或URL，尝试获取最新下载地址...")
        wiki_url = get_latest_wiki_url()
        print(f"使用URL: {wiki_url}")
    
    try:
        process_zhwiki(
            wiki_file=args.wiki_file,
            wiki_url=wiki_url,
            skip_download=args.skip_download
        )
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
