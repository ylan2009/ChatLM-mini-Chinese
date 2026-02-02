@echo off
REM 预训练数据集一键下载和处理脚本 (Windows版本)

echo ==========================================
echo   ChatLM-mini-Chinese 预训练数据集准备
echo ==========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 python，请先安装 Python 3
    pause
    exit /b 1
)

echo √ Python 环境检查通过
echo.

REM 检查并安装依赖
echo 检查依赖库...
pip install -q requests tqdm ujson pandas pyarrow fastparquet datasets opencc-python-reimplemented colorlog rich matplotlib

echo √ 依赖库安装完成
echo.

REM 询问是否下载维基百科数据
set /p download_wiki="是否下载维基百科数据？(文件约2.7GB，下载和处理时间较长) [y/N]: "

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 下载数据集
echo.
echo ==========================================
echo   步骤 1/2: 下载数据集
echo ==========================================
echo.

if /i "%download_wiki%"=="y" (
    python download_and_process_datasets.py --download-all
) else (
    python download_and_process_datasets.py --download webtext2019zh baike_qa chinese_medical belle zhihu_kol
)

if errorlevel 1 (
    echo 下载失败，请检查网络连接或查看日志文件
    pause
    exit /b 1
)

REM 处理数据集
echo.
echo ==========================================
echo   步骤 2/2: 处理数据集
echo ==========================================
echo.

python download_and_process_datasets.py --process

if errorlevel 1 (
    echo 处理失败，请查看日志文件
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   √ 所有任务完成！
echo ==========================================
echo.
echo 生成的文件：
echo   - 训练集: ..\data\my_train_dataset.parquet
echo   - 测试集: ..\data\my_test_dataset.parquet
echo   - 验证集: ..\data\my_valid_dataset.parquet
echo   - 语料库: ..\data\my_corpus.txt
echo.
echo 日志文件：
echo   - 下载日志: ..\logs\download_datasets.log
echo   - 处理日志: ..\logs\raw_data_process.log
echo.
echo 下一步：
echo   1. 训练 tokenizer: cd ..\tokenize ^&^& python train_tokenizer.py
echo   2. 开始预训练: cd ..\pretrain ^&^& python pretrain.py
echo.

pause
