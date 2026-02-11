#!/usr/bin/env python3
"""
检查 Parquet 文件的列名和数据格式
"""

import pandas as pd
import sys

def check_parquet(file_path):
    """检查 parquet 文件的结构"""
    print(f"正在检查文件: {file_path}")
    print("=" * 80)
    
    try:
        # 读取 parquet 文件
        df = pd.read_parquet(file_path)
        
        # 显示基本信息
        print(f"\n✓ 文件读取成功！")
        print(f"  总行数: {len(df):,}")
        print(f"  总列数: {len(df.columns)}")
        
        # 显示列名
        print(f"\n📋 列名列表:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        # 显示数据类型
        print(f"\n📊 数据类型:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # 显示前3行数据
        print(f"\n📝 前3行数据:")
        print("-" * 80)
        for i, row in df.head(3).iterrows():
            print(f"\n第 {i+1} 行:")
            for col in df.columns:
                value = str(row[col])
                if len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {col}: {value}")
        
        # 检查是否有空值
        print(f"\n🔍 空值检查:")
        null_counts = df.isnull().sum()
        for col in df.columns:
            null_count = null_counts[col]
            null_pct = (null_count / len(df)) * 100
            print(f"  {col}: {null_count:,} ({null_pct:.2f}%)")
        
        # 生成 dataset_info.json 配置建议
        print(f"\n" + "=" * 80)
        print(f"💡 dataset_info.json 配置建议:")
        print("-" * 80)
        
        # 尝试识别列名
        columns = list(df.columns)
        
        # 常见的列名映射
        prompt_candidates = ['input', 'prompt', 'question', 'text', 'instruction', 'query']
        response_candidates = ['target', 'response', 'answer', 'output', 'completion']
        
        prompt_col = None
        response_col = None
        
        for col in columns:
            col_lower = col.lower()
            if col_lower in prompt_candidates:
                prompt_col = col
            if col_lower in response_candidates:
                response_col = col
        
        if prompt_col and response_col:
            print(f"""
{{
  "custom_t5_dataset": {{
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {{
      "prompt": "{prompt_col}",
      "response": "{response_col}"
    }}
  }}
}}
""")
            print(f"✓ 自动识别到:")
            print(f"  - prompt 列: {prompt_col}")
            print(f"  - response 列: {response_col}")
        else:
            print(f"""
{{
  "custom_t5_dataset": {{
    "file_name": "data/my_train_dataset.parquet",
    "file_format": "parquet",
    "columns": {{
      "prompt": "{columns[0] if len(columns) > 0 else 'COLUMN_NAME'}",
      "response": "{columns[1] if len(columns) > 1 else 'COLUMN_NAME'}"
    }}
  }}
}}
""")
            print(f"⚠️ 无法自动识别，请根据实际列名修改")
            print(f"  可用列名: {', '.join(columns)}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/my_train_dataset.parquet"
    
    sys.exit(check_parquet(file_path))
