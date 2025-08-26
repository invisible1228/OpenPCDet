#!/usr/bin/env python3
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

def parse_evaluation_file(file_path):
    """解析评估结果文件"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        print(f"\n📄 Parsing file: {file_path}")
        print(f"File size: {len(content)} characters")
        
        if len(content) == 0:
            print("⚠️ File is empty!")
            return {}
        
        print(f"Full content:\n{'='*60}")
        print(content)
        print('='*60)
        
        # 提取mAP结果
        results = {}
        
        # 尝试不同的模式匹配
        patterns = [
            r'Car.*?moderate.*?(\d+\.\d+)',  # Car moderate
            r'car.*?moderate.*?(\d+\.\d+)',  # car moderate (小写)
            r'Car_3d_moderate.*?(\d+\.\d+)', # Car_3d_moderate
            r'moderate.*?(\d+\.\d+)',        # 任何moderate
            r'mAP.*?(\d+\.\d+)',             # mAP
        ]
        
        print(f"Trying regex patterns...")
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"  ✅ Found pattern '{pattern}': {matches}")
        
        # 按行分析
        lines = content.split('\n')
        print(f"Analyzing {len(lines)} lines...")
        for i, line in enumerate(lines):
            if line.strip():  # 非空行
                print(f"  Line {i:2d}: {line.strip()}")
                if 'car' in line.lower() or 'moderate' in line.lower() or any(char.isdigit() for char in line):
                    print(f"    ⭐ Important line: {line.strip()}")
        
        return results
    
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return {}

def analyze_all_results():
    """分析所有结果文件"""
    results_dir = "evaluation_results"
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # 找到所有结果文件
    result_files = []
    for file in os.listdir(results_dir):
        if file.startswith('eval_result_') and file.endswith('.txt'):
            result_files.append(file)
    
    if not result_files:
        print("❌ No result files found!")
        return
    
    print(f"📁 Found {len(result_files)} result files:")
    for file in result_files:
        print(f"  - {file}")
    
    # 解析每个文件
    all_results = {}
    for file in result_files:
        model_name = file.replace('eval_result_', '').replace('.txt', '')
        file_path = os.path.join(results_dir, file)
        results = parse_evaluation_file(file_path)
        all_results[model_name] = results
    
    return all_results

def inspect_dataframe_structure(df):
    """检查数据框结构"""
    print("📊 DataFrame structure:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {list(df.index)}")
    print("\nDataFrame content:")
    print(df)
    print("\nColumn types:")
    print(df.dtypes)

def safe_extract_metrics(results_dir):
    """安全地提取评估指标"""
    data = []
    
    for file in os.listdir(results_dir):
        if file.startswith('eval_result_') and file.endswith('.txt'):
            model_name = file.replace('eval_result_', '').replace('.txt', '')
            file_path = os.path.join(results_dir, file)
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # 尝试提取各种可能的指标
                metrics = {'model': model_name}
                
                # 查找包含数字的行
                lines = content.split('\n')
                for line in lines:
                    # 查找包含数字的行
                    numbers = re.findall(r'\d+\.\d+', line)
                    if numbers and ('car' in line.lower() or 'moderate' in line.lower()):
                        print(f"Model {model_name}: {line.strip()}")
                        
                        # 尝试提取具体指标
                        if 'moderate' in line.lower():
                            metrics['car_moderate'] = float(numbers[0]) if numbers else 0.0
                        if 'easy' in line.lower():
                            metrics['car_easy'] = float(numbers[0]) if numbers else 0.0
                        if 'hard' in line.lower():
                            metrics['car_hard'] = float(numbers[0]) if numbers else 0.0
                
                data.append(metrics)
                
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
    
    return pd.DataFrame(data)

def main():
    print("🔍 Starting result analysis debugging...")
    
    # 分析所有结果
    all_results = analyze_all_results()
    
    # 尝试创建数据框
    results_dir = "evaluation_results"
    if os.path.exists(results_dir):
        df = safe_extract_metrics(results_dir)
        
        if not df.empty:
            print("\n" + "="*50)
            print("📊 EXTRACTED DATAFRAME:")
            print("="*50)
            inspect_dataframe_structure(df)
            
            # 保存为CSV方便查看
            csv_file = os.path.join(results_dir, "extracted_metrics.csv")
            df.to_csv(csv_file, index=False)
            print(f"\n✅ Metrics saved to: {csv_file}")
        else:
            print("❌ No data extracted!")
    
    print("\n🔍 Debugging completed!")

if __name__ == "__main__":
    main()