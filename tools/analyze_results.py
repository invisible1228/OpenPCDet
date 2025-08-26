#!/usr/bin/env python3
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_kitti_results(result_text):
    """解析KITTI评估结果 - 增强版"""
    results = {}
    
    print(f"🔍 Parsing result text ({len(result_text)} chars)")
    if len(result_text.strip()) == 0:
        print("⚠️ Empty result text!")
        return results
    
    # 打印内容预览
    print("Content preview:")
    print("=" * 50)
    print(result_text[:1000])
    print("=" * 50)
    
    # 尝试多种可能的格式模式
    patterns_v1 = {
        'car_easy': r'Car AP@0\.70, 0\.70, 0\.70:\s*bbox AP:([\d\.]+), ([\d\.]+), ([\d\.]+)',
        'car_moderate': r'Car AP@0\.70, 0\.70, 0\.70:\s*bbox AP:[\d\.]+, ([\d\.]+), [\d\.]+',
        'car_hard': r'Car AP@0\.70, 0\.70, 0\.70:\s*bbox AP:[\d\.]+, [\d\.]+, ([\d\.]+)',
    }
    
    # 备用模式（更宽松的匹配）
    patterns_v2 = {
        'car_easy': r'Car.*?Easy.*?(\d+\.\d+)',
        'car_moderate': r'Car.*?Moderate.*?(\d+\.\d+)',
        'car_hard': r'Car.*?Hard.*?(\d+\.\d+)',
    }
    
    # 更通用的模式
    patterns_v3 = {
        'car_easy': r'Car.*?AP.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)',
        'car_moderate': r'Car.*?AP.*?\d+\.\d+.*?(\d+\.\d+).*?\d+\.\d+',
        'car_hard': r'Car.*?AP.*?\d+\.\d+.*?\d+\.\d+.*?(\d+\.\d+)',
    }
    
    # 数字提取模式
    patterns_v4 = {
        'car_results': r'Car.*?(\d+\.\d+)',
    }
    
    all_patterns = [
        ("Standard KITTI format", patterns_v1),
        ("Easy/Moderate/Hard format", patterns_v2), 
        ("General AP format", patterns_v3),
        ("Number extraction", patterns_v4)
    ]
    
    # 尝试每种模式
    for pattern_name, patterns in all_patterns:
        print(f"\n🔍 Trying {pattern_name}...")
        found_matches = False
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, result_text, re.IGNORECASE | re.DOTALL)
            if matches:
                print(f"  ✅ Found {key}: {matches}")
                if key == 'car_results':
                    # 如果只是数字提取，假设顺序是 easy, moderate, hard
                    numbers = [float(m) for m in matches[:3]]
                    if len(numbers) >= 3:
                        results['car_easy'] = numbers[0]
                        results['car_moderate'] = numbers[1] 
                        results['car_hard'] = numbers[2]
                else:
                    results[key] = float(matches[0] if isinstance(matches[0], str) else matches[0][0])
                found_matches = True
        
        if found_matches:
            print(f"✅ Successfully parsed with {pattern_name}")
            break
    
    # 如果所有模式都失败，尝试手动查找数字
    if not results:
        print("⚠️ Pattern matching failed, trying manual number extraction...")
        lines = result_text.split('\n')
        for line in lines:
            if 'car' in line.lower():
                numbers = re.findall(r'\d+\.\d+', line)
                if numbers:
                    print(f"  Found line with numbers: {line.strip()}")
                    print(f"  Numbers: {numbers}")
                    # 简单启发式：如果有3个数字，假设是easy, moderate, hard
                    if len(numbers) >= 3:
                        results['car_easy'] = float(numbers[0])
                        results['car_moderate'] = float(numbers[1])
                        results['car_hard'] = float(numbers[2])
                        break
    
    print(f"🎯 Final extracted results: {results}")
    return results

def safe_create_comparison_table():
    """安全创建对比表格"""
    results_dir = "evaluation_results"
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    # 读取所有结果文件
    models_results = {}
    
    result_files = [f for f in os.listdir(results_dir) if f.startswith('eval_result_') and f.endswith('.txt')]
    
    if not result_files:
        print("❌ No result files found!")
        return pd.DataFrame()
    
    print(f"📁 Found {len(result_files)} result files: {result_files}")
    
    for filename in result_files:
        model_name = filename.replace('eval_result_', '').replace('.txt', '')
        file_path = os.path.join(results_dir, filename)
        
        print(f"\n🔍 Processing {model_name}...")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if len(content.strip()) == 0:
                print(f"⚠️ Empty file: {filename}")
                # 尝试读取完整输出文件
                full_file = file_path.replace('eval_result_', 'eval_full_')
                if os.path.exists(full_file):
                    print(f"📄 Trying full output file: {os.path.basename(full_file)}")
                    with open(full_file, 'r') as f:
                        content = f.read()
            
            parsed_results = parse_kitti_results(content)
            models_results[model_name] = parsed_results
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            models_results[model_name] = {}
    
    # 创建DataFrame
    if not models_results or all(not results for results in models_results.values()):
        print("❌ No valid results extracted from any file!")
        return pd.DataFrame()
    
    df = pd.DataFrame(models_results).T
    df = df.round(2)
    
    print("\n📊 Created DataFrame:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {list(df.index)}")
    print("\nDataFrame content:")
    print(df)
    
    # 保存为CSV
    csv_file = f"{results_dir}/comparison_table.csv"
    df.to_csv(csv_file)
    print(f"✅ Comparison table saved to: {csv_file}")
    
    return df

def safe_create_visualizations(df, output_dir):
    """安全创建可视化图表"""
    if df.empty:
        print("⚠️ DataFrame is empty, skipping visualizations")
        return
    
    try:
        # 1. 柱状图对比
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PillarFocus Model Performance Comparison', fontsize=16)
        
        # Car detection performance
        car_metrics = [col for col in ['car_easy', 'car_moderate', 'car_hard'] if col in df.columns]
        
        if car_metrics:
            ax1 = axes[0, 0]
            car_data = df[car_metrics].fillna(0)
            car_data.plot(kind='bar', ax=ax1)
            ax1.set_title('Car Detection Performance')
            ax1.set_ylabel('AP (%)')
            ax1.legend([col.replace('car_', '').title() for col in car_metrics])
            ax1.tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No car metrics found', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # Pedestrian performance
        if 'pedestrian_easy' in df.columns:
            ax2 = axes[0, 1]
            df['pedestrian_easy'].plot(kind='bar', ax=ax2)
            ax2.set_title('Pedestrian Detection (Easy)')
            ax2.set_ylabel('AP (%)')
            ax2.tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No pedestrian metrics', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Cyclist performance  
        if 'cyclist_easy' in df.columns:
            ax3 = axes[1, 0]
            df['cyclist_easy'].plot(kind='bar', ax=ax3)
            ax3.set_title('Cyclist Detection (Easy)')
            ax3.set_ylabel('AP (%)')
            ax3.tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No cyclist metrics', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Overall comparison
        ax4 = axes[1, 1]
        if car_metrics:
            df[car_metrics].plot(kind='bar', ax=ax4)
            ax4.set_title('Overall Car Detection Performance')
            ax4.set_ylabel('AP (%)')
            ax4.legend([col.replace('car_', '').title() for col in car_metrics])
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No overall metrics', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Performance comparison chart saved")
        
        # 2. 改进效果图
        if 'baseline_pointpillar' in df.index and 'pillarfocus_full' in df.index:
            create_improvement_chart(df, output_dir)
        else:
            print("⚠️ Missing baseline or full model for improvement chart")
            
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def create_improvement_chart(df, output_dir):
    """创建改进效果图"""
    try:
        baseline = df.loc['baseline_pointpillar']
        full_model = df.loc['pillarfocus_full']
        
        improvements = {}
        for metric in baseline.index:
            if pd.notna(baseline[metric]) and pd.notna(full_model[metric]) and baseline[metric] != 0:
                improvement = full_model[metric] - baseline[metric]
                improvements[metric] = improvement
        
        if improvements:
            plt.figure(figsize=(10, 6))
            metrics = list(improvements.keys())
            values = list(improvements.values())
            
            colors = ['green' if v > 0 else 'red' for v in values]
            bars = plt.bar(metrics, values, color=colors, alpha=0.7)
            
            plt.title('PillarFocus Performance Improvement over Baseline')
            plt.ylabel('AP Improvement (%)')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'+{value:.2f}' if value > 0 else f'{value:.2f}',
                        ha='center', va='bottom' if value > 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/improvement_chart.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Improvement chart saved")
        else:
            print("⚠️ No valid improvements to chart")
            
    except Exception as e:
        print(f"❌ Error creating improvement chart: {e}")

def safe_generate_final_report():
    """安全生成最终报告"""
    print("🚀 Starting result analysis...")
    
    df = safe_create_comparison_table()
    
    if df.empty:
        print("❌ No data available for analysis!")
        return
    
    results_dir = "evaluation_results"
    
    # 创建可视化
    safe_create_visualizations(df, results_dir)
    
    # 生成报告
    report_file = f"{results_dir}/final_analysis_report.md"
    
    try:
        with open(report_file, 'w') as f:
            f.write("# PillarFocus模型详细分析报告\n\n")
            
            f.write("## 实验设置\n\n")
            f.write("- **数据集**: KITTI 3D Object Detection\n")
            f.write("- **基准模型**: PointPillar\n") 
            f.write("- **提出模型**: PillarFocus\n")
            f.write("- **训练轮次**: 80 epochs\n")
            f.write("- **评估指标**: Average Precision (AP) @ IoU=0.7 for Car, IoU=0.5 for Pedestrian/Cyclist\n\n")
            
            f.write("## 性能对比表\n\n")
            f.write(df.to_markdown())
            f.write("\n\n")
            
            # 安全地计算改进幅度
            if 'baseline_pointpillar' in df.index and 'pillarfocus_full' in df.index:
                f.write("## 改进分析\n\n")
                baseline = df.loc['baseline_pointpillar']
                full_model = df.loc['pillarfocus_full']
                
                for metric in baseline.index:
                    if pd.notna(baseline[metric]) and pd.notna(full_model[metric]) and baseline[metric] != 0:
                        improvement = full_model[metric] - baseline[metric]
                        relative_improvement = (improvement / baseline[metric]) * 100
                        f.write(f"- **{metric}**: {improvement:+.2f} AP ({relative_improvement:+.1f}%)\n")
                f.write("\n")
            else:
                f.write("## 改进分析\n\n")
                f.write("⚠️ 缺少基准模型或完整模型数据\n\n")
            
            # 消融实验分析
            f.write("## 消融实验分析\n\n")
            
            ablation_models = [name for name in df.index if 'ablation' in name]
            if ablation_models:
                f.write("消融实验显示了不同组件的贡献：\n\n")
                
                for model in ablation_models:
                    f.write(f"### {model}\n\n")
                    if 'no_dynamic' in model:
                        f.write("移除Dynamic Pillar Focus机制的影响：\n")
                    elif 'no_attention' in model:
                        f.write("移除空间注意力机制的影响：\n")
                    
                    model_results = df.loc[model]
                    for col in df.columns:
                        f.write(f"- {col}: {model_results.get(col, 'N/A')}\n")
                    f.write("\n")
            else:
                f.write("⚠️ 未找到消融实验数据\n\n")
            
            f.write("## 结论\n\n")
            f.write("1. **总体性能**: ")
            
            # 安全地访问数据
            if ('baseline_pointpillar' in df.index and 'pillarfocus_full' in df.index and 
                'car_moderate' in df.columns):
                baseline_car = df.loc['baseline_pointpillar', 'car_moderate']
                full_car = df.loc['pillarfocus_full', 'car_moderate']
                if pd.notna(baseline_car) and pd.notna(full_car):
                    improvement = full_car - baseline_car
                    if improvement > 0:
                        f.write(f"PillarFocus模型在车辆检测上取得了{improvement:.2f}% AP的提升\n")
                    else:
                        f.write(f"PillarFocus模型性能需要进一步优化\n")
                else:
                    f.write("数据不完整，无法计算改进幅度\n")
            else:
                f.write("缺少必要数据进行对比分析\n")
            
            f.write("2. **组件贡献**: 通过消融实验分析各组件的有效性\n")
            f.write("3. **适用场景**: 分析模型在不同难度场景下的表现\n\n")
            
            f.write("## 可视化图表\n\n")
            f.write("- ![性能对比](performance_comparison.png)\n")
            if os.path.exists(f"{results_dir}/improvement_chart.png"):
                f.write("- ![改进效果](improvement_chart.png)\n")
        
        print(f"✅ Final analysis report saved to: {report_file}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")

if __name__ == "__main__":
    safe_generate_final_report()