#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_comprehensive_metrics(text, model_name):
    """提取所有KITTI评估指标，特别关注PillarFocus改进点"""
    metrics = {}
    
    print(f"\n🔍 解析 {model_name} 的评估结果...")
    print(f"文件长度: {len(text)} 字符")
    
    if len(text.strip()) == 0:
        print("⚠️ 文件为空!")
        return metrics
    
    lines = text.split('\n')
    
    # 显示包含重要信息的行
    important_lines = []
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in [
            'car', 'pedestrian', 'cyclist', 'ap@', 'bbox', '3d', 'bev', 'aos',
            'easy', 'moderate', 'hard', 'recall', 'precision'
        ]):
            important_lines.append((i, line.strip()))
    
    print(f"📋 找到 {len(important_lines)} 行重要信息:")
    for line_num, line in important_lines[:15]:  # 显示前15行
        print(f"  {line_num:3d}: {line}")
    if len(important_lines) > 15:
        print(f"  ... 还有 {len(important_lines) - 15} 行")
    
    # 针对PillarFocus的特定指标提取
    current_category = None
    current_eval_type = None
    
    # 多种解析模式
    patterns = {
        # 标准KITTI格式
        'kitti_standard': {
            'car_3d': r'Car AP_R40@0\.70, 0\.70, 0\.70:\s*bbox AP:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*bev  AP:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*3d   AP:([\d\.]+), ([\d\.]+), ([\d\.]+)',
            'car_aos': r'Car AOS@0\.70, 0\.70, 0\.70:\s*bbox AOS:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*bev  AOS:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*3d   AOS:([\d\.]+), ([\d\.]+), ([\d\.]+)',
        },
        
        # 简化格式
        'simple_format': {
            'car_results': r'Car.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)',
            'pedestrian_results': r'Pedestrian.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)',
            'cyclist_results': r'Cyclist.*?(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)',
        },
        
        # 按行解析
        'line_by_line': {}
    }
    
    # 尝试标准KITTI格式
    print(f"\n🔍 尝试标准KITTI格式解析...")
    for key, pattern in patterns['kitti_standard'].items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            groups = match.groups()
            print(f"  ✅ 找到 {key}: {groups}")
            
            if key == 'car_3d':
                # bbox AP: easy, moderate, hard
                metrics['Car_bbox_easy'] = float(groups[0])
                metrics['Car_bbox_moderate'] = float(groups[1])
                metrics['Car_bbox_hard'] = float(groups[2])
                
                # bev AP: easy, moderate, hard
                metrics['Car_bev_easy'] = float(groups[3])
                metrics['Car_bev_moderate'] = float(groups[4])
                metrics['Car_bev_hard'] = float(groups[5])
                
                # 3d AP: easy, moderate, hard
                metrics['Car_3d_easy'] = float(groups[6])
                metrics['Car_3d_moderate'] = float(groups[7])
                metrics['Car_3d_hard'] = float(groups[8])
                
            elif key == 'car_aos':
                # AOS指标
                metrics['Car_aos_bbox_easy'] = float(groups[0])
                metrics['Car_aos_bbox_moderate'] = float(groups[1])
                metrics['Car_aos_bbox_hard'] = float(groups[2])
                
                metrics['Car_aos_bev_easy'] = float(groups[3])
                metrics['Car_aos_bev_moderate'] = float(groups[4])
                metrics['Car_aos_bev_hard'] = float(groups[5])
                
                metrics['Car_aos_3d_easy'] = float(groups[6])
                metrics['Car_aos_3d_moderate'] = float(groups[7])
                metrics['Car_aos_3d_hard'] = float(groups[8])
    
    # 如果标准格式失败，尝试简化格式
    if not metrics:
        print(f"\n🔍 尝试简化格式解析...")
        for key, pattern in patterns['simple_format'].items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                print(f"  ✅ 找到 {key}: {matches}")
                category = key.split('_')[0].capitalize()
                for i, match in enumerate(matches[:3]):  # 取前3个匹配
                    if len(match) >= 3:
                        metrics[f'{category}_unknown_{i}_easy'] = float(match[0])
                        metrics[f'{category}_unknown_{i}_moderate'] = float(match[1])
                        metrics[f'{category}_unknown_{i}_hard'] = float(match[2])
    
    # 按行解析（最后的尝试）
    if not metrics:
        print(f"\n🔍 尝试按行解析...")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # 查找包含类别名称和数字的行
            for category in ['Car', 'Pedestrian', 'Cyclist']:
                if category.lower() in line.lower():
                    numbers = re.findall(r'\d+\.\d+', line)
                    if len(numbers) >= 3:
                        # 尝试识别评估类型
                        eval_type = 'unknown'
                        if 'bbox' in line.lower():
                            eval_type = 'bbox'
                        elif '3d' in line.lower():
                            eval_type = '3d'
                        elif 'bev' in line.lower():
                            eval_type = 'bev'
                        elif 'aos' in line.lower():
                            eval_type = 'aos'
                        
                        base_key = f"{category}_{eval_type}_line{i}"
                        metrics[f"{base_key}_easy"] = float(numbers[0])
                        metrics[f"{base_key}_moderate"] = float(numbers[1])
                        metrics[f"{base_key}_hard"] = float(numbers[2])
                        
                        print(f"  📌 行 {i}: {line}")
                        print(f"     -> {base_key}: {numbers[0]}, {numbers[1]}, {numbers[2]}")
    
    print(f"🎯 {model_name} 提取了 {len(metrics)} 个指标")
    return metrics

def load_all_evaluation_results():
    """加载所有评估结果"""
    results_dir = "evaluation_results"
    
    model_files = {
        'baseline_pointpillar': 'eval_full_baseline_pointpillar.txt',
        'pillarfocus_full': 'eval_full_pillarfocus_full.txt',
        'ablation_no_dynamic': 'eval_full_ablation_no_dynamic.txt',
        'ablation_no_attention': 'eval_full_ablation_no_attention.txt'
    }
    
    all_results = {}
    
    print(f"📁 从 {results_dir} 加载评估结果...")
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(results_dir, filename)
        
        print(f"\n{'='*60}")
        print(f"📄 处理: {model_name}")
        print(f"📂 文件: {filename}")
        print(f"{'='*60}")
        
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filepath}")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_size = len(content)
            print(f"📊 文件大小: {file_size:,} 字符")
            
            if file_size == 0:
                print(f"⚠️ 文件为空: {filename}")
                continue
            
            metrics = extract_comprehensive_metrics(content, model_name)
            all_results[model_name] = metrics
            
        except Exception as e:
            print(f"❌ 处理文件时出错 {filename}: {e}")
            continue
    
    return all_results

def analyze_pillarfocus_improvements(all_results):
    """分析PillarFocus的具体改进效果"""
    
    if 'baseline_pointpillar' not in all_results:
        print("❌ 未找到基准模型结果!")
        return {}
    
    baseline = all_results['baseline_pointpillar']
    if not baseline:
        print("❌ 基准模型没有有效指标!")
        return {}
    
    print(f"\n📊 基准模型指标 ({len(baseline)} 个):")
    for key, value in sorted(baseline.items()):
        print(f"  {key}: {value:.4f}")
    
    improvements = {}
    
    # 分析每个模型相对于基准的改进
    for model_name, model_results in all_results.items():
        if model_name == 'baseline_pointpillar' or not model_results:
            continue
        
        print(f"\n{'🔍 分析 ' + model_name + ' 的改进'}")
        print("-" * 50)
        
        model_improvements = {}
        
        # 找到公共指标
        common_metrics = set(baseline.keys()) & set(model_results.keys())
        print(f"📈 公共指标数量: {len(common_metrics)}")
        
        if not common_metrics:
            print(f"❌ 与基准模型没有公共指标!")
            print(f"基准模型指标: {list(baseline.keys())[:5]}...")
            print(f"当前模型指标: {list(model_results.keys())[:5]}...")
            continue
        
        # 计算改进
        all_improvements = []
        for metric in common_metrics:
            baseline_val = baseline[metric]
            model_val = model_results[metric]
            
            if baseline_val > 0:  # 避免除零
                abs_improvement = model_val - baseline_val
                rel_improvement = (abs_improvement / baseline_val) * 100
                
                all_improvements.append({
                    'metric': metric,
                    'baseline': baseline_val,
                    'model': model_val,
                    'abs_improvement': abs_improvement,
                    'rel_improvement': rel_improvement
                })
        
        # 按绝对改进排序
        all_improvements.sort(key=lambda x: x['abs_improvement'], reverse=True)
        
        # 统计改进情况
        positive_improvements = [imp for imp in all_improvements if imp['abs_improvement'] > 0.001]
        significant_improvements = [imp for imp in all_improvements if imp['abs_improvement'] > 0.1]
        
        print(f"✅ 正向改进: {len(positive_improvements)}/{len(all_improvements)} 个指标")
        print(f"🚀 显著改进(>0.1): {len(significant_improvements)} 个指标")
        
        if positive_improvements:
            print(f"\n🏆 正向改进的指标:")
            for i, imp in enumerate(positive_improvements[:10], 1):  # 显示前10个
                print(f"  {i:2d}. {imp['metric']}")
                print(f"      {imp['baseline']:.4f} → {imp['model']:.4f} (+{imp['abs_improvement']:.4f}, +{imp['rel_improvement']:.2f}%)")
            
            # 特别关注与PillarFocus创新相关的指标
            focus_keywords = ['car', '3d', 'hard', 'moderate', 'bev']
            focus_improvements = []
            
            for imp in positive_improvements:
                if any(keyword in imp['metric'].lower() for keyword in focus_keywords):
                    focus_improvements.append(imp)
            
            if focus_improvements:
                print(f"\n🎯 PillarFocus重点关注的改进指标:")
                for i, imp in enumerate(focus_improvements[:5], 1):
                    print(f"  {i}. {imp['metric']}: +{imp['abs_improvement']:.4f} (+{imp['rel_improvement']:.2f}%)")
        
        else:
            print(f"❌ 未找到正向改进的指标")
            print(f"📊 样本对比 (前5个指标):")
            for imp in all_improvements[:5]:
                print(f"  {imp['metric']}: {imp['baseline']:.4f} → {imp['model']:.4f} ({imp['abs_improvement']:+.4f})")
        
        improvements[model_name] = all_improvements
    
    return improvements

def create_improvement_summary(improvements, all_results):
    """创建改进总结报告"""
    
    if not improvements:
        print("\n❌ 没有改进数据可以总结!")
        return
    
    print(f"\n📋 PILLARFOCUS改进总结报告")
    print("="*60)
    
    # 创建改进数据表
    improvement_data = []
    
    for model_name, model_improvements in improvements.items():
        print(f"\n🚀 {model_name.upper()}")
        print("-" * 40)
        
        positive_imps = [imp for imp in model_improvements if imp['abs_improvement'] > 0.001]
        
        if positive_imps:
            print(f"✅ 改进指标数量: {len(positive_imps)}")
            print(f"🏆 最佳改进: +{max(positive_imps, key=lambda x: x['abs_improvement'])['abs_improvement']:.4f}")
            print(f"📊 平均改进: +{np.mean([imp['abs_improvement'] for imp in positive_imps]):.4f}")
            
            # 添加到数据表
            for imp in positive_imps:
                improvement_data.append({
                    'Model': model_name,
                    'Metric': imp['metric'],
                    'Baseline': imp['baseline'],
                    'New_Value': imp['model'],
                    'Absolute_Improvement': imp['abs_improvement'],
                    'Relative_Improvement_Percent': imp['rel_improvement']
                })
        else:
            print("❌ 无改进指标")
    
    # 保存结果
    if improvement_data:
        results_dir = "evaluation_results"
        
        # 保存CSV
        df = pd.DataFrame(improvement_data)
        csv_file = os.path.join(results_dir, "pillarfocus_improvements.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n✅ 改进数据已保存到: {csv_file}")
        
        # 显示最佳改进
        print(f"\n🏆 TOP 10 最佳改进 (所有模型)")
        print("-" * 50)
        top_improvements = df.nlargest(10, 'Absolute_Improvement')
        for idx, (_, row) in enumerate(top_improvements.iterrows(), 1):
            print(f"{idx:2d}. {row['Model']} - {row['Metric']}")
            print(f"    {row['Baseline']:.4f} → {row['New_Value']:.4f} (+{row['Absolute_Improvement']:.4f})")
        
        # 针对不同改进类型的分析
        print(f"\n🎯 按PillarFocus创新点分类的改进:")
        print("-" * 40)
        
        categories = {
            '远距离/小目标检测(Hard难度)': df[df['Metric'].str.contains('hard', case=False)],
            '3D检测精度': df[df['Metric'].str.contains('3d', case=False)],
            'BEV检测精度': df[df['Metric'].str.contains('bev', case=False)],
            '车辆检测': df[df['Metric'].str.contains('car', case=False)],
            '整体性能(Moderate)': df[df['Metric'].str.contains('moderate', case=False)]
        }
        
        for cat_name, cat_data in categories.items():
            if not cat_data.empty:
                best_improvement = cat_data.loc[cat_data['Absolute_Improvement'].idxmax()]
                print(f"📈 {cat_name}:")
                print(f"   最佳: {best_improvement['Model']} - {best_improvement['Metric']}")
                print(f"   改进: +{best_improvement['Absolute_Improvement']:.4f} (+{best_improvement['Relative_Improvement_Percent']:.2f}%)")
    
    return improvement_data

def main():
    """主分析函数"""
    print("🔍 PillarFocus创新点针对性分析")
    print("="*80)
    print("🎯 重点关注:")
    print("  1. 动态Pillar → 稀疏点云场景改进")
    print("  2. 混合池化MPDC → 特征提取能力增强") 
    print("  3. PPFE多尺度融合 → 远距离小目标检测")
    print("  4. SCS-EM注意力 → 重要特征增强")
    print("="*80)
    
    # 加载所有结果
    all_results = load_all_evaluation_results()
    
    if not all_results:
        print("❌ 未找到任何评估结果!")
        return
    
    print(f"\n✅ 成功加载 {len(all_results)} 个模型的结果:")
    for model_name, metrics in all_results.items():
        print(f"  📊 {model_name}: {len(metrics)} 个指标")
    
    # 分析改进
    improvements = analyze_pillarfocus_improvements(all_results)
    
    # 生成总结报告
    improvement_data = create_improvement_summary(improvements, all_results)
    
    if improvement_data:
        print(f"\n🎉 分析完成! 找到了 {len(improvement_data)} 个改进指标")
        print(f"📁 详细结果请查看 evaluation_results/pillarfocus_improvements.csv")
    else:
        print(f"\n😞 抱歉，未找到明显的性能改进")
        print(f"💡 建议:")
        print(f"  1. 检查模型是否正确实现了创新点")
        print(f"  2. 验证训练是否充分收敛")
        print(f"  3. 考虑调整超参数")
        print(f"  4. 检查数据集是否适合验证这些创新点")

if __name__ == "__main__":
    main()