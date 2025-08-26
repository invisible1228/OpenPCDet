#!/usr/bin/env python3
import os
import subprocess
import argparse
import json
import pandas as pd
from datetime import datetime

class ModelTester:
    def __init__(self):
        self.results = {}
        
    def test_single_model(self, cfg_file, ckpt_path, model_name, test_set='val'):
        """测试单个模型"""
        
        print(f"🚀 开始测试模型: {model_name}")
        print(f"   配置文件: {cfg_file}")
        print(f"   模型权重: {ckpt_path}")
        
        # 检查文件是否存在
        if not os.path.exists(ckpt_path):
            print(f"❌ 模型文件不存在: {ckpt_path}")
            return None
            
        # 构建测试命令
        cmd = [
            'python', 'tools/test.py',
            '--cfg_file', cfg_file,
            '--ckpt', ckpt_path,
            '--batch_size', '1',
            '--workers', '4',
            '--extra_tag', f'{model_name}_test',
            '--eval_all',
            '--save_to_file'
        ]
        
        print(f"🔧 执行命令: {' '.join(cmd)}")
        
        try:
            # 运行测试
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ {model_name} 测试成功")
                
                # 解析结果
                output = result.stdout
                metrics = self.parse_test_output(output)
                
                # 保存详细输出
                with open(f'test_output_{model_name}.txt', 'w') as f:
                    f.write(output)
                
                return metrics
            else:
                print(f"❌ {model_name} 测试失败")
                print(f"错误信息: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_name} 测试超时")
            return None
        except Exception as e:
            print(f"❌ {model_name} 测试异常: {e}")
            return None
    
    def parse_test_output(self, output):
        """解析测试输出，提取关键指标"""
        import re
        
        metrics = {}
        
        # KITTI评估结果模式
        patterns = {
            'car_easy_3d': r'Car AP@0\.70, 0\.70, 0\.70:\s*bbox AP:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*bev AP:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*3d AP:([\d\.]+), ([\d\.]+), ([\d\.]+)',
            'pedestrian_3d': r'Pedestrian AP@0\.50, 0\.50, 0\.50:\s*bbox AP:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*bev AP:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*3d AP:([\d\.]+), ([\d\.]+), ([\d\.]+)',
            'cyclist_3d': r'Cyclist AP@0\.50, 0\.50, 0\.50:\s*bbox AP:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*bev AP:([\d\.]+), ([\d\.]+), ([\d\.]+)\s*3d AP:([\d\.]+), ([\d\.]+), ([\d\.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                class_name = key.split('_')[0]
                # 提取3D AP的三个难度值
                metrics[f'{class_name}_3d_easy'] = float(match.group(7))
                metrics[f'{class_name}_3d_moderate'] = float(match.group(8))
                metrics[f'{class_name}_3d_hard'] = float(match.group(9))
                
                # 提取BEV AP
                metrics[f'{class_name}_bev_easy'] = float(match.group(4))
                metrics[f'{class_name}_bev_moderate'] = float(match.group(5))
                metrics[f'{class_name}_bev_hard'] = float(match.group(6))
                
                # 提取2D AP
                metrics[f'{class_name}_2d_easy'] = float(match.group(1))
                metrics[f'{class_name}_2d_moderate'] = float(match.group(2))
                metrics[f'{class_name}_2d_hard'] = float(match.group(3))
        
        # 查找总体指标
        overall_pattern = r'Average.*AP.*:([\d\.]+)'
        overall_match = re.search(overall_pattern, output)
        if overall_match:
            metrics['overall_ap'] = float(overall_match.group(1))
        
        return metrics
    
    def test_multiple_models(self, model_configs):
        """测试多个模型"""
        
        print(f"📊 开始批量测试 {len(model_configs)} 个模型")
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*50}")
            print(f"测试模型: {model_name}")
            print(f"{'='*50}")
            
            metrics = self.test_single_model(
                config['cfg_file'],
                config['ckpt_path'],
                model_name
            )
            
            if metrics:
                self.results[model_name] = {
                    'metrics': metrics,
                    'config': config,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 打印关键指标
                print(f"📈 {model_name} 关键指标:")
                if 'car_3d_moderate' in metrics:
                    print(f"   Car 3D (Moderate): {metrics['car_3d_moderate']:.4f}")
                if 'pedestrian_3d_moderate' in metrics:
                    print(f"   Pedestrian 3D (Moderate): {metrics['pedestrian_3d_moderate']:.4f}")
                if 'cyclist_3d_moderate' in metrics:
                    print(f"   Cyclist 3D (Moderate): {metrics['cyclist_3d_moderate']:.4f}")
        
        # 生成对比报告
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """生成对比报告"""
        if not self.results:
            print("❌ 没有测试结果可用于生成报告")
            return
        
        print(f"\n📊 生成对比报告...")
        
        # 创建DataFrame
        df_data = []
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 保存为CSV
        df.to_csv('model_comparison_results.csv', index=False)
        print(f"✅ 详细结果已保存: model_comparison_results.csv")
        
        # 生成Markdown报告
        self.generate_markdown_report(df)
        
        # 生成可视化
        self.generate_plots(df)
    
    def generate_markdown_report(self, df):
        """生成Markdown格式的报告"""
        
        with open('model_test_report.md', 'w') as f:
            f.write("# 模型测试对比报告\n\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 测试配置\n\n")
            f.write("| 模型 | 配置文件 | 权重文件 |\n")
            f.write("|------|----------|----------|\n")
            
            for model_name, result in self.results.items():
                cfg_file = os.path.basename(result['config']['cfg_file'])
                ckpt_file = os.path.basename(result['config']['ckpt_path'])
                f.write(f"| {model_name} | {cfg_file} | {ckpt_file} |\n")
            
            f.write("\n## 3D检测结果 (Moderate难度)\n\n")
            f.write("| 模型 | Car | Pedestrian | Cyclist |\n")
            f.write("|------|-----|------------|--------|\n")
            
            for _, row in df.iterrows():
                model = row['Model']
                car_3d = row.get('car_3d_moderate', 0)
                ped_3d = row.get('pedestrian_3d_moderate', 0)
                cyc_3d = row.get('cyclist_3d_moderate', 0)
                f.write(f"| {model} | {car_3d:.4f} | {ped_3d:.4f} | {cyc_3d:.4f} |\n")
            
            f.write("\n## BEV检测结果 (Moderate难度)\n\n")
            f.write("| 模型 | Car | Pedestrian | Cyclist |\n")
            f.write("|------|-----|------------|--------|\n")
            
            for _, row in df.iterrows():
                model = row['Model']
                car_bev = row.get('car_bev_moderate', 0)
                ped_bev = row.get('pedestrian_bev_moderate', 0)
                cyc_bev = row.get('cyclist_bev_moderate', 0)
                f.write(f"| {model} | {car_bev:.4f} | {ped_bev:.4f} | {cyc_bev:.4f} |\n")
            
            # 添加改进分析
            if len(df) > 1:
                f.write("\n## 改进分析\n\n")
                baseline_idx = 0  # 假设第一个是基准
                baseline_name = df.iloc[baseline_idx]['Model']
                
                for i in range(1, len(df)):
                    model_name = df.iloc[i]['Model']
                    f.write(f"### {model_name} vs {baseline_name}\n\n")
                    
                    for metric in ['car_3d_moderate', 'pedestrian_3d_moderate', 'cyclist_3d_moderate']:
                        if metric in df.columns:
                            baseline_val = df.iloc[baseline_idx][metric]
                            current_val = df.iloc[i][metric]
                            improvement = current_val - baseline_val
                            percentage = (improvement / baseline_val * 100) if baseline_val > 0 else 0
                            
                            class_name = metric.split('_')[0].title()
                            f.write(f"- {class_name} 3D: {improvement:+.4f} ({percentage:+.2f}%)\n")
                    f.write("\n")
        
        print(f"✅ Markdown报告已保存: model_test_report.md")
    
    def generate_plots(self, df):
        """生成可视化图表"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 3D检测结果对比
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能对比', fontsize=16)
        
        # 3D AP对比
        metrics_3d = ['car_3d_moderate', 'pedestrian_3d_moderate', 'cyclist_3d_moderate']
        labels = ['Car', 'Pedestrian', 'Cyclist']
        
        if all(metric in df.columns for metric in metrics_3d):
            ax1 = axes[0, 0]
            x = np.arange(len(labels))
            width = 0.8 / len(df)
            
            for i, (_, row) in enumerate(df.iterrows()):
                values = [row[metric] for metric in metrics_3d]
                ax1.bar(x + i * width, values, width, label=row['Model'])
            
            ax1.set_xlabel('类别')
            ax1.set_ylabel('3D AP')
            ax1.set_title('3D检测AP对比 (Moderate)')
            ax1.set_xticks(x + width * (len(df) - 1) / 2)
            ax1.set_xticklabels(labels)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # BEV AP对比
        metrics_bev = ['car_bev_moderate', 'pedestrian_bev_moderate', 'cyclist_bev_moderate']
        
        if all(metric in df.columns for metric in metrics_bev):
            ax2 = axes[0, 1]
            
            for i, (_, row) in enumerate(df.iterrows()):
                values = [row[metric] for metric in metrics_bev]
                ax2.bar(x + i * width, values, width, label=row['Model'])
            
            ax2.set_xlabel('类别')
            ax2.set_ylabel('BEV AP')
            ax2.set_title('BEV检测AP对比 (Moderate)')
            ax2.set_xticks(x + width * (len(df) - 1) / 2)
            ax2.set_xticklabels(labels)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 难度对比 (以Car为例)
        if all(f'car_3d_{diff}' in df.columns for diff in ['easy', 'moderate', 'hard']):
            ax3 = axes[1, 0]
            difficulties = ['Easy', 'Moderate', 'Hard']
            x_diff = np.arange(len(difficulties))
            
            for i, (_, row) in enumerate(df.iterrows()):
                values = [row[f'car_3d_{diff}'] for diff in ['easy', 'moderate', 'hard']]
                ax3.bar(x_diff + i * width, values, width, label=row['Model'])
            
            ax3.set_xlabel('难度')
            ax3.set_ylabel('3D AP')
            ax3.set_title('Car 3D检测不同难度对比')
            ax3.set_xticks(x_diff + width * (len(df) - 1) / 2)
            ax3.set_xticklabels(difficulties)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 综合性能雷达图
        if len(df) <= 3:  # 只在模型数量不多时绘制雷达图
            ax4 = axes[1, 1]
            self.plot_radar_chart(df, ax4)
        
        plt.tight_layout()
        plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化图表已保存: model_comparison_plots.png")
    
    def plot_radar_chart(self, df, ax):
        """绘制雷达图"""
        import numpy as np
        
        metrics = ['car_3d_moderate', 'pedestrian_3d_moderate', 'cyclist_3d_moderate',
                   'car_bev_moderate', 'pedestrian_bev_moderate', 'cyclist_bev_moderate']
        
        available_metrics = [m for m in metrics if m in df.columns]
        
        if len(available_metrics) < 3:
            ax.text(0.5, 0.5, 'Not enough metrics\nfor radar chart', 
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 绘制每个模型
        for _, row in df.iterrows():
            values = [row[metric] for metric in available_metrics]
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
            ax.fill(angles, values, alpha=0.25)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n') for m in available_metrics])
        ax.legend()
        ax.set_title('综合性能雷达图')

def main():
    parser = argparse.ArgumentParser(description='批量测试模型精度')
    parser.add_argument('--models_config', type=str, default=None, help='模型配置JSON文件')
    args = parser.parse_args()
    
    tester = ModelTester()
    
    # 默认测试配置
    default_models = {
        'baseline_pointpillar': {
            'cfg_file': 'tools/cfgs/kitti_models/pointpillar.yaml',
            'ckpt_path': 'output/kitti_models/pointpillar/baseline_pointpillar/ckpt/checkpoint_epoch_80.pth'
        },
        'pillarfocus_full': {
            'cfg_file': 'tools/cfgs/custom_models/pillarfocus_pop_rcnn.yaml',
            'ckpt_path': 'output/custom_models/pillarfocus_pop_rcnn/pillarfocus_full/ckpt/checkpoint_epoch_80.pth'
        }
    }
    
    if args.models_config and os.path.exists(args.models_config):
        with open(args.models_config, 'r') as f:
            models_config = json.load(f)
    else:
        models_config = default_models
    
    # 执行批量测试
    tester.test_multiple_models(models_config)

if __name__ == "__main__":
    main()