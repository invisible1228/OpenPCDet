#!/usr/bin/env python3
import os
import subprocess
import argparse
import sys

def extract_eval_summary(output_text):
    """提取评估结果摘要"""
    lines = output_text.split('\n')
    summary_lines = []
    
    for line in lines:
        # 寻找包含mAP或评估结果的行
        if any(keyword in line.lower() for keyword in ['map', 'precision', 'recall', 'easy', 'moderate', 'hard']):
            summary_lines.append(line)
    
    return '\n'.join(summary_lines) if summary_lines else output_text

def run_evaluation(cfg_file, ckpt_path, extra_tag, output_dir):
    """运行单个模型的评估"""
    # 检查文件是否存在
    if not os.path.exists(cfg_file):
        print(f"❌ Config file not found: {cfg_file}")
        return None
        
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return None
    
    # 确保输出目录存在并且使用绝对路径
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试是否可以写入文件
    test_file = os.path.join(output_dir, f"test_{extra_tag}.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Output directory accessible: {output_dir}")
    except Exception as e:
        print(f"❌ Cannot write to output directory {output_dir}: {e}")
        return None
    
    # 切换到tools目录
    tools_dir = "/det/OpenPCDet/tools"
    original_dir = os.getcwd()
    os.chdir(tools_dir)
    
    cmd = [
        'python', 'test.py',
        '--cfg_file', cfg_file,
        '--ckpt', ckpt_path,
        '--extra_tag', extra_tag,
        '--save_to_file'
    ]
    
    print(f"🚀 Running evaluation for {extra_tag}...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # 使用更简单的方式执行命令
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=3600  # 1小时超时
        )
        
        # 恢复原始目录
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"✅ Evaluation completed for {extra_tag}")
            
            # 保存完整输出
            full_output_file = os.path.join(output_dir, f"eval_full_{extra_tag}.txt")
            try:
                with open(full_output_file, 'w') as f:
                    f.write(result.stdout)
                    f.write("\n--- STDERR ---\n")
                    f.write(result.stderr)
                print(f"✅ Full output saved to: {full_output_file}")
            except Exception as e:
                print(f"⚠️ Failed to save full output: {e}")
            
            # 提取并保存摘要
            summary = extract_eval_summary(result.stdout)
            summary_file = os.path.join(output_dir, f"eval_result_{extra_tag}.txt")
            try:
                with open(summary_file, 'w') as f:
                    f.write(summary)
                print(f"✅ Summary saved to: {summary_file}")
            except Exception as e:
                print(f"⚠️ Failed to save summary: {e}")
            
            return summary
        else:
            print(f"❌ Evaluation failed for {extra_tag}")
            print(f"Return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
            # 保存错误信息
            error_file = os.path.join(output_dir, f"eval_error_{extra_tag}.txt")
            try:
                with open(error_file, 'w') as f:
                    f.write(f"Return code: {result.returncode}\n")
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
                print(f"✅ Error info saved to: {error_file}")
            except Exception as e:
                print(f"⚠️ Failed to save error info: {e}")
            
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Evaluation timed out for {extra_tag}")
        os.chdir(original_dir)
        return None
    except Exception as e:
        print(f"❌ Exception occurred for {extra_tag}: {str(e)}")
        os.chdir(original_dir)
        return None

def test_single_model():
    """测试单个模型（用于调试）"""
    cfg_file = '/det/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml'
    ckpt_path = '/det/OpenPCDet/output/kitti_models/pointpillar/baseline_pointpillar/ckpt/checkpoint_epoch_80.pth'
    extra_tag = 'baseline_pointpillar'
    
    # 使用绝对路径创建输出目录
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, 'evaluation_results')
    print(f"📁 Using output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    result = run_evaluation(cfg_file, ckpt_path, extra_tag, output_dir)
    
    if result:
        print("✅ Test successful!")
        print("Result preview:")
        print(result[:500])
    else:
        print("❌ Test failed!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='只测试单个模型')
    args = parser.parse_args()
    
    if args.test:
        test_single_model()
        return
    
    # 创建结果目录 - 使用绝对路径
    current_dir = os.getcwd()
    results_dir = os.path.join(current_dir, "evaluation_results")
    print(f"📁 Creating results directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 测试目录权限
    try:
        test_file = os.path.join(results_dir, "test_permission.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Results directory is writable")
    except Exception as e:
        print(f"❌ Cannot write to results directory: {e}")
        return
    
    # 定义要评估的模型
    models_to_eval = [
        {
            'name': 'baseline_pointpillar',
            'cfg': '/det/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
            'ckpt': '/det/OpenPCDet/output/kitti_models/pointpillar/baseline_pointpillar/ckpt/checkpoint_epoch_80.pth'
        },
        {
            'name': 'pillarfocus_full',
            'cfg': '/det/OpenPCDet/tools/cfgs/kitti_models/pillarfocus_pop_rcnn.yaml',
            'ckpt': '/det/OpenPCDet/output/kitti_models/pillarfocus_pop_rcnn/pillarfocus_full/ckpt/checkpoint_epoch_80.pth'
        },
        {
            'name': 'ablation_no_dynamic',
            'cfg': '/det/OpenPCDet/tools/cfgs/kitti_models/pillarfocus_ablation_no_dynamic.yaml',
            'ckpt': '/det/OpenPCDet/output/kitti_models/pillarfocus_ablation_no_dynamic/ablation_no_dynamic/ckpt/checkpoint_epoch_80.pth'
        },
        {
            'name': 'ablation_no_attention',
            'cfg': '/det/OpenPCDet/tools/cfgs/kitti_models/pillarfocus_ablation_no_attention.yaml',
            'ckpt': '/det/OpenPCDet/output/kitti_models/pillarfocus_ablation_no_attention/ablation_no_attention/ckpt/checkpoint_epoch_80.pth'
        }
    ]
    
    results = {}
    
    for model in models_to_eval:
        print(f"\n{'='*50}")
        print(f"Processing: {model['name']}")
        print(f"{'='*50}")
        
        result = run_evaluation(
            model['cfg'], 
            model['ckpt'], 
            model['name'], 
            results_dir
        )
        results[model['name']] = result
    
    # 生成对比报告
    generate_comparison_report(results, results_dir)

def generate_comparison_report(results, output_dir):
    """生成对比报告"""
    report_file = f"{output_dir}/comparison_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# PillarFocus模型评估对比报告\n\n")
        f.write("## 模型配置\n\n")
        f.write("| 模型 | 描述 |\n")
        f.write("|------|------|\n")
        f.write("| baseline_pointpillar | 标准PointPillar基准模型 |\n")
        f.write("| pillarfocus_full | 完整PillarFocus模型 |\n")
        f.write("| ablation_no_dynamic | 无Dynamic Pillar机制 |\n")
        f.write("| ablation_no_attention | 无空间注意力机制 |\n\n")
        
        f.write("## 评估结果\n\n")
        
        for model_name, result in results.items():
            f.write(f"### {model_name}\n\n")
            if result:
                f.write("```\n")
                f.write(result)
                f.write("\n```\n\n")
            else:
                f.write("❌ 评估失败\n\n")
    
    print(f"📊 Comparison report saved to: {report_file}")

if __name__ == "__main__":
    print("🚀 Starting evaluation script...")
    main()
    print("✅ Script completed!")