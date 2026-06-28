#!/usr/bin/env python3
"""
分析num_steps超参数实验结果

生成：
1. IC-AUC vs num_steps曲线图
2. Accuracy vs num_steps曲线图
3. 统计表格
4. 最佳num_steps推荐
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir):
    """加载实验结果"""
    results_file = Path(results_dir) / 'all_results.json'

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r') as f:
        all_results = json.load(f)

    return all_results


def create_comparison_plots(results_dir):
    """创建对比图表"""

    # 加载数据
    summary_csv = Path(results_dir) / 'summary.csv'
    if not summary_csv.exists():
        print(f"Warning: {summary_csv} not found, skipping plots")
        return

    df = pd.read_csv(summary_csv)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('num_steps超参数实验结果对比 (Cora数据集)', fontsize=16, fontweight='bold')

    # 获取所有模型和num_steps值
    models = df['Model'].unique()
    num_steps_values = sorted(df['num_steps'].unique())

    # 定义颜色和标记
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    # 1. IC-AUC (Additive) vs num_steps
    ax1 = axes[0, 0]
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model].sort_values('num_steps')
        ax1.plot(model_data['num_steps'], model_data['IC_AUC_Add'],
                marker=markers[i % len(markers)], label=model,
                color=colors[i], linewidth=2, markersize=8)

    ax1.set_xlabel('num_steps', fontsize=12)
    ax1.set_ylabel('IC-AUC (Additive)', fontsize=12)
    ax1.set_title('IC-AUC (Additive) vs num_steps', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xticks(num_steps_values)
    ax1.set_xticklabels(num_steps_values)

    # 2. IC-AUC (Log) vs num_steps
    ax2 = axes[0, 1]
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model].sort_values('num_steps')
        ax2.plot(model_data['num_steps'], model_data['IC_AUC_Log'],
                marker=markers[i % len(markers)], label=model,
                color=colors[i], linewidth=2, markersize=8)

    ax2.set_xlabel('num_steps', fontsize=12)
    ax2.set_ylabel('IC-AUC (Log-ratio)', fontsize=12)
    ax2.set_title('IC-AUC (Log-ratio) vs num_steps', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xticks(num_steps_values)
    ax2.set_xticklabels(num_steps_values)

    # 3. Max Accuracy vs num_steps
    ax3 = axes[1, 0]
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model].sort_values('num_steps')
        ax3.plot(model_data['num_steps'], model_data['Max_Accuracy'],
                marker=markers[i % len(markers)], label=model,
                color=colors[i], linewidth=2, markersize=8)

    ax3.set_xlabel('num_steps', fontsize=12)
    ax3.set_ylabel('Max Accuracy', fontsize=12)
    ax3.set_title('Max Accuracy vs num_steps', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_xticks(num_steps_values)
    ax3.set_xticklabels(num_steps_values)

    # 4. Final Accuracy vs num_steps
    ax4 = axes[1, 1]
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model].sort_values('num_steps')
        ax4.plot(model_data['num_steps'], model_data['Final_Accuracy'],
                marker=markers[i % len(markers)], label=model,
                color=colors[i], linewidth=2, markersize=8)

    ax4.set_xlabel('num_steps', fontsize=12)
    ax4.set_ylabel('Final Accuracy (at max complexity)', fontsize=12)
    ax4.set_title('Final Accuracy vs num_steps', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_xticks(num_steps_values)
    ax4.set_xticklabels(num_steps_values)

    plt.tight_layout()

    # 保存图表
    plot_file = Path(results_dir) / 'comparison_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图表已保存: {plot_file}")

    plt.close()


def analyze_best_num_steps(results_dir):
    """分析最佳num_steps值"""

    summary_csv = Path(results_dir) / 'summary.csv'
    if not summary_csv.exists():
        print(f"Warning: {summary_csv} not found")
        return

    df = pd.read_csv(summary_csv)

    print(f"\n{'='*80}")
    print("最佳num_steps分析")
    print(f"{'='*80}\n")

    # 对每个模型找到最佳num_steps
    models = df['Model'].unique()

    best_results = []

    for model in models:
        model_data = df[df['Model'] == model]

        # 根据IC-AUC (Additive)找最佳
        best_ic_auc_add = model_data.loc[model_data['IC_AUC_Add'].idxmax()]
        best_ic_auc_log = model_data.loc[model_data['IC_AUC_Log'].idxmax()]
        best_accuracy = model_data.loc[model_data['Max_Accuracy'].idxmax()]

        best_results.append({
            'Model': model,
            'Best_num_steps_IC_AUC_Add': int(best_ic_auc_add['num_steps']),
            'Best_IC_AUC_Add': best_ic_auc_add['IC_AUC_Add'],
            'Best_num_steps_IC_AUC_Log': int(best_ic_auc_log['num_steps']),
            'Best_IC_AUC_Log': best_ic_auc_log['IC_AUC_Log'],
            'Best_num_steps_Accuracy': int(best_accuracy['num_steps']),
            'Best_Accuracy': best_accuracy['Max_Accuracy']
        })

    best_df = pd.DataFrame(best_results)

    # 保存最佳结果
    best_file = Path(results_dir) / 'best_num_steps.csv'
    best_df.to_csv(best_file, index=False, float_format='%.6f')
    print(f"✅ 最佳num_steps分析已保存: {best_file}")

    # 打印结果
    print("\n各模型最佳num_steps (按IC-AUC Additive):")
    print(best_df[['Model', 'Best_num_steps_IC_AUC_Add', 'Best_IC_AUC_Add']].to_string(index=False))

    # 统计最常见的最佳num_steps
    print(f"\n{'='*80}")
    print("num_steps值出现频率 (按IC-AUC Additive):")
    print(f"{'='*80}\n")

    num_steps_counts = best_df['Best_num_steps_IC_AUC_Add'].value_counts().sort_index()
    for num_steps, count in num_steps_counts.items():
        print(f"  num_steps={num_steps}: {count}/{len(models)} 模型 ({count/len(models)*100:.1f}%)")

    # 推荐
    recommended_num_steps = num_steps_counts.idxmax()
    print(f"\n💡 推荐的num_steps值: {recommended_num_steps}")
    print(f"   (在{num_steps_counts[recommended_num_steps]}个模型中表现最佳)")


def generate_report(results_dir):
    """生成完整报告"""

    print(f"\n{'='*80}")
    print("生成实验报告")
    print(f"{'='*80}\n")

    # 1. 创建对比图表
    try:
        create_comparison_plots(results_dir)
    except Exception as e:
        print(f"❌ 图表生成失败: {e}")

    # 2. 分析最佳num_steps
    try:
        analyze_best_num_steps(results_dir)
    except Exception as e:
        print(f"❌ 最佳num_steps分析失败: {e}")

    # 3. 生成Markdown报告
    try:
        generate_markdown_report(results_dir)
    except Exception as e:
        print(f"❌ Markdown报告生成失败: {e}")


def generate_markdown_report(results_dir):
    """生成Markdown格式的报告"""

    summary_csv = Path(results_dir) / 'summary.csv'
    if not summary_csv.exists():
        return

    df = pd.read_csv(summary_csv)

    report_lines = []
    report_lines.append("# num_steps 超参数实验报告\n")
    report_lines.append(f"**实验时间**: {Path(results_dir).name.replace('hyperparameter_num_steps_', '')}\n")
    report_lines.append(f"**数据集**: Cora\n")
    report_lines.append(f"**任务**: original label\n")
    report_lines.append(f"**下游模型**: GCN\n")
    report_lines.append(f"**测试模型数**: {len(df['Model'].unique())}\n")
    report_lines.append(f"**num_steps值**: {sorted(df['num_steps'].unique())}\n")

    report_lines.append("\n## IC-AUC (Additive) 结果\n\n")
    pivot_add = df.pivot(index='Model', columns='num_steps', values='IC_AUC_Add')
    report_lines.append(pivot_add.to_markdown(floatfmt='.4f'))

    report_lines.append("\n\n## IC-AUC (Log-ratio) 结果\n\n")
    pivot_log = df.pivot(index='Model', columns='num_steps', values='IC_AUC_Log')
    report_lines.append(pivot_log.to_markdown(floatfmt='.4f'))

    report_lines.append("\n\n## Max Accuracy 结果\n\n")
    pivot_acc = df.pivot(index='Model', columns='num_steps', values='Max_Accuracy')
    report_lines.append(pivot_acc.to_markdown(floatfmt='.4f'))

    report_lines.append("\n\n## 可视化\n\n")
    report_lines.append("![Comparison Plots](comparison_plots.png)\n")

    # 保存报告
    report_file = Path(results_dir) / 'REPORT.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Markdown报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='分析num_steps超参数实验结果')
    parser.add_argument('results_dir', type=str, help='实验结果目录路径')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        return

    print(f"分析结果目录: {results_dir}")

    generate_report(results_dir)

    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
