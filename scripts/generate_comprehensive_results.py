#!/usr/bin/env python3
"""
生成完整的超参数实验综合结果

从process_results中的TSV文件读取数据，计算IC-AUC并生成可视化
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_ic_auc(complexity, information):
    """计算Information-Complexity曲线下面积

    按照正确的方式计算：从complexity=0到1进行积分
    """
    if len(complexity) == 0 or len(information) == 0:
        return None

    if len(complexity) < 2:
        return 0.0

    # 按complexity从小到大排序（从0到1）
    sorted_pairs = sorted(zip(complexity, information))
    x_vals = np.array([pair[0] for pair in sorted_pairs])
    y_vals = np.array([pair[1] for pair in sorted_pairs])

    # 使用梯形法则计算AUC
    auc = 0.0
    for i in range(1, len(x_vals)):
        dx = x_vals[i] - x_vals[i-1]
        avg_y = (y_vals[i] + y_vals[i-1]) / 2
        auc += dx * avg_y

    return auc


def load_and_process_results(results_dir):
    """加载所有TSV结果文件并处理"""
    process_results_dir = Path(results_dir) / 'Cora_original_gcn' / 'process_results'

    all_data = []
    model_metrics = {}

    # 读取所有TSV文件
    for tsv_file in process_results_dir.glob('*_step_metrics.tsv'):
        df = pd.read_csv(tsv_file, sep='\t')

        if len(df) == 0:
            continue

        model_name = df['model'].iloc[0]

        # 提取指标
        complexity = df['complexity_metric'].values
        info_add = df['information_metric_additive'].values
        info_log = df['information_metric_log_ratio'].values
        accuracy = df['accuracy_metric'].values

        # 计算IC-AUC
        ic_auc_add = calculate_ic_auc(complexity, info_add)
        ic_auc_log = calculate_ic_auc(complexity, info_log)
        max_accuracy = np.max(accuracy)
        final_accuracy = accuracy[-1]

        model_metrics[model_name] = {
            'complexity': complexity,
            'info_add': info_add,
            'info_log': info_log,
            'accuracy': accuracy,
            'ic_auc_add': ic_auc_add,
            'ic_auc_log': ic_auc_log,
            'max_accuracy': max_accuracy,
            'final_accuracy': final_accuracy,
            'num_steps': len(complexity) - 1  # 减去step 0
        }

        all_data.append(df)

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    return combined_df, model_metrics


def create_comparison_plots(model_metrics, output_dir):
    """创建对比图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 模型简称映射
    model_display_names = {
        'gradient_based': 'IGPrune',
        'networkit_forest_fire': 'EFF',
        'networkit_local_degree': 'LD',
        'networkit_local_similarity': 'LS',
        'networkit_random_edge': 'RE',
        'networkit_random_node_edge': 'RN',
        'networkit_scan': 'SCAN',
        'networkit_simmelian': 'SO',
        'pri_graphs': 'PRI-graph'
    }

    # 颜色和标记
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_metrics)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    # 创建4个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. IC曲线 (Additive)
    ax1 = axes[0, 0]
    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        display_name = model_display_names.get(model_name, model_name)
        ax1.plot(metrics['complexity'], metrics['info_add'],
                marker=markers[idx % len(markers)],
                color=colors[idx],
                label=display_name,
                linewidth=2,
                markersize=6,
                alpha=0.8)
    ax1.set_xlabel('Complexity (Edge Retention Ratio)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Information (Additive)', fontsize=12, fontweight='bold')
    ax1.set_title('Information-Complexity Curve (Additive Normalization)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([-0.05, 1.05])

    # 2. IC曲线 (Log-ratio)
    ax2 = axes[0, 1]
    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        display_name = model_display_names.get(model_name, model_name)
        ax2.plot(metrics['complexity'], metrics['info_log'],
                marker=markers[idx % len(markers)],
                color=colors[idx],
                label=display_name,
                linewidth=2,
                markersize=6,
                alpha=0.8)
    ax2.set_xlabel('Complexity (Edge Retention Ratio)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Information (Log-ratio)', fontsize=12, fontweight='bold')
    ax2.set_title('Information-Complexity Curve (Log-ratio Normalization)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])

    # 3. Accuracy-Complexity曲线
    ax3 = axes[1, 0]
    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        display_name = model_display_names.get(model_name, model_name)
        ax3.plot(metrics['complexity'], metrics['accuracy'],
                marker=markers[idx % len(markers)],
                color=colors[idx],
                label=display_name,
                linewidth=2,
                markersize=6,
                alpha=0.8)
    ax3.set_xlabel('Complexity (Edge Retention Ratio)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy-Complexity Curve', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1.05])

    # 4. IC-AUC对比柱状图
    ax4 = axes[1, 1]
    model_names_sorted = sorted(model_metrics.keys(),
                                key=lambda x: model_metrics[x]['ic_auc_add'] or 0,
                                reverse=True)
    display_names_sorted = [model_display_names.get(m, m) for m in model_names_sorted]
    ic_auc_add_sorted = [model_metrics[m]['ic_auc_add'] for m in model_names_sorted]

    bars = ax4.bar(range(len(display_names_sorted)), ic_auc_add_sorted,
                   color=colors[:len(display_names_sorted)], alpha=0.8)
    ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('IC-AUC (Additive)', fontsize=12, fontweight='bold')
    ax4.set_title('IC-AUC Comparison (Additive Normalization)', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(display_names_sorted)))
    ax4.set_xticklabels(display_names_sorted, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    # 在柱子上添加数值
    for i, (bar, val) in enumerate(zip(bars, ic_auc_add_sorted)):
        if val is not None:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    output_file = output_dir / 'comparison_plots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图表已保存: {output_file}")
    plt.close()


def generate_summary_tables(model_metrics, output_dir):
    """生成汇总表格"""
    output_dir = Path(output_dir)

    # 模型简称映射
    model_display_names = {
        'gradient_based': 'IGPrune',
        'networkit_forest_fire': 'EFF',
        'networkit_local_degree': 'LD',
        'networkit_local_similarity': 'LS',
        'networkit_random_edge': 'RE',
        'networkit_random_node_edge': 'RN',
        'networkit_scan': 'SCAN',
        'networkit_simmelian': 'SO',
        'pri_graphs': 'PRI-graph'
    }

    # 创建汇总数据
    summary_data = []
    for model_name, metrics in model_metrics.items():
        display_name = model_display_names.get(model_name, model_name)
        summary_data.append({
            'Model': display_name,
            'IC_AUC_Add': metrics['ic_auc_add'],
            'IC_AUC_Log': metrics['ic_auc_log'],
            'Max_Accuracy': metrics['max_accuracy'],
            'Final_Accuracy': metrics['final_accuracy'],
            'Num_Steps': metrics['num_steps']
        })

    summary_df = pd.DataFrame(summary_data)

    # 按IC-AUC (Additive)排序
    summary_df = summary_df.sort_values('IC_AUC_Add', ascending=False)

    # 保存汇总表格
    summary_file = output_dir / 'summary_table.csv'
    summary_df.to_csv(summary_file, index=False, float_format='%.6f')
    print(f"✅ 汇总表格已保存: {summary_file}")

    # 打印汇总表格
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("="*80)

    return summary_df


def generate_markdown_report(model_metrics, summary_df, output_dir):
    """生成Markdown格式的报告"""
    output_dir = Path(output_dir)

    report = []
    report.append("# 超参数实验报告: num_steps对模型性能的影响")
    report.append("")
    report.append("## 实验配置")
    report.append("")
    report.append("- **数据集**: Cora")
    report.append("- **任务**: original label (节点分类)")
    report.append("- **下游模型**: GCN")
    report.append("- **num_steps**: 10")
    report.append("- **测试模型数**: 9")
    report.append("")

    report.append("## 实验结果汇总")
    report.append("")
    report.append("### 性能排名 (按IC-AUC Additive)")
    report.append("")
    report.append("| 排名 | 模型 | IC-AUC (Add) | IC-AUC (Log) | Max Accuracy | Final Accuracy |")
    report.append("|------|------|--------------|--------------|--------------|----------------|")

    for idx, row in summary_df.iterrows():
        rank = idx + 1 if isinstance(idx, int) else summary_df.index.get_loc(idx) + 1
        report.append(f"| {rank} | {row['Model']} | {row['IC_AUC_Add']:.4f} | {row['IC_AUC_Log']:.4f} | {row['Max_Accuracy']:.4f} | {row['Final_Accuracy']:.4f} |")

    report.append("")
    report.append("## 关键发现")
    report.append("")

    # 找出最佳模型
    best_model = summary_df.iloc[0]
    report.append(f"1. **最佳模型**: {best_model['Model']}")
    report.append(f"   - IC-AUC (Additive): {best_model['IC_AUC_Add']:.4f}")
    report.append(f"   - IC-AUC (Log-ratio): {best_model['IC_AUC_Log']:.4f}")
    report.append(f"   - 最高准确率: {best_model['Max_Accuracy']:.4f}")
    report.append("")

    report.append("2. **性能对比**:")
    report.append(f"   - 最佳IC-AUC: {summary_df['IC_AUC_Add'].max():.4f} ({summary_df.loc[summary_df['IC_AUC_Add'].idxmax(), 'Model']})")
    report.append(f"   - 最差IC-AUC: {summary_df['IC_AUC_Add'].min():.4f} ({summary_df.loc[summary_df['IC_AUC_Add'].idxmin(), 'Model']})")
    report.append(f"   - 平均IC-AUC: {summary_df['IC_AUC_Add'].mean():.4f}")
    report.append("")

    report.append("3. **准确率分析**:")
    report.append(f"   - 最高准确率: {summary_df['Max_Accuracy'].max():.4f}")
    report.append(f"   - 平均最高准确率: {summary_df['Max_Accuracy'].mean():.4f}")
    report.append(f"   - 平均最终准确率: {summary_df['Final_Accuracy'].mean():.4f}")
    report.append("")

    report.append("## 可视化")
    report.append("")
    report.append("详细的对比图表请查看: `comparison_plots.png`")
    report.append("")
    report.append("图表包括:")
    report.append("1. Information-Complexity曲线 (Additive归一化)")
    report.append("2. Information-Complexity曲线 (Log-ratio归一化)")
    report.append("3. Accuracy-Complexity曲线")
    report.append("4. IC-AUC对比柱状图")
    report.append("")

    report.append("## 结论")
    report.append("")
    report.append(f"本次实验使用num_steps=10对9个不同的图简化模型进行了评估。")
    report.append(f"结果表明，{best_model['Model']}在信息-复杂度权衡方面表现最佳，")
    report.append(f"IC-AUC达到{best_model['IC_AUC_Add']:.4f}。")
    report.append("")

    # 保存报告
    report_file = output_dir / 'REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✅ Markdown报告已保存: {report_file}")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = './results/hyperparameter_num_steps_20251007_124905'

    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    print("="*80)
    print("生成超参数实验综合结果")
    print("="*80)
    print(f"结果目录: {results_dir}")
    print()

    # 加载和处理结果
    print("📊 加载实验结果...")
    combined_df, model_metrics = load_and_process_results(results_dir)

    if not model_metrics:
        print("❌ 未找到任何实验结果")
        return

    print(f"✅ 已加载 {len(model_metrics)} 个模型的结果")
    print()

    # 生成汇总表格
    print("📋 生成汇总表格...")
    summary_df = generate_summary_tables(model_metrics, results_dir)
    print()

    # 创建对比图表
    print("📈 创建对比图表...")
    create_comparison_plots(model_metrics, results_dir)
    print()

    # 生成Markdown报告
    print("📝 生成Markdown报告...")
    generate_markdown_report(model_metrics, summary_df, results_dir)
    print()

    print("="*80)
    print("✅ 所有综合结果已生成完毕！")
    print("="*80)
    print(f"\n结果文件:")
    print(f"  - 汇总表格: {results_dir / 'summary_table.csv'}")
    print(f"  - 对比图表: {results_dir / 'comparison_plots.png'}")
    print(f"  - 详细报告: {results_dir / 'REPORT.md'}")
    print()


if __name__ == '__main__':
    main()
