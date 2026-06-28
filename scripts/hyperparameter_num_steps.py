#!/usr/bin/env python3
"""
超参数实验：num_steps对模型性能的影响

测试配置：
- 数据集: Cora
- 任务: original label
- 下游模型: GCN
- num_steps: [5, 10, 20, 50]
- 测试模型: INXplain + 8个baseline（共9个）

模型列表：
1. gradient_based (INXplain)
2. networkit_forest_fire (EFF)
3. networkit_local_degree (LD)
4. networkit_local_similarity (LS)
5. networkit_random_edge (RE)
6. networkit_random_node_edge (RN)
7. networkit_scan (SCAN)
8. networkit_simmelian (SO)
9. pri_graphs (PRI-graph)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GS.benchmark.unified import UnifiedBenchmark


def run_hyperparameter_experiment():
    """运行num_steps超参数实验"""

    # 实验配置
    dataset_name = 'Cora'
    task_type = 'original'
    downstream_model = 'gcn'
    num_steps_values = [5, 10, 20, 50]

    # 测试的所有模型
    models = [
        'gradient_based',           # INXplain
        'networkit_forest_fire',    # EFF
        'networkit_local_degree',   # LD
        'networkit_local_similarity', # LS
        'networkit_random_edge',    # RE
        'networkit_random_node_edge', # RN
        'networkit_scan',           # SCAN
        'networkit_simmelian',      # SO
        'pri_graphs'                # PRI-graph
    ]

    # 模型简称映射
    model_short_names = {
        'gradient_based': 'INXplain',
        'networkit_forest_fire': 'EFF',
        'networkit_local_degree': 'LD',
        'networkit_local_similarity': 'LS',
        'networkit_random_edge': 'RE',
        'networkit_random_node_edge': 'RN',
        'networkit_scan': 'SCAN',
        'networkit_simmelian': 'SO',
        'pri_graphs': 'PRI-graph'
    }

    # 创建结果目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'./results/hyperparameter_num_steps_{timestamp}'

    print("="*80)
    print("超参数实验: num_steps对模型性能的影响")
    print("="*80)
    print(f"数据集: {dataset_name}")
    print(f"任务: {task_type}")
    print(f"下游模型: {downstream_model}")
    print(f"num_steps值: {num_steps_values}")
    print(f"测试模型数量: {len(models)}")
    print(f"结果保存到: {results_dir}")
    print("="*80)
    print()

    # 初始化benchmark
    benchmark = UnifiedBenchmark(
        results_dir=results_dir,
        device='cuda',
        data_dir='./data',
        random_seed=42
    )

    # 存储所有实验结果
    all_results = []

    # 对每个num_steps值
    for num_steps in num_steps_values:
        print(f"\n{'='*80}")
        print(f"测试 num_steps = {num_steps}")
        print(f"{'='*80}\n")

        # 对每个模型
        for model_name in models:
            print(f"\n{'-'*60}")
            print(f"模型: {model_short_names[model_name]} (num_steps={num_steps})")
            print(f"{'-'*60}")

            try:
                # 运行单个模型实验
                result = benchmark.run_single_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    task_type=task_type,
                    downstream_model=downstream_model,
                    num_steps=num_steps,
                    epochs=100,  # 固定训练轮数
                    model_kwargs={}
                )

                if result and 'error' not in result:
                    # 提取关键指标
                    exp_result = {
                        'model': model_short_names[model_name],
                        'num_steps': num_steps,
                        'ic_auc_add': result.get('ic_auc_add', None),
                        'ic_auc_log': result.get('ic_auc_log', None),
                        'max_accuracy': max(result.get('accuracy_metrics', [0])) if result.get('accuracy_metrics') else None,
                        'final_accuracy': result.get('accuracy_metrics', [None])[-1] if result.get('accuracy_metrics') else None,
                        'complexity_metrics': result.get('complexity_metrics', []),
                        'information_metrics_add': result.get('information_metrics_add', []),
                        'information_metrics_log': result.get('information_metrics_log', []),
                        'accuracy_metrics': result.get('accuracy_metrics', []),
                        'exp_dir': str(result.get('exp_dir', '')),
                        'status': 'success'
                    }

                    all_results.append(exp_result)

                    print(f"✅ 完成")
                    print(f"   IC-AUC (add): {exp_result['ic_auc_add']:.4f}" if exp_result['ic_auc_add'] else "   IC-AUC (add): N/A")
                    print(f"   IC-AUC (log): {exp_result['ic_auc_log']:.4f}" if exp_result['ic_auc_log'] else "   IC-AUC (log): N/A")
                    print(f"   Max Accuracy: {exp_result['max_accuracy']:.4f}" if exp_result['max_accuracy'] else "   Max Accuracy: N/A")

                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                    all_results.append({
                        'model': model_short_names[model_name],
                        'num_steps': num_steps,
                        'status': 'failed',
                        'error': error_msg
                    })
                    print(f"❌ 失败: {error_msg}")

            except Exception as e:
                print(f"❌ 异常: {e}")
                all_results.append({
                    'model': model_short_names[model_name],
                    'num_steps': num_steps,
                    'status': 'exception',
                    'error': str(e)
                })
                import traceback
                traceback.print_exc()

    # 保存详细结果
    print(f"\n{'='*80}")
    print("保存实验结果")
    print(f"{'='*80}\n")

    # 保存为JSON
    results_file = Path(results_dir) / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ 详细结果已保存: {results_file}")

    # 创建汇总表格
    summary_data = []
    for result in all_results:
        if result.get('status') == 'success':
            summary_data.append({
                'Model': result['model'],
                'num_steps': result['num_steps'],
                'IC_AUC_Add': result['ic_auc_add'],
                'IC_AUC_Log': result['ic_auc_log'],
                'Max_Accuracy': result['max_accuracy'],
                'Final_Accuracy': result['final_accuracy']
            })

    summary_df = pd.DataFrame(summary_data)

    # 保存为CSV
    summary_csv = Path(results_dir) / 'summary.csv'
    summary_df.to_csv(summary_csv, index=False, float_format='%.6f')
    print(f"✅ 汇总表格已保存: {summary_csv}")

    # 创建pivot表格（按模型分组）
    if len(summary_df) > 0:
        for metric in ['IC_AUC_Add', 'IC_AUC_Log', 'Max_Accuracy']:
            pivot = summary_df.pivot(index='Model', columns='num_steps', values=metric)
            pivot_file = Path(results_dir) / f'{metric.lower()}_by_model.csv'
            pivot.to_csv(pivot_file, float_format='%.6f')
            print(f"✅ {metric} pivot表已保存: {pivot_file}")

    # 打印汇总统计
    print(f"\n{'='*80}")
    print("实验汇总")
    print(f"{'='*80}\n")

    print(f"总实验数: {len(all_results)}")
    print(f"成功: {sum(1 for r in all_results if r.get('status') == 'success')}")
    print(f"失败: {sum(1 for r in all_results if r.get('status') in ['failed', 'exception'])}")

    if len(summary_df) > 0:
        print("\nIC-AUC (Additive) 汇总:")
        print(summary_df.pivot(index='Model', columns='num_steps', values='IC_AUC_Add').to_string(float_format=lambda x: f'{x:.4f}'))

        print("\nMax Accuracy 汇总:")
        print(summary_df.pivot(index='Model', columns='num_steps', values='Max_Accuracy').to_string(float_format=lambda x: f'{x:.4f}'))

    print(f"\n{'='*80}")
    print("实验完成！")
    print(f"{'='*80}")
    print(f"\n所有结果保存在: {results_dir}")

    return results_dir, all_results


if __name__ == '__main__':
    results_dir, all_results = run_hyperparameter_experiment()
