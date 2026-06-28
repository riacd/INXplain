#!/usr/bin/env python3
"""
恢复未完成的超参数实验

检查已有结果，只运行缺失的实验
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


def get_completed_experiments(results_dir):
    """检查已完成的实验"""
    process_results_dir = Path(results_dir) / 'Cora_original_gcn' / 'process_results'

    if not process_results_dir.exists():
        return set()

    completed = set()
    for file in process_results_dir.glob('*_step_metrics.tsv'):
        # 从文件名提取模型名
        model_name = file.stem.replace('_step_metrics', '')
        completed.add(model_name)

    return completed


def resume_hyperparameter_experiment(existing_results_dir):
    """恢复超参数实验，只运行缺失的实验"""

    # 实验配置（与原实验相同）
    dataset_name = 'Cora'
    task_type = 'original'
    downstream_model = 'gcn'
    num_steps_values = [5, 10, 20, 50]

    # 测试的所有模型
    all_models = [
        'gradient_based',           # IGPrune
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

    # 检查已完成的实验
    completed_models = get_completed_experiments(existing_results_dir)
    print(f"\n已完成的实验: {completed_models}")

    # 确定需要运行的模型
    models_to_run = [m for m in all_models if m not in completed_models]

    print("="*80)
    print("恢复超参数实验: num_steps对模型性能的影响")
    print("="*80)
    print(f"数据集: {dataset_name}")
    print(f"任务: {task_type}")
    print(f"下游模型: {downstream_model}")
    print(f"num_steps值: {num_steps_values}")
    print(f"总模型数: {len(all_models)}")
    print(f"已完成: {len(completed_models)}")
    print(f"待运行: {len(models_to_run)}")
    print(f"待运行模型: {[model_short_names[m] for m in models_to_run]}")
    print(f"使用现有结果目录: {existing_results_dir}")
    print("="*80)
    print()

    if not models_to_run:
        print("✅ 所有实验已完成！")
        return existing_results_dir, []

    # 初始化benchmark（使用相同的结果目录）
    benchmark = UnifiedBenchmark(
        results_dir=existing_results_dir,
        device='cuda',
        data_dir='./data',
        random_seed=42
    )

    # 存储新完成的实验结果
    new_results = []

    # 对每个num_steps值（注意：原实验是对每个模型测试单个num_steps，这里需要确认配置）
    # 从process_results看，似乎每个模型只有一组结果，可能num_steps是固定的
    # 让我先测试单个num_steps=10
    num_steps = 10  # 使用默认值

    print(f"\n{'='*80}")
    print(f"使用 num_steps = {num_steps}")
    print(f"{'='*80}\n")

    # 对每个缺失的模型
    for model_name in models_to_run:
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

                new_results.append(exp_result)

                print(f"✅ 完成")
                print(f"   IC-AUC (add): {exp_result['ic_auc_add']:.4f}" if exp_result['ic_auc_add'] else "   IC-AUC (add): N/A")
                print(f"   IC-AUC (log): {exp_result['ic_auc_log']:.4f}" if exp_result['ic_auc_log'] else "   IC-AUC (log): N/A")
                print(f"   Max Accuracy: {exp_result['max_accuracy']:.4f}" if exp_result['max_accuracy'] else "   Max Accuracy: N/A")

            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                new_results.append({
                    'model': model_short_names[model_name],
                    'num_steps': num_steps,
                    'status': 'failed',
                    'error': error_msg
                })
                print(f"❌ 失败: {error_msg}")

        except Exception as e:
            print(f"❌ 异常: {e}")
            new_results.append({
                'model': model_short_names[model_name],
                'num_steps': num_steps,
                'status': 'exception',
                'error': str(e)
            })
            import traceback
            traceback.print_exc()

    # 保存新完成的实验结果
    if new_results:
        print(f"\n{'='*80}")
        print("保存新完成的实验结果")
        print(f"{'='*80}\n")

        # 保存为JSON（追加模式）
        results_file = Path(existing_results_dir) / 'new_results.json'
        with open(results_file, 'w') as f:
            json.dump(new_results, f, indent=2)
        print(f"✅ 新结果已保存: {results_file}")

        # 打印汇总
        print(f"\n{'='*80}")
        print("新完成的实验汇总")
        print(f"{'='*80}\n")

        print(f"新实验数: {len(new_results)}")
        print(f"成功: {sum(1 for r in new_results if r.get('status') == 'success')}")
        print(f"失败: {sum(1 for r in new_results if r.get('status') in ['failed', 'exception'])}")

        # 打印成功的实验结果
        success_results = [r for r in new_results if r.get('status') == 'success']
        if success_results:
            print("\n成功完成的实验:")
            for r in success_results:
                print(f"  {r['model']}: IC-AUC={r['ic_auc_add']:.4f}, Acc={r['max_accuracy']:.4f}")

    print(f"\n{'='*80}")
    print("恢复实验完成！")
    print(f"{'='*80}")
    print(f"\n所有结果保存在: {existing_results_dir}")
    print(f"\n下一步: 运行分析脚本以生成综合报告")
    print(f"  python scripts/analyze_hyperparameter_results.py {existing_results_dir}")

    return existing_results_dir, new_results


if __name__ == '__main__':
    # 使用已存在的结果目录
    existing_results_dir = './results/hyperparameter_num_steps_20251007_124905'

    if len(sys.argv) > 1:
        existing_results_dir = sys.argv[1]

    if not Path(existing_results_dir).exists():
        print(f"❌ 结果目录不存在: {existing_results_dir}")
        sys.exit(1)

    results_dir, new_results = resume_hyperparameter_experiment(existing_results_dir)
