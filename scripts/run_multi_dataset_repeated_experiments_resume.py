#!/usr/bin/env python3
"""
多数据集重复实验脚本（支持断点续传）

在多个数据集上测试指定模型，每个数据集重复多次（每次使用不同的随机种子），
计算benchmark各指标的平均值和标准误差，并报告时间消耗。

新增功能：
1. 自动跳过已完成的实验
2. 支持排除特定模型-数据集组合
3. 支持从中断处继续运行

使用示例:
    python scripts/run_multi_dataset_repeated_experiments_resume.py \
        --models networkit_forest_fire pri_graphs gradient_based \
        --datasets Cora CiteSeer PubMed KarateClub \
        --task original \
        --num-repeats 5 \
        --seeds 42 123 456 789 1024 \
        --exclude-combinations "pri_graphs:PubMed"
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time
from typing import List, Dict, Any, Set, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from GS.benchmark.unified import UnifiedBenchmark
from GS.models import model_registry


def build_downstream_kwargs(args) -> Dict[str, Any]:
    """Build downstream model kwargs from CLI options."""
    kwargs: Dict[str, Any] = {}

    if args.downstream == 'gat' and args.downstream_preset == 'gat_tuned':
        kwargs.update({
            'hidden_dim': 8,
            'heads': 8,
            'dropout': 0.6,
            'lr': 0.005,
            'weight_decay': 5e-4,
            'early_stopping_patience': 50,
        })

    if args.downstream_hidden_dim is not None:
        kwargs['hidden_dim'] = args.downstream_hidden_dim
    if args.downstream_dropout is not None:
        kwargs['dropout'] = args.downstream_dropout
    if args.downstream_lr is not None:
        kwargs['lr'] = args.downstream_lr
    if args.downstream_weight_decay is not None:
        kwargs['weight_decay'] = args.downstream_weight_decay
    if args.downstream_early_stopping_patience is not None:
        kwargs['early_stopping_patience'] = args.downstream_early_stopping_patience
    if args.downstream == 'gat' and args.gat_heads is not None:
        kwargs['heads'] = args.gat_heads

    return kwargs


def check_if_completed(
    dataset: str,
    model_name: str,
    seed: int,
    task: str,
    downstream: str,
    results_base_dir: str = './results/multi_dataset_repeated'
) -> bool:
    """检查某个实验是否已经完成"""
    result_dir = Path(results_base_dir) / f"{dataset}_{task}_{downstream}" / f"seed_{seed}" / f"{dataset}_{task}_{downstream}" / "process_results"

    # 检查是否存在step_metrics.tsv文件
    metrics_file = result_dir / f"{model_name}_step_metrics.tsv"

    return metrics_file.exists()


def parse_exclusions(exclusion_list: List[str]) -> Set[Tuple[str, str]]:
    """解析排除列表，格式为 'model:dataset'"""
    exclusions = set()
    if exclusion_list:
        for item in exclusion_list:
            if ':' in item:
                model, dataset = item.split(':', 1)
                exclusions.add((model.strip(), dataset.strip()))
            else:
                print(f"⚠️  警告: 忽略无效的排除项 '{item}' (应为 'model:dataset' 格式)")
    return exclusions


def run_single_experiment(
    model_name: str,
    dataset: str,
    task: str,
    downstream: str,
    num_steps: int,
    epochs: int,
    seed: int,
    device: str,
    downstream_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """运行单次实验"""
    try:
        benchmark = UnifiedBenchmark(
            device=device,
            random_seed=seed,
            results_dir=f'./results/multi_dataset_repeated/{dataset}_{task}_{downstream}/seed_{seed}'
        )

        result = benchmark.run_single_model(
            model_name=model_name,
            dataset_name=dataset,
            task_type=task,
            downstream_model=downstream,
            num_steps=num_steps,
            epochs=epochs,
            downstream_kwargs=downstream_kwargs
        )

        if result.get('success'):
            return {
                'success': True,
                'ic_auc_additive': result.get('ic_auc_additive', None),
                'ic_auc_log_ratio': result.get('ic_auc_log_ratio', None),
                'threshold_point_additive': result.get('threshold_point_additive', None),
                'threshold_point_log_ratio': result.get('threshold_point_log_ratio', None),
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Unknown error')
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def load_completed_result(
    dataset: str,
    model_name: str,
    seed: int,
    task: str,
    downstream: str,
    results_base_dir: str = './results/multi_dataset_repeated'
) -> Dict[str, Any]:
    """从已完成的实验中加载结果"""
    result_dir = Path(results_base_dir) / f"{dataset}_{task}_{downstream}" / f"seed_{seed}" / f"{dataset}_{task}_{downstream}" / "aggregate_results"

    # 读取aggregate_metrics.tsv
    metrics_file = result_dir / "aggregate_metrics.tsv"

    if not metrics_file.exists():
        return {'success': False, 'error': 'Metrics file not found'}

    try:
        df = pd.read_csv(metrics_file, sep='\t')
        model_data = df[df['model'] == model_name]

        if len(model_data) == 0:
            return {'success': False, 'error': 'Model not found in metrics'}

        row = model_data.iloc[0]

        return {
            'success': True,
            'ic_auc_additive': row.get('ic_auc_additive', None),
            'ic_auc_log_ratio': row.get('ic_auc_log_ratio', None),
            'threshold_point_additive': row.get('threshold_point_additive', None),
            'threshold_point_log_ratio': row.get('threshold_point_log_ratio', None),
        }
    except Exception as e:
        return {'success': False, 'error': f'Error loading result: {str(e)}'}


def run_multi_dataset_repeated_experiments(
    model_names: List[str],
    datasets: List[str],
    task: str = 'original',
    downstream: str = 'gcn',
    num_steps: int = 10,
    epochs: int = 30,
    num_repeats: int = 5,
    seeds: List[int] = None,
    device: str = 'cuda',
    skip_completed: bool = True,
    exclusions: Set[Tuple[str, str]] = None,
    downstream_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    运行多数据集重复实验（支持断点续传）
    """
    if exclusions is None:
        exclusions = set()

    # 生成随机种子列表
    if seeds is None:
        seeds = [42 + i * 100 for i in range(num_repeats)]
    elif len(seeds) < num_repeats:
        print(f"⚠️  警告: 提供的种子数量({len(seeds)})少于重复次数({num_repeats})")
        num_repeats = len(seeds)

    print("=" * 80)
    print(f"多数据集重复实验设置")
    print("=" * 80)
    print(f"数据集: {', '.join(datasets)}")
    print(f"任务: {task}")
    print(f"下游模型: {downstream}")
    print(f"模型列表: {', '.join(model_names)}")
    print(f"重复次数: {num_repeats}")
    print(f"随机种子: {seeds}")
    print(f"步数: {num_steps}, 训练轮数: {epochs}")
    print(f"设备: {device}")
    print(f"跳过已完成: {skip_completed}")
    if downstream_kwargs:
        print(f"下游模型参数: {downstream_kwargs}")
    if exclusions:
        print(f"排除组合: {', '.join([f'{m}:{d}' for m, d in exclusions])}")
    print("=" * 80)

    # 存储所有结果
    all_results = {
        'config': {
            'datasets': datasets,
            'task': task,
            'downstream': downstream,
            'num_steps': num_steps,
            'epochs': epochs,
            'num_repeats': num_repeats,
            'seeds': seeds,
            'device': device,
            'skip_completed': skip_completed,
            'exclusions': list(exclusions) if exclusions else [],
            'downstream_kwargs': downstream_kwargs or {}
        },
        'results': {}  # {dataset: {model: {...}}}
    }

    # 记录总时间
    total_start_time = time.time()
    skipped_count = 0
    excluded_count = 0
    run_count = 0

    # 对每个数据集
    for dataset_idx, dataset in enumerate(datasets, 1):
        print(f"\n{'='*80}")
        print(f"数据集 {dataset_idx}/{len(datasets)}: {dataset}")
        print(f"{'='*80}")

        all_results['results'][dataset] = {}

        # 对每个模型
        for model_idx, model_name in enumerate(model_names, 1):
            # 检查是否在排除列表中
            if (model_name, dataset) in exclusions:
                print(f"\n模型 {model_idx}/{len(model_names)}: {model_name}")
                print("-" * 60)
                print(f"  ⊗ 跳过 (在排除列表中)")
                excluded_count += 1
                continue

            print(f"\n模型 {model_idx}/{len(model_names)}: {model_name}")
            print("-" * 60)

            model_results = {
                'runs': [],
                'statistics': {}
            }

            # 重复实验
            for repeat_idx, seed in enumerate(seeds, 1):
                # 检查是否已完成
                if skip_completed and check_if_completed(dataset, model_name, seed, task, downstream):
                    print(f"  运行 {repeat_idx}/{num_repeats} (种子: {seed})... ⏭️  已完成，跳过", flush=True)

                    # 尝试加载已有结果
                    result = load_completed_result(dataset, model_name, seed, task, downstream)

                    if result['success']:
                        run_result = {
                            'seed': seed,
                            'repeat_idx': repeat_idx,
                            'success': True,
                            'ic_auc_additive': result['ic_auc_additive'],
                            'ic_auc_log_ratio': result['ic_auc_log_ratio'],
                            'threshold_point_additive': result['threshold_point_additive'],
                            'threshold_point_log_ratio': result['threshold_point_log_ratio'],
                            'run_time': 0.0,  # 已完成的不计时间
                            'skipped': True
                        }
                        model_results['runs'].append(run_result)
                        skipped_count += 1
                    else:
                        print(f"    ⚠️  无法加载已完成结果: {result.get('error')}")
                    continue

                print(f"  运行 {repeat_idx}/{num_repeats} (种子: {seed})...", end=" ", flush=True)

                run_start_time = time.time()

                result = run_single_experiment(
                    model_name=model_name,
                    dataset=dataset,
                    task=task,
                    downstream=downstream,
                    num_steps=num_steps,
                    epochs=epochs,
                    seed=seed,
                    device=device,
                    downstream_kwargs=downstream_kwargs
                )

                run_time = time.time() - run_start_time

                if result['success']:
                    run_result = {
                        'seed': seed,
                        'repeat_idx': repeat_idx,
                        'success': True,
                        'ic_auc_additive': result['ic_auc_additive'],
                        'ic_auc_log_ratio': result['ic_auc_log_ratio'],
                        'threshold_point_additive': result['threshold_point_additive'],
                        'threshold_point_log_ratio': result['threshold_point_log_ratio'],
                        'run_time': run_time,
                        'skipped': False
                    }
                    print(f"✅ ({run_time:.1f}s)")
                    run_count += 1
                else:
                    run_result = {
                        'seed': seed,
                        'repeat_idx': repeat_idx,
                        'success': False,
                        'error': result['error'],
                        'run_time': run_time,
                        'skipped': False
                    }
                    print(f"❌ {result['error']}")

                model_results['runs'].append(run_result)

            # 计算统计数据
            successful_runs = [r for r in model_results['runs'] if r.get('success', False)]

            if successful_runs:
                # 提取指标
                ic_auc_add_values = [r['ic_auc_additive'] for r in successful_runs if r['ic_auc_additive'] is not None]
                ic_auc_log_values = [r['ic_auc_log_ratio'] for r in successful_runs if r['ic_auc_log_ratio'] is not None]
                threshold_add_values = [r['threshold_point_additive'] for r in successful_runs if r['threshold_point_additive'] is not None]
                threshold_log_values = [r['threshold_point_log_ratio'] for r in successful_runs if r['threshold_point_log_ratio'] is not None]

                def compute_stats(values):
                    if len(values) == 0:
                        return {'mean': None, 'std': None, 'stderr': None, 'n': 0}
                    arr = np.array(values)
                    return {
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr, ddof=1)) if len(values) > 1 else 0.0,
                        'stderr': float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(values) > 1 else 0.0,
                        'n': len(values),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr))
                    }

                # 只计算新运行的时间
                new_runs = [r for r in model_results['runs'] if not r.get('skipped', False)]

                model_results['statistics'] = {
                    'num_successful': len(successful_runs),
                    'num_failed': len(model_results['runs']) - len(successful_runs),
                    'num_skipped': sum(1 for r in model_results['runs'] if r.get('skipped', False)),
                    'ic_auc_additive': compute_stats(ic_auc_add_values),
                    'ic_auc_log_ratio': compute_stats(ic_auc_log_values),
                    'threshold_point_additive': compute_stats(threshold_add_values),
                    'threshold_point_log_ratio': compute_stats(threshold_log_values),
                    'avg_run_time': np.mean([r['run_time'] for r in new_runs]) if new_runs else 0.0,
                    'total_run_time': np.sum([r['run_time'] for r in new_runs]) if new_runs else 0.0
                }

                # 打印统计
                stats = model_results['statistics']
                print(f"\n  统计结果 ({len(successful_runs)}/{num_repeats} 成功, {stats['num_skipped']} 跳过):")
                print(f"    IC-AUC (Add): {stats['ic_auc_additive']['mean']:.4f} ± {stats['ic_auc_additive']['stderr']:.4f}")
                print(f"    IC-AUC (Log): {stats['ic_auc_log_ratio']['mean']:.4f} ± {stats['ic_auc_log_ratio']['stderr']:.4f}")
                print(f"    Threshold (Add): {stats['threshold_point_additive']['mean']:.4f} ± {stats['threshold_point_additive']['stderr']:.4f}")
                print(f"    Threshold (Log): {stats['threshold_point_log_ratio']['mean']:.4f} ± {stats['threshold_point_log_ratio']['stderr']:.4f}")
                if new_runs:
                    print(f"    新运行时间: {stats['total_run_time']:.1f}s, 平均: {stats['avg_run_time']:.1f}s")
            else:
                model_results['statistics'] = {
                    'num_successful': 0,
                    'num_failed': len(model_results['runs']),
                    'num_skipped': sum(1 for r in model_results['runs'] if r.get('skipped', False))
                }
                print(f"\n  ❌ 所有运行均失败")

            all_results['results'][dataset][model_name] = model_results

    # 计算总时间
    total_time = time.time() - total_start_time
    all_results['total_time'] = total_time
    all_results['summary'] = {
        'total_experiments': run_count + skipped_count + excluded_count,
        'new_runs': run_count,
        'skipped': skipped_count,
        'excluded': excluded_count
    }

    print(f"\n{'='*80}")
    print(f"所有实验完成!")
    print(f"  总耗时: {total_time/3600:.2f} 小时")
    print(f"  新运行: {run_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  排除: {excluded_count}")
    print(f"{'='*80}")

    return all_results


def save_results(results: Dict[str, Any], output_dir: str = './results/multi_dataset_repeated'):
    """保存结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = results['config']
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # 保存完整JSON
    json_file = output_path / f"multi_dataset_repeated_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 保存完整结果到: {json_file}")

    # 生成汇总表格
    summary_data = []

    for dataset, models_data in results['results'].items():
        for model_name, model_results in models_data.items():
            stats = model_results['statistics']

            if stats.get('num_successful', 0) > 0:
                summary_data.append({
                    'Dataset': dataset,
                    'Model': model_name,
                    'Success_Rate': f"{stats['num_successful']}/{config['num_repeats']}",
                    'IC_AUC_Add_Mean': stats['ic_auc_additive']['mean'],
                    'IC_AUC_Add_StdErr': stats['ic_auc_additive']['stderr'],
                    'IC_AUC_Log_Mean': stats['ic_auc_log_ratio']['mean'],
                    'IC_AUC_Log_StdErr': stats['ic_auc_log_ratio']['stderr'],
                    'Threshold_Add_Mean': stats['threshold_point_additive']['mean'],
                    'Threshold_Add_StdErr': stats['threshold_point_additive']['stderr'],
                    'Threshold_Log_Mean': stats['threshold_point_log_ratio']['mean'],
                    'Threshold_Log_StdErr': stats['threshold_point_log_ratio']['stderr'],
                    'Total_Time_s': stats['total_run_time'],
                    'Avg_Time_s': stats['avg_run_time']
                })

    if summary_data:
        df = pd.DataFrame(summary_data)

        # 保存为TSV
        tsv_file = output_path / f"summary_{timestamp}.tsv"
        df.to_csv(tsv_file, sep='\t', index=False)
        print(f"✅ 保存汇总表格到: {tsv_file}")

        # 保存为CSV
        csv_file = output_path / f"summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ 保存汇总表格到: {csv_file}")

        # 打印汇总表格
        print(f"\n{'='*80}")
        print(f"汇总表格:")
        print(f"{'='*80}")
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='多数据集重复实验工具（支持断点续传）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 模型和数据集选择
    parser.add_argument('--models', nargs='+', required=True,
                       help='要测试的模型列表')

    parser.add_argument('--datasets', nargs='+', required=True,
                       choices=['Cora', 'CiteSeer', 'PubMed', 'KarateClub',
                                'IMDB', 'Reddit', 'PPI',
                                'SO_relation_ME', 'SO_relation_MT'],
                       help='要测试的数据集列表')

    # 实验参数
    parser.add_argument('--task',
                       choices=['original', 'degree', 'degree_centrality',
                                'pagerank', 'closeness_centrality'],
                       default='original',
                       help='任务类型 (默认: original)')

    parser.add_argument('--downstream',
                       choices=['gcn', 'gat'],
                       default='gcn',
                       help='下游任务模型 (默认: gcn)')

    parser.add_argument('--num-steps', type=int, default=10,
                       help='图总结步数 (默认: 10)')

    parser.add_argument('--epochs', type=int, default=30,
                       help='下游任务训练轮数 (默认: 30)')
    parser.add_argument('--downstream-preset',
                       choices=['default', 'gat_tuned'],
                       default='default',
                       help='下游模型参数预设 (默认: default)')
    parser.add_argument('--downstream-hidden-dim', type=int,
                       help='下游模型隐藏维度')
    parser.add_argument('--gat-heads', type=int,
                       help='GAT 注意力头数')
    parser.add_argument('--downstream-dropout', type=float,
                       help='下游模型 dropout')
    parser.add_argument('--downstream-lr', type=float,
                       help='下游模型学习率')
    parser.add_argument('--downstream-weight-decay', type=float,
                       help='下游模型 weight decay')
    parser.add_argument('--downstream-early-stopping-patience', type=int,
                       help='下游模型 early stopping patience')

    # 重复实验参数
    parser.add_argument('--num-repeats', type=int, default=5,
                       help='重复次数 (默认: 5)')

    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                       help='随机种子列表 (默认: 自动生成)')

    parser.add_argument('--device',
                       choices=['cpu', 'cuda'],
                       default='cuda',
                       help='计算设备 (默认: cuda)')

    parser.add_argument('--output-dir', type=str,
                       default='./results/multi_dataset_repeated',
                       help='结果输出目录')

    # 断点续传和排除参数
    parser.add_argument('--no-skip-completed', action='store_true',
                       help='不跳过已完成的实验')

    parser.add_argument('--exclude-combinations', nargs='+', default=[],
                       help='排除的模型-数据集组合，格式: model:dataset (例: pri_graphs:PubMed)')

    args = parser.parse_args()

    # 验证模型名称
    available_models = model_registry.list_models()
    invalid_models = [m for m in args.models if m not in available_models]

    if invalid_models:
        print(f"❌ 错误: 以下模型不存在: {', '.join(invalid_models)}")
        print(f"\n可用模型列表:")
        for model in available_models:
            print(f"  - {model}")
        return

    # 解析排除列表
    exclusions = parse_exclusions(args.exclude_combinations)
    downstream_kwargs = build_downstream_kwargs(args)

    # 运行实验
    results = run_multi_dataset_repeated_experiments(
        model_names=args.models,
        datasets=args.datasets,
        task=args.task,
        downstream=args.downstream,
        num_steps=args.num_steps,
        epochs=args.epochs,
        num_repeats=args.num_repeats,
        seeds=args.seeds,
        device=args.device,
        skip_completed=not args.no_skip_completed,
        exclusions=exclusions,
        downstream_kwargs=downstream_kwargs
    )

    # 保存结果
    save_results(results, args.output_dir)


if __name__ == '__main__':
    main()
