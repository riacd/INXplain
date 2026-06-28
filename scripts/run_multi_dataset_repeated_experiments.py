#!/usr/bin/env python3
"""
多数据集重复实验脚本

在多个数据集上测试指定模型，每个数据集重复多次（每次使用不同的随机种子），
计算benchmark各指标的平均值和标准误差，并报告时间消耗。

使用示例:
    python scripts/run_multi_dataset_repeated_experiments.py \
        --models networkit_forest_fire pri_graphs gradient_based \
        --datasets Cora CiteSeer PubMed KarateClub \
        --task original \
        --num-repeats 5 \
        --seeds 42 123 456 789 1024
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional

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


def build_model_kwargs(args) -> Dict[str, Any]:
    """Build graph summarization model kwargs from CLI options."""
    kwargs: Dict[str, Any] = {}

    if args.scoring_downstream is not None:
        kwargs['downstream_model_type'] = args.scoring_downstream
    if args.scoring_train_epochs is not None:
        kwargs['train_epochs'] = args.scoring_train_epochs
    if args.scoring_hidden_dim is not None:
        kwargs['hidden_dim'] = args.scoring_hidden_dim
    if args.scoring_dropout is not None:
        kwargs['dropout'] = args.scoring_dropout
    if args.scoring_lr is not None:
        kwargs['lr'] = args.scoring_lr
    if args.scoring_weight_decay is not None:
        kwargs['weight_decay'] = args.scoring_weight_decay
    if args.scoring_early_stopping_patience is not None:
        kwargs['early_stopping_patience'] = args.scoring_early_stopping_patience
    if args.sampling_repeats is not None:
        kwargs['sampling_repeats'] = args.sampling_repeats
    if args.sampling_subset_num is not None:
        kwargs['sampling_subset_num'] = args.sampling_subset_num
    if args.sampling_seed is not None:
        kwargs['sampling_seed'] = args.sampling_seed
    if args.stability_penalty is not None:
        kwargs['stability_penalty'] = args.stability_penalty
    if args.scoring_model_types is not None:
        kwargs['scoring_model_types'] = args.scoring_model_types

    return kwargs


def run_single_experiment(
    model_name: str,
    dataset: str,
    task: str,
    downstream: str,
    num_steps: int,
    epochs: int,
    seed: int,
    device: str,
    downstream_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    disable_adaptive_epochs: bool = False,
    min_original_accuracy: Optional[float] = None,
    min_accuracy_over_majority: float = 0.0,
    require_informative_reference: bool = True,
    output_dir: str = './results/multi_dataset_repeated'
) -> Dict[str, Any]:
    """运行单次实验"""
    try:
        seed_results_dir = Path(output_dir) / f'{dataset}_{task}_{downstream}' / f'seed_{seed}'
        model_info = model_registry.get_model_info(model_name)
        downstream_kwargs = dict(downstream_kwargs or {})
        if downstream in ('h2gcn', 'gcnii') and 'dataset_name' not in downstream_kwargs:
            downstream_kwargs['dataset_name'] = dataset
        effective_model_kwargs: Dict[str, Any] = {}
        if model_info['category'] == 'development':
            effective_model_kwargs.update(model_kwargs or {})
            if 'sampling_seed' in effective_model_kwargs:
                effective_model_kwargs['sampling_seed'] = seed
        elif model_name.startswith('networkit_'):
            effective_model_kwargs['seed'] = seed

        benchmark = UnifiedBenchmark(
            device=device,
            random_seed=seed,
            results_dir=str(seed_results_dir)
        )
        if disable_adaptive_epochs:
            benchmark.enable_memory_optimization = False

        result = benchmark.run_single_model(
            model_name=model_name,
            dataset_name=dataset,
            task_type=task,
            downstream_model=downstream,
            num_steps=num_steps,
            epochs=epochs,
            model_kwargs=effective_model_kwargs or None,
            downstream_kwargs=downstream_kwargs,
            min_original_accuracy=min_original_accuracy,
            min_accuracy_over_majority=min_accuracy_over_majority,
            require_informative_reference=require_informative_reference,
        )

        if result.get('success'):
            accuracy_metrics = result.get('accuracy_metrics') or []
            return {
                'success': True,
                'ic_auc_additive': result.get('ic_auc_additive', None),
                'ic_auc_log_ratio': result.get('ic_auc_log_ratio', None),
                'threshold_point_additive': result.get('threshold_point_additive', None),
                'threshold_point_log_ratio': result.get('threshold_point_log_ratio', None),
                'original_accuracy': accuracy_metrics[0] if accuracy_metrics else None,
                'empty_accuracy': accuracy_metrics[-1] if accuracy_metrics else None,
                'training_time': result.get('training_time'),
                'summarization_time': result.get('summarization_time'),
                'effective_model_kwargs': effective_model_kwargs,
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
    downstream_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    disable_adaptive_epochs: bool = False,
    min_original_accuracy: Optional[float] = None,
    min_accuracy_over_majority: float = 0.0,
    output_dir: str = './results/multi_dataset_repeated'
) -> Dict[str, Any]:
    """
    运行多数据集重复实验
    """
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
    if downstream_kwargs:
        print(f"下游模型参数: {downstream_kwargs}")
    if model_kwargs:
        print(f"剪枝模型参数: {model_kwargs}")
    if disable_adaptive_epochs:
        print("Adaptive epochs: disabled")
    print("=" * 80)

    recorded_model_kwargs = dict(model_kwargs or {})
    if 'sampling_seed' in recorded_model_kwargs:
        recorded_model_kwargs['sampling_seed'] = '<repeat_seed>'

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
            'downstream_kwargs': downstream_kwargs or {},
            'model_kwargs': recorded_model_kwargs,
            'disable_adaptive_epochs': disable_adaptive_epochs,
            'min_original_accuracy': min_original_accuracy,
            'min_accuracy_over_majority': min_accuracy_over_majority,
            'model_seed_policy': 'repeat_seed',
            'output_dir': output_dir
        },
        'results': {}  # {dataset: {model: {...}}}
    }

    # 记录总时间
    total_start_time = time.time()

    # 对每个数据集
    for dataset_idx, dataset in enumerate(datasets, 1):
        print(f"\n{'='*80}")
        print(f"数据集 {dataset_idx}/{len(datasets)}: {dataset}")
        print(f"{'='*80}")

        all_results['results'][dataset] = {}

        # 对每个模型
        for model_idx, model_name in enumerate(model_names, 1):
            print(f"\n模型 {model_idx}/{len(model_names)}: {model_name}")
            print("-" * 60)

            model_results = {
                'runs': [],
                'statistics': {}
            }

            # 重复实验
            for repeat_idx, seed in enumerate(seeds, 1):
                print(f"\n  运行 {repeat_idx}/{num_repeats} (种子: {seed})...", end=" ", flush=True)

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
                    downstream_kwargs=downstream_kwargs,
                    model_kwargs=model_kwargs,
                    disable_adaptive_epochs=disable_adaptive_epochs,
                    min_original_accuracy=min_original_accuracy,
                    min_accuracy_over_majority=min_accuracy_over_majority,
                    output_dir=output_dir
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
                        'original_accuracy': result['original_accuracy'],
                        'empty_accuracy': result['empty_accuracy'],
                        'training_time': result['training_time'],
                        'summarization_time': result['summarization_time'],
                        'effective_model_kwargs': result['effective_model_kwargs'],
                        'run_time': run_time
                    }
                    print(f"✅ ({run_time:.1f}s)")
                else:
                    run_result = {
                        'seed': seed,
                        'repeat_idx': repeat_idx,
                        'success': False,
                        'error': result['error'],
                        'run_time': run_time
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
                original_accuracy_values = [r['original_accuracy'] for r in successful_runs if r['original_accuracy'] is not None]
                empty_accuracy_values = [r['empty_accuracy'] for r in successful_runs if r['empty_accuracy'] is not None]
                run_time_values = [r['run_time'] for r in successful_runs]

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

                model_results['statistics'] = {
                    'num_successful': len(successful_runs),
                    'num_failed': len(model_results['runs']) - len(successful_runs),
                    'ic_auc_additive': compute_stats(ic_auc_add_values),
                    'ic_auc_log_ratio': compute_stats(ic_auc_log_values),
                    'threshold_point_additive': compute_stats(threshold_add_values),
                    'threshold_point_log_ratio': compute_stats(threshold_log_values),
                    'original_accuracy': compute_stats(original_accuracy_values),
                    'empty_accuracy': compute_stats(empty_accuracy_values),
                    'run_time': compute_stats(run_time_values),
                    'avg_run_time': np.mean([r['run_time'] for r in model_results['runs']]),
                    'total_run_time': np.sum([r['run_time'] for r in model_results['runs']])
                }

                # 打印统计
                stats = model_results['statistics']
                print(f"\n  统计结果 ({len(successful_runs)}/{num_repeats} 成功):")
                print(f"    IC-AUC (Add): {stats['ic_auc_additive']['mean']:.4f} ± {stats['ic_auc_additive']['stderr']:.4f}")
                print(f"    IC-AUC (Log): {stats['ic_auc_log_ratio']['mean']:.4f} ± {stats['ic_auc_log_ratio']['stderr']:.4f}")
                print(f"    Threshold (Add): {stats['threshold_point_additive']['mean']:.4f} ± {stats['threshold_point_additive']['stderr']:.4f}")
                print(f"    Threshold (Log): {stats['threshold_point_log_ratio']['mean']:.4f} ± {stats['threshold_point_log_ratio']['stderr']:.4f}")
                print(f"    总时间: {stats['total_run_time']:.1f}s, 平均: {stats['avg_run_time']:.1f}s")
            else:
                model_results['statistics'] = {
                    'num_successful': 0,
                    'num_failed': len(model_results['runs']),
                    'avg_run_time': np.mean([r['run_time'] for r in model_results['runs']]),
                    'total_run_time': np.sum([r['run_time'] for r in model_results['runs']])
                }
                print(f"\n  ❌ 所有运行均失败")

            all_results['results'][dataset][model_name] = model_results

    # 计算总时间
    total_time = time.time() - total_start_time
    all_results['total_time'] = total_time

    print(f"\n{'='*80}")
    print(f"所有实验完成! 总耗时: {total_time/3600:.2f} 小时")
    print(f"{'='*80}")

    return all_results


def _format_metric_with_stderr(mean, stderr) -> str:
    if mean is None:
        return ""
    if stderr is None:
        return f"{mean:.6f}"
    return f"{mean:.6f} +/- {stderr:.6f}"


def _write_markdown_summary(df: pd.DataFrame, output_path: Path, title: str):
    display_columns = [
        'Dataset', 'Task', 'Downstream', 'Model', 'Success_Rate',
        'IC_AUC_Add', 'IC_AUC_Log', 'Threshold_Add', 'Threshold_Log',
        'Original_Accuracy', 'Empty_Accuracy', 'Runtime_s',
        'Total_Time_s', 'Avg_Time_s', 'Config'
    ]
    md_df = df[display_columns].copy()

    def markdown_cell(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    with open(output_path, 'w') as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(display_columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(display_columns)) + " |\n")
        for _, row in md_df.iterrows():
            f.write("| " + " | ".join(markdown_cell(row[col]) for col in display_columns) + " |\n")


def save_results(
    results: Dict[str, Any],
    output_dir: str = './results/multi_dataset_repeated',
    summary_prefix: Optional[str] = None,
    write_markdown_summary: bool = False
):
    """保存结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = results['config']
    timestamp = time.strftime('%Y%m%d_%H%M%S') + f"_{time.time_ns() % 1_000_000_000:09d}"

    # 保存完整JSON
    json_file = output_path / f"multi_dataset_repeated_results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ 保存完整结果到: {json_file}")

    # 生成汇总表格
    summary_data = []

    for dataset, models_data in results['results'].items():
        for model_name, model_results in models_data.items():
            stats = model_results['statistics']
            model_info = model_registry.get_model_info(model_name)

            num_successful = stats.get('num_successful', 0)

            ic_auc_add = stats.get('ic_auc_additive', {})
            ic_auc_log = stats.get('ic_auc_log_ratio', {})
            threshold_add = stats.get('threshold_point_additive', {})
            threshold_log = stats.get('threshold_point_log_ratio', {})
            original_accuracy = stats.get('original_accuracy', {})
            empty_accuracy = stats.get('empty_accuracy', {})
            run_time = stats.get('run_time', {})

            config_summary = {
                'num_steps': config['num_steps'],
                'epochs': config['epochs'],
                'seeds': config['seeds'],
                'device': config['device'],
                'model_category': model_info['category'],
                'downstream_kwargs': config['downstream_kwargs'],
                'model_kwargs': config['model_kwargs'] if model_info['category'] == 'development' else {},
                'disable_adaptive_epochs': config['disable_adaptive_epochs'],
                'min_original_accuracy': config['min_original_accuracy'],
                'min_accuracy_over_majority': config['min_accuracy_over_majority'],
                'model_seed_policy': config['model_seed_policy'],
            }

            summary_data.append({
                'Dataset': dataset,
                'Task': config['task'],
                'Downstream': config['downstream'],
                'Model': model_name,
                'Success_Rate': f"{num_successful}/{config['num_repeats']}",
                'IC_AUC_Add_Mean': ic_auc_add.get('mean'),
                'IC_AUC_Add_StdErr': ic_auc_add.get('stderr'),
                'IC_AUC_Log_Mean': ic_auc_log.get('mean'),
                'IC_AUC_Log_StdErr': ic_auc_log.get('stderr'),
                'Threshold_Add_Mean': threshold_add.get('mean'),
                'Threshold_Add_StdErr': threshold_add.get('stderr'),
                'Threshold_Log_Mean': threshold_log.get('mean'),
                'Threshold_Log_StdErr': threshold_log.get('stderr'),
                'Original_Accuracy_Mean': original_accuracy.get('mean'),
                'Original_Accuracy_StdErr': original_accuracy.get('stderr'),
                'Empty_Accuracy_Mean': empty_accuracy.get('mean'),
                'Empty_Accuracy_StdErr': empty_accuracy.get('stderr'),
                'Runtime_s_Mean': run_time.get('mean'),
                'Runtime_s_StdErr': run_time.get('stderr'),
                'Total_Time_s': stats.get('total_run_time'),
                'Avg_Time_s': stats.get('avg_run_time'),
                'IC_AUC_Add': _format_metric_with_stderr(
                    ic_auc_add.get('mean'), ic_auc_add.get('stderr')),
                'IC_AUC_Log': _format_metric_with_stderr(
                    ic_auc_log.get('mean'), ic_auc_log.get('stderr')),
                'Threshold_Add': _format_metric_with_stderr(
                    threshold_add.get('mean'), threshold_add.get('stderr')),
                'Threshold_Log': _format_metric_with_stderr(
                    threshold_log.get('mean'), threshold_log.get('stderr')),
                'Original_Accuracy': _format_metric_with_stderr(
                    original_accuracy.get('mean'), original_accuracy.get('stderr')),
                'Empty_Accuracy': _format_metric_with_stderr(
                    empty_accuracy.get('mean'), empty_accuracy.get('stderr')),
                'Runtime_s': _format_metric_with_stderr(
                    run_time.get('mean'), run_time.get('stderr')),
                'Config': json.dumps(config_summary, sort_keys=True, default=str),
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

        if summary_prefix:
            fixed_tsv_file = output_path / f"{summary_prefix}.tsv"
            df.to_csv(fixed_tsv_file, sep='\t', index=False)
            print(f"✅ 保存固定汇总表格到: {fixed_tsv_file}")

            if write_markdown_summary:
                fixed_md_file = output_path / f"{summary_prefix}.md"
                _write_markdown_summary(df, fixed_md_file, summary_prefix)
                print(f"✅ 保存Markdown汇总到: {fixed_md_file}")

        # 打印汇总表格
        print(f"\n{'='*80}")
        print(f"汇总表格:")
        print(f"{'='*80}")
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='多数据集重复实验工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 模型和数据集选择
    parser.add_argument('--models', nargs='+', required=True,
                       help='要测试的模型列表')

    parser.add_argument('--datasets', nargs='+', required=True,
                       choices=['Cora', 'CiteSeer', 'PubMed', 'KarateClub',
                                'IMDB', 'Reddit', 'PPI',
                                'Cornell', 'Texas', 'Wisconsin',
                                'SO_relation_ME', 'SO_relation_MT',
                                'ogbn-arxiv'],
                       help='要测试的数据集列表')

    # 实验参数
    parser.add_argument('--task',
                       choices=['original', 'degree', 'degree_centrality',
                                'pagerank', 'closeness_centrality'],
                       default='original',
                       help='任务类型 (默认: original)')

    parser.add_argument('--downstream',
                       choices=['gcn', 'gat', 'sage', 'h2gcn', 'gcnii'],
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
    parser.add_argument('--scoring-train-epochs', type=int,
                       help='剪枝 scoring model 每个剪枝 step 的训练轮数')
    parser.add_argument('--scoring-hidden-dim', type=int,
                       help='剪枝 scoring model 隐藏维度')
    parser.add_argument('--scoring-dropout', type=float,
                       help='剪枝 scoring model dropout')
    parser.add_argument('--scoring-lr', type=float,
                       help='剪枝 scoring model 学习率')
    parser.add_argument('--scoring-weight-decay', type=float,
                       help='剪枝 scoring model weight decay')
    parser.add_argument('--scoring-early-stopping-patience', type=int,
                       help='剪枝 scoring model early stopping patience')
    parser.add_argument('--scoring-downstream',
                       choices=['gcn', 'gat', 'sage', 'h2gcn', 'gcnii'],
                       help='剪枝方法内部用于边/子集打分的下游模型')
    parser.add_argument('--sampling-repeats', type=int,
                       help='联合 subset 采样 repeat 次数')
    parser.add_argument('--sampling-subset-num', type=int,
                       help='每个 repeat 内最多评估的联合删除 subset 数；不设置时评估全部分割出的 subset')
    parser.add_argument('--sampling-seed', type=int,
                       help='联合 subset 采样随机种子')
    parser.add_argument('--stability-penalty', type=float,
                       help='稳定版边级打分的 std 惩罚权重')
    parser.add_argument('--scoring-model-types', nargs='+',
                       choices=['gcn', 'gat', 'sage', 'graphsage', 'graph_sage'],
                       help='多模型剪枝 scorer 列表；默认由模型决定')
    parser.add_argument('--disable-adaptive-epochs', action='store_true',
                       help='禁用benchmark内部的自适应epoch缩减，严格使用--epochs')
    parser.add_argument('--min-original-accuracy', type=float,
                       help='原图测试准确率最低要求；未达到则实验失败')
    parser.add_argument('--min-accuracy-over-majority', type=float, default=0.0,
                       help='原图准确率至少超过测试集多数类基线的幅度')

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
    parser.add_argument('--summary-prefix', type=str,
                       help='额外写出固定文件名的汇总TSV/Markdown前缀')
    parser.add_argument('--write-markdown-summary', action='store_true',
                       help='配合--summary-prefix写出Markdown汇总表')

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

    downstream_kwargs = build_downstream_kwargs(args)
    model_kwargs = build_model_kwargs(args)

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
        downstream_kwargs=downstream_kwargs,
        model_kwargs=model_kwargs,
        disable_adaptive_epochs=args.disable_adaptive_epochs,
        min_original_accuracy=args.min_original_accuracy,
        min_accuracy_over_majority=args.min_accuracy_over_majority,
        output_dir=args.output_dir
    )

    # 保存结果
    save_results(
        results,
        args.output_dir,
        summary_prefix=args.summary_prefix,
        write_markdown_summary=args.write_markdown_summary
    )


if __name__ == '__main__':
    main()
