#!/usr/bin/env python3
"""
综合基准测试脚本

用于运行大规模的模型对比实验，支持多数据集、多任务类型的系统性测试。
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional
import time
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from GS.benchmark.unified import UnifiedBenchmark
from GS.datasets.loaders import DatasetLoader


class ComprehensiveBenchmark:
    """综合基准测试类"""

    def __init__(self, results_dir: str = './results/comprehensive_benchmark', device: str = 'cpu'):
        """
        初始化综合基准测试

        Args:
            results_dir: 结果保存目录
            device: 计算设备
        """
        self.benchmark = UnifiedBenchmark(results_dir=results_dir, device=device)
        self.results_dir = Path(results_dir)

        # 预定义的模型组
        self.model_groups = {
            'networkit': [
                'networkit_forest_fire',
                'networkit_local_degree',
                'networkit_local_similarity',
                'networkit_random_edge',
                'networkit_random_node_edge',
                'networkit_scan',
                'networkit_simmelian'
            ],
            'baselines': [
                'networkit_forest_fire',
                'networkit_local_degree',
                'networkit_local_similarity',
                'networkit_random_edge',
                'networkit_random_node_edge',
                'networkit_scan',
                'networkit_simmelian',
                'pri_graphs'
            ],
            'neural_enhanced': [
                'neural_enhanced_main',
                'neural_enhanced_high_fusion',
                'neural_enhanced_low_fusion',
                'neural_enhanced_no_residual',
                'neural_enhanced_slow_gradient'
            ],
            'development': [
                'gradient_based'
            ],
            'gradient_and_baselines': [
                'gradient_based',
                'networkit_forest_fire',
                'networkit_local_degree',
                'networkit_local_similarity',
                'networkit_random_edge',
                'networkit_random_node_edge',
                'networkit_scan',
                'networkit_simmelian',
                'pri_graphs'
            ],
            'all_neural_and_development': [
                'neural_enhanced_main',
                'neural_enhanced_high_fusion',
                'neural_enhanced_low_fusion',
                'neural_enhanced_no_residual',
                'neural_enhanced_slow_gradient',
                'gradient_based'
            ],
            'all_models': [
                'networkit_forest_fire',
                'networkit_local_degree',
                'networkit_local_similarity',
                'networkit_random_edge',
                'networkit_random_node_edge',
                'networkit_scan',
                'networkit_simmelian',
                'pri_graphs',
                'neural_enhanced_main',
                'neural_enhanced_high_fusion',
                'neural_enhanced_low_fusion',
                'neural_enhanced_no_residual',
                'neural_enhanced_slow_gradient',
                'gradient_based'
            ],
            'all_models_no_pri': [
                'networkit_forest_fire',
                'networkit_local_degree',
                'networkit_local_similarity',
                'networkit_random_edge',
                'networkit_random_node_edge',
                'networkit_scan',
                'networkit_simmelian',
                'gradient_based'
            ]
        }

        # 预定义的数据集组
        self.dataset_groups = {
            'small': ['KarateClub'],
            'citation': ['Cora', 'CiteSeer', 'PubMed'],
            'social': ['KarateClub', 'IMDB'],
            'academic': ['WikiCS'],
            'so_relation': ['SO_relation_ME', 'SO_relation_MT'],
            'ogb_small': ['ogbn-arxiv'],
            'ogb_medium': ['ogbn-products', 'ogbn-proteins'],
            'ogb_large': ['ogbn-mag', 'ogbn-papers100M'],
            'traditional': ['KarateClub', 'Cora', 'CiteSeer', 'PubMed', 'IMDB', 'WikiCS'],
            'all_non_ogb': ['KarateClub', 'Cora', 'CiteSeer', 'PubMed', 'IMDB', 'WikiCS', 'SO_relation_ME', 'SO_relation_MT']
        }

        # 预定义的任务类型组
        self.task_groups = {
            'basic': ['original', 'degree'],
            'centrality': ['degree_centrality', 'pagerank', 'closeness_centrality'],
            'all': ['original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality']
        }

    def get_completed_experiments(self) -> Set[str]:
        """获取已完成的实验"""
        completed = set()
        if self.results_dir.exists():
            for exp_dir in self.results_dir.iterdir():
                if exp_dir.is_dir() and '_' in exp_dir.name:
                    completed.add(exp_dir.name)
        return completed

    def get_completed_model_experiments(self, experiment_key: str, models: List[str]) -> Set[str]:
        """
        获取特定实验设置下已完成的模型

        Args:
            experiment_key: 实验设置标识符 (如 "Cora_original_gcn")
            models: 需要检查的模型列表

        Returns:
            已完成的模型名称集合
        """
        completed_models = set()
        exp_dir = self.results_dir / experiment_key

        if not exp_dir.exists():
            return completed_models

        # 检查每个模型是否有完整的结果
        for model_name in models:
            if self._is_model_experiment_complete(exp_dir, model_name):
                completed_models.add(model_name)

        return completed_models

    def _is_model_experiment_complete(self, exp_dir: Path, model_name: str) -> bool:
        """
        检查特定模型的实验是否完整

        Args:
            exp_dir: 实验目录
            model_name: 模型名称

        Returns:
            True if 实验完整，False otherwise
        """
        # 检查过程结果文件是否存在
        process_results_dir = exp_dir / "process_results"
        if not process_results_dir.exists():
            return False

        # 检查关键文件是否存在
        step_metrics_file = process_results_dir / f"{model_name}_step_metrics.tsv"
        if not step_metrics_file.exists() or step_metrics_file.stat().st_size == 0:
            return False

        # 检查综合结果目录中是否有该模型的记录
        comprehensive_dir = exp_dir / "comprehensive_results"
        if comprehensive_dir.exists():
            ic_auc_file = comprehensive_dir / "ic_auc_comparison.tsv"
            if ic_auc_file.exists():
                try:
                    # 读取TSV文件检查是否包含该模型
                    df = pd.read_csv(ic_auc_file, sep='\t')
                    if 'model' in df.columns and model_name in df['model'].values:
                        return True
                except Exception:
                    pass

        return False

    def _load_existing_results(self, exp_dir: Path, model_names: List[str]) -> Dict:
        """
        从已有的实验结果中加载模型数据

        Args:
            exp_dir: 实验目录
            model_names: 需要加载的模型名称列表

        Returns:
            加载的结果字典
        """
        results = {}

        for model_name in model_names:
            try:
                # 从综合结果中读取 IC-AUC。snr_auc 字段仅用于兼容旧结果。
                snr_auc = None
                comprehensive_dir = exp_dir / "comprehensive_results"
                if comprehensive_dir.exists():
                    ic_auc_file = comprehensive_dir / "ic_auc_comparison.tsv"
                    if ic_auc_file.exists():
                        try:
                            df = pd.read_csv(ic_auc_file, sep='\t')
                            if 'model' in df.columns and model_name in df['model'].values:
                                model_row = df[df['model'] == model_name]
                                # 优先使用 ic_auc_additive，如果不存在则使用 snr_auc
                                if not model_row.empty:
                                    if 'ic_auc_additive' in df.columns:
                                        snr_auc = float(model_row['ic_auc_additive'].iloc[0])
                                    elif 'snr_auc' in df.columns:
                                        snr_auc = float(model_row['snr_auc'].iloc[0])
                        except Exception:
                            pass

                # 从过程结果中读取复杂度和信息指标
                complexity_metrics = []
                information_metrics = []
                process_dir = exp_dir / "process_results"
                if process_dir.exists():
                    step_metrics_file = process_dir / f"{model_name}_step_metrics.tsv"
                    if step_metrics_file.exists():
                        try:
                            df = pd.read_csv(step_metrics_file, sep='\t')
                            if 'complexity_metric' in df.columns and 'information_metric' in df.columns:
                                complexity_metrics = df['complexity_metric'].tolist()
                                information_metrics = df['information_metric'].tolist()
                        except Exception:
                            pass

                # 构建结果对象
                if snr_auc is not None or complexity_metrics:
                    results[model_name] = {
                        'model_name': model_name,
                        'snr_auc': snr_auc if snr_auc is not None else 0.0,
                        'complexity_metrics': complexity_metrics,
                        'information_metrics': information_metrics,
                        'success': True,
                        'loaded_from_cache': True
                    }
                else:
                    # 如果无法加载完整数据，标记为需要重新运行
                    results[model_name] = {
                        'model_name': model_name,
                        'success': False,
                        'error': 'Incomplete cached data',
                        'loaded_from_cache': False
                    }

            except Exception as e:
                results[model_name] = {
                    'model_name': model_name,
                    'success': False,
                    'error': f'Failed to load cached result: {e}',
                    'loaded_from_cache': False
                }

        return results

    def run_comprehensive_test(self,
                             model_group: str = 'networkit',
                             dataset_group: str = 'traditional',
                             task_group: str = 'basic',
                             downstream_model: str = 'gcn',
                             num_steps: int = 10,
                             epochs: int = 30,
                             skip_completed: bool = True,
                             dry_run: bool = False) -> Dict:
        """
        运行综合测试

        Args:
            model_group: 模型组名称
            dataset_group: 数据集组名称
            task_group: 任务组名称
            downstream_model: 下游任务模型
            num_steps: 图总结步数
            epochs: 训练轮数
            skip_completed: 是否跳过已完成的实验
            dry_run: 是否只打印计划而不执行

        Returns:
            测试结果字典
        """
        # 获取模型、数据集、任务列表
        models = self.model_groups.get(model_group, [model_group] if isinstance(model_group, str) else model_group)
        datasets = self.dataset_groups.get(dataset_group, [dataset_group] if isinstance(dataset_group, str) else dataset_group)
        tasks = self.task_groups.get(task_group, [task_group] if isinstance(task_group, str) else task_group)

        print(f'🧪 综合基准测试计划')
        print('=' * 80)
        print(f'模型组: {model_group} ({len(models)} 个模型)')
        print(f'  {models}')
        print(f'数据集组: {dataset_group} ({len(datasets)} 个数据集)')
        print(f'  {datasets}')
        print(f'任务组: {task_group} ({len(tasks)} 个任务类型)')
        print(f'  {tasks}')
        print(f'下游模型: {downstream_model}')
        print(f'总计实验: {len(models)} × {len(datasets)} × {len(tasks)} = {len(models) * len(datasets) * len(tasks)} 个')

        # 检查已完成的实验
        completed_experiments = self.get_completed_experiments() if skip_completed else set()
        if completed_experiments:
            print(f'\n📂 已完成的实验: {len(completed_experiments)} 个')

        # 计算需要执行的实验（更精确的模型级别检查）
        planned_experiments = []
        skipped_experiments_with_results = {}  # 存储跳过的实验的结果

        for dataset in datasets:
            for task_type in tasks:
                experiment_key = f'{dataset}_{task_type}_{downstream_model}'

                if skip_completed:
                    # 检查已完成的模型
                    completed_models = self.get_completed_model_experiments(experiment_key, models)
                    remaining_models = [m for m in models if m not in completed_models]

                    if not remaining_models:
                        # 所有模型都已完成，加载结果但不执行
                        print(f'✅ 跳过完整实验: {experiment_key} (所有模型已完成)')
                        exp_dir = self.results_dir / experiment_key
                        skipped_experiments_with_results[experiment_key] = self._load_existing_results(exp_dir, models)
                    else:
                        # 部分模型需要执行
                        planned_experiments.append((dataset, task_type, experiment_key, remaining_models, completed_models))
                else:
                    # 不跳过，所有模型都需要执行
                    planned_experiments.append((dataset, task_type, experiment_key, models, set()))

        print(f'\n📋 计划执行的实验: {len(planned_experiments)} 个')
        for i, experiment_data in enumerate(planned_experiments[:5]):
            dataset, task_type, exp_key = experiment_data[:3]
            print(f'  {i+1}. {exp_key}')
        if len(planned_experiments) > 5:
            print(f'  ... 还有 {len(planned_experiments)-5} 个实验')

        # 打印跳过的完整实验
        if skipped_experiments_with_results:
            print(f'\n⏭️  完全跳过的实验: {len(skipped_experiments_with_results)} 个')
            for exp_key in list(skipped_experiments_with_results.keys())[:3]:
                print(f'  - {exp_key}')
            if len(skipped_experiments_with_results) > 3:
                print(f'  ... 还有 {len(skipped_experiments_with_results)-3} 个')

        if dry_run:
            print('\n🏃 Dry run 模式 - 仅显示计划，不执行实验')
            return {'planned_experiments': len(planned_experiments), 'total_experiments': len(models) * len(datasets) * len(tasks)}

        print(f'\n⏱️  预计总时间: {len(planned_experiments) * len(models) * 2} 分钟')
        print('🚀 开始执行实验...\n')

        # 执行实验
        all_results = {}
        completed_count = 0

        # 首先添加跳过的完整实验结果
        all_results.update(skipped_experiments_with_results)

        for i, experiment_data in enumerate(planned_experiments):
            dataset, task_type, experiment_key, models_to_run, completed_models = experiment_data
            print(f'\n🔬 实验 {i+1}/{len(planned_experiments)}: {experiment_key}')
            print('-' * 60)

            if completed_models:
                print(f'⏭️  跳过已完成的模型: {list(completed_models)} ({len(completed_models)}/{len(models)})')
            print(f'🚀 需要运行的模型: {list(models_to_run)} ({len(models_to_run)}/{len(models)})')

            start_time = time.time()

            try:
                # 运行模型对比实验（只运行需要的模型）
                results = self.benchmark.compare_models(
                    model_names=list(models_to_run),
                    dataset_name=dataset,
                    task_type=task_type,
                    downstream_model=downstream_model,
                    num_steps=num_steps,
                    epochs=epochs
                )

                # 如果有跳过的模型，合并已有结果
                if completed_models:
                    exp_dir = self.results_dir / experiment_key
                    existing_results = self._load_existing_results(exp_dir, list(completed_models))
                    # 合并结果
                    for model_name, result in existing_results.items():
                        results[model_name] = result

                all_results[experiment_key] = results

                # 统计成功的模型
                successful = sum(1 for r in results.values() if r.get('success', False))
                elapsed = time.time() - start_time

                print(f'✅ 完成: {successful}/{len(models)} 个模型成功 (耗时: {elapsed:.1f}s)')
                completed_count += 1

            except Exception as e:
                elapsed = time.time() - start_time
                print(f'❌ 实验失败: {e} (耗时: {elapsed:.1f}s)')
                all_results[experiment_key] = {'error': str(e)}

            # 进度更新
            progress = (i + 1) / len(planned_experiments) * 100
            print(f'📊 总体进度: {i+1}/{len(planned_experiments)} ({progress:.1f}%)')

        # 生成总结
        self._generate_summary(all_results, model_group, dataset_group, task_group)

        return all_results

    def _generate_summary(self, results: Dict, model_group: str, dataset_group: str, task_group: str):
        """生成实验总结"""
        print(f'\n' + '=' * 80)
        print(f'🎯 综合测试结果总结')
        print(f'=' * 80)

        # 统计成功率
        total_experiments = len(results)
        successful_experiments = sum(1 for r in results.values()
                                   if isinstance(r, dict) and 'error' not in r)
        success_rate = successful_experiments / total_experiments * 100 if total_experiments > 0 else 0

        print(f'完成实验数: {successful_experiments}/{total_experiments}')
        print(f'实验成功率: {success_rate:.1f}%')

        # 各实验最佳模型
        print(f'\n📈 各实验最佳模型排名:')
        best_models = {}

        for exp_key, exp_results in results.items():
            if isinstance(exp_results, dict) and 'error' not in exp_results:
                # 找到最佳模型
                best_model = None
                best_auc = -1

                for model_name, result in exp_results.items():
                    if result.get('success', False):
                        auc = result.get('snr_auc', 0)
                        if auc > best_auc:
                            best_auc = auc
                            best_model = model_name

                if best_model:
                    best_models[exp_key] = (best_model, best_auc)
                    print(f'   {exp_key}: {best_model} (IC-AUC: {best_auc:.3f})')
                else:
                    print(f'   {exp_key}: 无成功模型')
            else:
                print(f'   {exp_key}: 实验失败')

        # 模型总体表现排名
        if best_models:
            print(f'\n🏆 模型总体表现统计:')
            model_stats = {}
            for (best_model, auc) in best_models.values():
                if best_model not in model_stats:
                    model_stats[best_model] = {'wins': 0, 'total_auc': 0, 'count': 0}
                model_stats[best_model]['wins'] += 1
                model_stats[best_model]['total_auc'] += auc
                model_stats[best_model]['count'] += 1

            # 按获胜次数排序
            sorted_models = sorted(model_stats.items(),
                                 key=lambda x: (x[1]['wins'], x[1]['total_auc']/x[1]['count']),
                                 reverse=True)

            for i, (model, stats) in enumerate(sorted_models):
                avg_auc = stats['total_auc'] / stats['count']
                print(f'   {i+1}. {model}: {stats["wins"]} 次获胜, 平均 IC-AUC: {avg_auc:.3f}')

        print(f'\n📁 详细结果保存在: {self.results_dir}')
        print(f'✅ {model_group} × {dataset_group} × {task_group} 综合测试完成!')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='综合基准测试脚本')
    parser.add_argument('--model-group', default='networkit',
                       choices=['networkit', 'baselines', 'neural_enhanced', 'development', 'gradient_and_baselines', 'all_neural_and_development', 'all_models', 'all_models_no_pri'],
                       help='模型组选择')
    parser.add_argument('--dataset-group', default='traditional',
                       choices=['small', 'citation', 'social', 'academic', 'so_relation', 'ogb_small', 'ogb_medium', 'ogb_large', 'traditional', 'all_non_ogb'],
                       help='数据集组选择')
    parser.add_argument('--task-group', default='basic',
                       choices=['basic', 'centrality', 'all'],
                       help='任务类型组选择')
    parser.add_argument('--downstream', default='gcn',
                       choices=['gcn', 'gat'],
                       help='下游任务模型')
    parser.add_argument('--steps', type=int, default=10,
                       help='图总结步数')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--results-dir', default='./results/comprehensive_benchmark',
                       help='结果保存目录')
    parser.add_argument('--device', default='cpu',
                       choices=['cpu', 'cuda'],
                       help='计算设备')
    parser.add_argument('--no-skip', action='store_true',
                       help='不跳过已完成的实验')
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示计划不执行')

    # 新增：单独计算综合结果的选项
    parser.add_argument('--compute-comprehensive-only', action='store_true',
                       help='仅计算综合结果（从已有的过程结果）')
    parser.add_argument('--experiment-pattern', default='*',
                       help='实验目录匹配模式（如 "Cora_*_gcn"），用于批量计算综合结果')
    parser.add_argument('--target-models', nargs='*',
                       help='指定要包含在综合结果中的模型名称列表')

    args = parser.parse_args()

    # 创建ComprehensiveBenchmark实例
    comprehensive = ComprehensiveBenchmark(results_dir=args.results_dir, device=args.device)

    if args.compute_comprehensive_only:
        # 单独计算综合结果模式
        print(f"🔄 单独计算综合结果模式")
        print(f"📂 结果目录: {args.results_dir}")
        print(f"🔍 实验模式: {args.experiment_pattern}")
        if args.target_models:
            print(f"🎯 指定模型: {args.target_models}")

        try:
            batch_results = comprehensive.benchmark.batch_compute_comprehensive_results(
                results_base_dir=args.results_dir,
                experiment_pattern=args.experiment_pattern,
                model_names=args.target_models
            )

            print(f"\n✅ 批量综合结果计算完成!")
            print(f"📊 处理了 {batch_results['success_count']}/{batch_results['total_count']} 个实验")

        except Exception as e:
            print(f"❌ 综合结果计算失败: {e}")

    else:
        # 正常运行综合测试模式
        results = comprehensive.run_comprehensive_test(
            model_group=args.model_group,
            dataset_group=args.dataset_group,
            task_group=args.task_group,
            downstream_model=args.downstream,
            num_steps=args.steps,
            epochs=args.epochs,
            skip_completed=not args.no_skip,
            dry_run=args.dry_run
        )


if __name__ == '__main__':
    main()


"""
使用示例：

1. 正常运行comprehensive benchmark（包含过程结果和综合结果）：
   python scripts/comprehensive_benchmark.py --model-group networkit --dataset-group all_non_ogb --task-group all --downstream gcn --device cuda --epochs 30 --steps 10

2. 仅计算综合结果（从已保存的过程结果）：
   python scripts/comprehensive_benchmark.py --compute-comprehensive-only --results-dir ./results/comprehensive_benchmark

3. 计算特定实验的综合结果：
   python scripts/comprehensive_benchmark.py --compute-comprehensive-only --experiment-pattern "Cora_*_gcn" --results-dir ./results/comprehensive_benchmark

4. 计算特定模型的综合结果：
   python scripts/comprehensive_benchmark.py --compute-comprehensive-only --target-models gradient_based networkit_local_similarity --results-dir ./results/comprehensive_benchmark

这个实现覆盖当前综合 benchmark 的主要要求：
- ✅ 实验设置分门别类
- ✅ 单独的过程结果计算和保存（TSV + CSV + PNG）
- ✅ 跳过已完成的实验
- ✅ 单独的综合结果计算接口
- ✅ IC 曲线对比图和 IC-AUC 表格
"""
