#!/usr/bin/env python3
"""
统一基准测试脚本

使用模型注册机制，支持开发模型和baseline模型的统一测试和比较。
所有模型都实现相同的GraphSummarizationModel接口，确保公平比较。

使用示例:
    # 列出所有可用模型
    python scripts/run_unified_benchmark.py --list-models
    
    # 测试单个模型
    python scripts/run_unified_benchmark.py --model gradient_based --dataset Cora
    
    # 比较开发模型
    python scripts/run_unified_benchmark.py --compare-dev --dataset Cora
    
    # 比较baseline模型  
    python scripts/run_unified_benchmark.py --compare-baselines --dataset Cora
    
    # 比较所有模型
    python scripts/run_unified_benchmark.py --compare-all --dataset Cora
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from GS.benchmark.unified import UnifiedBenchmark
from GS.models import model_registry


def build_downstream_kwargs(args) -> dict:
    """Build downstream model kwargs from CLI options."""
    kwargs = {}

    if args.downstream == 'gat' and args.downstream_preset == 'gat_tuned':
        kwargs.update({
            'hidden_dim': 8,
            'heads': 8,
            'dropout': 0.6,
            'attention_dropout': 0.6,
            'negative_slope': 0.2,
            'lr': 0.005,
            'weight_decay': 5e-4,
            'early_stopping_patience': 100,
        })

    if args.downstream_hidden_dim is not None:
        kwargs['hidden_dim'] = args.downstream_hidden_dim
    if args.downstream_dropout is not None:
        kwargs['dropout'] = args.downstream_dropout
    if args.attention_dropout is not None:
        kwargs['attention_dropout'] = args.attention_dropout
    if args.negative_slope is not None:
        kwargs['negative_slope'] = args.negative_slope
    if args.downstream_lr is not None:
        kwargs['lr'] = args.downstream_lr
    if args.downstream_weight_decay is not None:
        kwargs['weight_decay'] = args.downstream_weight_decay
    if args.downstream_early_stopping_patience is not None:
        kwargs['early_stopping_patience'] = args.downstream_early_stopping_patience
    if args.downstream == 'gat' and args.gat_heads is not None:
        kwargs['heads'] = args.gat_heads

    return kwargs


def build_model_kwargs(args) -> dict:
    """Build graph summarization model kwargs from CLI options."""
    kwargs = {}

    if args.scoring_downstream is not None:
        kwargs['downstream_model_type'] = args.scoring_downstream

    if args.scoring_downstream == 'gat' and args.scoring_preset == 'gat_tuned':
        kwargs.update({
            'hidden_dim': 8,
            'gat_heads': 8,
            'dropout': 0.6,
            'attention_dropout': 0.6,
            'negative_slope': 0.2,
            'lr': 0.005,
            'weight_decay': 5e-4,
            'early_stopping_patience': 100,
        })

    if args.scoring_hidden_dim is not None:
        kwargs['hidden_dim'] = args.scoring_hidden_dim
    if args.scoring_train_epochs is not None:
        kwargs['train_epochs'] = args.scoring_train_epochs
    if args.scoring_gat_heads is not None:
        kwargs['gat_heads'] = args.scoring_gat_heads
    if args.scoring_dropout is not None:
        kwargs['dropout'] = args.scoring_dropout
    if args.scoring_attention_dropout is not None:
        kwargs['attention_dropout'] = args.scoring_attention_dropout
    if args.scoring_negative_slope is not None:
        kwargs['negative_slope'] = args.scoring_negative_slope
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


def list_models():
    """列出所有可用的模型"""
    print("="*80)
    print("可用的Graph Summarization模型")
    print("="*80)
    
    dev_models = model_registry.list_development_models()
    baseline_models = model_registry.list_baseline_models()
    
    print(f"\n开发模型 ({len(dev_models)} 个):")
    for model in dev_models:
        info = model_registry.get_model_info(model)
        print(f"  - {model}: {info['description']}")
    
    print(f"\n基准模型 ({len(baseline_models)} 个):")
    for model in baseline_models:
        info = model_registry.get_model_info(model)  
        print(f"  - {model}: {info['description']}")
    
    print(f"\n总计: {len(dev_models) + len(baseline_models)} 个模型")


def test_single_model(model_name: str,
                     dataset: str = 'Cora',
                     task: str = 'original',
                     downstream: str = 'gcn',
                     num_steps: int = 10,
                     epochs: int = 30,
                     model_kwargs: Optional[Dict] = None,
                     downstream_kwargs: Optional[Dict] = None,
                     disable_adaptive_epochs: bool = False):
    """测试单个模型"""
    benchmark = UnifiedBenchmark(device='cuda' if 'cuda' in sys.argv else 'cpu')
    if disable_adaptive_epochs:
        benchmark.enable_memory_optimization = False
    
    result = benchmark.run_single_model(
        model_name=model_name,
        dataset_name=dataset,
        task_type=task,
        downstream_model=downstream,
        num_steps=num_steps,
        epochs=epochs,
        model_kwargs=model_kwargs,
        downstream_kwargs=downstream_kwargs
    )
    
    if result.get('success'):
        print(f"\n✅ 模型 {model_name} 测试完成")
        print(f"IC-AUC(additive): {result['ic_auc_additive']:.4f}")
    else:
        print(f"\n❌ 模型 {model_name} 测试失败: {result.get('error', 'Unknown error')}")


def compare_models(model_list: list,
                  dataset: str = 'Cora',
                  task: str = 'original',
                  downstream: str = 'gcn',
                  num_steps: int = 10,
                  epochs: int = 30,
                  model_kwargs: Optional[Dict] = None,
                  downstream_kwargs: Optional[Dict] = None,
                  disable_adaptive_epochs: bool = False):
    """比较多个模型"""
    if not model_list:
        print("❌ 没有指定要比较的模型")
        return
    
    benchmark = UnifiedBenchmark(device='cuda' if 'cuda' in sys.argv else 'cpu')
    if disable_adaptive_epochs:
        benchmark.enable_memory_optimization = False
    
    results = benchmark.compare_models(
        model_names=model_list,
        dataset_name=dataset,
        task_type=task,
        downstream_model=downstream,
        num_steps=num_steps,
        epochs=epochs,
        model_kwargs=model_kwargs,
        downstream_kwargs=downstream_kwargs
    )
    
    successful_count = sum(1 for r in results.values() if r.get('success'))
    print(f"\n✅ 成功测试 {successful_count}/{len(model_list)} 个模型")


def main():
    parser = argparse.ArgumentParser(
        description='统一基准测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 基本选项
    parser.add_argument('--list-models', action='store_true',
                       help='列出所有可用模型')
    
    parser.add_argument('--model', type=str,
                       help='测试单个模型')
    
    # 比较选项
    parser.add_argument('--compare-dev', action='store_true',
                       help='比较所有开发模型')
    
    parser.add_argument('--compare-baselines', action='store_true',
                       help='比较所有baseline模型')
    
    parser.add_argument('--compare-all', action='store_true',
                       help='比较所有模型')
    
    parser.add_argument('--compare-models', nargs='+',
                       help='比较指定的模型列表')
    
    # 实验参数
    parser.add_argument('--dataset',
                       type=str,
                       default='Cora',
                       help='数据集名称，支持内置数据集以及本地湖泊网络别名（如 HongL, XYH）')
    
    parser.add_argument('--downstream',
                       choices=['gcn', 'gat', 'sage'],
                       default='gcn',
                       help='下游任务模型 (默认: gcn)')

    parser.add_argument('--task',
                       choices=['original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality'],
                       default='original',
                       help='任务类型: original(原始标签), degree(度分布标签), degree_centrality(度中心性标签), pagerank(PageRank标签), closeness_centrality(接近中心性标签) (默认: original)')

    parser.add_argument('--num-steps', type=int, default=10,
                       help='图总结步数 (默认: 10)')

    parser.add_argument('--epochs', type=int, default=200,
                       help='下游任务训练轮数 (默认: 200)')
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
    parser.add_argument('--attention-dropout', type=float,
                       help='GAT attention dropout')
    parser.add_argument('--negative-slope', type=float,
                       help='GAT LeakyReLU negative slope')
    parser.add_argument('--downstream-lr', type=float,
                       help='下游模型学习率')
    parser.add_argument('--downstream-weight-decay', type=float,
                       help='下游模型 weight decay')
    parser.add_argument('--downstream-early-stopping-patience', type=int,
                       help='下游模型 early stopping patience')
    parser.add_argument('--scoring-downstream',
                       choices=['gcn', 'gat', 'sage'],
                       help='剪枝方法内部用于边/子集打分的下游模型；未设置时使用模型默认值')
    parser.add_argument('--scoring-preset',
                       choices=['default', 'gat_tuned'],
                       default='default',
                       help='剪枝 scoring model 参数预设 (默认: default)')
    parser.add_argument('--scoring-hidden-dim', type=int,
                       help='剪枝 scoring model 隐藏维度')
    parser.add_argument('--scoring-train-epochs', type=int,
                       help='剪枝 scoring model 每个剪枝 step 的训练轮数')
    parser.add_argument('--scoring-gat-heads', type=int,
                       help='剪枝 scoring GAT 注意力头数')
    parser.add_argument('--scoring-dropout', type=float,
                       help='剪枝 scoring model dropout')
    parser.add_argument('--scoring-attention-dropout', type=float,
                       help='剪枝 scoring GAT attention dropout')
    parser.add_argument('--scoring-negative-slope', type=float,
                       help='剪枝 scoring GAT LeakyReLU negative slope')
    parser.add_argument('--scoring-lr', type=float,
                       help='剪枝 scoring model 学习率')
    parser.add_argument('--scoring-weight-decay', type=float,
                       help='剪枝 scoring model weight decay')
    parser.add_argument('--scoring-early-stopping-patience', type=int,
                       help='剪枝 scoring model early stopping patience')
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
                       help='禁用benchmark的自适应降轮逻辑')
    
    parser.add_argument('--device',
                       choices=['cpu', 'cuda'],
                       default='cuda',
                       help='计算设备 (默认: cuda)')
    
    args = parser.parse_args()
    downstream_kwargs = build_downstream_kwargs(args)
    model_kwargs = build_model_kwargs(args)
    
    print("Graph Summarization统一基准测试工具")
    print(f"数据集: {args.dataset}, 任务: {args.task}, 下游模型: {args.downstream}")
    print(f"步数: {args.num_steps}, 训练轮数: {args.epochs}")
    if downstream_kwargs:
        print(f"下游模型参数: {downstream_kwargs}")
    if model_kwargs:
        print(f"剪枝 scoring model 参数: {model_kwargs}")
    
    if args.list_models:
        list_models()
        
    elif args.model:
        test_single_model(
            args.model,
            args.dataset,
            args.task,
            args.downstream,
            args.num_steps,
            args.epochs,
            model_kwargs,
            downstream_kwargs,
            args.disable_adaptive_epochs
        )
        
    elif args.compare_dev:
        dev_models = model_registry.list_development_models()
        compare_models(dev_models, args.dataset, args.task, args.downstream, args.num_steps, args.epochs, model_kwargs, downstream_kwargs, args.disable_adaptive_epochs)

    elif args.compare_baselines:
        baseline_models = model_registry.list_baseline_models()
        compare_models(baseline_models, args.dataset, args.task, args.downstream, args.num_steps, args.epochs, model_kwargs, downstream_kwargs, args.disable_adaptive_epochs)

    elif args.compare_all:
        all_models = model_registry.list_models()
        compare_models(all_models, args.dataset, args.task, args.downstream, args.num_steps, args.epochs, model_kwargs, downstream_kwargs, args.disable_adaptive_epochs)

    elif args.compare_models:
        compare_models(args.compare_models, args.dataset, args.task, args.downstream, args.num_steps, args.epochs, model_kwargs, downstream_kwargs, args.disable_adaptive_epochs)
        
    else:
        print("请指定要执行的操作。使用 --help 查看帮助信息。")
        parser.print_help()


if __name__ == '__main__':
    main()
