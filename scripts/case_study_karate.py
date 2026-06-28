"""
KarateClub Case Study Script

支持在 KarateClub 数据集上运行开发模型或 baseline 模型进行图简化，
并自动保存每一步 summary graph 的可视化结果。
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GS.benchmark.unified import UnifiedBenchmark
from GS.models import model_registry


DEFAULT_TASK_TYPES = [
    "original",
    "degree",
    "degree_centrality",
    "pagerank",
    "closeness_centrality",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KarateClub case study for graph pruning and visualization."
    )
    parser.add_argument("--dataset", default="KarateClub", help="Dataset name (default: KarateClub)")
    parser.add_argument("--downstream-model", default="GCN", help="Downstream model (default: GCN)")
    parser.add_argument("--num-steps", type=int, default=10, help="Pruning steps (default: 10)")
    parser.add_argument("--epochs", type=int, default=100, help="Downstream training epochs (default: 100)")
    parser.add_argument("--device", default="cuda", help="Device, e.g., cuda/cpu (default: cuda)")
    parser.add_argument("--data-dir", default="./data", help="Dataset directory")
    parser.add_argument("--results-dir", default="./results/case_study_karate", help="Output directory")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASK_TYPES,
        choices=DEFAULT_TASK_TYPES,
        help="Task types to run",
    )

    parser.add_argument(
        "--model-group",
        choices=["single", "baseline", "development", "all"],
        default="single",
        help="Model selection mode (default: single)",
    )
    parser.add_argument(
        "--model",
        default="gradient_based",
        help="Model name used when --model-group=single (default: gradient_based)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit model list (overrides --model-group/--model)",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List all registered baseline models and exit",
    )
    return parser.parse_args()


def _resolve_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        return args.models
    if args.model_group == "baseline":
        return model_registry.list_baseline_models()
    if args.model_group == "development":
        return model_registry.list_development_models()
    if args.model_group == "all":
        return model_registry.list_models()
    return [args.model]


def main() -> int:
    args = _parse_args()

    if args.list_baselines:
        baselines = model_registry.list_baseline_models()
        print("Registered baseline models:")
        for name in baselines:
            info = model_registry.get_model_info(name)
            print(f"  - {name}: {info.get('description', '')}")
        return 0

    models = _resolve_models(args)
    if not models:
        print("❌ No models to run.")
        return 1

    available = set(model_registry.list_models())
    missing = [name for name in models if name not in available]
    if missing:
        print(f"❌ Unknown model(s): {missing}")
        print(f"Available models: {sorted(available)}")
        return 1

    benchmark = UnifiedBenchmark(
        results_dir=args.results_dir,
        device=args.device,
        data_dir=args.data_dir,
        random_seed=args.random_seed,
    )

    print("=" * 80)
    print("KarateClub Case Study")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(models)}")
    print(f"Downstream Model: {args.downstream_model}")
    print(f"Number of Steps: {args.num_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Results Dir: {args.results_dir}")
    print("=" * 80)

    total = len(models) * len(args.tasks)
    succeeded = 0
    failed = []

    for model_name in models:
        print(f"\n{'#' * 80}")
        print(f"Model: {model_name}")
        print(f"{'#' * 80}")

        for task_type in args.tasks:
            print(f"\n{'=' * 80}")
            print(f"Running task: {task_type}")
            print(f"{'=' * 80}\n")

            try:
                result = benchmark.run_single_model(
                    model_name=model_name,
                    dataset_name=args.dataset,
                    task_type=task_type,
                    downstream_model=args.downstream_model,
                    num_steps=args.num_steps,
                    epochs=args.epochs,
                    model_kwargs={},
                )

                if result and "error" not in result:
                    succeeded += 1
                    print(f"\n✅ [{model_name}] task={task_type} completed")
                    print(f"Results saved to: {result.get('exp_dir', 'N/A')}")
                else:
                    error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else "Unknown error"
                    failed.append((model_name, task_type, error_msg))
                    print(f"\n⚠️ [{model_name}] task={task_type} failed: {error_msg}")

            except Exception as exc:
                failed.append((model_name, task_type, str(exc)))
                print(f"\n❌ [{model_name}] task={task_type} exception: {exc}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("KarateClub Case Study Completed")
    print("=" * 80)
    print(f"Total runs: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {len(failed)}")
    print(f"Results location: {args.results_dir}")
    print("\nGenerated artifacts include:")
    print("  - process_results/*_step_metrics.tsv")
    print("  - summary_graphs/*_step_*_edges.csv")
    print("  - graph_visualizations/*_step_*_graph.png")
    print("  - process_results/*_ic_curve.png")
    print("  - comprehensive_results/*")

    if failed:
        print("\nFailed runs:")
        for model_name, task_type, error_msg in failed:
            print(f"  - model={model_name}, task={task_type}, error={error_msg}")

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
