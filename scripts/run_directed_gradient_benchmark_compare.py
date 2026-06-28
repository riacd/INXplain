#!/usr/bin/env python3
"""
Run directed-gradient benchmark experiments and merge the new results with
previously completed baseline comparisons.

This script reruns the original directed IGPrune implementation
(`gradient_based_original`) on selected dataset/task pairs while preserving
edge direction. It then combines the new directed result with baseline rows
from existing unified benchmark outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from GS.benchmark.unified import UnifiedBenchmark


DIRECTED_MODEL_NAME = "gradient_based_original"
MERGED_MODEL_NAME = "gradient_based_directed"
MERGED_CATEGORY = "directed"
MERGED_DESCRIPTION = "Directed IGPrune benchmark rerun using the original gradient-based implementation"


def parse_dataset_task_pairs(values: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid dataset-task pair '{value}'. Expected DATASET:TASK")
        dataset, task = value.split(":", 1)
        pairs.append((dataset, task))
    return pairs


def exp_id(dataset: str, task: str, downstream: str) -> str:
    return f"{dataset}_{task}_{downstream}"


def build_directed_row(result: dict) -> pd.DataFrame:
    row = {
        "model": MERGED_MODEL_NAME,
        "category": MERGED_CATEGORY,
        "description": MERGED_DESCRIPTION,
        "ic_auc_log_ratio": result["ic_auc_log_ratio"],
        "ic_auc_additive": result["ic_auc_additive"],
        "threshold_point_log_ratio": result.get("threshold_point_log_ratio"),
        "threshold_point_additive": result.get("threshold_point_additive"),
        "training_time_seconds": result["training_time"],
        "summarization_time_seconds": result["summarization_time"],
        "dataset": result["dataset"],
        "task_type": result["task_type"],
        "downstream_model": result["downstream_model"],
        "source_model_name": result["model_name"],
        "preserve_edge_direction": True,
    }
    return pd.DataFrame([row])


def build_threshold_row(result: dict) -> pd.DataFrame:
    row = {
        "model": MERGED_MODEL_NAME,
        "category": MERGED_CATEGORY,
        "threshold_point_log_ratio": result.get("threshold_point_log_ratio"),
        "threshold_point_additive": result.get("threshold_point_additive"),
        "dataset": result["dataset"],
        "task_type": result["task_type"],
        "downstream_model": result["downstream_model"],
        "source_model_name": result["model_name"],
        "preserve_edge_direction": True,
    }
    return pd.DataFrame([row])


def write_summary(merged_df: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("Rank\tModel\tCategory\tIC-AUC(Add)\tIC-AUC(Log)\tThreshold(Add)\tThreshold(Log)")
    for idx, (_, row) in enumerate(merged_df.iterrows(), start=1):
        lines.append(
            f"{idx}\t{row['model']}\t{row['category']}\t"
            f"{row.get('ic_auc_additive', float('nan')):.6f}\t"
            f"{row.get('ic_auc_log_ratio', float('nan')):.6f}\t"
            f"{row.get('threshold_point_additive', float('nan')):.6f}\t"
            f"{row.get('threshold_point_log_ratio', float('nan')):.6f}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def merge_with_baselines(
    directed_result: dict,
    baseline_root: Path,
    comparison_dir: Path,
) -> None:
    dataset = directed_result["dataset"]
    task = directed_result["task_type"]
    downstream = directed_result["downstream_model"]
    current_exp_id = exp_id(dataset, task, downstream)

    baseline_ic_auc_path = baseline_root / current_exp_id / "comprehensive_results" / "ic_auc_comparison.tsv"
    baseline_threshold_path = baseline_root / current_exp_id / "comprehensive_results" / "threshold_points_comparison.tsv"

    if not baseline_ic_auc_path.exists():
        raise FileNotFoundError(f"Missing baseline comparison table: {baseline_ic_auc_path}")
    if not baseline_threshold_path.exists():
        raise FileNotFoundError(f"Missing baseline threshold table: {baseline_threshold_path}")

    baseline_ic_auc_df = pd.read_csv(baseline_ic_auc_path, sep="\t")
    baseline_ic_auc_df = baseline_ic_auc_df[baseline_ic_auc_df["category"] == "baseline"].copy()
    baseline_threshold_df = pd.read_csv(baseline_threshold_path, sep="\t")
    baseline_threshold_df = baseline_threshold_df[baseline_threshold_df["category"] == "baseline"].copy()

    merged_ic_auc = pd.concat(
        [baseline_ic_auc_df, build_directed_row(directed_result)],
        ignore_index=True,
        sort=False,
    )
    merged_ic_auc = merged_ic_auc.sort_values("ic_auc_additive", ascending=False)

    merged_threshold = pd.concat(
        [baseline_threshold_df, build_threshold_row(directed_result)],
        ignore_index=True,
        sort=False,
    )
    merged_threshold["threshold_sort"] = merged_threshold["threshold_point_additive"].fillna(float("inf"))
    merged_threshold = merged_threshold.sort_values("threshold_sort", ascending=True).drop(columns=["threshold_sort"])

    comparison_dir.mkdir(parents=True, exist_ok=True)
    merged_ic_auc.to_csv(comparison_dir / "ic_auc_comparison.tsv", sep="\t", index=False)
    merged_threshold.to_csv(comparison_dir / "threshold_points_comparison.tsv", sep="\t", index=False)
    write_summary(merged_ic_auc, comparison_dir / "ranking_summary.tsv")

    metadata = {
        "dataset": dataset,
        "task_type": task,
        "downstream_model": downstream,
        "baseline_root": str(baseline_root / current_exp_id),
        "directed_model_name": directed_result["model_name"],
        "merged_model_name": MERGED_MODEL_NAME,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (comparison_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run directed IGPrune benchmark and merge with completed baseline results.")
    parser.add_argument(
        "--dataset-tasks",
        nargs="+",
        default=[
            "SO_relation_ME:original",
            "SO_relation_MT:original",
            "HongL:degree",
            "XYH:degree",
        ],
        help="Dataset/task pairs in DATASET:TASK format.",
    )
    parser.add_argument("--downstream", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT / "results" / "unified_benchmark")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "directed_gradient_benchmark")
    parser.add_argument("--skip-completed", action="store_true", help="Skip rerun if directed step metrics already exist.")
    args = parser.parse_args()

    pairs = parse_dataset_task_pairs(args.dataset_tasks)

    for dataset, task in pairs:
        current_exp_id = exp_id(dataset, task, args.downstream)
        print(f"\n{'=' * 100}")
        print(f"Directed rerun: dataset={dataset}, task={task}, downstream={args.downstream}")
        print(f"{'=' * 100}")

        directed_exp_dir = args.results_root / current_exp_id
        directed_metrics_path = directed_exp_dir / "process_results" / f"{DIRECTED_MODEL_NAME}_step_metrics.tsv"

        if args.skip_completed and directed_metrics_path.exists():
            print(f"⏭️  Skipping rerun because directed metrics already exist: {directed_metrics_path}")
            directed_ic_auc_path = directed_exp_dir / "directed_only_result.json"
            if not directed_ic_auc_path.exists():
                raise FileNotFoundError(
                    f"Directed result metadata missing for skipped run: {directed_ic_auc_path}"
                )
            directed_result = json.loads(directed_ic_auc_path.read_text())
        else:
            benchmark = UnifiedBenchmark(
                results_dir=str(args.results_root),
                device=args.device,
            )
            directed_result = benchmark.run_single_model(
                model_name=DIRECTED_MODEL_NAME,
                dataset_name=dataset,
                task_type=task,
                downstream_model=args.downstream,
                num_steps=args.num_steps,
                epochs=args.epochs,
                preserve_edge_direction=True,
            )
            if not directed_result.get("success"):
                raise RuntimeError(
                    f"Directed benchmark failed for {dataset}:{task}: {directed_result.get('error')}"
                )

            directed_only_summary = {
                "model_name": directed_result["model_name"],
                "dataset": directed_result["dataset"],
                "task_type": directed_result["task_type"],
                "downstream_model": directed_result["downstream_model"],
                "ic_auc_log_ratio": directed_result["ic_auc_log_ratio"],
                "ic_auc_additive": directed_result["ic_auc_additive"],
                "threshold_point_log_ratio": directed_result.get("threshold_point_log_ratio"),
                "threshold_point_additive": directed_result.get("threshold_point_additive"),
                "training_time": directed_result["training_time"],
                "summarization_time": directed_result["summarization_time"],
                "preserve_edge_direction": True,
                "success": True,
            }
            directed_exp_dir.mkdir(parents=True, exist_ok=True)
            (directed_exp_dir / "directed_only_result.json").write_text(
                json.dumps(directed_only_summary, indent=2) + "\n"
            )

        comparison_dir = directed_exp_dir / "comparison_with_existing_baselines"
        merge_with_baselines(directed_result, args.baseline_root, comparison_dir)
        print(f"✅ Comparison saved to: {comparison_dir}")


if __name__ == "__main__":
    main()
