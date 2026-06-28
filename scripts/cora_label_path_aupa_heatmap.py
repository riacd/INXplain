#!/usr/bin/env python3
"""
Generate a Cora task-label pruning-path consistency heatmap.

The compared label choices are DatasetLoader task types:
original, degree, degree_centrality, pagerank and closeness_centrality. Each
task type produces one pruning path with the selected summarization model for
each seed. Pairwise AUPA_mid matrices are averaged across seeds, and the mean
matrix is plotted as a heatmap.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from GS.datasets import DatasetLoader
from GS.models import model_registry
from scripts.path_stability_scoring_models import (
    path_metrics,
    save_path,
    set_seed,
    undirected_edge_set,
)


TASK_TYPES = [
    "original",
    "degree",
    "degree_centrality",
    "pagerank",
    "closeness_centrality",
]


def parse_task_types(values: Sequence[str]) -> List[str]:
    task_types = []
    for value in values:
        for item in value.split(","):
            task_type = item.strip()
            if not task_type:
                continue
            if task_type not in TASK_TYPES:
                raise argparse.ArgumentTypeError(
                    f"Unsupported task type {task_type!r}. Choices: {TASK_TYPES}"
                )
            task_types.append(task_type)
    if not task_types:
        raise argparse.ArgumentTypeError("must contain at least one task type")
    return task_types


def label_counts(labels: torch.Tensor) -> Dict[str, int]:
    counts = torch.bincount(labels.detach().cpu().long())
    return {str(idx): int(value.item()) for idx, value in enumerate(counts)}


def load_task_graphs(
    dataset: str,
    task_types: Sequence[str],
    device: torch.device,
) -> List[Dict]:
    loader = DatasetLoader()
    tasks = []
    for task_type in task_types:
        graph, train_mask, val_mask, test_mask = loader.load_dataset(dataset, task_type=task_type)
        graph = loader.preprocess_for_summarization(graph, to_undirected_graph=True).to(device)
        train_mask = train_mask.detach().cpu().bool()
        val_mask = val_mask.detach().cpu().bool()
        test_mask = test_mask.detach().cpu().bool()

        if int(train_mask.sum().item()) == 0 or int(val_mask.sum().item()) == 0:
            raise ValueError(
                f"Task {task_type} has train={int(train_mask.sum().item())}, "
                f"val={int(val_mask.sum().item())}; cannot score validation-loss paths."
            )

        tasks.append({
            "task_type": task_type,
            "display_name": task_type,
            "graph": graph,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "label_counts": label_counts(graph.y),
        })
    return tasks


def generate_task_path(
    model_name: str,
    task: Dict,
    seed: int,
    num_steps: int,
    train_epochs: int,
    scoring_model: str,
    sampling_repeats: int,
    sampling_subset_num: Optional[int],
    stability_penalty: Optional[float],
    device: torch.device,
) -> Tuple[List[set], float]:
    set_seed(seed)
    graph = task["graph"]
    model_kwargs = {
        "input_dim": graph.x.size(1),
        "downstream_model_type": scoring_model,
        "train_epochs": train_epochs,
        "device": device,
    }
    if "joint" in model_name:
        model_kwargs["sampling_repeats"] = sampling_repeats
        model_kwargs["sampling_seed"] = seed
        if sampling_subset_num is not None:
            model_kwargs["sampling_subset_num"] = sampling_subset_num
    if stability_penalty is not None and model_name.endswith("_stable"):
        model_kwargs["stability_penalty"] = stability_penalty

    model = model_registry.create_model(model_name, **model_kwargs)
    model.train_mask = task["train_mask"].to(device)
    model.val_mask = task["val_mask"].to(device)
    model.labels = graph.y.to(device)

    start_time = time.time()
    summary_graphs = model.summarize(graph, num_steps=num_steps)
    elapsed = time.time() - start_time
    return [undirected_edge_set(item.edge_index) for item in summary_graphs], elapsed


def compute_pairwise_aupa_matrix(
    paths: Sequence[Sequence[set]],
    original_edges: set,
) -> Tuple[np.ndarray, List[List[Dict]]]:
    count = len(paths)
    matrix = np.zeros((count, count), dtype=float)
    details: List[List[Dict]] = [[{} for _ in range(count)] for _ in range(count)]
    for i in range(count):
        for j in range(count):
            metrics = path_metrics(paths[i], paths[j], original_edges)
            matrix[i, j] = 1.0 if i == j else metrics["aupa_mid"]
            details[i][j] = metrics
    return matrix, details


def save_matrix_table(
    output_dir: Path,
    stem: str,
    matrix: np.ndarray,
    task_types: Sequence[str],
) -> None:
    df = pd.DataFrame(matrix, index=task_types, columns=task_types)
    df.index.name = "task_type"
    df.to_csv(output_dir / f"{stem}.csv")
    df.to_csv(output_dir / f"{stem}.tsv", sep="\t")


def save_pairwise_metrics(
    output_dir: Path,
    task_types: Sequence[str],
    seeded_details: Sequence[Dict],
    mean_matrix: np.ndarray,
    std_matrix: np.ndarray,
) -> None:
    rows = []
    for seed_result in seeded_details:
        seed = seed_result["seed"]
        details = seed_result["details"]
        for i, left_task in enumerate(task_types):
            for j, right_task in enumerate(task_types):
                metrics = details[i][j]
                rows.append({
                    "seed": seed,
                    "left_task_type": left_task,
                    "right_task_type": right_task,
                    "aupa_mid": metrics["aupa_mid"],
                    "deleted_aupa_mid": metrics["deleted_aupa_mid"],
                    "deletion_time_spearman": metrics["deletion_time_spearman"],
                    "final_non_empty_retained_jaccard": metrics["final_non_empty_retained_jaccard"],
                })
    pd.DataFrame(rows).to_csv(output_dir / "pairwise_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "pairwise_metrics.tsv", index=False, sep="\t")

    summary_rows = []
    for i, left_task in enumerate(task_types):
        for j, right_task in enumerate(task_types):
            summary_rows.append({
                "left_task_type": left_task,
                "right_task_type": right_task,
                "aupa_mid_mean": mean_matrix[i, j],
                "aupa_mid_std": std_matrix[i, j],
            })
    pd.DataFrame(summary_rows).to_csv(output_dir / "pairwise_metrics_summary.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "pairwise_metrics_summary.tsv", index=False, sep="\t")


def plot_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_dir: Path,
) -> None:
    display_labels = [
        "PageRank" if label.lower() == "pagerank" else label.replace("_", " ").title()
        for label in labels
    ]
    trajectory_cmap = LinearSegmentedColormap.from_list(
        "igprune_trajectory_agreement",
        ["#718699", "#F2D675"],
    )
    size = max(6.5, 0.7 * len(labels) + 3.5)
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(matrix, cmap=trajectory_cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Task Label", fontsize=13)
    ax.set_ylabel("Task Label", fontsize=13)
    ax.set_title(title, fontsize=16, pad=10)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if value < 0.55 else "black"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Trajectory Agreement", fontsize=13)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(output_dir / "aupa_mid_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "aupa_mid_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--task-types", nargs="+", default=TASK_TYPES, type=str,
                        help="Task labels to compare. Accepts space-separated names or comma-separated lists.")
    parser.add_argument("--model", default="gradient_based_joint_edge_score_stable")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed kept for backward compatibility. Ignored when --seeds is set.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--scoring-model", choices=["gcn", "gat", "sage", "h2gcn", "gcnii"], default="gcn")
    parser.add_argument("--sampling-repeats", type=int, default=3)
    parser.add_argument("--sampling-subset-num", type=int, default=None)
    parser.add_argument("--stability-penalty", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", default="results/cora_task_label_path_aupa_heatmap")
    args = parser.parse_args()
    args.task_types = parse_task_types(args.task_types)
    if args.seeds is None:
        args.seeds = [args.seed if args.seed is not None else 42]
    return args


def main() -> None:
    args = parse_args()
    if args.num_steps < 2:
        raise ValueError("AUPA_mid needs at least two pruning steps; set --num-steps >= 2.")
    set_seed(args.seeds[0])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    paths_dir = output_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_task_graphs(args.dataset, args.task_types, device)
    config = vars(args).copy()
    config["device_resolved"] = str(device)
    config["task_label_counts"] = {
        task["task_type"]: task["label_counts"] for task in tasks
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    task_types = [task["task_type"] for task in tasks]
    matrices = []
    seeded_details = []
    path_records = []

    for seed in args.seeds:
        seed_paths_dir = paths_dir / f"seed_{seed}"
        seed_paths_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for task in tasks:
            task_type = task["task_type"]
            print(f"\n===== {args.dataset} seed={seed} task_type={task_type} =====", flush=True)
            path_sets, elapsed = generate_task_path(
                model_name=args.model,
                task=task,
                seed=seed,
                num_steps=args.num_steps,
                train_epochs=args.train_epochs,
                scoring_model=args.scoring_model,
                sampling_repeats=args.sampling_repeats,
                sampling_subset_num=args.sampling_subset_num,
                stability_penalty=args.stability_penalty,
                device=device,
            )
            paths.append(path_sets)
            save_path(path_sets, seed_paths_dir / f"{task_type}.json")
            path_records.append({
                "seed": seed,
                "task_type": task_type,
                "train_nodes": int(task["train_mask"].sum().item()),
                "val_nodes": int(task["val_mask"].sum().item()),
                "test_nodes": int(task["test_mask"].sum().item()),
                "label_counts": task["label_counts"],
                "path_time_seconds": elapsed,
                "edge_counts_by_step": [len(edge_set) for edge_set in path_sets],
            })
            (output_dir / "path_records.json").write_text(json.dumps(path_records, indent=2) + "\n")

        original_edges = paths[0][0]
        matrix, details = compute_pairwise_aupa_matrix(paths, original_edges)
        matrices.append(matrix)
        seeded_details.append({"seed": seed, "details": details})
        save_matrix_table(seed_paths_dir, "aupa_mid_matrix", matrix, task_types)

    matrix_stack = np.stack(matrices, axis=0)
    mean_matrix = np.nanmean(matrix_stack, axis=0)
    std_matrix = np.nanstd(matrix_stack, axis=0)
    save_matrix_table(output_dir, "aupa_mid_matrix", mean_matrix, task_types)
    save_matrix_table(output_dir, "aupa_mid_mean_matrix", mean_matrix, task_types)
    save_matrix_table(output_dir, "aupa_mid_std_matrix", std_matrix, task_types)
    save_pairwise_metrics(output_dir, task_types, seeded_details, mean_matrix, std_matrix)

    title = "Cross-label trajectory agreement"
    plot_heatmap(mean_matrix, task_types, title, output_dir)

    off_diag = mean_matrix[~np.eye(mean_matrix.shape[0], dtype=bool)]
    std_off_diag = std_matrix[~np.eye(std_matrix.shape[0], dtype=bool)]
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "seeds": args.seeds,
        "num_seeds": len(args.seeds),
        "task_types": task_types,
        "num_task_types": len(task_types),
        "mean_off_diagonal_aupa_mid": float(np.nanmean(off_diag)) if off_diag.size else float("nan"),
        "std_across_pairs_off_diagonal_aupa_mid": float(np.nanstd(off_diag)) if off_diag.size else float("nan"),
        "mean_seed_std_off_diagonal_aupa_mid": float(np.nanmean(std_off_diag)) if std_off_diag.size else float("nan"),
        "min_off_diagonal_aupa_mid": float(np.nanmin(off_diag)) if off_diag.size else float("nan"),
        "max_off_diagonal_aupa_mid": float(np.nanmax(off_diag)) if off_diag.size else float("nan"),
        "outputs": {
            "heatmap_png": str(output_dir / "aupa_mid_heatmap.png"),
            "heatmap_pdf": str(output_dir / "aupa_mid_heatmap.pdf"),
            "matrix_csv": str(output_dir / "aupa_mid_matrix.csv"),
            "mean_matrix_csv": str(output_dir / "aupa_mid_mean_matrix.csv"),
            "std_matrix_csv": str(output_dir / "aupa_mid_std_matrix.csv"),
            "pairwise_metrics_csv": str(output_dir / "pairwise_metrics.csv"),
            "pairwise_metrics_summary_csv": str(output_dir / "pairwise_metrics_summary.csv"),
            "paths_dir": str(paths_dir),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n===== Task-label AUPA heatmap summary =====")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
