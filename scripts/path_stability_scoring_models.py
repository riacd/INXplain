#!/usr/bin/env python3
"""
Measure pruning-path stability when the gradient-based scorer uses GCN vs GAT,
and optionally sweep the subset-sampling repeat count used at each pruning step.

The benchmark evaluator is intentionally not used here. This script isolates the
pruning trajectory itself: for each method and seed it generates one path with
GCN scoring and one path with GAT scoring, then compares retained/deleted edge
sets and edge deletion times across the two paths.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from GS.datasets import DatasetLoader
from GS.models import model_registry


Edge = Tuple[int, int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def undirected_edge_set(edge_index: torch.Tensor) -> Set[Edge]:
    edges = set()
    edge_index = edge_index.detach().cpu()
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0, i].item())
        dst = int(edge_index[1, i].item())
        if src == dst:
            continue
        if src > dst:
            src, dst = dst, src
        edges.add((src, dst))
    return edges


def jaccard(left: Set[Edge], right: Set[Edge]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) == 0:
        return float("nan")
    left_ranks = rankdata(left)
    right_ranks = rankdata(right)
    left_std = left_ranks.std()
    right_std = right_ranks.std()
    if left_std == 0 or right_std == 0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def deletion_times(path_sets: Sequence[Set[Edge]], original_edges: Set[Edge]) -> Dict[Edge, int]:
    times = {}
    num_steps = len(path_sets) - 1
    for edge in original_edges:
        deleted_at = num_steps + 1
        for step in range(1, len(path_sets)):
            if edge not in path_sets[step]:
                deleted_at = step
                break
        times[edge] = deleted_at
    return times


def path_metrics(
    gcn_sets: Sequence[Set[Edge]],
    gat_sets: Sequence[Set[Edge]],
    original_edges: Set[Edge]
) -> Dict:
    retained_jaccard = []
    deleted_jaccard = []
    complexity = []

    total_edges = max(1, len(original_edges))
    for step in range(len(gcn_sets)):
        retained_jaccard.append(jaccard(gcn_sets[step], gat_sets[step]))
        complexity.append((len(gcn_sets[step]) + len(gat_sets[step])) / (2.0 * total_edges))
        if step == 0:
            deleted_jaccard.append(1.0)
        else:
            gcn_deleted = gcn_sets[step - 1] - gcn_sets[step]
            gat_deleted = gat_sets[step - 1] - gat_sets[step]
            deleted_jaccard.append(jaccard(gcn_deleted, gat_deleted))

    mid_indices = [
        idx for idx, value in enumerate(complexity)
        if 0.2 <= value <= 0.8 and idx not in (0, len(complexity) - 1)
    ]
    if mid_indices:
        aupa_mid = float(np.mean([retained_jaccard[idx] for idx in mid_indices]))
        deleted_aupa_mid = float(np.mean([deleted_jaccard[idx] for idx in mid_indices]))
    else:
        aupa_mid = float(np.mean(retained_jaccard[1:-1])) if len(retained_jaccard) > 2 else float("nan")
        deleted_aupa_mid = float(np.mean(deleted_jaccard[1:-1])) if len(deleted_jaccard) > 2 else float("nan")

    gcn_times = deletion_times(gcn_sets, original_edges)
    gat_times = deletion_times(gat_sets, original_edges)
    ordered_edges = sorted(original_edges)
    deletion_time_spearman = spearman(
        [gcn_times[edge] for edge in ordered_edges],
        [gat_times[edge] for edge in ordered_edges],
    )

    return {
        "retained_jaccard_by_step": retained_jaccard,
        "deleted_jaccard_by_step": deleted_jaccard,
        "complexity_by_step": complexity,
        "aupa_mid": aupa_mid,
        "deleted_aupa_mid": deleted_aupa_mid,
        "deletion_time_spearman": deletion_time_spearman,
        "final_non_empty_retained_jaccard": retained_jaccard[-2] if len(retained_jaccard) >= 2 else float("nan"),
        "mid_step_indices": mid_indices,
    }


def random_path(
    original_edges: Set[Edge],
    counts_by_step: Sequence[int],
    rng: np.random.Generator
) -> List[Set[Edge]]:
    ordered_edges = sorted(original_edges)
    permutation = rng.permutation(len(ordered_edges))
    deletion_rank = {
        ordered_edges[int(edge_idx)]: rank
        for rank, edge_idx in enumerate(permutation)
    }
    path = []
    for count in counts_by_step:
        kept = {
            edge for edge in ordered_edges
            if deletion_rank[edge] >= len(ordered_edges) - count
        }
        path.append(kept)
    return path


def random_baseline_metrics(
    original_edges: Set[Edge],
    gcn_counts: Sequence[int],
    gat_counts: Sequence[int],
    repeats: int,
    seed: int
) -> Dict:
    rng = np.random.default_rng(seed)
    metrics = []
    for _ in range(repeats):
        left = random_path(original_edges, gcn_counts, rng)
        right = random_path(original_edges, gat_counts, rng)
        metrics.append(path_metrics(left, right, original_edges))

    keys = ["aupa_mid", "deleted_aupa_mid", "deletion_time_spearman", "final_non_empty_retained_jaccard"]
    summary = {}
    for key in keys:
        values = np.asarray([item[key] for item in metrics], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    return summary


def save_path(path_sets: Sequence[Set[Edge]], output_path: Path) -> None:
    serializable = [
        [[src, dst] for src, dst in sorted(edge_set)]
        for edge_set in path_sets
    ]
    output_path.write_text(json.dumps(serializable) + "\n")


def generate_path(
    model_name: str,
    scoring_model: str,
    seed: int,
    graph: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    num_steps: int,
    train_epochs: int,
    sampling_repeats: int,
    sampling_subset_num: Optional[int],
    stability_penalty: Optional[float],
    device: torch.device,
) -> Tuple[List[Set[Edge]], float]:
    set_seed(seed)
    model_kwargs = {
        "input_dim": graph.x.size(1),
        "downstream_model_type": scoring_model,
        "train_epochs": train_epochs,
        "device": device,
    }
    if "joint" in model_name:
        model_kwargs["sampling_repeats"] = sampling_repeats
        if sampling_subset_num is not None:
            model_kwargs["sampling_subset_num"] = sampling_subset_num
        model_kwargs["sampling_seed"] = seed
    if stability_penalty is not None and model_name.endswith("_stable"):
        model_kwargs["stability_penalty"] = stability_penalty
    model = model_registry.create_model(model_name, **model_kwargs)
    model.train_mask = train_mask
    model.val_mask = val_mask
    model.labels = graph.y

    start = time.time()
    summary_graphs = model.summarize(graph, num_steps=num_steps)
    elapsed = time.time() - start
    return [undirected_edge_set(item.edge_index) for item in summary_graphs], elapsed


def aggregate_records(records: Sequence[Dict]) -> List[Dict]:
    grouped = {}
    for record in records:
        grouped.setdefault(record["model"], []).append(record)

    summary = []
    keys = ["aupa_mid", "deleted_aupa_mid", "deletion_time_spearman", "final_non_empty_retained_jaccard"]
    for model, items in grouped.items():
        row = {"model": model, "count": len(items)}
        for key in keys:
            values = np.asarray([item["metrics"][key] for item in items], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values))
            random_values = np.asarray([
                item["random_baseline"][f"{key}_mean"]
                for item in items
            ], dtype=float)
            row[f"random_{key}_mean"] = float(np.nanmean(random_values))
        row["gcn_time_mean"] = float(np.mean([item["gcn_time"] for item in items]))
        row["gat_time_mean"] = float(np.mean([item["gat_time"] for item in items]))
        summary.append(row)
    return summary


def aggregate_records_by_sampling_repeats(records: Sequence[Dict]) -> List[Dict]:
    grouped = {}
    for record in records:
        key = (record.get("sampling_subset_num"), record["sampling_repeats"], record["model"])
        grouped.setdefault(key, []).append(record)

    summary = []
    keys = ["aupa_mid", "deleted_aupa_mid", "deletion_time_spearman", "final_non_empty_retained_jaccard"]
    for (sampling_subset_num, sampling_repeats, model), items in sorted(
        grouped.items(),
        key=lambda item: (
            -1 if item[0][0] is None else item[0][0],
            item[0][1],
            item[0][2],
        )
    ):
        row = {
            "sampling_subset_num": sampling_subset_num,
            "sampling_repeats": sampling_repeats,
            "model": model,
            "count": len(items),
        }
        for key in keys:
            values = np.asarray([item["metrics"][key] for item in items], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(values))
            row[f"{key}_std"] = float(np.nanstd(values))
            random_values = np.asarray([
                item["random_baseline"][f"{key}_mean"]
                for item in items
            ], dtype=float)
            row[f"random_{key}_mean"] = float(np.nanmean(random_values))
        row["gcn_time_mean"] = float(np.mean([item["gcn_time"] for item in items]))
        row["gat_time_mean"] = float(np.mean([item["gat_time"] for item in items]))
        summary.append(row)
    return summary


def save_summary_tables(output_dir: Path, rows: List[Dict], prefix: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"
    tsv_path = output_dir / f"{prefix}.tsv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    df.to_csv(csv_path, index=False)
    df.to_csv(tsv_path, index=False, sep="\t")


def run_path_stability_for_sampling_repeats(
    args: argparse.Namespace,
    sampling_subset_num: Optional[int],
    sampling_repeats: int,
    base_output_dir: Path,
    graph: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
) -> Dict:
    subset_dir_name = f"subset_num_{sampling_subset_num}" if sampling_subset_num is not None else "subset_num_all"
    output_dir = base_output_dir / subset_dir_name / f"sampling_repeats_{sampling_repeats}"
    paths_dir = output_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for model_name in args.models:
        for seed in args.seeds:
            print(f"\n===== repeats={sampling_repeats} | {model_name} seed={seed}: GCN scoring =====", flush=True)
            gcn_sets, gcn_time = generate_path(
                model_name=model_name,
                scoring_model="gcn",
                seed=seed,
                graph=graph,
                train_mask=train_mask,
                val_mask=val_mask,
                num_steps=args.num_steps,
                train_epochs=args.train_epochs,
                sampling_repeats=sampling_repeats,
                sampling_subset_num=sampling_subset_num,
                stability_penalty=args.stability_penalty,
                device=device,
            )
            save_path(gcn_sets, paths_dir / f"{model_name}_seed_{seed}_scoring_gcn.json")

            print(f"\n===== repeats={sampling_repeats} | {model_name} seed={seed}: GAT scoring =====", flush=True)
            gat_sets, gat_time = generate_path(
                model_name=model_name,
                scoring_model="gat",
                seed=seed,
                graph=graph,
                train_mask=train_mask,
                val_mask=val_mask,
                num_steps=args.num_steps,
                train_epochs=args.train_epochs,
                sampling_repeats=sampling_repeats,
                sampling_subset_num=sampling_subset_num,
                stability_penalty=args.stability_penalty,
                device=device,
            )
            save_path(gat_sets, paths_dir / f"{model_name}_seed_{seed}_scoring_gat.json")

            original_edges = gcn_sets[0]
            metrics = path_metrics(gcn_sets, gat_sets, original_edges)
            random_metrics = random_baseline_metrics(
                original_edges=original_edges,
                gcn_counts=[len(item) for item in gcn_sets],
                gat_counts=[len(item) for item in gat_sets],
                repeats=args.random_repeats,
                seed=seed + 100000,
            )
            record = {
                "model": model_name,
                "seed": seed,
                "dataset": args.dataset,
                "task": args.task,
                "num_steps": args.num_steps,
                "train_epochs": args.train_epochs,
                "sampling_subset_num": sampling_subset_num,
                "sampling_repeats": sampling_repeats,
                "random_repeats": args.random_repeats,
                "gcn_time": gcn_time,
                "gat_time": gat_time,
                "metrics": metrics,
                "random_baseline": random_metrics,
            }
            records.append(record)
            (output_dir / "records.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record, indent=2), flush=True)

    summary = aggregate_records(records)
    result = {
        "config": {
            **vars(args),
            "sampling_subset_num": sampling_subset_num,
            "sampling_repeats": sampling_repeats,
        },
        "summary": summary,
        "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("\n===== Path stability summary =====")
    print(json.dumps(summary, indent=2))
    return {
        "sampling_subset_num": sampling_subset_num,
        "sampling_repeats": sampling_repeats,
        "output_dir": str(output_dir),
        "result": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[
        "gradient_based_joint_subset_best",
        "gradient_based_joint_edge_score",
        "gradient_based_joint_edge_score_stable",
        "gradient_based_joint_product_importance",
    ])
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--task", default="original")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--sampling-repeats", type=int, default=5)
    parser.add_argument("--sampling-repeats-values", nargs="+", type=int, default=None)
    parser.add_argument("--sampling-subset-num", type=int, default=None)
    parser.add_argument("--sampling-subset-num-values", nargs="+", type=int, default=None)
    parser.add_argument("--random-repeats", type=int, default=50)
    parser.add_argument("--stability-penalty", type=float, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", default="results/path_stability_scoring_models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader()
    graph, train_mask, val_mask, _ = loader.load_dataset(args.dataset, task_type=args.task)
    graph = loader.preprocess_for_summarization(graph, to_undirected_graph=True).to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    sampling_repeats_values = args.sampling_repeats_values or [args.sampling_repeats]
    sampling_subset_num_values = (
        args.sampling_subset_num_values
        if args.sampling_subset_num_values is not None
        else [args.sampling_subset_num]
    )
    sweep_runs = []
    for sampling_subset_num in sampling_subset_num_values:
        for sampling_repeats in sampling_repeats_values:
            print(
                f"\n########## sampling_subset_num={sampling_subset_num or 'all'} "
                f"| sampling_repeats={sampling_repeats} ##########",
                flush=True,
            )
            sweep_runs.append(
                run_path_stability_for_sampling_repeats(
                    args=args,
                    sampling_subset_num=sampling_subset_num,
                    sampling_repeats=sampling_repeats,
                    base_output_dir=output_dir,
                    graph=graph,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    device=device,
                )
            )

    if len(sweep_runs) > 1:
        combined_records = []
        for run in sweep_runs:
            combined_records.extend(run["result"]["records"])
        combined_summary = aggregate_records_by_sampling_repeats(combined_records)
        combined_result = {
            "config": {
                **vars(args),
                "sampling_subset_num_values": sampling_subset_num_values,
                "sampling_repeats_values": sampling_repeats_values,
            },
            "summary": combined_summary,
            "records": combined_records,
            "runs": [
                {
                    "sampling_subset_num": run["sampling_subset_num"],
                    "sampling_repeats": run["sampling_repeats"],
                    "output_dir": run["output_dir"],
                    "summary": run["result"]["summary"],
                }
                for run in sweep_runs
            ],
        }
        (output_dir / "sampling_repeats_sweep_summary.json").write_text(
            json.dumps(combined_result, indent=2) + "\n"
        )
        save_summary_tables(output_dir, combined_summary, "sampling_repeats_sweep_summary")
        print("\n===== Sampling repeats sweep summary =====")
        print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
