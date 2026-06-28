#!/usr/bin/env python3
"""Analyze seed-to-seed pruning path stability from saved path JSON files."""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd

Edge = Tuple[int, int]


def load_path(path: Path) -> List[Set[Edge]]:
    data = json.loads(path.read_text())
    return [set(tuple(edge) for edge in step_edges) for step_edges in data]


def jaccard(left: Set[Edge], right: Set[Edge]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def deletion_times(path_sets: Sequence[Set[Edge]]) -> Dict[Edge, int]:
    original_edges = path_sets[0]
    final_step = len(path_sets) - 1
    times = {edge: final_step for edge in original_edges}
    previous = path_sets[0]
    for step_idx in range(1, len(path_sets)):
        removed = previous - path_sets[step_idx]
        for edge in removed:
            times[edge] = step_idx
        previous = path_sets[step_idx]
    return times


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return float("nan")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var == 0 or right_var == 0:
        return 1.0
    cov = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    return cov / math.sqrt(left_var * right_var)


def path_metrics(left: Sequence[Set[Edge]], right: Sequence[Set[Edge]]) -> Dict[str, float]:
    original_edges = left[0]
    num_steps = len(left)
    mid_indices = list(range(max(1, num_steps // 3), min(num_steps - 1, (2 * num_steps) // 3) + 1))

    retained_jaccard = [jaccard(left[idx], right[idx]) for idx in range(num_steps)]
    deleted_jaccard = [
        jaccard(original_edges - left[idx], original_edges - right[idx])
        for idx in range(num_steps)
    ]

    final_non_empty_idx = 0
    for idx in range(1, num_steps):
        if left[idx] and right[idx]:
            final_non_empty_idx = idx

    left_times = deletion_times(left)
    right_times = deletion_times(right)
    ordered_edges = sorted(original_edges)
    deletion_corr = pearson(
        [left_times[edge] for edge in ordered_edges],
        [right_times[edge] for edge in ordered_edges],
    )

    return {
        "aupa_mid": sum(retained_jaccard[idx] for idx in mid_indices) / len(mid_indices),
        "deleted_aupa_mid": sum(deleted_jaccard[idx] for idx in mid_indices) / len(mid_indices),
        "deletion_time_corr": deletion_corr,
        "final_non_empty_retained_jaccard": retained_jaccard[final_non_empty_idx],
    }


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return (
        float(sum(values) / len(values)),
        float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--sampling-repeats-values", nargs="+", type=int, required=True)
    parser.add_argument("--sampling-subset-num-values", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--scoring-models", nargs="+", default=["gcn", "gat"])
    args = parser.parse_args()

    rows = []
    for sampling_subset_num in args.sampling_subset_num_values:
        subset_dir_name = (
            f"subset_num_{sampling_subset_num}"
            if sampling_subset_num != "all"
            else "subset_num_all"
        )
        for sampling_repeats in args.sampling_repeats_values:
            nested_paths_dir = args.input_dir / subset_dir_name / f"sampling_repeats_{sampling_repeats}" / "paths"
            legacy_paths_dir = args.input_dir / f"sampling_repeats_{sampling_repeats}" / "paths"
            paths_dir = nested_paths_dir if nested_paths_dir.exists() else legacy_paths_dir
            for model in args.models:
                for scoring_model in args.scoring_models:
                    paths = {}
                    missing = []
                    for seed in args.seeds:
                        path = paths_dir / f"{model}_seed_{seed}_scoring_{scoring_model}.json"
                        if not path.exists():
                            missing.append(str(path))
                        else:
                            paths[seed] = load_path(path)
                    if missing:
                        rows.append({
                            "sampling_subset_num": sampling_subset_num,
                            "sampling_repeats": sampling_repeats,
                            "model": model,
                            "scoring_model": scoring_model,
                            "num_pairs": 0,
                            "missing": ";".join(missing),
                        })
                        continue

                    pair_metrics = [
                        path_metrics(paths[left_seed], paths[right_seed])
                        for left_seed, right_seed in itertools.combinations(args.seeds, 2)
                    ]
                    row = {
                        "sampling_subset_num": sampling_subset_num,
                        "sampling_repeats": sampling_repeats,
                        "model": model,
                        "scoring_model": scoring_model,
                        "num_pairs": len(pair_metrics),
                        "missing": "",
                    }
                    for metric_name in [
                        "aupa_mid",
                        "deleted_aupa_mid",
                        "deletion_time_corr",
                        "final_non_empty_retained_jaccard",
                    ]:
                        mean, std = mean_std([item[metric_name] for item in pair_metrics])
                        row[f"{metric_name}_mean"] = mean
                        row[f"{metric_name}_std"] = std
                    rows.append(row)

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_prefix.with_suffix(".tsv"), sep="\t", index=False)
    df.to_csv(output_prefix.with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
