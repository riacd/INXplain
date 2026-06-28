#!/usr/bin/env python3
"""Measure seed-to-seed pruning-path AUPA for Cora baselines."""

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from GS.datasets import DatasetLoader
from GS.models import model_registry


Edge = Tuple[int, int]
DEFAULT_BASELINES = [
    "networkit_forest_fire",
    "networkit_local_degree",
    "networkit_local_similarity",
    "networkit_random_edge",
    "networkit_random_node_edge",
    "networkit_scan",
    "networkit_simmelian",
    "pri_graphs",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def undirected_edge_set(edge_index: torch.Tensor) -> Set[Edge]:
    result = set()
    for src, dst in edge_index.detach().cpu().t().tolist():
        if src != dst:
            result.add((min(src, dst), max(src, dst)))
    return result


def save_path(path_sets: Sequence[Set[Edge]], output_path: Path) -> None:
    serializable = [
        [[src, dst] for src, dst in sorted(edge_set)]
        for edge_set in path_sets
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serializable) + "\n")


def load_path(input_path: Path) -> List[Set[Edge]]:
    return [
        {tuple(edge) for edge in step}
        for step in json.loads(input_path.read_text())
    ]


def jaccard(left: Set[Edge], right: Set[Edge]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def pair_metrics(left: Sequence[Set[Edge]], right: Sequence[Set[Edge]]) -> Dict:
    if len(left) != len(right):
        raise ValueError(f"Path lengths differ: {len(left)} != {len(right)}")
    if left[0] != right[0]:
        raise ValueError("Seed paths do not start from the same original edge set")

    total_edges = max(1, len(left[0]))
    retained_jaccard = [jaccard(a, b) for a, b in zip(left, right)]
    complexity = [
        (len(a) + len(b)) / (2.0 * total_edges)
        for a, b in zip(left, right)
    ]
    mid_indices = [
        idx for idx, value in enumerate(complexity)
        if 0.2 <= value <= 0.8 and idx not in (0, len(complexity) - 1)
    ]
    interior = list(range(1, max(1, len(retained_jaccard) - 1)))
    selected = mid_indices or interior
    return {
        "aupa_mid": float(np.mean([retained_jaccard[idx] for idx in selected])),
        "aupa_all_interior": float(np.mean([retained_jaccard[idx] for idx in interior])),
        "retained_jaccard_by_step": retained_jaccard,
        "complexity_by_step": complexity,
        "mid_step_indices": mid_indices,
    }


def create_model(model_name: str, seed: int, device: torch.device):
    kwargs = {"seed": seed, "device": str(device)}
    return model_registry.create_model(model_name, **kwargs)


def generate_path(
    model_name: str,
    seed: int,
    graph: Data,
    num_steps: int,
    device: torch.device,
) -> Tuple[List[Set[Edge]], float]:
    set_seed(seed)
    model = create_model(model_name, seed, device)
    start = time.time()
    summary_graphs = model.summarize(graph, num_steps=num_steps)
    elapsed = time.time() - start
    expected_length = num_steps + 1
    if len(summary_graphs) != expected_length:
        raise ValueError(
            f"{model_name} returned {len(summary_graphs)} graphs; expected {expected_length}"
        )
    return [undirected_edge_set(item.edge_index) for item in summary_graphs], elapsed


def aggregate(output_dir: Path, models: Sequence[str], seeds: Sequence[int]) -> List[Dict]:
    pair_rows = []
    summary_rows = []
    for model_name in models:
        paths = {}
        missing = []
        for seed in seeds:
            path_file = output_dir / "paths" / model_name / f"seed_{seed}.json"
            if path_file.exists():
                paths[seed] = load_path(path_file)
            else:
                missing.append(seed)

        if missing:
            summary_rows.append({
                "model": model_name,
                "num_seeds": len(paths),
                "num_pairs": 0,
                "missing_seeds": " ".join(map(str, missing)),
            })
            continue

        model_rows = []
        for left_seed, right_seed in itertools.combinations(seeds, 2):
            metrics = pair_metrics(paths[left_seed], paths[right_seed])
            row = {
                "model": model_name,
                "left_seed": left_seed,
                "right_seed": right_seed,
                **metrics,
            }
            pair_rows.append(row)
            model_rows.append(row)

        aupa_mid = np.asarray([row["aupa_mid"] for row in model_rows])
        aupa_all = np.asarray([row["aupa_all_interior"] for row in model_rows])
        summary_rows.append({
            "model": model_name,
            "num_seeds": len(seeds),
            "num_pairs": len(model_rows),
            "missing_seeds": "",
            "aupa_mid_mean": float(aupa_mid.mean()),
            "aupa_mid_std": float(aupa_mid.std(ddof=1)) if len(aupa_mid) > 1 else 0.0,
            "aupa_mid_min": float(aupa_mid.min()),
            "aupa_mid_max": float(aupa_mid.max()),
            "aupa_all_interior_mean": float(aupa_all.mean()),
            "aupa_all_interior_std": float(aupa_all.std(ddof=1)) if len(aupa_all) > 1 else 0.0,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_df = pd.DataFrame(pair_rows)
    summary_df = pd.DataFrame(summary_rows)
    pair_df.to_csv(output_dir / "pairwise_aupa.csv", index=False)
    pair_df.to_csv(output_dir / "pairwise_aupa.tsv", index=False, sep="\t")
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    summary_df.to_csv(output_dir / "summary.tsv", index=False, sep="\t")
    (output_dir / "pairwise_aupa.json").write_text(json.dumps(pair_rows, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    print(summary_df.to_string(index=False), flush=True)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--task", default="original")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cora_baseline_seed_aupa"))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) < 2:
        raise ValueError("At least two distinct seeds are required")
    unknown = sorted(set(args.models) - set(model_registry.list_models()))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "output_dir": str(args.output_dir), "device_used": str(device)}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    if not args.aggregate_only:
        loader = DatasetLoader()
        graph, _, _, _ = loader.load_dataset(args.dataset, task_type=args.task)
        graph = loader.preprocess_for_summarization(
            graph, to_undirected_graph=True
        ).to(device)
        runtime_rows = []
        for model_name in args.models:
            for seed in args.seeds:
                output_path = args.output_dir / "paths" / model_name / f"seed_{seed}.json"
                if output_path.exists() and not args.force:
                    print(f"SKIP {model_name} seed={seed}", flush=True)
                    continue
                print(f"RUN {model_name} seed={seed}", flush=True)
                path_sets, elapsed = generate_path(
                    model_name, seed, graph, args.num_steps, device
                )
                save_path(path_sets, output_path)
                runtime_rows.append({
                    "model": model_name,
                    "seed": seed,
                    "runtime_seconds": elapsed,
                })
                pd.DataFrame(runtime_rows).to_csv(
                    args.output_dir / "runtimes_latest_invocation.csv", index=False
                )
                aggregate(args.output_dir, args.models, args.seeds)

    aggregate(args.output_dir, args.models, args.seeds)


if __name__ == "__main__":
    main()
