#!/usr/bin/env python3
"""Plot per-edge appearance probabilities across repeated KarateClub pruning runs."""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from GS.datasets import DatasetLoader
from GS.models import model_registry


Edge = Tuple[int, int]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_LABEL_COLORS: Sequence[str] = ("#395B3F", "#E2DCD0", "#D3A228", "#BED7E6")
DEFAULT_REFERENCE_SUMMARY_DIR = (
    PROJECT_ROOT
    / "results"
    / "case_study_karate"
    / "KarateClub_original_GCN"
    / "summary_graphs"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def undirected_edge_set(edge_index: torch.Tensor) -> Set[Edge]:
    edges = set()
    for src, dst in edge_index.detach().cpu().t().tolist():
        if src == dst:
            continue
        edges.add((min(src, dst), max(src, dst)))
    return edges


def edge_set_to_index(edges: Set[Edge], device: torch.device) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    directed = [(src, dst) for edge in sorted(edges) for src, dst in (edge, edge[::-1])]
    return torch.tensor(directed, dtype=torch.long, device=device).t().contiguous()


def add_random_edges(
    graph: Data,
    fraction: float,
    seed: int,
) -> Tuple[Data, Set[Edge]]:
    """Add ceil(fraction * |E|) unique non-edges to an undirected graph."""
    original_edges = undirected_edge_set(graph.edge_index)
    add_count = int(math.ceil(len(original_edges) * fraction))
    candidates = [
        (src, dst)
        for src in range(graph.num_nodes)
        for dst in range(src + 1, graph.num_nodes)
        if (src, dst) not in original_edges
    ]
    if add_count > len(candidates):
        raise ValueError(
            f"Cannot add {add_count} edges: only {len(candidates)} non-edges are available."
        )

    rng = random.Random(seed)
    added_edges = set(rng.sample(candidates, add_count))
    perturbed = graph.clone()
    perturbed.edge_index = edge_set_to_index(
        original_edges | added_edges,
        graph.edge_index.device,
    )
    if hasattr(perturbed, "edge_attr"):
        perturbed.edge_attr = None
    return perturbed, added_edges


def aggregate_edge_probabilities(
    paths: Sequence[Sequence[Set[Edge]]],
) -> Tuple[List[Edge], np.ndarray]:
    if not paths:
        raise ValueError("At least one pruning path is required.")
    num_states = len(paths[0])
    if any(len(path) != num_states for path in paths):
        raise ValueError("All pruning paths must contain the same number of states.")

    edge_universe = sorted(set().union(*(set().union(*path) for path in paths)))
    probabilities = np.zeros((len(edge_universe), num_states), dtype=float)
    for edge_idx, edge in enumerate(edge_universe):
        for step in range(num_states):
            probabilities[edge_idx, step] = np.mean(
                [edge in path[step] for path in paths]
            )
    return edge_universe, probabilities


def run_pruning(
    graph: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    model_name: str,
    seed: int,
    num_steps: int,
    train_epochs: int,
    scoring_model: str,
    device: torch.device,
) -> Tuple[List[Set[Edge]], float]:
    set_seed(seed)
    model = model_registry.create_model(
        model_name,
        input_dim=graph.x.size(1),
        downstream_model_type=scoring_model,
        train_epochs=train_epochs,
        device=device,
    )
    model.set_training_data(train_mask, val_mask, graph.y)

    start = time.time()
    summaries = model.summarize(graph, num_steps=num_steps)
    elapsed = time.time() - start
    return [undirected_edge_set(item.edge_index) for item in summaries], elapsed


def save_run_paths(output_dir: Path, records: Sequence[Dict]) -> None:
    paths_dir = output_dir / "paths"
    paths_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        payload = {
            "case": record["case"],
            "seed": record["seed"],
            "runtime_seconds": record["runtime_seconds"],
            "states": [
                [list(edge) for edge in sorted(edge_set)]
                for edge_set in record["path"]
            ],
        }
        path = paths_dir / f"seed_{record['seed']}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")


def save_probability_tables(
    output_dir: Path,
    case_name: str,
    edges: Sequence[Edge],
    probabilities: np.ndarray,
) -> None:
    rows = []
    for edge_idx, (src, dst) in enumerate(edges):
        for step, probability in enumerate(probabilities[edge_idx]):
            rows.append({
                "case": case_name,
                "src": src,
                "dst": dst,
                "step": step,
                "appearance_probability": float(probability),
            })

    fieldnames = list(rows[0]) if rows else [
        "case", "src", "dst", "step", "appearance_probability"
    ]
    with (output_dir / "edge_appearance_probabilities.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "edge_appearance_probabilities.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


def to_builtin_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def build_label_color_map(
    labels: Sequence[Any],
    specified_colors: Sequence[str] = DEFAULT_LABEL_COLORS,
) -> Dict[Any, Any]:
    unique_labels = sorted({to_builtin_scalar(label) for label in labels}, key=lambda x: str(x))
    label_colors: Dict[Any, Any] = {}
    for idx, label in enumerate(unique_labels):
        if idx < len(specified_colors):
            label_colors[label] = specified_colors[idx]
        else:
            cmap = plt.cm.tab10 if len(unique_labels) <= 10 else plt.cm.tab20
            label_colors[label] = cmap(idx % cmap.N)
    return label_colors


def load_reference_positions(
    reference_summary_dir: Optional[Path],
    num_nodes: int,
) -> Dict[int, np.ndarray]:
    if reference_summary_dir is not None and reference_summary_dir.exists():
        candidates = sorted(reference_summary_dir.glob("*_step_0_edges.csv"))
        if candidates:
            layout_graph = nx.Graph()
            layout_graph.add_nodes_from(range(num_nodes))
            with candidates[0].open() as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    layout_graph.add_edge(int(row["source"]), int(row["target"]))
            return nx.spring_layout(layout_graph, k=1.0, iterations=100, seed=42)

    layout_graph = nx.karate_club_graph()
    layout_graph.add_nodes_from(range(num_nodes))
    return nx.spring_layout(layout_graph, k=1.0, iterations=100, seed=42)


def load_probability_table(input_dir: Path) -> Tuple[List[Edge], np.ndarray]:
    path = input_dir / "edge_appearance_probabilities.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing probability table: {path}")

    records = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append((
                (int(row["src"]), int(row["dst"])),
                int(row["step"]),
                float(row["appearance_probability"]),
            ))
    if not records:
        return [], np.zeros((0, 0), dtype=float)

    edges = sorted({edge for edge, _, _ in records})
    steps = sorted({step for _, step, _ in records})
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    step_to_idx = {step: idx for idx, step in enumerate(steps)}
    probabilities = np.zeros((len(edges), len(steps)), dtype=float)
    for edge, step, probability in records:
        probabilities[edge_to_idx[edge], step_to_idx[step]] = probability
    return edges, probabilities


def draw_probability_graph(
    axis,
    num_nodes: int,
    positions: Dict[int, np.ndarray],
    edges: Sequence[Edge],
    edge_probabilities: Sequence[float],
    node_labels: Sequence[int],
    step: int,
    label_colors: Dict[Any, Any],
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    visible = [
        (edge, probability)
        for edge, probability in zip(edges, edge_probabilities)
        if probability > 0
    ]
    graph.add_edges_from(edge for edge, _ in visible)
    all_nodes = set(range(num_nodes))
    connected_nodes = {node for edge, _ in visible for node in edge}
    isolated_nodes = all_nodes - connected_nodes
    unique_labels = sorted({to_builtin_scalar(label) for label in node_labels}, key=lambda x: str(x))

    for label in unique_labels:
        label_connected_nodes = [
            node for node in connected_nodes
            if to_builtin_scalar(node_labels[node]) == label
        ]
        if label_connected_nodes:
            nx.draw_networkx_nodes(
                graph,
                positions,
                ax=axis,
                nodelist=label_connected_nodes,
                node_size=800,
                node_color=[label_colors[label]],
                edgecolors="black",
                linewidths=2,
                alpha=1.0,
            )

    for label in unique_labels:
        label_isolated_nodes = [
            node for node in isolated_nodes
            if to_builtin_scalar(node_labels[node]) == label
        ]
        if label_isolated_nodes:
            nx.draw_networkx_nodes(
                graph,
                positions,
                ax=axis,
                nodelist=label_isolated_nodes,
                node_size=400,
                node_color=[label_colors[label]],
                edgecolors="black",
                linewidths=1,
                alpha=1.0,
            )

    if visible:
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=[edge for edge, _ in visible],
            edge_color=[probability for _, probability in visible],
            edge_cmap=plt.cm.Blues,
            edge_vmin=0.0,
            edge_vmax=1.0,
            width=[0.6 + 2.8 * probability for _, probability in visible],
            style="solid",
            alpha=0.9,
        )
    axis.set_title(f"Pruning step {step}", fontsize=10)
    axis.axis("off")


def set_reference_axis_limits(axis, positions: Dict[int, np.ndarray], padding: float = 0.08) -> None:
    coords = np.asarray(list(positions.values()), dtype=float)
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-9)
    axis.set_xlim(min_xy[0] - padding * span[0], max_xy[0] + padding * span[0])
    axis.set_ylim(min_xy[1] - padding * span[1], max_xy[1] + padding * span[1])


def save_probability_plots(
    output_dir: Path,
    case_name: str,
    num_nodes: int,
    edges: Sequence[Edge],
    probabilities: np.ndarray,
    node_labels: Sequence[int],
    reference_summary_dir: Optional[Path] = DEFAULT_REFERENCE_SUMMARY_DIR,
) -> None:
    labels = [f"{src}-{dst}" for src, dst in edges]
    heatmap_height = max(8.0, 0.18 * len(edges))
    fig, axis = plt.subplots(figsize=(10, heatmap_height))
    image = axis.imshow(probabilities, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(probabilities.shape[1]))
    axis.set_xticklabels(range(probabilities.shape[1]))
    axis.set_yticks(range(len(edges)))
    axis.set_yticklabels(labels, fontsize=6)
    axis.set_xlabel("Pruning step")
    axis.set_ylabel("Undirected edge")
    axis.set_title(f"{case_name}: mean edge appearance probability")
    fig.colorbar(image, ax=axis, label="Appearance probability")
    fig.tight_layout()
    fig.savefig(output_dir / "edge_probability_heatmap.png", dpi=220)
    plt.close(fig)

    positions = load_reference_positions(reference_summary_dir, num_nodes)
    label_colors = build_label_color_map(node_labels)
    num_states = probabilities.shape[1]
    columns = min(3, num_states)
    rows = math.ceil(num_states / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 5.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    step_dir = output_dir / "step_graphs"
    step_dir.mkdir(parents=True, exist_ok=True)
    for step in range(num_states):
        draw_probability_graph(
            axes[step], num_nodes, positions, edges, probabilities[:, step],
            node_labels, step, label_colors,
        )
        set_reference_axis_limits(axes[step], positions)

        single_fig, single_axis = plt.subplots(figsize=(12, 9))
        draw_probability_graph(
            single_axis, num_nodes, positions, edges, probabilities[:, step],
            node_labels, step, label_colors,
        )
        set_reference_axis_limits(single_axis, positions)
        divider = make_axes_locatable(single_axis)
        colorbar_axis = divider.append_axes("right", size="3%", pad=0.12)
        single_fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap="Blues"),
            cax=colorbar_axis,
            label="Appearance probability",
        )
        single_axis.legend(
            handles=[
                Patch(facecolor=label_colors[label], edgecolor="black",
                      label=f"Ground truth label {label}")
                for label in sorted({to_builtin_scalar(label) for label in node_labels}, key=lambda x: str(x))
            ],
            loc="upper right",
            fontsize=7,
        )
        single_fig.tight_layout()
        single_fig.savefig(step_dir / f"step_{step:02d}.png", dpi=220)
        plt.close(single_fig)
    for axis in axes[num_states:]:
        axis.axis("off")
    fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="Blues"),
        ax=list(axes[:num_states]),
        label="Appearance probability",
        fraction=0.018,
        pad=0.015,
    )
    fig.legend(
        handles=[
            Patch(facecolor=label_colors[label], edgecolor="black",
                  label=f"Ground truth label {label}")
            for label in sorted({to_builtin_scalar(label) for label in node_labels}, key=lambda x: str(x))
        ],
        loc="lower center",
        ncol=len(set(node_labels)),
        fontsize=8,
    )
    fig.suptitle(f"{case_name}: edge stability across pruning steps", fontsize=14)
    fig.subplots_adjust(wspace=0.16, hspace=0.28, bottom=0.08, top=0.93)
    fig.savefig(output_dir / "edge_probability_graphs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_case(
    case_name: str,
    args: argparse.Namespace,
    base_graph: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
) -> None:
    output_dir = Path(args.output_dir) / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    original_edges = undirected_edge_set(base_graph.edge_index)
    records = []

    for seed in args.seeds:
        graph = base_graph.clone()
        added_edges: Set[Edge] = set()
        if case_name == "perturbed":
            graph, added_edges = add_random_edges(graph, args.add_edge_fraction, seed)
        graph = graph.to(device)

        print(
            f"\n[{case_name}] seed={seed}, edges={len(undirected_edge_set(graph.edge_index))}, "
            f"added={len(added_edges)}",
            flush=True,
        )
        path, runtime = run_pruning(
            graph=graph,
            train_mask=train_mask,
            val_mask=val_mask,
            model_name=args.model,
            seed=seed,
            num_steps=args.num_steps,
            train_epochs=args.train_epochs,
            scoring_model=args.scoring_model,
            device=device,
        )
        records.append({
            "case": case_name,
            "seed": seed,
            "added_edges": added_edges,
            "path": path,
            "runtime_seconds": runtime,
        })
        save_run_paths(output_dir, records)

    edges, probabilities = aggregate_edge_probabilities(
        [record["path"] for record in records]
    )
    save_probability_tables(output_dir, case_name, edges, probabilities)
    save_probability_plots(
        output_dir,
        case_name,
        base_graph.num_nodes,
        edges,
        probabilities,
        base_graph.y.detach().cpu().tolist(),
        Path(args.reference_summary_dir),
    )
    summary = {
        "case": case_name,
        "model": args.model,
        "scoring_model": args.scoring_model,
        "seeds": args.seeds,
        "num_repeats": len(args.seeds),
        "num_steps": args.num_steps,
        "train_epochs": args.train_epochs,
        "original_undirected_edges": len(original_edges),
        "add_edge_fraction": args.add_edge_fraction if case_name == "perturbed" else 0.0,
        "added_edges_per_run": (
            [len(record["added_edges"]) for record in records]
            if case_name == "perturbed" else [0] * len(records)
        ),
        "edge_universe_size": len(edges),
        "runtime_seconds": [record["runtime_seconds"] for record in records],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {case_name} stability results to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=["original", "perturbed"],
                        default=["original", "perturbed"])
    parser.add_argument("--model", default="gradient_based",
                        help="Single-edge INXplain model registry name")
    parser.add_argument("--scoring-model", choices=["gcn", "gat", "sage"], default="gcn")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--add-edge-fraction", type=float, default=0.20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/karate_edge_stability")
    parser.add_argument(
        "--reference-summary-dir",
        default=str(DEFAULT_REFERENCE_SUMMARY_DIR),
        help="Reference summary_graphs directory used to match Karate node layout/style.",
    )
    parser.add_argument(
        "--redraw-existing",
        action="store_true",
        help="Only redraw plots from existing edge_appearance_probabilities.csv files.",
    )
    return parser.parse_args()


def redraw_existing_cases(args: argparse.Namespace, base_graph: Data) -> None:
    node_labels = base_graph.y.detach().cpu().tolist()
    for case_name in args.cases:
        output_dir = Path(args.output_dir) / case_name
        edges, probabilities = load_probability_table(output_dir)
        save_probability_plots(
            output_dir=output_dir,
            case_name=case_name,
            num_nodes=base_graph.num_nodes,
            edges=edges,
            probabilities=probabilities,
            node_labels=node_labels,
            reference_summary_dir=Path(args.reference_summary_dir),
        )
        print(f"Redrew {case_name} plots in {output_dir}")


def main() -> None:
    args = parse_args()
    if len(args.seeds) != 5:
        print(f"Warning: requested {len(args.seeds)} repeats; the case study specifies 5.")
    if not 0 < args.add_edge_fraction <= 1:
        raise ValueError("--add-edge-fraction must be in (0, 1].")
    if args.model not in model_registry.list_models():
        raise ValueError(f"Unknown model {args.model!r}.")

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    loader = DatasetLoader(root_dir=args.data_dir)
    graph, train_mask, val_mask, _ = loader.load_dataset("KarateClub", task_type="original")
    graph = loader.preprocess_for_summarization(graph, to_undirected_graph=True)
    if args.redraw_existing:
        redraw_existing_cases(args, graph)
        return

    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    for case_name in args.cases:
        run_case(case_name, args, graph, train_mask, val_mask, device)


if __name__ == "__main__":
    main()
