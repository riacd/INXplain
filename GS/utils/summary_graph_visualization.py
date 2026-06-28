"""
Reusable utilities for summary-graph visualization.

This module supports:
1) plotting from in-memory torch_geometric summary graphs
2) plotting from saved CSV files in a `summary_graphs/` directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import re

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


DEFAULT_LABEL_COLORS: Sequence[str] = ("#395B3F", "#E2DCD0", "#D3A228", "#BED7E6")
STEP_FILE_PATTERN = re.compile(r"^(?P<model>.+)_step_(?P<step>\d+)_edges\.csv$")


def _to_builtin_scalar(value: Any) -> Any:
    """Convert numpy scalars to Python scalars for stable dict keys/comparisons."""
    return value.item() if hasattr(value, "item") else value


def _build_label_color_map(
    labels: Iterable[Any],
    specified_colors: Sequence[str] = DEFAULT_LABEL_COLORS,
) -> Dict[Any, Any]:
    unique_labels = sorted({_to_builtin_scalar(label) for label in labels}, key=lambda x: str(x))
    num_labels = len(unique_labels)

    label_colors: Dict[Any, Any] = {}
    for idx, label in enumerate(unique_labels):
        if idx < len(specified_colors):
            label_colors[label] = specified_colors[idx]
        else:
            if num_labels <= 10:
                cmap = plt.cm.tab10
            elif num_labels <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.hsv
            label_colors[label] = cmap(idx % cmap.N)
    return label_colors


def _load_accuracy_by_step(step_metrics_path: Optional[Path]) -> Optional[Dict[int, float]]:
    if step_metrics_path is None or not step_metrics_path.exists():
        return None

    df = pd.read_csv(step_metrics_path, sep="\t")
    if "accuracy_metric" not in df.columns:
        return None

    if "step" in df.columns:
        steps = df["step"].tolist()
    else:
        steps = list(range(len(df)))

    accuracy_by_step: Dict[int, float] = {}
    for step, acc in zip(steps, df["accuracy_metric"].tolist()):
        if pd.isna(acc):
            continue
        accuracy_by_step[int(step)] = float(acc)
    return accuracy_by_step


def _discover_step_csv_files(summary_graphs_dir: Path, model_name: Optional[str]) -> Tuple[str, List[Tuple[int, Path]]]:
    discovered: List[Tuple[str, int, Path]] = []
    for csv_path in summary_graphs_dir.glob("*_step_*_edges.csv"):
        match = STEP_FILE_PATTERN.match(csv_path.name)
        if not match:
            continue
        discovered.append((match.group("model"), int(match.group("step")), csv_path))

    if not discovered:
        raise FileNotFoundError(f"No step edge CSV files found in: {summary_graphs_dir}")

    if model_name is None:
        model_candidates = sorted({item[0] for item in discovered})
        if len(model_candidates) != 1:
            raise ValueError(
                "Multiple models found in summary_graphs directory. "
                "Please set model_name explicitly. "
                f"Found: {model_candidates}"
            )
        model_name = model_candidates[0]

    step_files = [(step, path) for model, step, path in discovered if model == model_name]
    if not step_files:
        raise FileNotFoundError(
            f"No CSV files for model '{model_name}' found in {summary_graphs_dir}"
        )

    step_files.sort(key=lambda x: x[0])
    return model_name, step_files


def _load_node_labels(node_info_path: Path) -> Dict[Any, Any]:
    if not node_info_path.exists():
        return {}

    node_df = pd.read_csv(node_info_path)
    required = {"node_id", "label"}
    if not required.issubset(set(node_df.columns)):
        raise ValueError(
            f"{node_info_path} must contain columns {sorted(required)}; "
            f"got {node_df.columns.tolist()}"
        )

    node_labels: Dict[Any, Any] = {}
    for node_id, label in zip(node_df["node_id"].tolist(), node_df["label"].tolist()):
        node_labels[_to_builtin_scalar(node_id)] = _to_builtin_scalar(label)
    return node_labels


def _load_step_graphs_from_csv(step_files: Sequence[Tuple[int, Path]]) -> Dict[int, nx.Graph]:
    step_graphs: Dict[int, nx.Graph] = {}
    for step, csv_path in step_files:
        df = pd.read_csv(csv_path)
        if not {"source", "target"}.issubset(df.columns):
            raise ValueError(
                f"{csv_path} must contain 'source' and 'target' columns; "
                f"got {df.columns.tolist()}"
            )

        graph = nx.Graph()
        for src, dst in df[["source", "target"]].itertuples(index=False, name=None):
            graph.add_edge(_to_builtin_scalar(src), _to_builtin_scalar(dst))
        step_graphs[step] = graph
    return step_graphs


def render_summary_graphs(
    step_graphs: Mapping[int, nx.Graph],
    node_labels: Mapping[Any, Any],
    output_dir: Path,
    model_name: str,
    accuracy_by_step: Optional[Mapping[int, float]] = None,
    layout_seed: int = 42,
    layout_k: float = 1.0,
    layout_iterations: int = 100,
    specified_colors: Sequence[str] = DEFAULT_LABEL_COLORS,
    dpi: int = 150,
) -> List[Path]:
    """
    Render summary graph sequence with fixed node layout and label-based colors.

    Returns:
        List of generated PNG paths.
    """
    if not step_graphs:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use node_info as canonical node set when available; otherwise infer from graphs.
    if node_labels:
        all_nodes = set(node_labels.keys())
    else:
        all_nodes = set()
        for graph in step_graphs.values():
            all_nodes.update(graph.nodes())
        node_labels = {node: 0 for node in all_nodes}

    # Backfill labels for nodes that may appear in CSV but not in node_info.
    for graph in step_graphs.values():
        for node in graph.nodes():
            if node not in node_labels:
                node_labels[node] = "unknown"
                all_nodes.add(node)

    label_colors = _build_label_color_map(node_labels.values(), specified_colors=specified_colors)
    unique_labels = sorted({node_labels[node] for node in all_nodes}, key=lambda x: str(x))

    # Build fixed layout from step-0 graph when available; otherwise use union graph.
    if 0 in step_graphs:
        layout_graph = step_graphs[0].copy()
    else:
        layout_graph = nx.Graph()
        for graph in step_graphs.values():
            layout_graph.add_edges_from(graph.edges())
    layout_graph.add_nodes_from(all_nodes)
    fixed_pos = nx.spring_layout(layout_graph, k=layout_k, iterations=layout_iterations, seed=layout_seed)

    generated_pngs: List[Path] = []
    for step in sorted(step_graphs.keys()):
        graph = step_graphs[step]
        connected_nodes = set(graph.nodes())
        isolated_nodes = all_nodes - connected_nodes

        # Include all nodes for node drawing while keeping edges from current step graph.
        plot_graph = nx.Graph()
        plot_graph.add_nodes_from(all_nodes)
        plot_graph.add_edges_from(graph.edges())

        plt.figure(figsize=(12, 9))

        # Draw connected nodes by label.
        for label in unique_labels:
            label_connected_nodes = [node for node in connected_nodes if node_labels.get(node) == label]
            if not label_connected_nodes:
                continue
            nx.draw_networkx_nodes(
                plot_graph,
                fixed_pos,
                nodelist=label_connected_nodes,
                node_size=800,
                node_color=[label_colors[label]],
                edgecolors="black",
                linewidths=2,
                alpha=1.0,
            )

        # Draw isolated nodes by label (smaller nodes).
        for label in unique_labels:
            label_isolated_nodes = [node for node in isolated_nodes if node_labels.get(node) == label]
            if not label_isolated_nodes:
                continue
            nx.draw_networkx_nodes(
                plot_graph,
                fixed_pos,
                nodelist=label_isolated_nodes,
                node_size=400,
                node_color=[label_colors[label]],
                edgecolors="black",
                linewidths=1,
                alpha=1.0,
            )

        # Draw edges for the current step.
        if graph.number_of_edges() > 0:
            nx.draw_networkx_edges(graph, fixed_pos, alpha=0.4, width=0.8)

        title = f"Step {step}: {len(all_nodes)} nodes, {graph.number_of_edges()} edges"
        if accuracy_by_step is not None and step in accuracy_by_step:
            title += f", Accuracy: {accuracy_by_step[step]:.4f}"

        # Increase title font size for better readability in exported figures.
        plt.title(title, fontsize=28)
        plt.axis("off")

        png_path = output_dir / f"{model_name}_step_{step}_graph.png"
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        generated_pngs.append(png_path)

    return generated_pngs


def visualize_from_summary_csv(
    summary_graphs_dir: Path | str,
    model_name: Optional[str] = None,
    output_dir: Optional[Path | str] = None,
    node_info_path: Optional[Path | str] = None,
    step_metrics_path: Optional[Path | str] = None,
    max_nodes: int = 200,
) -> List[Path]:
    """
    Re-draw summary graph visualizations from saved CSV files.
    """
    summary_graphs_dir = Path(summary_graphs_dir)
    if not summary_graphs_dir.exists():
        raise FileNotFoundError(f"summary_graphs directory does not exist: {summary_graphs_dir}")

    model_name, step_files = _discover_step_csv_files(summary_graphs_dir, model_name=model_name)
    step_graphs = _load_step_graphs_from_csv(step_files)

    if node_info_path is None:
        node_info_path = summary_graphs_dir / "node_info.csv"
    node_labels = _load_node_labels(Path(node_info_path))

    # Infer node count from CSV if node_info.csv is missing.
    if node_labels:
        num_nodes = len(node_labels)
    else:
        all_nodes: set[Any] = set()
        for graph in step_graphs.values():
            all_nodes.update(graph.nodes())
        num_nodes = len(all_nodes)

    if num_nodes > max_nodes:
        return []

    if output_dir is None:
        output_dir = summary_graphs_dir.parent / "graph_visualizations"

    if step_metrics_path is None:
        inferred_step_metrics = summary_graphs_dir.parent / "process_results" / f"{model_name}_step_metrics.tsv"
        step_metrics_path = inferred_step_metrics if inferred_step_metrics.exists() else None

    accuracy_by_step = _load_accuracy_by_step(Path(step_metrics_path)) if step_metrics_path else None

    return render_summary_graphs(
        step_graphs=step_graphs,
        node_labels=node_labels,
        output_dir=Path(output_dir),
        model_name=model_name,
        accuracy_by_step=accuracy_by_step,
    )


def visualize_from_torch_summary_graphs(
    summary_graphs: Sequence[Any],
    original_graph: Any,
    output_dir: Path | str,
    model_name: str,
    accuracy_metrics: Optional[Sequence[float]] = None,
    max_nodes: int = 200,
) -> List[Path]:
    """
    Reusable plotting entry from in-memory torch_geometric graphs.
    """
    if getattr(original_graph, "num_nodes", 0) > max_nodes:
        return []
    if not hasattr(original_graph, "y") or original_graph.y is None:
        return []

    # Import lazily so this module can still be used for CSV-only plotting.
    from torch_geometric.utils import to_networkx

    node_labels: Dict[int, Any] = {}
    labels = original_graph.y.cpu().numpy()
    for node_id in range(original_graph.num_nodes):
        node_labels[int(node_id)] = _to_builtin_scalar(labels[node_id])

    step_graphs: Dict[int, nx.Graph] = {}
    for step, graph in enumerate(summary_graphs):
        step_graphs[step] = to_networkx(graph, to_undirected=True)

    accuracy_by_step: Optional[Dict[int, float]] = None
    if accuracy_metrics is not None:
        accuracy_by_step = {
            int(step): float(acc)
            for step, acc in enumerate(accuracy_metrics)
            if acc is not None
        }

    return render_summary_graphs(
        step_graphs=step_graphs,
        node_labels=node_labels,
        output_dir=Path(output_dir),
        model_name=model_name,
        accuracy_by_step=accuracy_by_step,
    )
