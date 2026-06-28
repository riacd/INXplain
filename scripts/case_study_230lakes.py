"""
Case Study: Simplify 230 Lake Networks using IGPrune (Gradient-Based) Model

This script processes 230 directed graphs from the 230lake_networks directory,
applying the gradient-based graph summarization model (IGPrune) for 30 steps.
The output directory mirrors the structure produced for the 50-lake case study.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from GS.models.gradient_based import GradientBasedGraphSummarization


def load_directed_graph_from_tsv(tsv_path: Path) -> Data:
    """
    Load a directed graph from a TSV file with columns [consumer, resource, weight].
    """
    df = pd.read_csv(tsv_path, sep="\t")

    sources = df["consumer"].values
    targets = df["resource"].values

    # Map node names to indices to ensure contiguous ids.
    all_nodes = sorted(set(sources.tolist() + targets.tolist()))
    node_to_id = {node: idx for idx, node in enumerate(all_nodes)}

    edge_index = torch.tensor(
        [
            [node_to_id[src] for src in sources],
            [node_to_id[tgt] for tgt in targets],
        ],
        dtype=torch.long,
    )

    num_nodes = len(all_nodes)
    x = torch.eye(num_nodes, dtype=torch.float)

    # Generate simple degree-based labels to enable gradient computation.
    in_degrees = torch.bincount(edge_index[1], minlength=num_nodes).float()
    degree_sorted = torch.argsort(in_degrees)
    labels = torch.zeros(num_nodes, dtype=torch.long)

    third = max(num_nodes // 3, 1)
    labels[degree_sorted[:third]] = 0
    labels[degree_sorted[third : 2 * third]] = 1
    labels[degree_sorted[2 * third :]] = 2

    # Create train/val/test masks with a 60/20/20 split.
    train_size = max(int(0.6 * num_nodes), 1)
    val_size = max(int(0.2 * num_nodes), 1)

    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size : train_size + val_size]] = True
    test_mask[perm[train_size + val_size :]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Persist node mappings for writing outputs.
    data.node_names = all_nodes
    data.node_to_id = node_to_id

    return data


def save_simplified_graph(graph_data: Data, output_path: Path, node_names: list[str]) -> None:
    """
    Save simplified graph in TSV format with columns [consumer, resource, weight].
    """
    edge_index = graph_data.edge_index.cpu().numpy()
    sources = [node_names[idx] for idx in edge_index[0]]
    targets = [node_names[idx] for idx in edge_index[1]]
    weights = [1] * len(sources)

    df = pd.DataFrame({"consumer": sources, "resource": targets, "weight": weights})
    df.to_csv(output_path, sep="\t", index=False)


def summarize_graph(
    model: GradientBasedGraphSummarization,
    graph_data: Data,
    num_steps: int,
) -> list[Data]:
    """
    Apply IGPrune summarization and return the sequence of simplified graphs.
    """
    model.train_mask = graph_data.train_mask
    model.val_mask = graph_data.val_mask
    model.labels = graph_data.y

    return model.summarize(graph_data, num_steps=num_steps)


def format_percentage_removed(original_edges: int, current_edges: int) -> float:
    if original_edges == 0:
        return 0.0
    removed = original_edges - current_edges
    return (removed / original_edges) * 100.0


def process_graph(
    tsv_file: Path,
    output_dir: Path,
    model: GradientBasedGraphSummarization,
    num_steps: int,
    overwrite: bool = False,
) -> None:
    graph_name = tsv_file.stem.replace("_network", "")
    graph_output_dir = output_dir / graph_name
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = graph_output_dir / f"{graph_name}_summary.txt"
    if summary_file.exists() and not overwrite:
        print(f"Skipping {graph_name} (summary exists, use --overwrite to regenerate)")
        return

    print(f"\n{'=' * 72}\nProcessing graph: {graph_name}\n{'=' * 72}")

    graph_data = load_directed_graph_from_tsv(tsv_file)
    print(f"  Nodes: {graph_data.num_nodes}")
    print(f"  Edges: {graph_data.num_edges}")

    summarized_graphs = summarize_graph(model, graph_data, num_steps=num_steps)
    original_edge_count = summarized_graphs[0].num_edges

    print("\nSaving simplified graphs...")
    for step_idx, simplified_graph in enumerate(summarized_graphs):
        output_file = graph_output_dir / f"{graph_name}_step{step_idx:02d}.tsv"
        save_simplified_graph(simplified_graph, output_file, graph_data.node_names)
        print(f"  Step {step_idx:02d}: {simplified_graph.num_edges} edges -> {output_file.name}")

    print("Writing summary statistics...")
    with open(summary_file, "w") as f:
        f.write(f"Graph: {graph_name}\n")
        f.write(f"Original nodes: {graph_data.num_nodes}\n")
        f.write(f"Original edges: {original_edge_count}\n")
        f.write(f"Simplification steps: {num_steps}\n")
        f.write("\nEdge counts per step:\n")
        for step_idx, simplified_graph in enumerate(summarized_graphs):
            edges = simplified_graph.num_edges
            percent_removed = format_percentage_removed(original_edge_count, edges)
            f.write(
                f"  Step {step_idx:02d}: {edges:6d} edges "
                f"({percent_removed:5.1f}% removed)\n"
            )

    print(f"Summary saved to {summary_file}")


def run_case_study(
    input_dir: Path,
    output_dir: Path,
    num_steps: int,
    hidden_dim: int,
    train_epochs: int,
    device: str | None,
    overwrite: bool,
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_obj}")
    print(f"Simplification steps: {num_steps}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(glob.glob(str(input_dir / "*.tsv")))
    print(f"\nFound {len(tsv_files)} TSV files to process")

    model = GradientBasedGraphSummarization(
        input_dim=None,
        downstream_model_type="gcn",
        hidden_dim=hidden_dim,
        train_epochs=train_epochs,
        device=device_obj,
    )

    for tsv_file in tqdm(tsv_files, desc="Processing lake networks"):
        process_graph(Path(tsv_file), output_dir, model, num_steps=num_steps, overwrite=overwrite)

    print(f"\nProcessing complete. Results stored in: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simplify 230 lake networks using the IGPrune directed graph summarization model (30 steps)."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "230lake_networks",
        help="Directory containing input TSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "case_study_230lakes",
        help="Directory to store simplification outputs.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of simplification steps to perform.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for the downstream GCN model.",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=10,
        help="Number of training epochs for the downstream model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit device string, e.g. 'cuda' or 'cpu'. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if summaries already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_case_study(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        hidden_dim=args.hidden_dim,
        train_epochs=args.train_epochs,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
