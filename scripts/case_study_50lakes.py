"""
Case Study: Simplify 50 Lake Networks using INXplain (Gradient-Based) Model

This script processes 50 directed graphs from the 50lake_networks directory,
applying the gradient-based graph summarization model (INXplain).
"""

import os
import sys
import glob
import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch_geometric.data import Data
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from GS.models.gradient_based import GradientBasedGraphSummarization


def load_directed_graph_from_tsv(tsv_path):
    """
    Load a directed graph from TSV file

    Args:
        tsv_path: Path to TSV file with columns: consumer, resource, weight

    Returns:
        Data: PyTorch Geometric Data object
    """
    # Read TSV file
    df = pd.read_csv(tsv_path, sep='\t')

    # Extract consumer (source) and resource (target) columns
    sources = df['consumer'].values
    targets = df['resource'].values

    # Create node ID mapping
    all_nodes = sorted(set(list(sources) + list(targets)))
    node_to_id = {node: idx for idx, node in enumerate(all_nodes)}

    # Convert to edge index
    edge_index = torch.tensor([
        [node_to_id[src] for src in sources],
        [node_to_id[tgt] for tgt in targets]
    ], dtype=torch.long)

    # Create node features (use identity matrix as default)
    num_nodes = len(all_nodes)
    x = torch.eye(num_nodes, dtype=torch.float)

    # Create dummy labels for gradient computation (use degree-based labels)
    # Calculate in-degree for each node
    in_degrees = torch.bincount(edge_index[1], minlength=num_nodes).float()

    # Divide into 3 classes based on degree (low, medium, high)
    degree_sorted = torch.argsort(in_degrees)
    labels = torch.zeros(num_nodes, dtype=torch.long)

    third = num_nodes // 3
    labels[degree_sorted[:third]] = 0  # low degree
    labels[degree_sorted[third:2*third]] = 1  # medium degree
    labels[degree_sorted[2*third:]] = 2  # high degree

    # Create train/val/test masks (60/20/20 split)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True

    # Create graph data
    data = Data(
        x=x,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    # Store node name mapping for later use
    data.node_names = all_nodes
    data.node_to_id = node_to_id

    return data


def save_simplified_graph(graph_data, output_path, node_names):
    """
    Save simplified graph in TSV format (consumer, resource, weight)

    Args:
        graph_data: PyTorch Geometric Data object
        output_path: Output TSV file path
        node_names: List of node names
    """
    edge_index = graph_data.edge_index.cpu().numpy()

    # Convert edge indices back to node names
    sources = [node_names[idx] for idx in edge_index[0]]
    targets = [node_names[idx] for idx in edge_index[1]]
    weights = [1] * len(sources)  # All edges have weight 1

    # Create DataFrame
    df = pd.DataFrame({
        'consumer': sources,
        'resource': targets,
        'weight': weights
    })

    # Save to TSV
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved simplified graph to {output_path}")


def main(num_steps=2):
    """Main function to process 50 lake networks

    Args:
        num_steps: Number of simplification steps (default: 2)
    """

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Number of simplification steps: {num_steps}")

    # Input and output directories
    input_dir = project_root / "data" / "50lake_networks"
    output_dir = project_root / "results" / f"case_study_50lakes_steps{num_steps}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all TSV files
    tsv_files = sorted(glob.glob(str(input_dir / "*.tsv")))
    print(f"\nFound {len(tsv_files)} TSV files to process")

    # Process each graph
    for tsv_file in tqdm(tsv_files, desc="Processing lake networks"):
        # Extract graph name
        graph_name = Path(tsv_file).stem.replace('_network', '')

        # Create output directory for this graph
        graph_output_dir = output_dir / graph_name
        graph_output_dir.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        summary_file = graph_output_dir / f"{graph_name}_summary.txt"
        if summary_file.exists():
            print(f"\nSkipping {graph_name} (already processed)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {graph_name}")
        print(f"{'='*60}")

        try:
            # Load graph
            print(f"Loading graph from {tsv_file}...")
            graph_data = load_directed_graph_from_tsv(tsv_file)
            print(f"  Nodes: {graph_data.num_nodes}")
            print(f"  Edges: {graph_data.num_edges}")

            # Initialize gradient-based model (INXplain)
            print(f"\nInitializing INXplain model...")
            model = GradientBasedGraphSummarization(
                input_dim=graph_data.x.size(1),
                downstream_model_type='gcn',
                hidden_dim=64,  # Smaller hidden dim for faster computation
                train_epochs=10,  # Fewer epochs for faster computation
                device=device
            )

            # Set training data (required for gradient computation)
            model.train_mask = graph_data.train_mask
            model.val_mask = graph_data.val_mask
            model.labels = graph_data.y

            # Apply graph summarization
            print(f"\nApplying graph summarization ({num_steps} steps)...")
            summarized_graphs = model.summarize(graph_data, num_steps=num_steps)

            print(f"\nGenerated {len(summarized_graphs)} simplified graphs:")
            for i, g in enumerate(summarized_graphs):
                print(f"  Step {i}: {g.num_edges} edges")

            # Save simplified graphs
            print(f"\nSaving simplified graphs...")
            for step_idx, simplified_graph in enumerate(summarized_graphs):
                output_file = graph_output_dir / f"{graph_name}_step{step_idx}.tsv"
                save_simplified_graph(
                    simplified_graph,
                    output_file,
                    graph_data.node_names
                )

            # Save summary statistics
            stats_file = graph_output_dir / f"{graph_name}_summary.txt"
            with open(stats_file, 'w') as f:
                f.write(f"Graph: {graph_name}\n")
                f.write(f"Original nodes: {graph_data.num_nodes}\n")
                f.write(f"Original edges: {graph_data.num_edges}\n")
                f.write(f"Simplification steps: {num_steps}\n")
                f.write(f"\nEdge counts per step:\n")
                for i, g in enumerate(summarized_graphs):
                    f.write(f"  Step {i}: {g.num_edges} edges\n")

            print(f"Saved summary statistics to {stats_file}")

        except Exception as e:
            print(f"Error processing {graph_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simplify 50 lake networks using INXplain model')
    parser.add_argument('--num_steps', type=int, default=2,
                        help='Number of simplification steps (default: 2)')
    args = parser.parse_args()

    main(num_steps=args.num_steps)
