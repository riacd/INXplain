"""
Generate dataset statistics table in LaTeX format.

This script collects statistics for the datasets used in the experiments
and outputs a LaTeX table in the same format as the reference table.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from pathlib import Path

def get_feature_scale(x):
    """Get the scale/range of features."""
    import torch

    min_val = x.min().item()
    max_val = x.max().item()

    # Check if binary
    unique_vals = torch.unique(x)
    if len(unique_vals) == 2 and set(unique_vals.tolist()) == {0.0, 1.0}:
        return r"$\{0,1\}$"

    # Check if all 0 or 1
    if torch.all((x == 0) | (x == 1)):
        return r"$\{0,1\}$"

    # Return range
    return f"$[{min_val:.3f}, {max_val:.3f}]$"

def collect_dataset_stats():
    """Collect statistics for all required datasets."""
    from GS.datasets.loaders import DatasetLoader

    loader = DatasetLoader(root_dir='./data')

    # Datasets to analyze: Cora, CiteSeer, PubMed, KarateClub, WebKB, ME, MT
    dataset_names = [
        'Cora', 'CiteSeer', 'PubMed', 'KarateClub',
        'Texas', 'Wisconsin', 'Cornell',
        'SO_relation_ME', 'SO_relation_MT',
    ]
    display_names = {
        'Cora': 'Cora',
        'CiteSeer': 'Citeseer',
        'PubMed': 'PubMed',
        'KarateClub': 'KarateClub',
        'Texas': 'Texas',
        'Wisconsin': 'Wisconsin',
        'Cornell': 'Cornell',
        'SO_relation_ME': 'SO-ME',
        'SO_relation_MT': 'SO-MT'
    }

    stats = {}

    for dataset_name in dataset_names:
        print(f"\nProcessing {dataset_name}...")
        try:
            # Load with original task
            data, train_mask, val_mask, test_mask = loader.load_dataset(
                dataset_name,
                task_type='original',
                normalize_features=False  # Get original features for scale
            )

            num_nodes = data.num_nodes
            num_edges = data.edge_index.size(1) // 2  # Undirected edges
            num_features = data.x.size(1)
            num_classes = int(data.y.max()) + 1
            train_nodes = int(train_mask.sum())
            val_nodes = int(val_mask.sum())
            test_nodes = int(test_mask.sum())
            label_rate = train_nodes / num_nodes
            feature_scale = get_feature_scale(data.x)

            stats[display_names[dataset_name]] = {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_features': num_features,
                'num_classes': num_classes,
                'train_nodes': train_nodes,
                'val_nodes': val_nodes,
                'test_nodes': test_nodes,
                'label_rate': label_rate,
                'feature_scale': feature_scale
            }

            print(f"  Nodes: {num_nodes}")
            print(f"  Edges: {num_edges}")
            print(f"  Features: {num_features}")
            print(f"  Classes: {num_classes}")
            print(f"  Train/Val/Test: {train_nodes}/{val_nodes}/{test_nodes}")
            print(f"  Label Rate: {label_rate:.3f}")
            print(f"  Feature Scale: {feature_scale}")

        except Exception as e:
            print(f"  Error: {e}")
            stats[display_names[dataset_name]] = None

    return stats


def _find_lake_network_file(lake_name, data_root='./data'):
    """Find a lake network TSV file in 50/230 lake directories."""
    data_root_path = Path(data_root)
    search_dirs = [
        data_root_path / '50lake_networks',
        data_root_path / '230lake_networks'
    ]

    matches = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for path in sorted(search_dir.glob('*.tsv')):
            if lake_name.lower() in path.stem.lower():
                matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"No network TSV found for lake '{lake_name}' in {search_dirs}"
        )

    # Prefer exact *_<lake_name>_network naming if multiple matches exist.
    exact_suffix = f"_{lake_name.lower()}_network"
    exact_matches = [p for p in matches if p.stem.lower().endswith(exact_suffix)]
    if len(exact_matches) == 1:
        return exact_matches[0]

    return matches[0]


def collect_lake_stats(lake_names, data_root='./data'):
    """Collect node/edge statistics for selected lake network TSV files."""
    stats = {}
    for lake_name in lake_names:
        print(f"\nProcessing lake: {lake_name}...")
        try:
            tsv_path = _find_lake_network_file(lake_name, data_root=data_root)
            df = pd.read_csv(tsv_path, sep='\t')

            required_columns = {'consumer', 'resource'}
            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"Missing required columns in {tsv_path}: {required_columns}"
                )

            consumers = df['consumer'].astype(str)
            resources = df['resource'].astype(str)
            num_nodes = len(set(consumers).union(set(resources)))
            num_edges = len(df)
            train_nodes = int(num_nodes * 0.6)
            val_nodes = int(num_nodes * 0.2)
            test_nodes = num_nodes - train_nodes - val_nodes
            label_rate = (train_nodes / num_nodes) if num_nodes > 0 else 0.0

            stats[lake_name] = {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'train_nodes': train_nodes,
                'val_nodes': val_nodes,
                'test_nodes': test_nodes,
                'label_rate': label_rate,
                'source_file': str(tsv_path)
            }

            print(f"  Source: {tsv_path}")
            print(f"  Nodes: {num_nodes}")
            print(f"  Edges: {num_edges}")
            print(f"  Train/Val/Test: {train_nodes}/{val_nodes}/{test_nodes}")
            print(f"  Label Rate: {label_rate:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            stats[lake_name] = None

    return stats

def generate_latex_table(stats):
    """Generate LaTeX table from statistics."""

    # Order of datasets in table
    dataset_order = [
        'Cora', 'Citeseer', 'PubMed', 'KarateClub',
        'Texas', 'Wisconsin', 'Cornell',
        'SO-ME', 'SO-MT',
    ]

    # Start building LaTeX table
    latex = r"""\begin{table*}[!htbp]
\caption{Summary of the datasets for node classification tasks.}
\label{tab:stats:node_classification}
\begin{center}
\resizebox{0.9\textwidth}{!}{
    \begin{tabular}{l""" + "c" * len(dataset_order) + r"""}
    \toprule
"""

    # Header row
    header = "    & " + " & ".join([r"\textbf{" + name + "}" for name in dataset_order]) + r" \\"
    latex += header + "\n"
    latex += r"    \midrule" + "\n"

    # Row labels
    rows = [
        ('# Nodes', 'num_nodes', lambda x: f"${x:,}$"),
        ('# Edges', 'num_edges', lambda x: f"${x:,}$"),
        ('# Features', 'num_features', lambda x: f"${x:,}$"),
        ('# Classes', 'num_classes', lambda x: f"${x}$"),
        ('# Training Nodes', 'train_nodes', lambda x: f"${x:,}$"),
        ('# Validation Nodes', 'val_nodes', lambda x: f"${x:,}$"),
        ('# Test Nodes', 'test_nodes', lambda x: f"${x:,}$"),
        ('Label Rate', 'label_rate', lambda x: f"${x:.3f}$"),
        ('Feature Scale', 'feature_scale', lambda x: x),  # Already formatted
    ]

    # Generate data rows
    for label, key, formatter in rows:
        row = f"    {label}"
        for dataset in dataset_order:
            if stats.get(dataset) and stats[dataset] is not None:
                value = stats[dataset].get(key, '-')
                if value == '-':
                    row += " & -"
                else:
                    row += f" & {formatter(value)}"
            else:
                row += " & -"
        row += r" \\"
        latex += row + "\n"

    # Close table
    latex += r"""    \bottomrule
    \end{tabular}
}
\end{center}
\end{table*}"""

    return latex


def generate_lake_latex_table(stats, lake_order):
    """Generate a compact LaTeX table for selected lake datasets."""
    latex = r"""\begin{table}[!htbp]
\caption{Summary of selected lake networks.}
\label{tab:stats:lakes}
\begin{center}
\resizebox{0.7\textwidth}{!}{
    \begin{tabular}{l""" + "c" * len(lake_order) + r"""}
    \toprule
"""

    header = "    & " + " & ".join([r"\textbf{" + name + "}" for name in lake_order]) + r" \\"
    latex += header + "\n"
    latex += r"    \midrule" + "\n"

    rows = [
        ('# Nodes', 'num_nodes', lambda x: f"${x:,}$"),
        ('# Edges', 'num_edges', lambda x: f"${x:,}$"),
        ('# Training Nodes', 'train_nodes', lambda x: f"${x:,}$"),
        ('# Validation Nodes', 'val_nodes', lambda x: f"${x:,}$"),
        ('# Test Nodes', 'test_nodes', lambda x: f"${x:,}$"),
        ('Label Rate', 'label_rate', lambda x: f"${x:.3f}$")
    ]

    for label, key, formatter in rows:
        row = f"    {label}"
        for lake in lake_order:
            if stats.get(lake) and stats[lake] is not None:
                row += f" & {formatter(stats[lake][key])}"
            else:
                row += " & -"
        row += r" \\"
        latex += row + "\n"

    latex += r"""    \bottomrule
    \end{tabular}
}
\end{center}
\end{table}"""

    return latex

def main():
    parser = argparse.ArgumentParser(description="Generate dataset statistics")
    parser.add_argument(
        '--lake_names',
        nargs='+',
        default=None,
        help="Optional lake names (e.g., HongL XYH) searched in 50lake_networks/230lake_networks."
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='./data',
        help="Data root directory (default: ./data)"
    )
    args = parser.parse_args()

    print("="*80)
    print("Dataset Statistics Collection")
    print("="*80)

    if args.lake_names:
        stats = collect_lake_stats(args.lake_names, data_root=args.data_root)
        print("\n" + "="*80)
        print("Generating Lake LaTeX Table")
        print("="*80)
        latex_table = generate_lake_latex_table(stats, args.lake_names)
        output_file = './results/lake_dataset_statistics_table.tex'
    else:
        # Collect default statistics
        stats = collect_dataset_stats()
        print("\n" + "="*80)
        print("Generating LaTeX Table")
        print("="*80)
        latex_table = generate_latex_table(stats)
        output_file = './results/dataset_statistics_table.tex'

    # Print and save
    print("\n" + latex_table)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\n\nLaTeX table saved to: {output_file}")

if __name__ == "__main__":
    main()
