#!/usr/bin/env python3
"""
Case Study Script for SO_relation datasets using gradient_based model.

Requirements:
- Use gradient_based model on ME & MT datasets
- All downstream tasks (original pathway labels + centrality-based labels)
- num_steps = 20 + 5 additional exponential deletion steps
- Output simplified graphs in same CSV format as original TSV files
- Preserve KO ID mapping
"""

import sys
import os
sys.path.append('.')

import torch
import pandas as pd
from GS.benchmark import UnifiedBenchmark
from GS.datasets import DatasetLoader
from GS.models import get_model_class
import numpy as np
import matplotlib.pyplot as plt


def save_graph_as_tsv(data, simplified_graphs, ko_mapping, output_dir, step_info):
    """
    Save simplified graphs in the same TSV format as original SO_relation files.

    Args:
        data: Original data object
        simplified_graphs: List of simplified graph edge indices
        ko_mapping: Mapping from node indices to KO IDs
        output_dir: Directory to save the TSV files
        step_info: List of dictionaries with step information
    """
    os.makedirs(output_dir, exist_ok=True)

    for step, (graph_edges, info) in enumerate(zip(simplified_graphs, step_info)):
        # Convert edge indices to KO IDs
        edges_with_kos = []

        if graph_edges.size(1) > 0:  # Check if there are edges
            # Only keep unique edges (remove duplicates from undirected graph)
            edge_list = graph_edges.t().numpy()
            unique_edges = set()

            for edge in edge_list:
                ko1 = ko_mapping[edge[0]]
                ko2 = ko_mapping[edge[1]]
                # Sort to ensure consistent ordering
                if ko1 < ko2:
                    unique_edges.add((ko1, ko2))
                else:
                    unique_edges.add((ko2, ko1))

            # Create edge dataframe with weights (set to 1.0 for simplified graphs)
            for ko1, ko2 in sorted(unique_edges):
                edges_with_kos.append({
                    'KO1': ko1,
                    'KO2': ko2,
                    'Weight': 1.0  # Simplified graphs are unweighted
                })

        # Create DataFrame
        df = pd.DataFrame(edges_with_kos)

        # Save as TSV
        filename = f"simplified_graph_step_{step:03d}.tsv"

        filepath = os.path.join(output_dir, filename)

        if len(edges_with_kos) > 0:
            df.to_csv(filepath, sep='\t', index=False)
        else:
            # Create empty file with header for completely disconnected graphs
            pd.DataFrame(columns=['KO1', 'KO2', 'Weight']).to_csv(filepath, sep='\t', index=False)

        print(f"Saved {len(edges_with_kos)} edges to {filename}")



def run_case_study_single_dataset(dataset_name, task_types, model_name='gradient_based',
                                  downstream_model='gcn', num_steps=100):
    """
    Run case study on a single SO_relation dataset with all task types.
    """
    print(f"\n{'='*80}")
    print(f"Running Case Study: {dataset_name}")
    print(f"Model: {model_name}, Downstream: {downstream_model}")
    print(f"Steps: {num_steps}")
    print(f"{'='*80}")

    results = {}

    for task_type in task_types:
        print(f"\n--- Task: {task_type} ---")

        try:
            # Create benchmark
            benchmark = UnifiedBenchmark(
                results_dir=f'./results/case_study_so_relation',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                memory_monitor=True
            )

            # Load dataset and get data
            loader = DatasetLoader('./data')
            data, train_mask, val_mask, test_mask = loader.load_dataset(dataset_name, task_type)

            # Get model class and create model
            model_class = get_model_class(model_name)
            model = model_class()

            # Run the experiment
            result = benchmark.run_single_model(
                model_name=model_name,
                dataset_name=dataset_name,
                task_type=task_type,
                downstream_model=downstream_model,
                num_steps=num_steps,
                epochs=50
            )

            if result.get('success', False):
                print(f"✅ {task_type} successful! IC-AUC: {result['snr_auc']:.4f}")
                results[task_type] = result

                # Save the simplified graphs in TSV format
                print(f"Saving simplified graphs as TSV files...")

                try:
                    # Get the simplified graphs from the benchmark result
                    result_dir = f'./results/case_study_so_relation/{dataset_name}_{task_type}_gcn'
                    summary_graphs_dir = f'{result_dir}/summary_graphs'

                    if os.path.exists(summary_graphs_dir):
                        # Read the existing CSV files and convert to edge indices
                        simplified_graphs = []

                        # Load CSV files for steps 0 to num_steps
                        for step in range(num_steps + 1):
                            csv_file = f'{summary_graphs_dir}/gradient_based_step_{step}_edges.csv'
                            if os.path.exists(csv_file):
                                # Read CSV and convert to edge_index format
                                df = pd.read_csv(csv_file)
                                if len(df) > 0:
                                    # Get the idx_to_ko mapping for converting indices to KO IDs
                                    idx_to_ko = data._idx_to_ko
                                    edge_list = []

                                    for _, row in df.iterrows():
                                        # CSV files use 'source', 'target' columns with node indices
                                        if 'source' in df.columns and 'target' in df.columns:
                                            idx1, idx2 = int(row['source']), int(row['target'])
                                            edge_list.extend([(idx1, idx2), (idx2, idx1)])

                                    if edge_list:
                                        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                                    else:
                                        edge_index = torch.empty((2, 0), dtype=torch.long)
                                else:
                                    edge_index = torch.empty((2, 0), dtype=torch.long)

                                simplified_graphs.append(edge_index)
                            else:
                                print(f"❌ CSV file not found: {csv_file}")
                                simplified_graphs.append(torch.empty((2, 0), dtype=torch.long))

                        # Create step information (only regular steps)
                        step_info = []
                        for i in range(num_steps + 1):
                            step_info.append({'step': i, 'type': 'regular'})

                        # Save graphs as TSV files
                        output_dir = f'./results/case_study_so_relation/{dataset_name}_{task_type}_summary_graphs_tsv'
                        ko_mapping = data._idx_to_ko

                        save_graph_as_tsv(data, simplified_graphs, ko_mapping, output_dir, step_info)

                        print(f"Saved {len(simplified_graphs)} simplified graphs to {output_dir}")
                    else:
                        print(f"❌ Summary graphs directory not found: {summary_graphs_dir}")

                except Exception as e:
                    print(f"❌ Error saving TSV files: {str(e)}")

            else:
                print(f"❌ {task_type} failed: {result.get('error', 'Unknown error')}")
                results[task_type] = result

        except Exception as e:
            print(f"❌ Error in {task_type}: {str(e)}")
            results[task_type] = {'success': False, 'error': str(e)}

    return results


def main():
    """Main function to run case study on both ME and MT datasets."""

    # Configuration
    datasets = ['SO_relation_ME', 'SO_relation_MT']
    task_types = ['original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality']
    model_name = 'gradient_based'
    downstream_model = 'gcn'
    num_steps = 100

    print("SO_relation Case Study")
    print("=" * 50)
    print(f"Datasets: {datasets}")
    print(f"Task types: {task_types}")
    print(f"Model: {model_name}")
    print(f"Downstream model: {downstream_model}")
    print(f"Steps: {num_steps}")
    print("=" * 50)

    all_results = {}

    for dataset_name in datasets:
        print(f"\n🔄 Processing {dataset_name}...")
        dataset_results = run_case_study_single_dataset(
            dataset_name=dataset_name,
            task_types=task_types,
            model_name=model_name,
            downstream_model=downstream_model,
            num_steps=num_steps
        )
        all_results[dataset_name] = dataset_results

    # Print summary
    print(f"\n{'='*80}")
    print("CASE STUDY SUMMARY")
    print(f"{'='*80}")

    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name}:")
        for task_type, result in dataset_results.items():
            if result.get('success', False):
                snr_auc = result.get('snr_auc', 0)
                print(f"  ✅ {task_type:20} IC-AUC: {snr_auc:.4f}")
            else:
                print(f"  ❌ {task_type:20} Failed: {result.get('error', 'Unknown')}")

    print(f"\nCase study completed! Results saved in ./results/case_study_so_relation/")
    print(f"TSV files with simplified graphs saved in *_summary_graphs_tsv/ directories")
    print(f"TSV format matches: ko_relation_min0_network_*.tsv (KO1, KO2, Weight columns)")


if __name__ == '__main__':
    main()
