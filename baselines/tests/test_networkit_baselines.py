#!/usr/bin/env python
"""
Test script for NetworKit sparsification baselines.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Import GS modules
from GS.datasets import DatasetLoader
from GS.models import GCNDownstreamModel
from GS.metrics import ComplexityMetric, InformationMetric, ICAnalysis

# Import NetworKit wrapper
from networkit_wrapper import NetworKitSparsifier


def test_single_method(method_name: str, dataset_name: str = 'Cora', device: str = 'cuda'):
    """
    Test a single NetworKit sparsification method.
    
    Args:
        method_name: NetworKit method to test
        dataset_name: Dataset to use
        device: Device to run on
    """
    print(f"\n{'='*80}")
    print(f"Testing NetworKit {method_name.upper()} on {dataset_name} with GCN")
    print(f"{'='*80}")
    
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device_obj}")
    
    # Load dataset
    print("\nLoading dataset...")
    loader = DatasetLoader('./data')
    
    try:
        original_graph, train_mask, val_mask, test_mask = loader.load_dataset(dataset_name)
        original_graph = loader.preprocess_for_summarization(original_graph)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    input_dim = original_graph.x.size(1)
    print(f"Graph: {original_graph.num_nodes} nodes, {original_graph.edge_index.shape[1]} edges")
    print(f"Features: {input_dim} dimensions")
    
    # Initialize NetworKit sparsifier
    print(f"\nInitializing NetworKit {method_name} sparsifier...")
    model = NetworKitSparsifier(
        method=method_name,
        seed=42,
        device=device
    )
    
    # Get method info
    method_info = model.get_method_info()
    print(f"Method: {method_info['name']}")
    print(f"Description: {method_info['description']}")
    
    # Number of steps
    num_steps = 5
    
    # Generate summary graphs
    print(f"\nGenerating {num_steps+1} summary graphs...")
    start_time = time.time()
    
    try:
        summary_graphs = model.summarize(original_graph, num_steps=num_steps)
        print(f"Summary generation took {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None
    
    # Print edge counts
    print("\nEdge counts in summary graphs:")
    for i, graph in enumerate(summary_graphs):
        print(f"  Step {i}: {graph.edge_index.shape[1]} edges")
    
    # Initialize downstream model
    print("\nInitializing downstream model...")
    downstream_model = GCNDownstreamModel(input_dim=input_dim, device=device_obj)
    
    # Compute metrics
    print("\nComputing metrics...")
    complexity_metric = ComplexityMetric()
    info_metric = InformationMetric(downstream_model, device_obj)
    
    complexity_metrics = complexity_metric.compute_list(summary_graphs)
    
    print("Training downstream models for information metric...")
    info_metrics = info_metric.compute_list(
        summary_graphs, train_mask, val_mask, test_mask,
        original_graph.y, epochs=30
    )
    
    # Calculate IC-AUC
    ic_auc = ICAnalysis.compute_ic_auc(complexity_metrics, info_metrics)
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Method: {method_info['name']}")
    print(f"IC-AUC: {ic_auc:.4f}")
    print(f"Complexity Metrics: {[f'{x:.0f}' for x in complexity_metrics]}")
    print(f"Information Metrics: {[f'{x:.4f}' for x in info_metrics]}")
    
    return {
        'method': method_name,
        'method_name': method_info['name'],
        'dataset': dataset_name,
        'ic_auc': ic_auc,
        'snr_auc': ic_auc,  # Backward-compatible key for older result consumers.
        'complexity_metrics': complexity_metrics,
        'information_metrics': info_metrics
    }


def test_all_methods(dataset_name: str = 'Cora', device: str = 'cuda'):
    """
    Test all NetworKit sparsification methods.
    
    Args:
        dataset_name: Dataset to test on
        device: Device to run on
    """
    methods = NetworKitSparsifier.SUPPORTED_METHODS
    results = []
    
    print(f"\n{'='*100}")
    print(f"TESTING ALL NETWORKIT METHODS ON {dataset_name.upper()}")
    print(f"{'='*100}")
    print(f"Methods to test: {', '.join([m.upper() for m in methods])}")
    
    for method in methods:
        try:
            result = test_single_method(method, dataset_name, device)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR testing {method}: {e}")
            continue
    
    if not results:
        print("No successful results to analyze!")
        return
    
    # Save results
    results_dir = Path('results/baselines/networkit')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'method': result['method'],
            'method_name': result['method_name'],
            'dataset': result['dataset'],
            'ic_auc': result['ic_auc']
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_file = results_dir / f'networkit_summary_{dataset_name.lower()}.tsv'
    df_summary.to_csv(summary_file, sep='\t', index=False)
    print(f"\nSummary results saved to: {summary_file}")
    
    # Create detailed results
    detailed_data = []
    for result in results:
        for i, (complexity, information) in enumerate(zip(result['complexity_metrics'], 
                                                         result['information_metrics'])):
            detailed_data.append({
                'method': result['method'],
                'method_name': result['method_name'],
                'dataset': result['dataset'],
                'step': i,
                'complexity': complexity,
                'information': information
            })
    
    df_detailed = pd.DataFrame(detailed_data)
    detailed_file = results_dir / f'networkit_detailed_{dataset_name.lower()}.csv'
    df_detailed.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    # Create comparison plot
    try:
        plt.figure(figsize=(14, 10))
        
        for result in results:
            plt.plot(result['complexity_metrics'], result['information_metrics'], 
                    'o-', linewidth=2, markersize=6, 
                    label=f"{result['method_name']} (IC-AUC={result['ic_auc']:.1f})")
        
        plt.xlabel('Complexity Metric (L0 Norm)', fontsize=12)
        plt.ylabel('Information Metric', fontsize=12)
        plt.title(f'IC Curves - NetworKit Methods on {dataset_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_file = results_dir / f'networkit_comparison_{dataset_name.lower()}.png'
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to: {plot_file}")
    except Exception as e:
        print(f"Could not save comparison plot: {e}")
    
    # Print final summary
    print(f"\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")
    print(f"{'Method':<25} {'Name':<30} {'IC-AUC':<10}")
    print("-" * 70)
    for result in sorted(results, key=lambda x: x['ic_auc'], reverse=True):
        print(f"{result['method']:<25} {result['method_name']:<30} {result['ic_auc']:<10.2f}")
    
    best_method = max(results, key=lambda x: x['ic_auc'])
    print(f"\nBest performing method: {best_method['method_name']} "
          f"(IC-AUC: {best_method['ic_auc']:.4f})")


def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NetworKit sparsification baselines')
    parser.add_argument('--method', type=str, default='all', 
                       choices=['all'] + NetworKitSparsifier.SUPPORTED_METHODS,
                       help='Method to test (default: all)')
    parser.add_argument('--dataset', type=str, default='Cora',
                       choices=['Cora', 'CiteSeer', 'PubMed'],
                       help='Dataset to use (default: Cora)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    if args.method == 'all':
        test_all_methods(args.dataset, args.device)
    else:
        test_single_method(args.method, args.dataset, args.device)


if __name__ == '__main__':
    main()
