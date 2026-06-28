#!/usr/bin/env python
"""
Simplified test script for PRI-Graphs baseline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
from pathlib import Path

# Import GS modules
from GS.datasets import DatasetLoader
from GS.models import GCNDownstreamModel
from GS.metrics import ComplexityMetric, InformationMetric, ICAnalysis

# Import PRI-Graphs model
from pri_graphs_model import PRIGraphsModel


def main():
    print("\n" + "="*60)
    print("Testing PRI-Graphs on Cora with GCN")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    loader = DatasetLoader('./data')
    
    # Try to load dataset, handle timeout gracefully
    try:
        original_graph, train_mask, val_mask, test_mask = loader.load_dataset('Cora')
        original_graph = loader.preprocess_for_summarization(original_graph)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Dataset may not be available. Please check network connection or use cached data.")
        return
    
    input_dim = original_graph.x.size(1)
    print(f"Graph: {original_graph.num_nodes} nodes, {original_graph.edge_index.shape[1]} edges")
    print(f"Features: {input_dim} dimensions")
    
    # Initialize PRI-Graphs model
    print("\nInitializing PRI-Graphs model...")
    model = PRIGraphsModel(
        tau=0.1,
        beta=0.5,
        alpha=0.1,
        lr=0.01,
        epochs=20,  # Reduced for faster testing
        n_samples=1,
        seed=42,
        device=device
    )
    
    # Number of steps
    num_steps = 3
    
    # Generate summary graphs
    print(f"\nGenerating {num_steps+1} summary graphs...")
    start_time = time.time()
    
    try:
        summary_graphs = model.summarize(original_graph, num_steps=num_steps)
        print(f"Summary generation took {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error during summarization: {e}")
        return
    
    # Print edge counts
    print("\nEdge counts in summary graphs:")
    for i, graph in enumerate(summary_graphs):
        print(f"  Step {i}: {graph.edge_index.shape[1]} edges")
    
    # Initialize downstream model
    print("\nInitializing downstream model...")
    downstream_model = GCNDownstreamModel(input_dim=input_dim, device=device)
    
    # Compute metrics
    print("\nComputing metrics...")
    complexity_metric = ComplexityMetric()
    info_metric = InformationMetric(downstream_model, device)
    
    complexity_metrics = complexity_metric.compute_list(summary_graphs)
    
    print("Training downstream models for information metric...")
    info_metrics = info_metric.compute_list(
        summary_graphs, train_mask, val_mask, test_mask,
        original_graph.y, epochs=30  # Reduced epochs for testing
    )
    
    # Calculate IC-AUC
    ic_auc = ICAnalysis.compute_ic_auc(complexity_metrics, info_metrics)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"IC-AUC: {ic_auc:.4f}")
    print(f"Complexity Metrics: {[f'{x:.0f}' for x in complexity_metrics]}")
    print(f"Information Metrics: {[f'{x:.4f}' for x in info_metrics]}")
    
    # Save results
    results_dir = Path('results/baselines/pri_graphs')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    df = pd.DataFrame({
        'step': list(range(len(complexity_metrics))),
        'complexity': complexity_metrics,
        'information': info_metrics
    })
    
    csv_path = results_dir / 'pri_graphs_cora_gcn_simple.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Plot IC curve
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(complexity_metrics, info_metrics, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Complexity Metric (L0 Norm)')
        plt.ylabel('Information Metric')
        plt.title(f'IC Curve - PRI-Graphs on Cora (IC-AUC={ic_auc:.4f})')
        plt.grid(True, alpha=0.3)
        
        plot_path = results_dir / 'ic_curve_cora_gcn_simple.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    print("\nTest completed successfully!")


if __name__ == '__main__':
    main()
