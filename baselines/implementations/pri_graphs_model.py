"""
Standalone PRI-Graphs model implementation for benchmark testing.
This integrates PRI-Graphs baseline into the GS framework without modifying existing model files.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'PRI-Graphs'))

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from typing import List
import copy
import warnings
warnings.filterwarnings('ignore')

# Import from PRI-Graphs
try:
    # Add PRI-Graphs directory to path
    pri_graphs_path = os.path.join(os.path.dirname(__file__), '..', 'PRI-Graphs')
    if pri_graphs_path not in sys.path:
        sys.path.insert(0, pri_graphs_path)
    
    from graph_sparse import sparsification_stochastic, incidence
    print("✓ PRI-Graphs module loaded successfully")
except ImportError as e:
    raise ImportError(f"PRI-Graphs graph_sparse module not available: {e}. Please ensure PRI-Graphs is properly installed.")

# Import base class from GS
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from GS.models.base import GraphSummarizationModel


class PRIGraphsModel(GraphSummarizationModel):
    """
    PRI-Graphs baseline model for graph summarization.
    Uses Principle of Relevant Information for graph sparsification.
    """
    
    def __init__(self, tau=0.1, beta=0.5, alpha=0.1, lr=0.01, 
                 epochs=50, n_samples=1, seed=42, device='cuda'):
        """
        Initialize PRI-Graphs model.
        
        Args:
            tau: Temperature parameter for Gumbel-softmax
            beta: Regularization parameter for entropy loss
            alpha: Connectivity loss weight
            lr: Learning rate for optimization
            epochs: Number of epochs for sparsification
            n_samples: Number of samples for stochastic optimization
            seed: Random seed
            device: Device to use for computation
        """
        # Verify dependencies are available
        if sparsification_stochastic is None or incidence is None:
            raise ImportError("PRI-Graphs dependencies not available. Cannot initialize PRIGraphsModel.")
        
        self.tau = tau
        self.beta = beta
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.n_samples = n_samples
        self.seed = seed
        self.device = device if torch.cuda.is_available() else 'cpu'
        
    def reset(self):
        """Reset model state. PRI-Graphs doesn't maintain state."""
        pass
    
    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        Generate a series of simplified graphs using PRI-Graphs method.
        
        Args:
            original_graph: Input graph to be simplified
            num_steps: Number of simplification steps (N_step)
            
        Returns:
            List[Data]: List of simplified graphs from original to empty
        """
        # Move graph to CPU for NetworkX processing
        edge_index = original_graph.edge_index.cpu()
        num_nodes = original_graph.num_nodes
        
        # Convert to NetworkX for PRI-Graphs
        edges = edge_index.t().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges)
        
        # Get total number of edges
        num_edges = G.number_of_edges()
        
        # Generate summary graphs
        summary_graphs = []
        
        # Step 0: Original graph with float dtype
        orig_graph_copy = Data(
            x=original_graph.x.float(),
            edge_index=original_graph.edge_index,
            y=original_graph.y,
            num_nodes=original_graph.x.size(0)
        )
        summary_graphs.append(orig_graph_copy)
        
        if num_edges == 0 or num_steps <= 0:
            # If no edges or no steps, return original graph repeated
            for _ in range(num_steps):
                summary_graphs.append(copy.deepcopy(original_graph))
            return summary_graphs
        
        # Calculate sparsity levels for each step
        # We want to go from 100% edges to 0% edges over num_steps
        sparsity_levels = []
        for step in range(1, num_steps + 1):
            # Linear interpolation from 1.0 to 0.0
            remaining_ratio = 1.0 - (step / num_steps)
            sparsity_levels.append(remaining_ratio)
        
        # Generate graphs at each sparsity level
        for step, target_sparsity in enumerate(sparsity_levels, 1):
            if target_sparsity <= 0.0:
                # Empty graph
                empty_graph = Data(
                    x=original_graph.x.float(),  # Ensure float dtype
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    y=original_graph.y,
                    num_nodes=num_nodes
                )
                summary_graphs.append(empty_graph)
            else:
                # Apply PRI-Graphs sparsification
                target_edges = max(0, int(num_edges * target_sparsity))
                
                if target_edges == 0:
                    # Empty graph
                    empty_graph = Data(
                        x=original_graph.x.float(),  # Ensure float dtype
                        edge_index=torch.zeros((2, 0), dtype=torch.long),
                        y=original_graph.y,
                        num_nodes=num_nodes
                    )
                    summary_graphs.append(empty_graph)
                    continue
                
                try:
                    # Run PRI-Graphs sparsification
                    E = torch.tensor(incidence(G), dtype=torch.float32)
                    
                    # Initialize weights
                    w = torch.ones(E.shape[1], dtype=torch.float32)
                    
                    # Optimize weights using PRI approach
                    w_best, sigma, E_result, theta, probs, cost_vec, history = sparsification_stochastic(
                        G,
                        tau=self.tau,
                        n_samples=self.n_samples,
                        epochs=min(self.epochs, 30),  # Limit for efficiency
                        lr=self.lr,
                        beta=self.beta,
                        alpha=self.alpha * target_sparsity,  # Scale alpha with sparsity
                        loss_type='vn',
                        seed=self.seed + step,
                        verbose=False,
                        hard=True,
                        plot_flag=False
                    )
                    
                    # Select top-k edges based on weights
                    if len(w_best) > target_edges:
                        _, top_indices = torch.topk(w_best, target_edges)
                        keep_mask = torch.zeros(len(w_best), dtype=torch.bool)
                        keep_mask[top_indices] = True
                    else:
                        keep_mask = torch.ones(len(w_best), dtype=torch.bool)
                    
                    # Convert edge indices
                    edge_list = list(G.edges())
                    kept_edges = [edge_list[i] for i in range(len(edge_list)) if keep_mask[i]]
                    
                    if kept_edges:
                        new_edge_index = torch.tensor(kept_edges, dtype=torch.long).t()
                        # Add reverse edges for undirected graph
                        new_edge_index = torch.cat([new_edge_index, new_edge_index[[1, 0]]], dim=1)
                    else:
                        new_edge_index = torch.zeros((2, 0), dtype=torch.long)
                    
                    sparse_graph = Data(
                        x=original_graph.x.float(),  # Ensure float dtype
                        edge_index=new_edge_index,
                        y=original_graph.y,
                        num_nodes=num_nodes
                    )
                    summary_graphs.append(sparse_graph)
                    
                except Exception as e:
                    print(f"Error: PRI-Graphs failed at step {step}: {e}")
                    raise e  # Don't use fallback, let the error propagate
        
        return summary_graphs