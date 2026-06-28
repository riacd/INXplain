import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'PRI-Graphs'))

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from graph_sparse import sparsification_stochastic, incidence, get_graph_from_incidence


class PRIGraphsWrapper(nn.Module):
    """
    Wrapper for PRI-Graphs baseline model.
    Implements graph sparsification using the PRI (Principle of Relevant Information) approach.
    """
    
    def __init__(self, n_steps: int = 5, tau: float = 0.1, beta: float = 0.5, 
                 alpha: float = 0.1, lr: float = 0.01, epochs: int = 100,
                 n_samples: int = 1, seed: int = 42):
        """
        Initialize PRI-Graphs wrapper.
        
        Args:
            n_steps: Number of sparsification steps
            tau: Temperature parameter for Gumbel-softmax
            beta: Regularization parameter
            alpha: Connectivity loss weight
            lr: Learning rate
            epochs: Number of training epochs
            n_samples: Number of samples for stochastic optimization
            seed: Random seed
        """
        super().__init__()
        self.n_steps = n_steps
        self.tau = tau
        self.beta = beta
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.n_samples = n_samples
        self.seed = seed
        
    def forward(self, edge_index: torch.Tensor, num_nodes: int, 
                x: torch.Tensor = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate a sequence of sparsified graphs.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes in the graph
            x: Node features (not used by PRI-Graphs)
            
        Returns:
            List of (edge_index, edge_weight) tuples for each sparsification step
        """
        # Convert edge_index to NetworkX graph
        edges = edge_index.t().cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edges)
        
        # Get incidence matrix
        E = torch.tensor(incidence(G), dtype=torch.float32)
        num_edges = E.shape[1]
        
        if num_edges == 0:
            # Return empty graphs if no edges
            return [(torch.zeros((2, 0), dtype=torch.long), 
                    torch.zeros(0)) for _ in range(self.n_steps + 1)]
        
        # Calculate target sparsity levels
        sparsity_levels = np.linspace(1.0, 0.0, self.n_steps + 1)
        
        graph_sequence = []
        
        for step, target_sparsity in enumerate(sparsity_levels):
            if target_sparsity == 1.0:
                # Original graph
                graph_sequence.append((edge_index, torch.ones(edge_index.shape[1])))
            elif target_sparsity == 0.0:
                # Empty graph
                graph_sequence.append((torch.zeros((2, 0), dtype=torch.long), 
                                      torch.zeros(0)))
            else:
                # Apply PRI-Graphs sparsification
                target_edges = int(num_edges * target_sparsity)
                
                # Run sparsification
                try:
                    sigma, rho, theta, w_best = sparsification_stochastic(
                        G, 
                        tau=self.tau,
                        n_samples=self.n_samples,
                        epochs=min(self.epochs, 50),  # Limit epochs for efficiency
                        lr=self.lr,
                        beta=self.beta,
                        alpha=self.alpha,
                        loss_type='vn',
                        seed=self.seed + step,
                        verbose=False,
                        hard=True,
                        plot_flag=False
                    )
                    
                    # Get sparse graph from weights
                    w_sorted = torch.sort(w_best, descending=True)[0]
                    threshold = w_sorted[min(target_edges, len(w_sorted)-1)] if target_edges > 0 else 1.0
                    w_sparse = (w_best >= threshold).float() * w_best
                    
                    # Create sparse edge index
                    edge_mask = w_sparse > 0
                    sparse_edge_index = edge_index[:, edge_mask]
                    sparse_edge_weight = w_sparse[edge_mask]
                    
                    graph_sequence.append((sparse_edge_index, sparse_edge_weight))
                    
                except Exception as e:
                    # Fallback to random sparsification if PRI-Graphs fails
                    print(f"Warning: PRI-Graphs failed at step {step}, using random sparsification: {e}")
                    perm = torch.randperm(edge_index.shape[1])[:target_edges]
                    sparse_edge_index = edge_index[:, perm]
                    sparse_edge_weight = torch.ones(target_edges)
                    graph_sequence.append((sparse_edge_index, sparse_edge_weight))
        
        return graph_sequence
    
    def train_step(self, *args, **kwargs):
        """PRI-Graphs doesn't require external training."""
        pass