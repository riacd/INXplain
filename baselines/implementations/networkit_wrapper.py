#!/usr/bin/env python
"""
NetworKit-based graph sparsification models for benchmarking.
"""

import torch
import numpy as np
import networkit as nk
from typing import List, Tuple, Optional, Dict
from torch_geometric.data import Data
import copy
import sys
import os

# Add GS package to path for importing base class
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from GS.models.base import GraphSummarizationModel


class NetworKitSparsifier(GraphSummarizationModel):
    """
    NetworKit图稀疏化模型包装器
    
    实现统一的GraphSummarizationModel接口，支持NetworKit库的多种稀疏化方法。
    """
    
    SUPPORTED_METHODS = [
        'forest_fire',
        'local_degree', 
        'local_similarity',
        'random_edge',
        'random_node_edge',
        'scan',
        'simmelian'
    ]
    
    def __init__(self, 
                 method: str = 'random_edge',
                 seed: int = 42,
                 device: str = 'cpu'):
        """
        Initialize NetworKit sparsifier.
        
        Args:
            method: Sparsification method to use
            seed: Random seed for reproducibility
            device: Device to use (cpu/cuda)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method {method} not supported. Choose from {self.SUPPORTED_METHODS}")
        
        self.method = method
        self.seed = seed
        self.device = device
        nk.setSeed(seed, useThreadId=False)
        
    def _torch_to_networkit(self, edge_index: torch.Tensor, num_nodes: int) -> nk.Graph:
        """
        Convert PyTorch Geometric edge_index to NetworKit graph.
        
        Args:
            edge_index: Edge indices in COO format (2, num_edges)
            num_nodes: Number of nodes
            
        Returns:
            NetworKit graph
        """
        # Create NetworKit graph
        G = nk.Graph(num_nodes, weighted=False, directed=False)
        
        # Add edges (only add one direction for undirected graph)
        edges_set = set()
        edge_array = edge_index.cpu().numpy()
        
        for i in range(edge_array.shape[1]):
            u, v = int(edge_array[0, i]), int(edge_array[1, i])
            if u != v:  # Skip self-loops
                edge_pair = (min(u, v), max(u, v))
                if edge_pair not in edges_set:
                    edges_set.add(edge_pair)
                    G.addEdge(u, v)
        
        return G
    
    def _networkit_to_torch(self, G: nk.Graph) -> torch.Tensor:
        """
        Convert NetworKit graph to PyTorch edge_index.
        
        Args:
            G: NetworKit graph
            
        Returns:
            Edge indices in COO format
        """
        edges = []
        for u, v in G.iterEdges():
            edges.append([u, v])
            edges.append([v, u])  # Add reverse edge for undirected
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return edge_index
    
    def _get_sparsifier(self, G: nk.Graph, target_ratio: float):
        """
        Get the appropriate NetworKit sparsifier based on method.
        
        Args:
            G: NetworKit graph
            target_ratio: Target edge retention ratio (0 to 1)
            
        Returns:
            Sparsifier object
        """
        if self.method == 'forest_fire':
            # Forest Fire Sparsifier
            burn_prob = 0.7  # Probability of burning
            target_burnt = 1.0 - target_ratio  # How much to burn
            return nk.sparsification.ForestFireSparsifier(
                burnProbability=burn_prob,
                targetBurntRatio=target_burnt
            )
            
        elif self.method == 'local_degree':
            # Local Degree Sparsifier
            return nk.sparsification.LocalDegreeSparsifier()
            
        elif self.method == 'local_similarity':
            # Local Similarity Sparsifier  
            return nk.sparsification.LocalSimilaritySparsifier()
            
        elif self.method == 'random_edge':
            # Random Edge Sparsifier
            return nk.sparsification.RandomEdgeSparsifier()
            
        elif self.method == 'random_node_edge':
            # Random Node Edge Sparsifier
            return nk.sparsification.RandomNodeEdgeSparsifier()
            
        elif self.method == 'scan':
            # SCAN Structural Similarity Sparsifier
            return nk.sparsification.SCANSparsifier()
            
        elif self.method == 'simmelian':
            # Simmelian Overlap Sparsifier
            return nk.sparsification.SimmelianSparsifierNonParametric()
        
        else:
            raise ValueError(f"Method {self.method} not implemented")
    
    def _sparsify_to_target_edges(self, G: nk.Graph, target_edges: int) -> nk.Graph:
        """
        Sparsify graph to achieve target edge count using edge scores.
        
        Args:
            G: NetworKit graph
            target_edges: Target number of edges
            
        Returns:
            Sparsified graph with approximately target_edges edges
        """
        current_edges = G.numberOfEdges()
        
        if target_edges >= current_edges:
            return G  # No sparsification needed
        
        # Compute edge scores using the specified method
        if self.method == 'random_edge':
            scorer = nk.sparsification.RandomEdgeScore(G)
        elif self.method == 'local_degree':
            scorer = nk.sparsification.LocalDegreeScore(G)
        elif self.method == 'local_similarity':
            # Need triangle scores for local similarity
            triangle_scorer = nk.sparsification.TriangleEdgeScore(G)
            triangle_scorer.run()
            triangles = triangle_scorer.scores()
            scorer = nk.sparsification.LocalSimilarityScore(G, triangles)
        elif self.method == 'forest_fire':
            scorer = nk.sparsification.ForestFireScore(G, 0.7, 5.0)
        elif self.method == 'random_node_edge':
            scorer = nk.sparsification.RandomNodeEdgeScore(G)
        elif self.method == 'scan':
            # SCAN needs triangles parameter
            triangle_scorer = nk.sparsification.TriangleEdgeScore(G)
            triangle_scorer.run()
            triangles = triangle_scorer.scores()
            scorer = nk.sparsification.SCANStructuralSimilarityScore(G, triangles)
        elif self.method == 'simmelian':
            # Simmelian needs triangles and maxRank parameters
            triangle_scorer = nk.sparsification.TriangleEdgeScore(G)
            triangle_scorer.run()
            triangles = triangle_scorer.scores()
            maxRank = 5  # Set reasonable maxRank
            scorer = nk.sparsification.SimmelianOverlapScore(G, triangles, maxRank)
        else:
            # Fallback to random for unknown methods
            scorer = nk.sparsification.RandomEdgeScore(G)
        
        # Run the scorer
        scorer.run()
        scores = scorer.scores()
        
        # Sort edges by their scores and keep the top target_edges
        edge_scores = []
        for u, v in G.iterEdges():
            edge_id = G.edgeId(u, v)
            edge_scores.append((scores[edge_id], u, v))
        
        # Sort by score (higher scores = more important edges)
        edge_scores.sort(reverse=True)
        
        # Create new graph with top target_edges edges
        G_sparse = nk.Graph(G.numberOfNodes(), weighted=False, directed=False)
        edges_added = 0
        
        for score, u, v in edge_scores:
            if edges_added >= target_edges:
                break
            G_sparse.addEdge(u, v)
            edges_added += 1
        
        return G_sparse
    
    def summarize(self, 
                  original_graph: Data, 
                  num_steps: int = 5) -> List[Data]:
        """
        Generate summary graphs at different sparsity levels with uniform progression.
        
        Args:
            original_graph: Original graph data
            num_steps: Number of sparsification steps
            
        Returns:
            List of sparsified graphs with uniformly decreasing edge counts
        """
        num_nodes = original_graph.x.size(0)
        edge_index = original_graph.edge_index
        
        # Convert to NetworKit graph
        G = self._torch_to_networkit(edge_index, num_nodes)
        G.indexEdges()  # Index edges for sparsification methods
        original_edges = G.numberOfEdges()
        
        summary_graphs = []
        
        # Step 0: Original graph
        orig_graph_copy = Data(
            x=original_graph.x.float(),
            edge_index=original_graph.edge_index,
            y=original_graph.y,
            num_nodes=num_nodes
        )
        summary_graphs.append(orig_graph_copy)
        
        # Generate target edge counts uniformly distributed from original_edges to 0
        target_edge_counts = []
        for step in range(1, num_steps + 1):
            if step == num_steps:
                target_edge_counts.append(0)  # Last step is always empty graph
            else:
                # Linear interpolation: from original_edges to 0
                target_edges = int(original_edges * (1.0 - step / num_steps))
                target_edge_counts.append(target_edges)
        
        print(f"Target edge counts: {[original_edges] + target_edge_counts}")
        
        # Generate sparsified graphs to match target edge counts
        for step_idx, target_edges in enumerate(target_edge_counts):
            step = step_idx + 1
            
            if target_edges <= 0:
                # Empty graph
                empty_graph = Data(
                    x=original_graph.x.float(),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    y=original_graph.y,
                    num_nodes=num_nodes
                )
                summary_graphs.append(empty_graph)
            else:
                # Use NetworKit sparsification to achieve target edge count
                G_sparse = self._sparsify_to_target_edges(G, target_edges)
                
                # Convert back to PyTorch
                sparse_edge_index = self._networkit_to_torch(G_sparse)
                
                # Create sparse graph
                sparse_graph = Data(
                    x=original_graph.x.float(),
                    edge_index=sparse_edge_index,
                    y=original_graph.y,
                    num_nodes=num_nodes
                )
                summary_graphs.append(sparse_graph)
        
        return summary_graphs
    
    def reset(self) -> None:
        """重置模型状态（NetworKit模型无状态，无需重置）"""
        pass
    
    def get_method_info(self) -> Dict[str, str]:
        """
        Get information about the current method.
        
        Returns:
            Dictionary with method information
        """
        method_info = {
            'forest_fire': {
                'name': 'Edge Forest Fire (EFF)',
                'description': 'Forest Fire sparsification based on random walks'
            },
            'local_degree': {
                'name': 'Local Degree (LD)', 
                'description': 'Determines maximum parameter value for edge retention'
            },
            'local_similarity': {
                'name': 'Local Similarity (LS)',
                'description': 'Sparsification based on local node similarity'
            },
            'random_edge': {
                'name': 'Random Edge (RE)',
                'description': 'Randomly delete edges'
            },
            'random_node_edge': {
                'name': 'Random Node Edge (RN)',
                'description': 'Randomly delete edges based on nodes'
            },
            'scan': {
                'name': 'SCAN Structural Similarity',
                'description': 'Structural Clustering Algorithm for Networks'
            },
            'simmelian': {
                'name': 'Simmelian Overlap (SO)',
                'description': 'Calculates minimum parameter for edge retention'
            }
        }
        
        return method_info.get(self.method, {'name': self.method, 'description': 'Unknown method'})