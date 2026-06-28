"""
Main Graph Summarization Model Implementation

Implements the LearnableGraphSummarization model according to detailed specifications in MODEL.md,
including complete neural network architecture and training strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, GATConv, SAGEConv
from typing import List, Optional, Dict, Any, Callable
import copy
import math
import numpy as np
from abc import abstractmethod

from .base import GraphSummarizationModel


class GINLayer(nn.Module):
    """
    GIN layer implementation, including BatchNorm, ReLU and Dropout.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 eps: float = 0.0, 
                 train_eps: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        
        self.gin_conv = GINConv(
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            ),
            eps=eps,
            train_eps=train_eps
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.gin_conv(x, edge_index)
        x = self.dropout(x)
        return x


class EdgeScorer(nn.Module):
    """
    Edge classifier MLP, outputs removal probability for each edge.
    """
    
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.mlp = nn.Sequential(
            # FC1: input -> 512
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            
            # FC2: 512 -> 256  
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            # FC3: 256 -> 1
            nn.Linear(256, 1)
        )
        
    def forward(self, edge_features):
        return self.mlp(edge_features)


class LearnableGraphSummarization(GraphSummarizationModel, nn.Module):
    """
    Learnable graph summarization model, fully implements the architecture defined in MODEL.md.

    Architecture:
    - Node Encoder: 3-layer GIN with hidden_dim=256
    - Step Embedding: Learnable lookup table with d_s=32
    - Edge Scorer: 3-layer MLP (input -> 512 -> 256 -> 1)
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 step_emb_dim: int = 32,
                 num_gin_layers: int = 3,
                 dropout: float = 0.2,
                 max_steps: int = 20,
                 device: str = 'cpu',
                 node_encoder_type: str = 'gin',  # 'gin', 'gat', 'sage'
                 use_step_embedding: bool = True,
                 use_edge_diff: bool = True,
                 **kwargs):
        """
        Initialize the model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (default 256)
            step_emb_dim: Step embedding dimension (default 32)
            num_gin_layers: Number of GIN layers (default 3)
            dropout: Dropout rate (default 0.2)
            max_steps: Maximum number of steps
            device: Computing device
            node_encoder_type: Node encoder type
            use_step_embedding: Whether to use step embedding
            use_edge_diff: Whether to use edge difference features
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.step_emb_dim = step_emb_dim
        self.num_gin_layers = num_gin_layers
        self.dropout = dropout
        self.max_steps = max_steps
        self.device = device
        self.node_encoder_type = node_encoder_type
        self.use_step_embedding = use_step_embedding
        self.use_edge_diff = use_edge_diff
        
        # Initialize node encoder
        self._build_node_encoder()

        # Step embedding
        if self.use_step_embedding:
            self.step_embedding = nn.Embedding(max_steps + 1, step_emb_dim)

        # Edge representation dimension calculation
        edge_repr_dim = 2 * hidden_dim  # h_u + h_v
        if self.use_edge_diff:
            edge_repr_dim += hidden_dim  # |h_u - h_v|
        if self.use_step_embedding:
            edge_repr_dim += step_emb_dim  # step embedding

        # Edge classifier
        self.edge_scorer = EdgeScorer(edge_repr_dim, dropout)

        # Xavier initialization
        self._initialize_weights()

        # Ensure all model parameters are float32 type
        self.float()
    
    def _build_node_encoder(self):
        """Build node encoder"""
        if self.node_encoder_type == 'gin':
            # Input projection layer
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

            # GIN layers
            self.gin_layers = nn.ModuleList([
                GINLayer(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    eps=0.0,
                    train_eps=True,
                    dropout=self.dropout
                ) for _ in range(self.num_gin_layers)
            ])
            
        elif self.node_encoder_type == 'gat':
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
            self.gat_layers = nn.ModuleList([
                GATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    dropout=self.dropout
                ) for _ in range(self.num_gin_layers)
            ])
            
        elif self.node_encoder_type == 'sage':
            self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
            self.sage_layers = nn.ModuleList([
                SAGEConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim
                ) for _ in range(self.num_gin_layers)
            ])
        else:
            raise ValueError(f"Unsupported encoder type: {self.node_encoder_type}")
    
    def _initialize_weights(self):
        """Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Node encoding.

        Args:
            x: Node features [n, d_in]
            edge_index: Edge index [2, num_edges]

        Returns:
            Node embeddings [n, d_h]
        """
        # Ensure input is float32 type
        if x.dtype != torch.float32:
            x = x.float()

        # Input projection
        h = self.input_projection(x)

        # Through encoder layers
        if self.node_encoder_type == 'gin':
            for gin_layer in self.gin_layers:
                h = gin_layer(h, edge_index)
        elif self.node_encoder_type == 'gat':
            for gat_layer in self.gat_layers:
                h = F.relu(gat_layer(h, edge_index))
                h = F.dropout(h, p=self.dropout, training=self.training)
        elif self.node_encoder_type == 'sage':
            for sage_layer in self.sage_layers:
                h = F.relu(sage_layer(h, edge_index))
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def compute_edge_features(self,
                            node_embeddings: torch.Tensor,
                            edge_index: torch.Tensor,
                            step: int) -> torch.Tensor:
        """
        Compute edge feature representations.

        Args:
            node_embeddings: Node embeddings [n, d_h]
            edge_index: Edge index [2, num_edges]
            step: Current step

        Returns:
            Edge features [num_edges, edge_feature_dim]
        """
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_u = node_embeddings[src_nodes]  # [num_edges, d_h]
        h_v = node_embeddings[dst_nodes]  # [num_edges, d_h]

        # Basic edge features: [h_u; h_v]
        edge_features = [h_u, h_v]

        # Difference features: |h_u - h_v|
        if self.use_edge_diff:
            edge_diff = torch.abs(h_u - h_v)
            edge_features.append(edge_diff)

        # Step embedding
        if self.use_step_embedding:
            step_tensor = torch.full((edge_index.size(1),), step,
                                   dtype=torch.long, device=edge_index.device)
            step_emb = self.step_embedding(step_tensor)  # [num_edges, d_s]
            edge_features.append(step_emb)

        # Concatenate all features
        edge_repr = torch.cat(edge_features, dim=1)
        return edge_repr
    
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                step: int) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x: Node features [n, d_in]
            edge_index: Edge index [2, num_edges]
            step: Current step

        Returns:
            Edge scores [num_edges]
        """
        # Ensure data type consistency
        if x.dtype != torch.float32:
            x = x.float()

        # Node encoding
        node_embeddings = self.encode_nodes(x, edge_index)

        # Edge feature computation
        edge_features = self.compute_edge_features(node_embeddings, edge_index, step)

        # Edge classification
        edge_scores = self.edge_scorer(edge_features).squeeze(-1)

        return edge_scores
    
    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        Generate graph summary sequence.

        Args:
            original_graph: Original graph
            num_steps: Number of summary steps

        Returns:
            Summary graph list, including original graph, total of num_steps+1 graphs
        """
        self.eval()
        summary_graphs = []
        current_graph = copy.deepcopy(original_graph)

        # Step 0: Original graph
        summary_graphs.append(current_graph)

        with torch.no_grad():
            for step in range(1, num_steps + 1):
                if current_graph.edge_index.size(1) == 0:
                    # Already empty graph, subsequent ones are all empty
                    empty_graph = Data(
                        x=original_graph.x,
                        edge_index=torch.zeros((2, 0), dtype=torch.long,
                                             device=original_graph.edge_index.device),
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                    summary_graphs.append(empty_graph)
                    continue

                # Compute edge scores
                edge_scores = self.forward(current_graph.x, current_graph.edge_index, step)

                # Determine number of edges to keep
                current_edges = current_graph.edge_index.size(1)
                keep_ratio = 1.0 - (step / num_steps)
                num_keep = max(0, int(current_edges * keep_ratio))

                if num_keep == 0:
                    # Empty graph
                    empty_graph = Data(
                        x=original_graph.x,
                        edge_index=torch.zeros((2, 0), dtype=torch.long,
                                             device=original_graph.edge_index.device),
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                    summary_graphs.append(empty_graph)
                    current_graph = empty_graph
                else:
                    # Select edges with highest scores
                    _, top_indices = torch.topk(edge_scores, num_keep, largest=True)
                    kept_edge_index = current_graph.edge_index[:, top_indices]

                    next_graph = Data(
                        x=original_graph.x,
                        edge_index=kept_edge_index,
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                    summary_graphs.append(next_graph)
                    current_graph = next_graph

        return summary_graphs

    def reset(self) -> None:
        """Reset model parameters"""
        self._initialize_weights()


# Ablation experiment variants
class LearnableGraphSummarization_GAT(LearnableGraphSummarization):
    """Variant using GAT as node encoder"""

    def __init__(self, *args, **kwargs):
        kwargs['node_encoder_type'] = 'gat'
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_SAGE(LearnableGraphSummarization):
    """Variant using GraphSAGE as node encoder"""

    def __init__(self, *args, **kwargs):
        kwargs['node_encoder_type'] = 'sage'
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_NoStepEmb(LearnableGraphSummarization):
    """Variant without step embedding"""

    def __init__(self, *args, **kwargs):
        kwargs['use_step_embedding'] = False
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_NoEdgeDiff(LearnableGraphSummarization):
    """Variant without edge difference features"""

    def __init__(self, *args, **kwargs):
        kwargs['use_edge_diff'] = False
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_SmallHidden(LearnableGraphSummarization):
    """Variant with smaller hidden dimension"""

    def __init__(self, input_dim, *args, **kwargs):
        kwargs['hidden_dim'] = 128
        super().__init__(input_dim, *args, **kwargs)


class LearnableGraphSummarization_LargeHidden(LearnableGraphSummarization):
    """Variant with larger hidden dimension"""

    def __init__(self, input_dim, *args, **kwargs):
        kwargs['hidden_dim'] = 512
        super().__init__(input_dim, *args, **kwargs)


class LearnableGraphSummarization_DeepGIN(LearnableGraphSummarization):
    """Variant with deeper GIN network"""

    def __init__(self, *args, **kwargs):
        kwargs['num_gin_layers'] = 5
        super().__init__(*args, **kwargs)


class LearnableGraphSummarization_ShallowGIN(LearnableGraphSummarization):
    """Variant with shallower GIN network"""

    def __init__(self, *args, **kwargs):
        kwargs['num_gin_layers'] = 2
        super().__init__(*args, **kwargs)