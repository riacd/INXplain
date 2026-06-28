"""
Downstream task model implementations

Contains models such as GCN and GAT for downstream tasks like node classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.models import GAE, VGAE
from torch_geometric.utils import negative_sampling, to_undirected
from typing import Optional
from tqdm import tqdm
from .base import DownstreamModel
import importlib
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


DOWNSTREAM_MODEL_ALIASES = {
    'gcn': 'gcn',
    'gat': 'gat',
    'sage': 'sage',
    'graphsage': 'sage',
    'graph_sage': 'sage',
    'h2gcn': 'h2gcn',
    'gcnii': 'gcnii',
    'gcnii_conv': 'gcnii',
    'gcn2': 'gcnii',
}


def normalize_downstream_model_name(model_type: str) -> str:
    """Return the canonical CLI/config name for a downstream model."""
    normalized = model_type.lower()
    if normalized not in DOWNSTREAM_MODEL_ALIASES:
        supported = ', '.join(sorted(set(DOWNSTREAM_MODEL_ALIASES.values())))
        raise ValueError(
            f"Unsupported downstream model type: {model_type}. "
            f"Supported canonical names: {supported}"
        )
    return DOWNSTREAM_MODEL_ALIASES[normalized]


def create_downstream_model(model_type: str, input_dim: int, device=None, **kwargs):
    """Create a downstream evaluator while preserving architecture defaults."""
    normalized = normalize_downstream_model_name(model_type)
    model_classes = {
        'gcn': GCNDownstreamModel,
        'gat': GATDownstreamModel,
        'sage': GraphSAGEDownstreamModel,
        'h2gcn': H2GCNDownstreamModel,
        'gcnii': GCNIIDownstreamModel,
    }
    return model_classes[normalized](input_dim=input_dim, device=device, **kwargs)


class NodeClassificationDownstreamMixin:
    """Shared full-batch node-classification training loop."""

    def train_model(self,
                    graph: Data,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    labels: torch.Tensor,
                    epochs: int = 200) -> None:
        if self.output_dim is None:
            self.output_dim = int(labels.max()) + 1
            self._build_model()

        self.model.train()
        graph.x = graph.x.float()
        graph = graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        labels = labels.to(self.device)
        best_state = None
        best_val_loss = None
        patience_counter = 0

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            try:
                out = self.model(graph.x, graph.edge_index)
                loss = F.nll_loss(out[train_mask], labels[train_mask])
                loss.backward()
                self.optimizer.step()

                if epoch % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if self.early_stopping_patience is not None and val_mask.sum() > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(graph.x, graph.edge_index)
                        val_loss = F.nll_loss(val_out[val_mask], labels[val_mask]).item()
                    self.model.train()

                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{self.__class__.__name__} memory error at epoch {epoch}, clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise e

        if best_state is not None:
            self.model.load_state_dict({
                key: value.to(self.device)
                for key, value in best_state.items()
            })

    def evaluate(self,
                 graph: Data,
                 test_mask: torch.Tensor,
                 labels: torch.Tensor) -> float:
        self.model.eval()
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        test_mask = test_mask.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            test_loss = F.nll_loss(out[test_mask], labels[test_mask])
        return float(test_loss)

    def predict(self, graph: Data) -> torch.Tensor:
        self.model.eval()
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        with torch.no_grad():
            return self.model(graph.x, graph.edge_index)

    def reset(self) -> None:
        if self.model is not None:
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._build_model()


class GCNDownstreamModel(DownstreamModel):
    """
    GCN model for downstream tasks like node classification.
    
    This model uses a 2-layer GCN architecture and can be trained
    on any graph structure.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 16,
                 dropout: float = 0.5,
                 lr: float = 0.01,
                 weight_decay: float = 5e-4,
                 output_dim: Optional[int] = None,
                 early_stopping_patience: Optional[int] = 10,
                 device: Optional[torch.device] = None):
        """
        Initialize GCN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            device: Computation device
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device is not None else torch.device('cpu')
        
        self.model = None
        self.optimizer = None
        self._build_model()
        
    def _build_model(self):
        """Build the GCN model."""
        if self.output_dim is None:
            # Will be set during first training call
            return
            
        self.model = GCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim, 
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device).float()
        
        self.optimizer = torch.optim.Adam([
            {'params': self.model.conv1.parameters(), 'weight_decay': self.weight_decay},
            {'params': self.model.conv2.parameters(), 'weight_decay': 0.0},
        ], lr=self.lr)
    
    def train_model(self,
                    graph: Data,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    labels: torch.Tensor,
                    epochs: int = 200) -> None:
        """Train the GCN model with memory optimization."""
        # Set output dimension if not set
        if self.output_dim is None:
            self.output_dim = int(labels.max()) + 1
            self._build_model()

        self.model.train()

        # Ensure float32 dtype for compatibility and move to device
        if graph.x.dtype == torch.float64:
            graph.x = graph.x.float()
        graph = graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        labels = labels.to(self.device)

        best_state = None
        best_val_loss = None
        patience_counter = 0

        # Memory-optimized training loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            try:
                # Forward pass
                out = self.model(graph.x, graph.edge_index)
                loss = F.nll_loss(out[train_mask], labels[train_mask])

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Clear intermediate gradients to save memory
                if epoch % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if self.early_stopping_patience is not None and val_mask.sum() > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(graph.x, graph.edge_index)
                        val_loss = F.nll_loss(val_out[val_mask], labels[val_mask]).item()
                    self.model.train()

                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ Memory error at epoch {epoch}, clearing cache and continuing...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Retry with gradient checkpointing or skip this epoch
                    continue
                else:
                    raise e

        if best_state is not None:
            self.model.load_state_dict({
                key: value.to(self.device)
                for key, value in best_state.items()
            })
    
    def evaluate(self,
                 graph: Data,
                 test_mask: torch.Tensor,
                 labels: torch.Tensor) -> float:
        """Evaluate model on test set."""
        self.model.eval()

        # Ensure float32 dtype for compatibility
        if graph.x.dtype == torch.float64:
            graph.x = graph.x.float()

        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            test_loss = F.nll_loss(out[test_mask], labels[test_mask].to(out.device))

        return float(test_loss)

    def predict(self, graph: Data) -> torch.Tensor:
        """Generate predictions for all nodes in the graph."""
        self.model.eval()

        # Ensure float32 dtype for compatibility
        if graph.x.dtype == torch.float64:
            graph.x = graph.x.float()

        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)

        return out

    def reset(self) -> None:
        """Reset model parameters with consistent initialization and memory cleanup."""
        if self.model is not None:
            # Clear old model from memory first
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Rebuild model which will use the current random seed state
            self._build_model()


class GCNModel(nn.Module):
    """Simple 2-layer GCN model for node classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Ensure float32 for compatibility 
        if x.dtype == torch.float64:
            x = x.float()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GATDownstreamModel(DownstreamModel):
    """
    GAT model for downstream tasks.
    
    Uses Graph Attention Networks for node classification.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 8,
                 output_dim: Optional[int] = None,
                 heads: int = 8,
                 dropout: float = 0.6,
                 attention_dropout: Optional[float] = None,
                 negative_slope: float = 0.2,
                 lr: float = 0.005,
                 weight_decay: float = 5e-4,
                 early_stopping_patience: Optional[int] = 100,
                 device: Optional[torch.device] = None):
        """Initialize GAT model."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout if attention_dropout is not None else dropout
        self.negative_slope = negative_slope
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device is not None else torch.device('cpu')
        
        self.model = None
        self.optimizer = None
        self._build_model()
    
    def _build_model(self):
        """Build the GAT model."""
        if self.output_dim is None:
            return
            
        self.model = GATModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            heads=self.heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            negative_slope=self.negative_slope
        ).to(self.device).float()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    
    def train_model(self,
                    graph: Data,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    labels: torch.Tensor,
                    epochs: int = 200) -> None:
        """Train the GAT model with memory optimization."""
        if self.output_dim is None:
            self.output_dim = int(labels.max()) + 1
            self._build_model()

        self.model.train()

        # Move to device and ensure proper dtype
        graph.x = graph.x.float()
        graph = graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        labels = labels.to(self.device)
        best_state = None
        best_val_loss = None
        patience_counter = 0

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            try:
                out = self.model(graph.x, graph.edge_index)
                loss = F.nll_loss(out[train_mask], labels[train_mask])

                loss.backward()
                self.optimizer.step()

                # Memory cleanup every 10 epochs
                if epoch % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if self.early_stopping_patience is not None and val_mask.sum() > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(graph.x, graph.edge_index)
                        val_loss = F.nll_loss(val_out[val_mask], labels[val_mask]).item()
                    self.model.train()

                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ GAT Memory error at epoch {epoch}, clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        if best_state is not None:
            self.model.load_state_dict({
                key: value.to(self.device)
                for key, value in best_state.items()
            })
    
    def evaluate(self,
                 graph: Data,
                 test_mask: torch.Tensor,
                 labels: torch.Tensor) -> float:
        """Evaluate GAT model."""
        self.model.eval()
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        test_mask = test_mask.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            test_loss = F.nll_loss(out[test_mask], labels[test_mask])
        return float(test_loss)

    def predict(self, graph: Data) -> torch.Tensor:
        """Generate predictions for all nodes in the graph."""
        self.model.eval()
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
        return out

    def reset(self) -> None:
        """Reset GAT model with consistent initialization and memory cleanup."""
        if self.model is not None:
            # Clear old model from memory first
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Rebuild model which will use the current random seed state
            self._build_model()


class GATModel(nn.Module):
    """2-layer GAT model for node classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        heads: int = 1,
        dropout: float = 0.6,
        attention_dropout: Optional[float] = None,
        negative_slope: float = 0.2
    ):
        super().__init__()
        self.dropout = dropout
        self.attention_dropout = attention_dropout if attention_dropout is not None else dropout
        self.negative_slope = negative_slope
        self.conv1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=self.attention_dropout,
            negative_slope=self.negative_slope
        )
        self.conv2 = GATConv(
            hidden_dim * heads,
            output_dim,
            heads=1,
            concat=False,
            dropout=self.attention_dropout,
            negative_slope=self.negative_slope
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGEDownstreamModel(DownstreamModel):
    """
    GraphSAGE model for downstream node classification tasks.

    This model provides a representative inductive mean-aggregation scorer for
    cross-model pruning experiments.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 dropout: float = 0.5,
                 lr: float = 0.01,
                 weight_decay: float = 5e-4,
                 output_dim: Optional[int] = None,
                 early_stopping_patience: Optional[int] = 10,
                 device: Optional[torch.device] = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.output_dim = output_dim
        self.early_stopping_patience = early_stopping_patience
        self.device = device if device is not None else torch.device('cpu')

        self.model = None
        self.optimizer = None
        self._build_model()

    def _build_model(self):
        """Build the GraphSAGE model."""
        if self.output_dim is None:
            return

        self.model = GraphSAGEModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device).float()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_model(self,
                    graph: Data,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    labels: torch.Tensor,
                    epochs: int = 200) -> None:
        """Train the GraphSAGE model with memory optimization."""
        if self.output_dim is None:
            self.output_dim = int(labels.max()) + 1
            self._build_model()

        self.model.train()

        graph.x = graph.x.float()
        graph = graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        labels = labels.to(self.device)
        best_state = None
        best_val_loss = None
        patience_counter = 0

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            try:
                out = self.model(graph.x, graph.edge_index)
                loss = F.nll_loss(out[train_mask], labels[train_mask])

                loss.backward()
                self.optimizer.step()

                if epoch % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if self.early_stopping_patience is not None and val_mask.sum() > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(graph.x, graph.edge_index)
                        val_loss = F.nll_loss(val_out[val_mask], labels[val_mask]).item()
                    self.model.train()

                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in self.model.state_dict().items()
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= self.early_stopping_patience:
                            break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ GraphSAGE Memory error at epoch {epoch}, clearing cache...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        if best_state is not None:
            self.model.load_state_dict({
                key: value.to(self.device)
                for key, value in best_state.items()
            })

    def evaluate(self,
                 graph: Data,
                 test_mask: torch.Tensor,
                 labels: torch.Tensor) -> float:
        """Evaluate GraphSAGE model."""
        self.model.eval()
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        test_mask = test_mask.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
            test_loss = F.nll_loss(out[test_mask], labels[test_mask])
        return float(test_loss)

    def predict(self, graph: Data) -> torch.Tensor:
        """Generate predictions for all nodes in the graph."""
        self.model.eval()
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        with torch.no_grad():
            out = self.model(graph.x, graph.edge_index)
        return out

    def reset(self) -> None:
        """Reset GraphSAGE model with consistent initialization and memory cleanup."""
        if self.model is not None:
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._build_model()


class GraphSAGEModel(nn.Module):
    """Simple 2-layer GraphSAGE model for node classification."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


WEBKB_GCNII_OFFICIAL_PARAMS = {
    'cornell': {'num_layers': 16, 'lamda': 1.0, 'weight_decay': 1e-3},
    'texas': {'num_layers': 32, 'lamda': 1.5, 'weight_decay': 1e-4},
    'wisconsin': {'num_layers': 16, 'lamda': 1.0, 'weight_decay': 5e-4},
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _edge_index_to_scipy_adjacency(edge_index: torch.Tensor, num_nodes: int) -> sp.coo_matrix:
    edge_index = to_undirected(edge_index.detach().cpu(), num_nodes=num_nodes)
    rows = edge_index[0].numpy()
    cols = edge_index[1].numpy()
    values = np.ones(rows.shape[0], dtype=np.float32)
    return sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes))


def _official_gcnii_adjacency(edge_index: torch.Tensor, num_nodes: int, device: torch.device):
    official_path = _repo_root() / 'third_party' / 'gcnii_official'
    if str(official_path) not in sys.path:
        sys.path.insert(0, str(official_path))
    from utils import sparse_mx_to_torch_sparse_tensor, sys_normalized_adjacency

    adjacency = _edge_index_to_scipy_adjacency(edge_index, num_nodes)
    adjacency = sys_normalized_adjacency(adjacency)
    return sparse_mx_to_torch_sparse_tensor(adjacency).to(device)


class GCNIIDownstreamModel(DownstreamModel):
    """GCNII evaluator using the official chennnM/GCNII PyTorch model."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: Optional[int] = None,
                 dropout: float = 0.5,
                 lr: float = 0.01,
                 weight_decay: Optional[float] = None,
                 output_dim: Optional[int] = None,
                 early_stopping_patience: Optional[int] = 100,
                 alpha: float = 0.5,
                 lamda: Optional[float] = None,
                 variant: bool = False,
                 dataset_name: Optional[str] = None,
                 device: Optional[torch.device] = None):
        official_params = WEBKB_GCNII_OFFICIAL_PARAMS.get(
            dataset_name.lower(), {}
        ) if dataset_name else {}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = (
            num_layers
            if num_layers is not None
            else official_params.get('num_layers', 64)
        )
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = (
            weight_decay
            if weight_decay is not None
            else official_params.get('weight_decay', 0.01)
        )
        self.output_dim = output_dim
        self.early_stopping_patience = early_stopping_patience
        self.alpha = alpha
        self.lamda = lamda if lamda is not None else official_params.get('lamda', 0.5)
        self.variant = variant
        self.dataset_name = dataset_name
        self.device = device if device is not None else torch.device('cpu')
        self.model = None
        self.optimizer = None
        self._build_model()

    def _build_model(self):
        if self.output_dim is None:
            return
        official_path = _repo_root() / 'third_party' / 'gcnii_official'
        if str(official_path) not in sys.path:
            sys.path.insert(0, str(official_path))
        from model import GCNII

        self.model = GCNIIModel(
            official_model=GCNII(
                nfeat=self.input_dim,
                nlayers=self.num_layers,
                nhidden=self.hidden_dim,
                nclass=self.output_dim,
                dropout=self.dropout,
                lamda=self.lamda,
                alpha=self.alpha,
                variant=self.variant,
            )
        ).to(self.device).float()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def train_model(self,
                    graph: Data,
                    train_mask: torch.Tensor,
                    val_mask: torch.Tensor,
                    labels: torch.Tensor,
                    epochs: int = 200) -> None:
        if self.output_dim is None:
            self.output_dim = int(labels.max()) + 1
            self._build_model()

        self.model.train()
        graph.x = graph.x.float()
        graph = graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        labels = labels.to(self.device)
        adj = _official_gcnii_adjacency(graph.edge_index, graph.num_nodes, self.device)
        best_state = None
        best_val_loss = None
        patience_counter = 0

        for _ in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(graph.x, adj)
            loss = F.nll_loss(out[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()

            if self.early_stopping_patience is not None and val_mask.sum() > 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(graph.x, adj)
                    val_loss = F.nll_loss(val_out[val_mask], labels[val_mask]).item()
                self.model.train()

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break

        if best_state is not None:
            self.model.load_state_dict({
                key: value.to(self.device)
                for key, value in best_state.items()
            })

    def evaluate(self,
                 graph: Data,
                 test_mask: torch.Tensor,
                 labels: torch.Tensor) -> float:
        self.model.eval()
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        test_mask = test_mask.to(self.device)
        labels = labels.to(self.device)
        adj = _official_gcnii_adjacency(graph.edge_index, graph.num_nodes, self.device)
        with torch.no_grad():
            out = self.model(graph.x, adj)
            test_loss = F.nll_loss(out[test_mask], labels[test_mask])
        return float(test_loss)

    def predict(self, graph: Data) -> torch.Tensor:
        self.model.eval()
        graph = graph.to(self.device)
        graph.x = graph.x.float()
        adj = _official_gcnii_adjacency(graph.edge_index, graph.num_nodes, self.device)
        with torch.no_grad():
            return self.model(graph.x, adj)

    def reset(self) -> None:
        if self.model is not None:
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._build_model()


class GCNIIModel(nn.Module):
    """Thin wrapper around the official chennnM/GCNII model."""

    def __init__(self, official_model: nn.Module):
        super().__init__()
        self.official_model = official_model

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.layout != torch.sparse_coo:
            adj = _official_gcnii_adjacency(adj, x.size(0), x.device)
        return self.official_model(x, adj)


WEBKB_H2GCN_PYTORCH_PARAMS = {
    'cornell': {'lr': 0.01, 'weight_decay': 5e-4, 'dropout': 0.5, 'use_relu': False},
    'texas': {'lr': 0.1, 'weight_decay': 0.0, 'dropout': 0.0, 'use_relu': False},
    'wisconsin': {'lr': 0.05, 'weight_decay': 5e-4, 'dropout': 0.5, 'use_relu': False},
}


def _edge_index_to_sparse_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device
) -> torch.Tensor:
    edge_index = to_undirected(edge_index.detach(), num_nodes=num_nodes).to(device)
    values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        edge_index,
        values,
        (num_nodes, num_nodes),
        device=device,
    ).coalesce()


def _load_h2gcn_pytorch_class():
    module_path = _repo_root() / 'third_party' / 'h2gcn_pytorch' / 'model.py'
    spec = importlib.util.spec_from_file_location('h2gcn_pytorch_model', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.H2GCN


class H2GCNPyTorchModel(nn.Module):
    """Wrapper for GitEventhandler/H2GCN-PyTorch with edge_index input."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 k: int = 2,
                 dropout: float = 0.5,
                 use_relu: bool = True):
        super().__init__()
        h2gcn_class = _load_h2gcn_pytorch_class()
        self.model = h2gcn_class(
            feat_dim=input_dim,
            hidden_dim=hidden_dim,
            class_dim=output_dim,
            k=k,
            dropout=dropout,
            use_relu=use_relu,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj = _edge_index_to_sparse_adjacency(edge_index, x.size(0), x.device)
        self.model.initialized = False
        self.model.a1 = None
        self.model.a2 = None
        return self.model(adj, x.float())


class H2GCNDownstreamModel(NodeClassificationDownstreamMixin, DownstreamModel):
    """H2GCN evaluator using patched GitEventhandler/H2GCN-PyTorch."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 k: int = 2,
                 dropout: float = 0.5,
                 lr: float = 0.01,
                 weight_decay: float = 5e-4,
                 output_dim: Optional[int] = None,
                 early_stopping_patience: Optional[int] = 50,
                 use_relu: bool = True,
                 dataset_name: Optional[str] = None,
                 device: Optional[torch.device] = None):
        h2gcn_params = WEBKB_H2GCN_PYTORCH_PARAMS.get(
            dataset_name.lower(), {}
        ) if dataset_name else {}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.dropout = h2gcn_params.get('dropout', dropout)
        self.lr = h2gcn_params.get('lr', lr)
        self.weight_decay = h2gcn_params.get('weight_decay', weight_decay)
        self.output_dim = output_dim
        self.early_stopping_patience = early_stopping_patience
        self.use_relu = h2gcn_params.get('use_relu', use_relu)
        self.dataset_name = dataset_name
        self.device = device if device is not None else torch.device('cpu')
        self.model = None
        self.optimizer = None
        self._build_model()

    def _build_model(self):
        if self.output_dim is None:
            return
        self.model = H2GCNPyTorchModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            k=self.k,
            dropout=self.dropout,
            use_relu=self.use_relu,
        ).to(self.device).float()
        self.optimizer = torch.optim.Adam(
            [{'params': self.model.model.params, 'weight_decay': self.weight_decay}],
            lr=self.lr,
        )
