"""Downstream models matching the original main-benchmark training protocol."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from .base import DownstreamModel


class LegacyTwoLayerModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.model_type = model_type
        if model_type == "gcn":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif model_type == "gat":
            self.conv1 = GATConv(input_dim, hidden_dim, heads=1, dropout=0.6)
            self.conv2 = GATConv(
                hidden_dim, output_dim, heads=1, concat=False, dropout=0.6
            )
        elif model_type == "sage":
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported legacy downstream model: {model_type}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.model_type == "gat":
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class LegacyDownstreamModel(DownstreamModel):
    """Two-layer evaluator with the optimizer and training loop used in 2025."""

    def __init__(
        self,
        model_type: str,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.model_type = normalize_legacy_model_name(model_type)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = None
        self.optimizer = None
        self._build_model()

    def _build_model(self) -> None:
        if self.output_dim is None:
            return
        self.model = LegacyTwoLayerModel(
            self.model_type,
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
        ).to(self.device).float()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4
        )

    def train_model(
        self,
        graph: Data,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
    ) -> None:
        if self.output_dim is None:
            self.output_dim = int(labels.max()) + 1
            self._build_model()

        graph = graph.to(self.device)
        graph.x = graph.x.float()
        train_mask = train_mask.to(self.device)
        labels = labels.to(self.device)
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(graph.x, graph.edge_index)
            loss = F.nll_loss(out[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evaluate(
        self, graph: Data, test_mask: torch.Tensor, labels: torch.Tensor
    ) -> float:
        graph = graph.to(self.device)
        test_mask = test_mask.to(self.device)
        labels = labels.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(graph.x.float(), graph.edge_index)
            return float(F.nll_loss(out[test_mask], labels[test_mask]))

    def predict(self, graph: Data) -> torch.Tensor:
        graph = graph.to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(graph.x.float(), graph.edge_index)

    def reset(self) -> None:
        if self.model is not None:
            del self.model
            del self.optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._build_model()


def normalize_legacy_model_name(model_type: str) -> str:
    aliases = {
        "gcn": "gcn",
        "gat": "gat",
        "sage": "sage",
        "graphsage": "sage",
        "graph_sage": "sage",
    }
    normalized = model_type.lower()
    if normalized not in aliases:
        raise ValueError(f"Unsupported legacy downstream model: {model_type}")
    return aliases[normalized]


def create_legacy_downstream_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int = 128,
    output_dim: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> LegacyDownstreamModel:
    return LegacyDownstreamModel(
        model_type=model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        device=device,
    )
