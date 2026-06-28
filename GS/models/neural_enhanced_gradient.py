"""
Neural-Enhanced Gradient-Based Graph Summarization Model

Hybrid model combining gradient information from Development Model 2 with neural network learning capability.
This model uses gradient method to provide prior knowledge of edge importance, then uses neural network for fine-tuning optimization.

Training strategy follows the fixed reweighting and dynamic reweighting strategies of Development Model 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Optional, Tuple, Dict, Any
import copy
import numpy as np
from tqdm import tqdm

from .base import GraphSummarizationModel
from .gradient_based import GradientBasedGraphSummarization
from .main_model import LearnableGraphSummarization


class EdgeImportanceRefiner(nn.Module):
    """
    Edge importance refinement network

    This network accepts:
    1. Edge importance scores computed by gradient method
    2. Edge structural features (node embeddings, etc.)
    3. Step information

    Outputs refined edge importance scores
    """

    def __init__(self,
                 node_emb_dim: int = 256,
                 step_emb_dim: int = 32,
                 hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        self.node_emb_dim = node_emb_dim
        self.step_emb_dim = step_emb_dim
        self.hidden_dim = hidden_dim

        # Edge feature dimension:2*node_emb + |node_diff| + step_emb + gradient_score
        edge_feature_dim = 3 * node_emb_dim + step_emb_dim + 1

        # Refinement network
        self.refiner = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output range [-1, 1], as adjustment factor
        )

    def forward(self,
                node_embeddings: torch.Tensor,
                edge_index: torch.Tensor,
                gradient_scores: torch.Tensor,
                step_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation

        Args:
            node_embeddings: Node embeddings [n, node_emb_dim]
            edge_index: Edge index [2, num_edges]
            gradient_scores: Edge scores computed by gradient method [num_edges]
            step_embedding: Step embedding [step_emb_dim]

        Returns:
            Adjustment factor [num_edges]，Range [-1, 1]
        """
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        h_u = node_embeddings[src_nodes]  # [num_edges, node_emb_dim]
        h_v = node_embeddings[dst_nodes]  # [num_edges, node_emb_dim]

        # Edge features
        edge_diff = torch.abs(h_u - h_v)  # [num_edges, node_emb_dim]
        gradient_scores_expanded = gradient_scores.unsqueeze(-1)  # [num_edges, 1]

        # 扩展Step embedding到所有Edge
        num_edges = edge_index.size(1)
        step_emb_expanded = step_embedding.unsqueeze(0).expand(num_edges, -1)  # [num_edges, step_emb_dim]

        # Concatenate all features
        edge_features = torch.cat([
            h_u, h_v, edge_diff, step_emb_expanded, gradient_scores_expanded
        ], dim=1)  # [num_edges, edge_feature_dim]

        # Ensure data type consistency
        edge_features = edge_features.float()

        # 通过Refinement network
        adjustment = self.refiner(edge_features).squeeze(-1)  # [num_edges]

        return adjustment


class NeuralEnhancedGradientModel(GraphSummarizationModel, nn.Module):
    """
    Neural Network Enhanced Gradient-Based Graph Summarization Model

    This model combines:
    1. Accuracy and interpretability of gradient method
    2. Learning capability and adaptability of neural networks

    Workflow:
    1. Use gradient method to compute initial edge importance (compute once and cache)
    2. Use neural network to learn how to refine these scores
    3. Obtain final edge selection through fusion strategy

    Optimization features:
    - Gradient computation caching: Gradient scores for the same graph are computed only once
    - Efficient inference: Inference speed after training is much faster than pure gradient method

    Training strategy:
    - Follows fixed reweighting and dynamic reweighting strategies of Development Model 1
    - Supports uniform and cosine fixed weights
    - Supports Frank-Wolfe and UGD dynamic weight solving
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 step_emb_dim: int = 32,
                 num_gin_layers: int = 3,
                 dropout: float = 0.2,
                 max_steps: int = 20,
                 device: str = 'cpu',
                 gradient_train_epochs: int = 20,
                 fusion_weight: float = 0.3,  # Weight of neural network adjustment
                 use_residual_learning: bool = True,
                 fast_gradient_computation: bool = True,  # Whether to use fast gradient computation
                 **kwargs):
        """
        Initialize hybrid model

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            step_emb_dim: Step embedding维度
            num_gin_layers: GINNumber of layers
            dropout: DropoutRate
            max_steps: Maximum steps
            device: Computing device
            gradient_train_epochs: Number of gradient method training epochs
            fusion_weight: Weight of neural network adjustment
            use_residual_learning: Whether to use residual learning
            fast_gradient_computation: Whether to use fast gradient computation
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.step_emb_dim = step_emb_dim
        self.dropout = dropout
        self.max_steps = max_steps
        self.device = device
        self.fusion_weight = fusion_weight
        self.use_residual_learning = use_residual_learning
        self.fast_gradient_computation = fast_gradient_computation

        # Gradient method model (for computing prior edge importance)
        self.gradient_model = GradientBasedGraphSummarization(
            input_dim=input_dim,
            train_epochs=gradient_train_epochs,
            device=torch.device(device) if isinstance(device, str) else device
        )

        # Neural network node encoder (reuses LearnableGraphSummarization architecture)
        self.neural_encoder = LearnableGraphSummarization(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            step_emb_dim=step_emb_dim,
            num_gin_layers=num_gin_layers,
            dropout=dropout,
            max_steps=max_steps,
            device=device,
            **kwargs
        )

        # Edge importance refinement network
        self.importance_refiner = EdgeImportanceRefiner(
            node_emb_dim=hidden_dim,
            step_emb_dim=step_emb_dim,
            hidden_dim=128,
            dropout=dropout
        )

        # Training related
        self.train_mask = None
        self.val_mask = None
        self.labels = None

        # Static gradient feature storage
        self._static_gradient_features = None
        self._original_edge_index = None
        self._is_gradient_features_computed = False

        # Move to device and ensure float type
        self.to(device)
        self.float()

    def set_training_data(self, train_mask, val_mask, labels):
        """Set training data"""
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.labels = labels

        # Also set training data for gradient model
        self.gradient_model.train_mask = train_mask
        self.gradient_model.val_mask = val_mask
        self.gradient_model.labels = labels

    def precompute_gradient_features(self, graph: Data):
        """
        Precompute gradient features as static features

        Args:
            graph: Original graph data
        """
        print("Precomputing gradient features...")

        # Ensure graph data is on correct device
        graph = graph.to(self.device)

        # 存储原始Edge index
        self._original_edge_index = graph.edge_index.clone().to(self.device)

        # Compute gradient features
        if self.fast_gradient_computation:
            gradient_features = self._compute_gradient_importance_fast(graph, graph.edge_index)
        else:
            gradient_features = self.gradient_model._compute_edge_gradients_sparse(graph, graph.edge_index)

        # Store as static features
        self._static_gradient_features = gradient_features.clone().detach().to(self.device)
        self._is_gradient_features_computed = True

        print(f"GradientFeature precomputation completed, total{len(gradient_features)} edges")

    def _get_gradient_features_for_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get gradient features for given edges

        Args:
            edge_index: 当前Edge index

        Returns:
            Gradient features for corresponding edges
        """
        if not self._is_gradient_features_computed:
            raise ValueError("GradientFeatures not yet precomputed, please call firstprecompute_gradient_features()")

        # If it is the complete edge set of the original graph, return directly
        if torch.equal(edge_index, self._original_edge_index):
            return self._static_gradient_features

        # Otherwise need to find corresponding edges
        return self._extract_gradient_features_for_subgraph(edge_index)

    def _extract_gradient_features_for_subgraph(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Extract corresponding gradient features for subgraph

        Args:
            edge_index: 子图Edge index

        Returns:
            Gradient features for subgraph edges
        """
        # Create edge dictionary mapping:(src, dst) -> feature_index
        original_edges = self._original_edge_index.t()  # [num_edges, 2]
        edge_to_idx = {}
        for i, (src, dst) in enumerate(original_edges):
            edge_to_idx[(src.item(), dst.item())] = i

        # Find corresponding feature indices for subgraph edges
        subgraph_edges = edge_index.t()  # [num_subgraph_edges, 2]
        feature_indices = []

        for src, dst in subgraph_edges:
            key = (src.item(), dst.item())
            if key in edge_to_idx:
                feature_indices.append(edge_to_idx[key])
            else:
                # If corresponding edge not found, this is a newly generated edge or reverse edge
                # Try reverse edge
                reverse_key = (dst.item(), src.item())
                if reverse_key in edge_to_idx:
                    feature_indices.append(edge_to_idx[reverse_key])
                else:
                    # If neither found, use average value or recompute
                    print(f"Warning: Edge ({src}, {dst}) does not exist in original graph, using average gradient features")
                    feature_indices.append(0)  # Temporarily use first feature

        feature_indices = torch.tensor(feature_indices, device=self.device)
        return self._static_gradient_features[feature_indices]


    def _compute_gradient_importance_fast(self, graph: Data, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Fast version of gradient importance computation (for training)

        Fast approximation based on structural features, rather than actually training downstream model
        """
        # Ensure edge_index is on correct device
        edge_index = edge_index.to(self.device)
        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        # Compute node degrees
        degree = torch.zeros(graph.x.size(0), device=self.device, dtype=torch.float)
        degree.scatter_add_(0, src_nodes.to(self.device), torch.ones_like(src_nodes, dtype=torch.float, device=self.device))
        degree.scatter_add_(0, dst_nodes.to(self.device), torch.ones_like(dst_nodes, dtype=torch.float, device=self.device))

        # Edge重要性 = 两端节点度数的几何平均
        edge_importance = torch.sqrt(degree[src_nodes] * degree[dst_nodes])

        # Normalize to reasonable range
        if edge_importance.max() > 0:
            edge_importance = edge_importance / edge_importance.max()

        return edge_importance

    def _compute_neural_adjustment(self,
                                 graph: Data,
                                 edge_index: torch.Tensor,
                                 gradient_scores: torch.Tensor,
                                 step: int) -> torch.Tensor:
        """
        使用神经网络计算Adjustment factor

        Args:
            graph: Graph data
            edge_index: Edge index
            gradient_scores: Gradient method scores
            step: Current step

        Returns:
            Adjustment factor
        """
        # 获取Node embeddings
        # 确保输入数据在正确的设备上
        graph_x = graph.x.to(self.device)
        edge_index = edge_index.to(self.device)
        node_embeddings = self.neural_encoder.encode_nodes(graph_x, edge_index)

        # 获取Step embedding
        step_tensor = torch.tensor(step, dtype=torch.long, device=self.device)
        step_embedding = self.neural_encoder.step_embedding(step_tensor)

        # 计算Adjustment factor
        adjustment = self.importance_refiner(
            node_embeddings, edge_index, gradient_scores, step_embedding
        )

        return adjustment

    def _fuse_scores(self,
                    gradient_scores: torch.Tensor,
                    neural_adjustment: torch.Tensor) -> torch.Tensor:
        """
        Fuse gradient scores with neural network adjustment

        Args:
            gradient_scores: Gradient method scores
            neural_adjustment: 神经网络Adjustment factor

        Returns:
            Fused scores
        """
        if self.use_residual_learning:
            # Residual learning: Adjust based on gradient scores
            adjusted_scores = gradient_scores + self.fusion_weight * neural_adjustment
        else:
            # Direct weighted fusion
            adjusted_scores = (1 - self.fusion_weight) * gradient_scores + \
                            self.fusion_weight * neural_adjustment

        return adjusted_scores

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                step: int) -> torch.Tensor:
        """
        Forward propagation，返回融合后的Edge分数

        Args:
            x: Node features [n, d_in]
            edge_index: Edge index [2, num_edges]
            step: Current step

        Returns:
            融合后的Edge分数 [num_edges]
        """
        # Ensure data type consistency和设备一致性
        if x.dtype != torch.float32:
            x = x.float()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)

        # 创建临时Graph data
        temp_graph = Data(x=x, edge_index=edge_index)

        # 1. Get precomputed gradient features (static features)
        gradient_scores = self._get_gradient_features_for_edges(edge_index)

        # 2. Compute neural network adjustment
        neural_adjustment = self._compute_neural_adjustment(
            temp_graph, edge_index, gradient_scores, step
        )

        # 3. Fuse scores
        final_scores = self._fuse_scores(gradient_scores, neural_adjustment)

        return final_scores

    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        Generate graph summary sequence

        Args:
            original_graph: Original graph
            num_steps: Number of summary steps

        Returns:
            Summary graph list
        """
        if self.train_mask is None or self.val_mask is None or self.labels is None:
            raise ValueError("必须先Set training data")

        # Ensure gradient features are precomputed
        if not self._is_gradient_features_computed:
            self.precompute_gradient_features(original_graph)

        self.eval()
        summary_graphs = []
        current_graph = copy.deepcopy(original_graph.to(self.device))

        # Step 0: Original graph
        summary_graphs.append(current_graph)

        print(f"开始Neural-Enhanced图简化: 总Edge数 {current_graph.edge_index.size(1)}")

        with torch.no_grad():
            for step in range(1, num_steps + 1):
                if current_graph.edge_index.size(1) == 0:
                    # Empty graph
                    empty_graph = Data(
                        x=original_graph.x,
                        edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                    summary_graphs.append(empty_graph)
                    continue

                # 使用Forward propagation获取融合的Edge分数
                edge_scores = self.forward(current_graph.x, current_graph.edge_index, step)

                # 确定保留Edge数
                if step == num_steps:
                    num_keep = 0
                else:
                    keep_ratio = 1.0 - (step / num_steps)
                    num_keep = max(0, int(current_graph.edge_index.size(1) * keep_ratio))

                if num_keep == 0:
                    current_graph = Data(
                        x=original_graph.x,
                        edge_index=torch.zeros((2, 0), dtype=torch.long, device=self.device),
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )
                else:
                    # 选择分数最高的Edge
                    _, top_indices = torch.topk(edge_scores, num_keep, largest=True)
                    kept_edge_index = current_graph.edge_index[:, top_indices]

                    current_graph = Data(
                        x=original_graph.x,
                        edge_index=kept_edge_index,
                        y=original_graph.y,
                        num_nodes=original_graph.x.size(0)
                    )

                summary_graphs.append(current_graph)

        return summary_graphs

    def reset(self) -> None:
        """Reset model"""
        self.neural_encoder.reset()
        self.gradient_model.reset()
        self.train_mask = None
        self.val_mask = None
        self.labels = None
        # Clear static gradient features
        self._static_gradient_features = None
        self._original_edge_index = None
        self._is_gradient_features_computed = False

    def get_gradient_features_info(self) -> Dict[str, Any]:
        """Get gradient feature information for debugging"""
        return {
            'is_computed': self._is_gradient_features_computed,
            'num_edges': len(self._static_gradient_features) if self._static_gradient_features is not None else 0,
            'feature_shape': self._static_gradient_features.shape if self._static_gradient_features is not None else None,
            'device': self._static_gradient_features.device if self._static_gradient_features is not None else None
        }


# Create variants with different configurations
class NeuralEnhancedGradientModel_HighFusion(NeuralEnhancedGradientModel):
    """High fusion weight variant"""
    def __init__(self, *args, **kwargs):
        kwargs['fusion_weight'] = 0.6
        super().__init__(*args, **kwargs)


class NeuralEnhancedGradientModel_LowFusion(NeuralEnhancedGradientModel):
    """Low fusion weight variant"""
    def __init__(self, *args, **kwargs):
        kwargs['fusion_weight'] = 0.1
        super().__init__(*args, **kwargs)


class NeuralEnhancedGradientModel_NoResidual(NeuralEnhancedGradientModel):
    """Variant without residual learning"""
    def __init__(self, *args, **kwargs):
        kwargs['use_residual_learning'] = False
        super().__init__(*args, **kwargs)


class NeuralEnhancedGradientModel_SlowGradient(NeuralEnhancedGradientModel):
    """Variant with exact gradient computation"""
    def __init__(self, *args, **kwargs):
        kwargs['fast_gradient_computation'] = False
        super().__init__(*args, **kwargs)


# Create wrapper class for compatibility with existing training_strategies.py
class TrainableNeuralEnhancedGradientModel:
    """
    Trainable Neural-Enhanced model wrapper class

    This class wraps NeuralEnhancedGradientModel, enabling it to work with existing
    GraphSummarizationTrainerand training strategies seamlessly
    """

    def __init__(self,
                 model: NeuralEnhancedGradientModel,
                 training_strategy: str = 'fixed_uniform',
                 solver_type: str = 'frank_wolfe'):
        """
        Initialize trainable wrapper

        Args:
            model: Neural-EnhancedModel instance
            training_strategy: Training strategy
                - 'fixed_uniform': Fixed weight (uniform)
                - 'fixed_cosine': Fixed weight (cosine)
                - 'dynamic_frank_wolfe': Dynamic weight (Frank-Wolfe)
                - 'dynamic_ugd': Dynamic weight (UGD)
            solver_type: Solver type for dynamic strategy
        """
        from .training_strategies import (
            FixedReweightingStrategy,
            DynamicReweightingStrategy,
            GraphSummarizationTrainer
        )

        self.model = model
        self.training_strategy_name = training_strategy

        # 创建Training strategy
        if training_strategy.startswith('fixed'):
            strategy_type = training_strategy.split('_')[1]  # 'uniform' or 'cosine'
            self.strategy = FixedReweightingStrategy(strategy_type=strategy_type)
        elif training_strategy.startswith('dynamic'):
            self.strategy = DynamicReweightingStrategy(solver_type=solver_type)
        else:
            raise ValueError(f"Unknown training strategy: {training_strategy}")

        # Create trainer
        self.trainer = GraphSummarizationTrainer(
            model=model,
            strategy=self.strategy,
            device=model.device
        )

    def train_model(self,
                   graph: Data,
                   train_mask: torch.Tensor,
                   val_mask: torch.Tensor,
                   labels: torch.Tensor,
                   epochs: int = 30,
                   num_steps: int = 10,
                   downstream_epochs: int = 30):
        """
        Train model

        Args:
            graph: Training graph
            train_mask: Training mask
            val_mask: Validation mask
            labels: Labels
            epochs: Number of epochs for graph summarization network training
            num_steps: Number of simplification steps
            downstream_epochs: Number of epochs for downstream model training
        """
        # Set training data
        self.model.set_training_data(train_mask, val_mask, labels)

        # 预Compute gradient features（训练前一次性计算）
        if not self.model._is_gradient_features_computed:
            self.model.precompute_gradient_features(graph)

        # Train using trainer
        history = self.trainer.train(
            graph=graph,
            train_labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            num_epochs=epochs,
            num_steps=num_steps,
            downstream_epochs=downstream_epochs
        )

        return history

    def summarize(self, graph: Data, num_steps: int = 10) -> List[Data]:
        """Generate graph summary sequence"""
        return self.model.summarize(graph, num_steps)

    def reset(self):
        """Reset model"""
        self.model.reset()

    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()

    def to(self, device):
        """Move to device"""
        self.model.to(device)
        return self

    def __getattr__(self, name):
        """Proxy to internal model"""
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")