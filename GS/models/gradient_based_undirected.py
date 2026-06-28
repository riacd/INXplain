"""
Gradient-Based Graph Simplification Model - 无向图版本

Correctly removes edges at the undirected graph level, ensuring both directions are deleted simultaneously.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from typing import List, Optional, Tuple
import copy
import numpy as np
import time
from tqdm import tqdm

from .base import GraphSummarizationModel
from .downstream import create_downstream_model, normalize_downstream_model_name


class GradientBasedUndirectedGraphSummarization(GraphSummarizationModel):
    """
    Gradient-Based Undirected Graph Simplification Model

    与标准gradient_based模型的区别：
    1. 在无向 edges层面计算梯度（对 edges对(u,v)和(v,u)取平均）
    2. 删 edges时同时Remove两个方向，保持图的无向性
    3. 计算复杂度时除以2，反映真实的无向 edges数

    这确保了：
    - 图始终保持无向性（不会出现半 edges）
    - 删 edges数量在无向 edges层面是均匀的
    - Complexity Metric 正确反映无向图的 edges数
    """

    def __init__(self,
                 input_dim: int = None,
                 downstream_model_type: str = 'gcn',
                 hidden_dim: Optional[int] = None,
                 train_epochs: int = 200,
                 gat_heads: int = 8,
                 dropout: Optional[float] = None,
                 attention_dropout: Optional[float] = None,
                 negative_slope: float = 0.2,
                 lr: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 early_stopping_patience: Optional[int] = None,
                 dataset_name: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        初始化Gradient-Based Undirected Graph Simplification Model

        Args:
            input_dim: Input feature dimension (for benchmark compatibility)
            downstream_model_type: Scoring model type used inside pruning ('gcn' or 'gat')
            hidden_dim: Hidden layer dimension
            train_epochs: Number of training epochs
            gat_heads: Number of GAT attention heads when downstream_model_type='gat'
            dropout: Dropout used by GAT scoring model
            attention_dropout: Attention dropout used by GAT scoring model
            negative_slope: LeakyReLU negative slope used by GAT scoring model
            lr: Learning rate used by the scoring model
            weight_decay: Weight decay used by the scoring model
            early_stopping_patience: Optional early stopping patience for the scoring model
            device: Computing device
        """
        self.input_dim = input_dim
        self.downstream_model_type = downstream_model_type
        self.hidden_dim = hidden_dim
        self.train_epochs = train_epochs
        self.gat_heads = gat_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.negative_slope = negative_slope
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.dataset_name = dataset_name
        self.device = device if device is not None else torch.device('cpu')

        # Store reference to training data (will be set during training)
        self.train_mask = None
        self.val_mask = None
        self.labels = None

        # Flag indicating whether model has been trained
        self.is_trained = False

    def _create_scoring_downstream_model(
        self,
        input_dim: int,
        output_dim: int,
        model_type: Optional[str] = None
    ):
        """Create the downstream model used inside pruning to score edge removals."""
        model_type = normalize_downstream_model_name(
            model_type or self.downstream_model_type
        )
        defaults = {
            'gcn': {'hidden_dim': 16, 'dropout': 0.5, 'lr': 0.01, 'patience': 10, 'weight_decay': 5e-4},
            'gat': {'hidden_dim': 8, 'dropout': 0.6, 'lr': 0.005, 'patience': 100, 'weight_decay': 5e-4},
            'sage': {'hidden_dim': 64, 'dropout': 0.5, 'lr': 0.01, 'patience': 10, 'weight_decay': 5e-4},
            'h2gcn': {'hidden_dim': 64, 'dropout': 0.5, 'lr': 0.01, 'patience': 50, 'weight_decay': 5e-4},
            'gcnii': {'hidden_dim': 64, 'dropout': 0.5, 'lr': 0.01, 'patience': 100, 'weight_decay': None},
        }[model_type]
        kwargs = {
            'hidden_dim': (
                self.hidden_dim
                if self.hidden_dim is not None
                else defaults['hidden_dim']
            ),
            'output_dim': output_dim,
            'dropout': self.dropout if self.dropout is not None else defaults['dropout'],
            'lr': self.lr if self.lr is not None else defaults['lr'],
            'early_stopping_patience': (
                self.early_stopping_patience
                if self.early_stopping_patience is not None
                else defaults['patience']
            ),
        }
        if self.weight_decay is not None or defaults['weight_decay'] is not None:
            kwargs['weight_decay'] = (
                self.weight_decay
                if self.weight_decay is not None
                else defaults['weight_decay']
            )
        if model_type == 'gat':
            kwargs.update({
                'heads': self.gat_heads,
                'attention_dropout': self.attention_dropout,
                'negative_slope': self.negative_slope,
            })
        if model_type in ('h2gcn', 'gcnii') and self.dataset_name is not None:
            kwargs['dataset_name'] = self.dataset_name
        return create_downstream_model(
            model_type,
            input_dim=input_dim,
            device=self.device,
            **kwargs,
        )

    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        Generate a series of simplified graphs（无向图版本）

        Args:
            original_graph: Original input graph
            num_steps: Number of simplification steps

        Returns:
            List[Data]: List of simplified graphs, including original graph and stepwise simplification results
        """
        if self.train_mask is None or self.val_mask is None or self.labels is None:
            raise ValueError("Must set training data first. Please ensure the model has been trained through TrainableGraphSummarizationModel.")

        # Move graph to specified device
        graph = original_graph.to(self.device)

        # Ensure edge index is undirected
        edge_index = to_undirected(graph.edge_index)

        # Result list, starting from original graph
        summarized_graphs = [copy.deepcopy(graph)]
        summarized_graphs[0].edge_index = edge_index

        current_edges = edge_index.clone()

        # Calculate number of undirected edges (half of directed edges)
        total_undirected_edges = current_edges.shape[1] // 2
        undirected_edges_per_step = total_undirected_edges // num_steps

        print(f"开始无向图梯度法简化: 总无向 edges数 {total_undirected_edges}, Remove per step {undirected_edges_per_step}  edges无向 edges")
        print(f"(对应有向 edges: {current_edges.shape[1]}  edges，Remove per step {undirected_edges_per_step * 2}  edges)")
        print("This may take some time, please be patient...")

        for step in tqdm(range(num_steps), desc="Graph simplification progress"):
            if current_edges.shape[1] <= 0:
                # 如果没有 edges了，添加空图
                empty_graph = copy.deepcopy(graph)
                empty_graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                summarized_graphs.append(empty_graph)
                continue

            # 计算当前Step要Remove的无向 edges数
            current_undirected_edges = current_edges.shape[1] // 2
            if step == num_steps - 1:
                # 最后一步Remove所有剩余 edges
                undirected_edges_to_remove = current_undirected_edges
            else:
                undirected_edges_to_remove = min(undirected_edges_per_step, current_undirected_edges)

            print(f"Step {step+1}: 当前无向 edges数 {current_undirected_edges}, Remove {undirected_edges_to_remove}  edges")

            # 获取 edges的梯度信息（无向 edges版本）
            print(f"  Computing {current_undirected_edges}  edges无向 edges的重要性...")
            start_time = time.time()
            edge_pair_gradients, edge_pairs = self._compute_undirected_edge_gradients(graph, current_edges)
            elapsed_time = time.time() - start_time
            print(f"   edges重要性计算完成，耗时 {elapsed_time:.1f}  seconds")

            # 选择要Remove的无向 edges（梯度最小的 edges）
            current_edges = self._remove_undirected_edges_by_gradient(
                current_edges, edge_pair_gradients, edge_pairs, undirected_edges_to_remove
            )

            # Create new graph data
            new_graph = copy.deepcopy(graph)
            new_graph.edge_index = current_edges

            summarized_graphs.append(new_graph)

        return summarized_graphs

    def _compute_undirected_edge_gradients(
        self, graph: Data, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        计算无向 edges的梯度（对 edges对取平均）

        Args:
            graph: Current graph data
            edge_index: 当前 edges索引（有向表示）

        Returns:
            edge_pair_gradients: 每 edges无向 edges的梯度值
            edge_pairs: 无向 edges对的列表 [(src, dst), ...]
        """
        # Create downstream task model
        input_dim = graph.x.size(1)
        output_dim = int(self.labels.max()) + 1

        downstream_model = self._create_scoring_downstream_model(input_dim, output_dim)

        # Create temporary graph data for training
        temp_graph = copy.deepcopy(graph)
        temp_graph.edge_index = edge_index.detach()

        # Train downstream model
        downstream_model.train_model(
            temp_graph,
            self.train_mask,
            self.val_mask,
            self.labels,
            epochs=self.train_epochs
        )

        # Build the canonical undirected-edge mapping once for this pruning step.
        edge_pair_tensor, edge_pair_inverse = self._build_undirected_edge_index(
            edge_index
        )
        edge_pairs = [tuple(pair) for pair in edge_pair_tensor.cpu().tolist()]
        num_undirected_edges = len(edge_pairs)

        edge_pair_gradients = torch.zeros(num_undirected_edges, device=self.device)

        downstream_model.model.eval()

        # 批量计算无向 edges的重要性
        with torch.no_grad():
            # Compute performance on complete graph as baseline
            base_out = downstream_model.model(graph.x.float(), edge_index)
            base_loss = F.nll_loss(base_out[self.val_mask], self.labels[self.val_mask].to(base_out.device))

            # 对每 edges无向 edges计算Remove后的性能变化
            batch_size = min(50, num_undirected_edges)
            print(f"      Using batch size {batch_size} to process {num_undirected_edges}  edges无向 edges...")

            for i in tqdm(range(0, num_undirected_edges, batch_size), desc="       edges重要性计算", leave=False):
                end_idx = min(i + batch_size, num_undirected_edges)
                batch_gradients = []

                for edge_pair_idx in range(i, end_idx):
                    # Remove both directions using the precomputed canonical ID.
                    reduced_edge_index = edge_index[
                        :, edge_pair_inverse != edge_pair_idx
                    ]

                    # 计算Remove edges后的性能
                    if reduced_edge_index.shape[1] > 0:
                        out = downstream_model.model(graph.x.float(), reduced_edge_index)
                        loss = F.nll_loss(out[self.val_mask], self.labels[self.val_mask].to(out.device))
                        gradient = loss - base_loss  # 损失增加越多， edges越重要
                    else:
                        gradient = float('inf')  # 如果Remove这 edges edges导致图断开，设为无穷大

                    batch_gradients.append(gradient)

                edge_pair_gradients[i:end_idx] = torch.tensor(batch_gradients, device=self.device)

        return edge_pair_gradients, edge_pairs

    @staticmethod
    def _build_undirected_edge_index(
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sorted canonical edge pairs and each directed edge's pair ID."""
        if edge_index.shape[1] == 0:
            return (
                torch.empty((0, 2), dtype=torch.long, device=edge_index.device),
                torch.empty((0,), dtype=torch.long, device=edge_index.device),
            )

        src, dst = edge_index
        low = torch.minimum(src, dst)
        high = torch.maximum(src, dst)
        valid = low != high
        inverse = torch.full(
            (edge_index.shape[1],), -1, dtype=torch.long, device=edge_index.device
        )
        if not valid.any():
            return (
                torch.empty((0, 2), dtype=torch.long, device=edge_index.device),
                inverse,
            )

        node_base = int(torch.maximum(low.max(), high.max()).item()) + 1
        canonical_keys = low[valid] * node_base + high[valid]
        unique_keys, valid_inverse = torch.unique(
            canonical_keys, sorted=True, return_inverse=True
        )
        inverse[valid] = valid_inverse
        edge_pairs = torch.stack(
            (unique_keys // node_base, unique_keys % node_base), dim=1
        )
        return edge_pairs, inverse

    def _extract_undirected_edges(self, edge_index: torch.Tensor) -> List[Tuple[int, int]]:
        """
        从有向 edges索引中提取无向 edges对

        Args:
            edge_index: 有向 edges索引 [2, num_edges]

        Returns:
            List of (src, dst) tuples where src < dst (to avoid duplicates)
        """
        edge_pairs, _ = self._build_undirected_edge_index(edge_index)
        return [tuple(pair) for pair in edge_pairs.cpu().tolist()]

    def _remove_undirected_edge_from_index(
        self, edge_index: torch.Tensor, src: int, dst: int
    ) -> torch.Tensor:
        """
        从 edges索引中Remove一 edges无向 edges（两个方向）

        Args:
            edge_index: 当前 edges索引
            src: Source node
            dst: Target node

        Returns:
            Remove edges后的 edges索引
        """
        edge_src, edge_dst = edge_index
        remove_mask = (
            ((edge_src == src) & (edge_dst == dst))
            | ((edge_src == dst) & (edge_dst == src))
        )
        return edge_index[:, ~remove_mask]

    def _remove_undirected_edges_by_gradient(
        self,
        edge_index: torch.Tensor,
        gradients: torch.Tensor,
        edge_pairs: List[Tuple[int, int]],
        num_undirected_edges_to_remove: int
    ) -> torch.Tensor:
        """
        根据梯度信息Remove无向 edges

        Args:
            edge_index: 当前 edges索引
            gradients: 无向 edges的梯度信息
            edge_pairs: 无向 edges对列表
            num_undirected_edges_to_remove: 要Remove的无向 edges数

        Returns:
            torch.Tensor: Remove edges后的 edges索引
        """
        if num_undirected_edges_to_remove >= len(edge_pairs):
            # Remove所有 edges
            return torch.empty((2, 0), dtype=torch.long, device=self.device)

        # 找到梯度最小的无向 edges
        _, indices_to_remove = torch.topk(gradients, k=num_undirected_edges_to_remove, largest=False)

        _, edge_pair_inverse = self._build_undirected_edge_index(edge_index)
        indices_to_remove = indices_to_remove.to(edge_index.device)
        keep_mask = ~torch.isin(edge_pair_inverse, indices_to_remove)
        return edge_index[:, keep_mask]

    def set_training_data(self, train_mask, val_mask, labels):
        """Set training data"""
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.labels = labels
        self.is_trained = True

    def reset(self):
        """Reset model state"""
        self.train_mask = None
        self.val_mask = None
        self.labels = None
        self.is_trained = False


class JointSubsetGradientBase(GradientBasedUndirectedGraphSummarization):
    """
    Base class for joint-subset gradient pruning variants.

    Instead of evaluating one edge at a time, each pruning step samples subsets
    of the same size as the step deletion target and evaluates their joint
    validation loss impact.
    """
    sampling_subset_num_controls_split_count = False

    def __init__(self,
                 input_dim: int = None,
                 downstream_model_type: str = 'gcn',
                 hidden_dim: Optional[int] = None,
                 train_epochs: int = 200,
                 sampling_repeats: int = 5,
                 sampling_subset_num: Optional[int] = None,
                 sampling_seed: int = 42,
                 gat_heads: int = 8,
                 dropout: Optional[float] = None,
                 attention_dropout: Optional[float] = None,
                 negative_slope: float = 0.2,
                 lr: Optional[float] = None,
                 weight_decay: float = 5e-4,
                 early_stopping_patience: Optional[int] = None,
                 device: Optional[torch.device] = None):
        super().__init__(
            input_dim=input_dim,
            downstream_model_type=downstream_model_type,
            hidden_dim=hidden_dim,
            train_epochs=train_epochs,
            gat_heads=gat_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            negative_slope=negative_slope,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=device
        )
        self.sampling_repeats = sampling_repeats
        self.sampling_subset_num = sampling_subset_num
        self.sampling_seed = sampling_seed

    def _train_downstream_for_edges(self, graph: Data, edge_index: torch.Tensor):
        input_dim = graph.x.size(1)
        output_dim = int(self.labels.max()) + 1

        downstream_model = self._create_scoring_downstream_model(input_dim, output_dim)

        temp_graph = copy.deepcopy(graph)
        temp_graph.edge_index = edge_index.detach()
        downstream_model.train_model(
            temp_graph,
            self.train_mask,
            self.val_mask,
            self.labels,
            epochs=self.train_epochs
        )
        downstream_model.model.eval()
        return downstream_model

    def _sample_edge_subsets(
        self,
        edge_pairs: List[Tuple[int, int]],
        subset_size: int,
        step: int
    ) -> List[List[Tuple[int, int]]]:
        if subset_size <= 0:
            return []

        subsets = []
        num_edges = len(edge_pairs)
        generator = torch.Generator(device='cpu')

        for repeat in range(self.sampling_repeats):
            generator.manual_seed(self.sampling_seed + step * 1000 + repeat)
            permutation = torch.randperm(num_edges, generator=generator).tolist()

            if (
                self.sampling_subset_num_controls_split_count
                and self.sampling_subset_num is not None
            ):
                num_subsets = max(1, min(self.sampling_subset_num, num_edges))
                for subset_indices in torch.tensor_split(
                    torch.tensor(permutation, dtype=torch.long),
                    num_subsets
                ):
                    if subset_indices.numel() == 0:
                        continue
                    subsets.append([edge_pairs[int(idx)] for idx in subset_indices.tolist()])
            else:
                for start in range(0, num_edges - subset_size + 1, subset_size):
                    subset_indices = permutation[start:start + subset_size]
                    subsets.append([edge_pairs[idx] for idx in subset_indices])

        return subsets

    def _remove_undirected_edge_subset(
        self,
        edge_index: torch.Tensor,
        edge_subset: List[Tuple[int, int]]
    ) -> torch.Tensor:
        if not edge_subset:
            return edge_index

        src, dst = edge_index
        low = torch.minimum(src, dst)
        high = torch.maximum(src, dst)
        node_base = int(torch.maximum(low.max(), high.max()).item()) + 1
        edge_keys = low * node_base + high
        subset_tensor = torch.tensor(
            edge_subset, dtype=torch.long, device=edge_index.device
        )
        subset_low = torch.minimum(subset_tensor[:, 0], subset_tensor[:, 1])
        subset_high = torch.maximum(subset_tensor[:, 0], subset_tensor[:, 1])
        subset_keys = subset_low * node_base + subset_high
        return edge_index[:, ~torch.isin(edge_keys, subset_keys)]

    def _score_joint_subsets(
        self,
        graph: Data,
        edge_index: torch.Tensor,
        edge_subsets: List[List[Tuple[int, int]]]
    ) -> Tuple[torch.Tensor, object]:
        downstream_model = self._train_downstream_for_edges(graph, edge_index)

        subset_scores = torch.zeros(len(edge_subsets), device=self.device)
        with torch.no_grad():
            base_out = downstream_model.model(graph.x.float(), edge_index)
            base_loss = F.nll_loss(
                base_out[self.val_mask],
                self.labels[self.val_mask].to(base_out.device)
            )

            for subset_idx, edge_subset in enumerate(
                tqdm(edge_subsets, desc="       joint subset scoring", leave=False)
            ):
                reduced_edge_index = self._remove_undirected_edge_subset(edge_index, edge_subset)
                if reduced_edge_index.shape[1] > 0:
                    out = downstream_model.model(graph.x.float(), reduced_edge_index)
                    loss = F.nll_loss(out[self.val_mask], self.labels[self.val_mask].to(out.device))
                    subset_scores[subset_idx] = loss - base_loss
                else:
                    subset_scores[subset_idx] = float('inf')

        return subset_scores, downstream_model

    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        if self.train_mask is None or self.val_mask is None or self.labels is None:
            raise ValueError("Must set training data first. Please ensure the model has been trained through TrainableGraphSummarizationModel.")

        graph = original_graph.to(self.device)
        edge_index = to_undirected(graph.edge_index)

        summarized_graphs = [copy.deepcopy(graph)]
        summarized_graphs[0].edge_index = edge_index

        current_edges = edge_index.clone()
        total_undirected_edges = current_edges.shape[1] // 2
        undirected_edges_per_step = total_undirected_edges // num_steps

        print(f"开始联合采样无向图简化: 总无向 edges数 {total_undirected_edges}, Remove per step {undirected_edges_per_step}  edges无向 edges")
        print(f"Sampling repeats: {self.sampling_repeats}")
        print(f"Sampling subset num per repeat: {self.sampling_subset_num or 'all'}")

        for step in tqdm(range(num_steps), desc="Joint-subset graph simplification progress"):
            if current_edges.shape[1] <= 0:
                empty_graph = copy.deepcopy(graph)
                empty_graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                summarized_graphs.append(empty_graph)
                continue

            current_undirected_edges = current_edges.shape[1] // 2
            if step == num_steps - 1:
                undirected_edges_to_remove = current_undirected_edges
            else:
                undirected_edges_to_remove = min(undirected_edges_per_step, current_undirected_edges)

            print(f"Step {step+1}: 当前无向 edges数 {current_undirected_edges}, Remove {undirected_edges_to_remove} edges")

            if undirected_edges_to_remove >= current_undirected_edges:
                current_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else:
                edge_pairs = self._extract_undirected_edges(current_edges)
                edge_subsets = self._sample_edge_subsets(
                    edge_pairs,
                    undirected_edges_to_remove,
                    step
                )
                print(f"  Sampled {len(edge_subsets)} joint deletion subsets")

                if not edge_subsets:
                    raise ValueError("No joint deletion subsets sampled; check subset size and edge count.")

                start_time = time.time()
                subset_scores, _ = self._score_joint_subsets(graph, current_edges, edge_subsets)
                elapsed_time = time.time() - start_time
                print(f"  Joint subset scoring completed in {elapsed_time:.1f} seconds")

                current_edges = self._select_and_remove_edges(
                    current_edges,
                    edge_pairs,
                    edge_subsets,
                    subset_scores,
                    undirected_edges_to_remove
                )

            new_graph = copy.deepcopy(graph)
            new_graph.edge_index = current_edges
            summarized_graphs.append(new_graph)

        return summarized_graphs

    def _select_and_remove_edges(
        self,
        edge_index: torch.Tensor,
        edge_pairs: List[Tuple[int, int]],
        edge_subsets: List[List[Tuple[int, int]]],
        subset_scores: torch.Tensor,
        num_undirected_edges_to_remove: int
    ) -> torch.Tensor:
        raise NotImplementedError


class JointSubsetBestGradientSummarization(JointSubsetGradientBase):
    """Delete the sampled edge subset with the smallest joint loss impact."""

    def _select_and_remove_edges(
        self,
        edge_index: torch.Tensor,
        edge_pairs: List[Tuple[int, int]],
        edge_subsets: List[List[Tuple[int, int]]],
        subset_scores: torch.Tensor,
        num_undirected_edges_to_remove: int
    ) -> torch.Tensor:
        best_subset_idx = int(torch.argmin(subset_scores).item())
        best_subset = edge_subsets[best_subset_idx]
        print(f"  Best subset score: {subset_scores[best_subset_idx].item():.6f}")
        return self._remove_undirected_edge_subset(edge_index, best_subset)


class JointSubsetEdgeScoreGradientSummarization(JointSubsetGradientBase):
    """Aggregate sampled subset scores back to edge scores, then delete bottom-k edges."""
    sampling_subset_num_controls_split_count = True

    def _select_and_remove_edges(
        self,
        edge_index: torch.Tensor,
        edge_pairs: List[Tuple[int, int]],
        edge_subsets: List[List[Tuple[int, int]]],
        subset_scores: torch.Tensor,
        num_undirected_edges_to_remove: int
    ) -> torch.Tensor:
        score_sums = torch.zeros(len(edge_pairs), device=self.device)
        score_counts = torch.zeros(len(edge_pairs), device=self.device)
        edge_to_idx = {edge_pair: idx for idx, edge_pair in enumerate(edge_pairs)}

        for subset, subset_score in zip(edge_subsets, subset_scores):
            for edge_pair in subset:
                edge_idx = edge_to_idx[edge_pair]
                score_sums[edge_idx] += subset_score
                score_counts[edge_idx] += 1

        edge_scores = score_sums / torch.clamp(score_counts, min=1)
        edge_scores[score_counts == 0] = float('inf')
        print(f"  Edge score coverage: {(score_counts > 0).sum().item()}/{len(edge_pairs)} edges")
        return self._remove_undirected_edges_by_gradient(
            edge_index,
            edge_scores,
            edge_pairs,
            num_undirected_edges_to_remove
        )


class JointSubsetStabilityAwareEdgeScoreGradientSummarization(JointSubsetGradientBase):
    """
    Aggregate sampled subset scores back to edge scores with a stability penalty.

    Edges are scored by mean subset loss delta plus a variance penalty, so edges
    that are consistently low-impact across sampled subsets are preferred.
    """
    sampling_subset_num_controls_split_count = True

    def __init__(
        self,
        input_dim: int = None,
        downstream_model_type: str = 'gcn',
        hidden_dim: Optional[int] = None,
        train_epochs: int = 200,
        sampling_repeats: int = 5,
        sampling_subset_num: Optional[int] = None,
        sampling_seed: int = 42,
        gat_heads: int = 8,
        dropout: Optional[float] = None,
        attention_dropout: Optional[float] = None,
        negative_slope: float = 0.2,
        lr: Optional[float] = None,
        weight_decay: float = 5e-4,
        early_stopping_patience: Optional[int] = None,
        stability_penalty: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            downstream_model_type=downstream_model_type,
            hidden_dim=hidden_dim,
            train_epochs=train_epochs,
            sampling_repeats=sampling_repeats,
            sampling_subset_num=sampling_subset_num,
            sampling_seed=sampling_seed,
            gat_heads=gat_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            negative_slope=negative_slope,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=device,
        )
        self.stability_penalty = stability_penalty

    def _select_and_remove_edges(
        self,
        edge_index: torch.Tensor,
        edge_pairs: List[Tuple[int, int]],
        edge_subsets: List[List[Tuple[int, int]]],
        subset_scores: torch.Tensor,
        num_undirected_edges_to_remove: int
    ) -> torch.Tensor:
        score_sums = torch.zeros(len(edge_pairs), device=self.device)
        score_sumsq = torch.zeros(len(edge_pairs), device=self.device)
        score_counts = torch.zeros(len(edge_pairs), device=self.device)
        edge_to_idx = {edge_pair: idx for idx, edge_pair in enumerate(edge_pairs)}

        for subset, subset_score in zip(edge_subsets, subset_scores):
            if not torch.isfinite(subset_score):
                continue
            for edge_pair in subset:
                edge_idx = edge_to_idx[edge_pair]
                score_sums[edge_idx] += subset_score
                score_sumsq[edge_idx] += subset_score * subset_score
                score_counts[edge_idx] += 1

        mean_scores = score_sums / torch.clamp(score_counts, min=1)
        variance = score_sumsq / torch.clamp(score_counts, min=1) - mean_scores.pow(2)
        variance = torch.clamp(variance, min=0.0)
        std_scores = torch.sqrt(variance)
        stable_scores = mean_scores + self.stability_penalty * std_scores
        stable_scores[score_counts == 0] = float('inf')

        print(
            f"  Stable edge score coverage: {(score_counts > 0).sum().item()}/{len(edge_pairs)} edges, "
            f"penalty={self.stability_penalty:.3f}"
        )
        return self._remove_undirected_edges_by_gradient(
            edge_index,
            stable_scores,
            edge_pairs,
            num_undirected_edges_to_remove
        )


class JointSubsetProductImportanceGradientSummarization(JointSubsetGradientBase):
    """
    Cross-model product-importance joint-subset variant.

    The same candidate deletion subsets are scored by representative GNNs
    (GCN, GAT and GraphSAGE by default). Each model's loss deltas are converted
    to within-model percentile importance scores, then multiplied. The selected
    subset is the one with the lowest product importance.
    """

    def __init__(self,
                 input_dim: int = None,
                 downstream_model_type: str = 'gcn',
                 hidden_dim: Optional[int] = None,
                 train_epochs: int = 200,
                 sampling_repeats: int = 5,
                 sampling_subset_num: Optional[int] = None,
                 sampling_seed: int = 42,
                 gat_heads: int = 8,
                 dropout: Optional[float] = None,
                 attention_dropout: Optional[float] = None,
                 negative_slope: float = 0.2,
                 lr: Optional[float] = None,
                 weight_decay: float = 5e-4,
                 early_stopping_patience: Optional[int] = None,
                 scoring_model_types: Optional[List[str]] = None,
                 device: Optional[torch.device] = None):
        super().__init__(
            input_dim=input_dim,
            downstream_model_type=downstream_model_type,
            hidden_dim=hidden_dim,
            train_epochs=train_epochs,
            sampling_repeats=sampling_repeats,
            sampling_subset_num=sampling_subset_num,
            sampling_seed=sampling_seed,
            gat_heads=gat_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            negative_slope=negative_slope,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            device=device
        )
        if scoring_model_types is None:
            self.scoring_model_types = ['gcn', 'gat', 'sage']
        elif isinstance(scoring_model_types, str):
            self.scoring_model_types = [scoring_model_types.lower()]
        else:
            self.scoring_model_types = [
                model_type.lower() for model_type in scoring_model_types
            ]
        if not self.scoring_model_types:
            raise ValueError("scoring_model_types must contain at least one model type")
        self.product_score_history = []

    def _train_scoring_model_for_edges(
        self,
        graph: Data,
        edge_index: torch.Tensor,
        model_type: str
    ):
        input_dim = graph.x.size(1)
        output_dim = int(self.labels.max()) + 1

        downstream_model = self._create_scoring_downstream_model(
            input_dim,
            output_dim,
            model_type=model_type
        )

        temp_graph = copy.deepcopy(graph)
        temp_graph.edge_index = edge_index.detach()
        downstream_model.train_model(
            temp_graph,
            self.train_mask,
            self.val_mask,
            self.labels,
            epochs=self.train_epochs
        )
        downstream_model.model.eval()
        return downstream_model

    def _score_joint_subsets_with_model(
        self,
        graph: Data,
        edge_index: torch.Tensor,
        edge_subsets: List[List[Tuple[int, int]]],
        model_type: str
    ) -> torch.Tensor:
        downstream_model = self._train_scoring_model_for_edges(
            graph,
            edge_index,
            model_type
        )

        subset_scores = torch.zeros(len(edge_subsets), device=self.device)
        with torch.no_grad():
            base_out = downstream_model.model(graph.x.float(), edge_index)
            base_loss = F.nll_loss(
                base_out[self.val_mask],
                self.labels[self.val_mask].to(base_out.device)
            )

            for subset_idx, edge_subset in enumerate(
                tqdm(edge_subsets, desc=f"       {model_type} product scoring", leave=False)
            ):
                reduced_edge_index = self._remove_undirected_edge_subset(edge_index, edge_subset)
                if reduced_edge_index.shape[1] > 0:
                    out = downstream_model.model(graph.x.float(), reduced_edge_index)
                    loss = F.nll_loss(out[self.val_mask], self.labels[self.val_mask].to(out.device))
                    subset_scores[subset_idx] = loss - base_loss
                else:
                    subset_scores[subset_idx] = float('inf')

        return subset_scores

    def _scores_to_percentile_importance(self, subset_scores: torch.Tensor) -> torch.Tensor:
        finite_mask = torch.isfinite(subset_scores)
        importance = torch.ones_like(subset_scores)

        finite_scores = subset_scores[finite_mask]
        if finite_scores.numel() == 0:
            return importance

        order = torch.argsort(finite_scores, stable=True)
        ranks = torch.empty_like(finite_scores)
        ranks[order] = torch.arange(
            1,
            finite_scores.numel() + 1,
            device=self.device,
            dtype=finite_scores.dtype
        )
        importance[finite_mask] = ranks / float(finite_scores.numel())
        return torch.clamp(importance, min=1e-6, max=1.0)

    def _score_joint_subsets(
        self,
        graph: Data,
        edge_index: torch.Tensor,
        edge_subsets: List[List[Tuple[int, int]]]
    ) -> Tuple[torch.Tensor, object]:
        importance_scores = []
        raw_score_summary = {}

        for model_type in self.scoring_model_types:
            print(f"  Product importance scoring with {model_type}")
            raw_scores = self._score_joint_subsets_with_model(
                graph,
                edge_index,
                edge_subsets,
                model_type
            )
            importance = self._scores_to_percentile_importance(raw_scores)
            importance_scores.append(importance)
            finite_raw = raw_scores[torch.isfinite(raw_scores)]
            raw_score_summary[model_type] = {
                "raw_min": float(finite_raw.min().item()) if finite_raw.numel() else None,
                "raw_mean": float(finite_raw.mean().item()) if finite_raw.numel() else None,
                "raw_max": float(finite_raw.max().item()) if finite_raw.numel() else None,
                "importance_min": float(importance.min().item()),
                "importance_mean": float(importance.mean().item()),
                "importance_max": float(importance.max().item()),
            }

        product_scores = torch.stack(importance_scores, dim=0).prod(dim=0)
        product_summary = {
            "scoring_model_types": list(self.scoring_model_types),
            "per_model": raw_score_summary,
            "product_min": float(product_scores.min().item()),
            "product_mean": float(product_scores.mean().item()),
            "product_max": float(product_scores.max().item()),
        }
        self.product_score_history.append(product_summary)
        print(
            "  Product importance score: "
            f"min={product_summary['product_min']:.6f}, "
            f"mean={product_summary['product_mean']:.6f}, "
            f"max={product_summary['product_max']:.6f}"
        )
        return product_scores, None

    def _select_and_remove_edges(
        self,
        edge_index: torch.Tensor,
        edge_pairs: List[Tuple[int, int]],
        edge_subsets: List[List[Tuple[int, int]]],
        subset_scores: torch.Tensor,
        num_undirected_edges_to_remove: int
    ) -> torch.Tensor:
        best_subset_idx = int(torch.argmin(subset_scores).item())
        best_subset = edge_subsets[best_subset_idx]
        print(f"  Best product importance: {subset_scores[best_subset_idx].item():.6f}")
        return self._remove_undirected_edge_subset(edge_index, best_subset)


class JointSubsetModelStableGradientSummarization(JointSubsetProductImportanceGradientSummarization):
    """
    INXplain(stable): cross-model rank-mean edge scoring.

    GCN, GAT and GraphSAGE score the same candidate deletion subsets. Each
    model's subset loss deltas are converted to percentile ranks, then those
    ranks are averaged back to edge-level scores over all sampled subsets that
    contain each edge. The final edge score is the mean edge rank across
    scoring models, and each step deletes the bottom-k edges.
    """
    sampling_subset_num_controls_split_count = True

    def __init__(
        self,
        input_dim: int = None,
        downstream_model_type: str = 'gcn',
        hidden_dim: Optional[int] = None,
        train_epochs: int = 200,
        sampling_repeats: int = 5,
        sampling_subset_num: Optional[int] = None,
        sampling_seed: int = 42,
        gat_heads: int = 8,
        dropout: Optional[float] = None,
        attention_dropout: Optional[float] = None,
        negative_slope: float = 0.2,
        lr: Optional[float] = None,
        weight_decay: float = 5e-4,
        early_stopping_patience: Optional[int] = None,
        scoring_model_types: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            downstream_model_type=downstream_model_type,
            hidden_dim=hidden_dim,
            train_epochs=train_epochs,
            sampling_repeats=sampling_repeats,
            sampling_subset_num=sampling_subset_num,
            sampling_seed=sampling_seed,
            gat_heads=gat_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            negative_slope=negative_slope,
            lr=lr,
            weight_decay=weight_decay,
            early_stopping_patience=early_stopping_patience,
            scoring_model_types=scoring_model_types,
            device=device,
        )
        self.model_stable_score_history = []

    def _score_joint_subsets(
        self,
        graph: Data,
        edge_index: torch.Tensor,
        edge_subsets: List[List[Tuple[int, int]]]
    ) -> Tuple[torch.Tensor, object]:
        rank_scores = []
        raw_score_summary = {}

        for model_type in self.scoring_model_types:
            print(f"  Model-stable rank scoring with {model_type}")
            raw_scores = self._score_joint_subsets_with_model(
                graph,
                edge_index,
                edge_subsets,
                model_type
            )
            ranks = self._scores_to_percentile_importance(raw_scores)
            rank_scores.append(ranks)

            finite_raw = raw_scores[torch.isfinite(raw_scores)]
            raw_score_summary[model_type] = {
                "raw_min": float(finite_raw.min().item()) if finite_raw.numel() else None,
                "raw_mean": float(finite_raw.mean().item()) if finite_raw.numel() else None,
                "raw_max": float(finite_raw.max().item()) if finite_raw.numel() else None,
                "rank_min": float(ranks.min().item()),
                "rank_mean": float(ranks.mean().item()),
                "rank_max": float(ranks.max().item()),
            }

        subset_rank_scores = torch.stack(rank_scores, dim=0)
        model_mean_subset_ranks = subset_rank_scores.mean(dim=1)
        summary = {
            "scoring_model_types": list(self.scoring_model_types),
            "per_model": raw_score_summary,
            "model_mean_subset_ranks": [
                float(value.item()) for value in model_mean_subset_ranks
            ],
        }
        self.model_stable_score_history.append(summary)
        print(
            "  Model-stable subset ranks: "
            + ", ".join(
                f"{model}={rank:.6f}"
                for model, rank in zip(
                    self.scoring_model_types,
                    summary["model_mean_subset_ranks"]
                )
            )
        )
        return subset_rank_scores, None

    def _select_and_remove_edges(
        self,
        edge_index: torch.Tensor,
        edge_pairs: List[Tuple[int, int]],
        edge_subsets: List[List[Tuple[int, int]]],
        subset_scores: torch.Tensor,
        num_undirected_edges_to_remove: int
    ) -> torch.Tensor:
        edge_to_idx = {edge_pair: idx for idx, edge_pair in enumerate(edge_pairs)}
        num_models = subset_scores.shape[0]
        per_model_edge_scores = []
        coverage = None

        for model_idx in range(num_models):
            score_sums = torch.zeros(len(edge_pairs), device=self.device)
            score_counts = torch.zeros(len(edge_pairs), device=self.device)

            for subset, subset_score in zip(edge_subsets, subset_scores[model_idx]):
                for edge_pair in subset:
                    edge_idx = edge_to_idx[edge_pair]
                    score_sums[edge_idx] += subset_score
                    score_counts[edge_idx] += 1

            edge_scores = score_sums / torch.clamp(score_counts, min=1)
            edge_scores[score_counts == 0] = float('inf')
            per_model_edge_scores.append(edge_scores)
            if coverage is None:
                coverage = score_counts

        model_stable_edge_scores = torch.stack(per_model_edge_scores, dim=0).mean(dim=0)
        covered_edges = (coverage > 0).sum().item() if coverage is not None else 0

        print(
            f"  Model-stable edge rank coverage: {covered_edges}/{len(edge_pairs)} edges, "
            f"models={','.join(self.scoring_model_types)}"
        )
        return self._remove_undirected_edges_by_gradient(
            edge_index,
            model_stable_edge_scores,
            edge_pairs,
            num_undirected_edges_to_remove
        )
