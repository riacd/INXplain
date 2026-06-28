"""
Gradient-Based Graph Simplification Model

Implements Development Model 2: Graph simplification method that progressively removes edges using gradient information.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from typing import List, Optional
import copy
import numpy as np
import time
from tqdm import tqdm

from .base import GraphSummarizationModel
from .legacy_downstream import create_legacy_downstream_model


class GradientBasedGraphSummarization(GraphSummarizationModel):
    """
    Gradient-Based Graph Simplification Model
    
    This model implements graph simplification through the following steps:
    1. Train downstream task model on training set
    2. Predict on validation set, compute loss
    3. Backpropagate to obtain gradients of edge weights
    4. Remove edges with smallest gradients (preserve edges with minimal performance impact)
    5. Repeat the above process until all edges are removed
    
    Note: This model learns how to select edges during training, but unlike learnable models,
    it uses gradient information rather than neural network parameters to decide which edges to remove.
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 downstream_model_type: str = 'gcn',
                 hidden_dim: int = 128,
                 train_epochs: int = 30,
                 device: Optional[torch.device] = None):
        """
        初始化Gradient-Based Graph Simplification Model

        Args:
            input_dim: Input feature dimension (for benchmark compatibility)
            downstream_model_type: Downstream task model type ('gcn' 或 'gat')
            hidden_dim: Hidden layer dimension
            train_epochs: Number of training epochs
            device: Computing device
        """
        self.input_dim = input_dim
        self.downstream_model_type = downstream_model_type
        self.hidden_dim = hidden_dim
        self.train_epochs = train_epochs
        self.device = device if device is not None else torch.device('cpu')

        # Store reference to training data (will be set during training)
        self.train_mask = None
        self.val_mask = None  
        self.labels = None
        
        # Flag indicating whether model has been trained
        self.is_trained = False
    
    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        Generate a series of simplified graphs
        
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
        
        # Result list, starting from original graph
        summarized_graphs = [copy.deepcopy(graph)]
        
        # Use sparse representation for efficiency
        edge_index = graph.edge_index
        current_edges = edge_index.clone()
        
        # Calculate number of edges to remove per step
        total_edges = current_edges.shape[1]
        edges_per_step = total_edges // num_steps
        remaining_edges = total_edges
        
        print(f"Starting gradient-based graph simplification: Total edges {total_edges}, Remove per step {edges_per_step}  edges")
        print("This may take some time, please be patient...")
        
        for step in tqdm(range(num_steps), desc="Graph simplification progress"):
            if current_edges.shape[1] <= 0:
                # 如果没有 edges了，添加空图
                empty_graph = copy.deepcopy(graph)
                empty_graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                summarized_graphs.append(empty_graph)
                continue
            
            # 计算当前Step要Remove的 edges数
            if step == num_steps - 1:
                # 最后一步Remove所有剩余 edges
                edges_to_remove = current_edges.shape[1]
            else:
                edges_to_remove = min(edges_per_step, current_edges.shape[1])
            
            print(f"Step {step+1}: 当前 edges数 {current_edges.shape[1]}, Remove {edges_to_remove}  edges")

            # 获取 edges的梯度信息（使用稀疏方法）
            print(f"  Computing {current_edges.shape[1]}  edges edges的重要性...")
            start_time = time.time()
            edge_gradients = self._compute_edge_gradients_sparse(graph, current_edges)
            elapsed_time = time.time() - start_time
            print(f"   edges重要性计算完成，耗时 {elapsed_time:.1f}  seconds")
            
            # 选择要Remove的 edges（梯度最小的 edges）
            current_edges = self._remove_edges_by_gradient_sparse(current_edges, edge_gradients, edges_to_remove)
            
            # Create new graph data
            new_graph = copy.deepcopy(graph)
            new_graph.edge_index = current_edges
            
            summarized_graphs.append(new_graph)
        
        return summarized_graphs
    
    def _compute_edge_gradients_sparse(self, graph: Data, edge_index: torch.Tensor) -> torch.Tensor:
        """
        使用稀疏表示计算 edges权重的梯度
        
        Args:
            graph: Current graph data
            edge_index: 当前 edges索引
            
        Returns:
            torch.Tensor: 每 edges edges的梯度值
        """
        # Create downstream task model
        input_dim = graph.x.size(1)
        output_dim = int(self.labels.max()) + 1
        
        downstream_model = create_legacy_downstream_model(
            model_type=self.downstream_model_type,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            device=self.device,
        )

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

        # 为每 edges edges计算梯度
        num_edges = edge_index.shape[1]
        edge_gradients = torch.zeros(num_edges, device=self.device)
        
        downstream_model.model.eval()
        
        # 批量计算 edges的重要性（简化版本：通过Remove每 edges edges看性能下降）
        with torch.no_grad():
            # Compute performance on complete graph as baseline
            base_out = downstream_model.model(graph.x.float(), edge_index)
            base_loss = F.nll_loss(base_out[self.val_mask], self.labels[self.val_mask].to(base_out.device))
            
            # 对每 edges edges计算Remove后的性能变化
            batch_size = min(50, num_edges)  # 减小批次大小以节省内存
            print(f"      Using batch size {batch_size} to process {num_edges}  edges edges...")

            for i in tqdm(range(0, num_edges, batch_size), desc="       edges重要性计算", leave=False):
                end_idx = min(i + batch_size, num_edges)
                batch_gradients = []
                
                for edge_idx in range(i, end_idx):
                    # 创建Remove当前 edges的 edges索引
                    mask = torch.ones(num_edges, dtype=torch.bool, device=self.device)
                    mask[edge_idx] = False
                    reduced_edge_index = edge_index[:, mask]
                    
                    # 计算Remove edges后的性能
                    if reduced_edge_index.shape[1] > 0:
                        out = downstream_model.model(graph.x.float(), reduced_edge_index)
                        loss = F.nll_loss(out[self.val_mask], self.labels[self.val_mask].to(out.device))
                        gradient = loss - base_loss  # 损失增加越多， edges越重要
                    else:
                        gradient = float('inf')  # 如果Remove这 edges edges导致图断开，设为无穷大
                    
                    batch_gradients.append(gradient)
                
                edge_gradients[i:end_idx] = torch.tensor(batch_gradients, device=self.device)
        
        return edge_gradients
    
    def _remove_edges_by_gradient_sparse(self, edge_index: torch.Tensor, gradients: torch.Tensor, num_edges_to_remove: int) -> torch.Tensor:
        """
        根据梯度信息Remove edges（稀疏版本）
        
        Args:
            edge_index: 当前 edges索引
            gradients:  edges的梯度信息
            num_edges_to_remove: 要Remove的 edges数
            
        Returns:
            torch.Tensor: Remove edges后的 edges索引
        """
        if num_edges_to_remove >= edge_index.shape[1]:
            # Remove所有 edges
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        # 找到梯度最小的 edges（Remove对性能影响最小的 edges）
        _, indices_to_remove = torch.topk(gradients, k=num_edges_to_remove, largest=False)
        
        # 创建保留 edges的掩码
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=self.device)
        mask[indices_to_remove] = False
        
        # 返回保留的 edges
        return edge_index[:, mask]
    
    def _compute_edge_gradients(self, graph: Data, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算 edges权重的梯度
        
        Args:
            graph: Current graph data
            adj_matrix: Current adjacency matrix (requires gradient)
            
        Returns:
            torch.Tensor:  edges的梯度信息
        """
        # Create downstream task model
        input_dim = graph.x.size(1)
        output_dim = int(self.labels.max()) + 1
        
        downstream_model = create_legacy_downstream_model(
            model_type=self.downstream_model_type,
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            device=self.device,
        )

        # 使用当前邻接矩阵创建 edges索引
        edge_index, _ = dense_to_sparse(adj_matrix.detach())
        temp_graph = copy.deepcopy(graph)
        temp_graph.edge_index = edge_index
        
        # Train downstream model
        downstream_model.train_model(
            temp_graph, 
            self.train_mask, 
            self.val_mask, 
            self.labels, 
            epochs=self.train_epochs
        )
        
        # Compute loss on validation set and calculate gradients
        downstream_model.model.eval()
        
        # Recreate adjacency matrix that requires gradients
        adj_for_grad = adj_matrix.clone().detach().requires_grad_(True)
        
        # Forward propagation
        edge_index_grad, _ = dense_to_sparse(adj_for_grad)
        out = downstream_model.model(graph.x.float(), edge_index_grad)
        
        # Compute validation loss
        val_loss = F.nll_loss(out[self.val_mask], self.labels[self.val_mask].to(out.device))
        
        # Backpropagation
        val_loss.backward()
        
        # Check if gradient exists
        if adj_for_grad.grad is None:
            print("Warning: Adjacency matrix gradient is None, returning zero gradient")
            return torch.zeros_like(adj_for_grad)
        
        # Return gradient
        return adj_for_grad.grad.clone()
    
    def _remove_edges_by_gradient(self, adj_matrix: torch.Tensor, gradients: torch.Tensor, num_edges_to_remove: int) -> torch.Tensor:
        """
        根据梯度信息Remove edges
        
        Args:
            adj_matrix: 当前邻接矩阵
            gradients:  edges的梯度信息
            num_edges_to_remove: 要Remove的 edges数
            
        Returns:
            torch.Tensor: Remove edges后的邻接矩阵
        """
        # Only consider upper triangular part (undirected graph)
        mask = torch.triu(adj_matrix.bool(), diagonal=1)
        
        # 获取现有 edges的位置和对应的梯度
        edge_positions = mask.nonzero(as_tuple=False)
        
        if len(edge_positions) == 0:
            return adj_matrix
        
        # 获取这些 edges对应的梯度值
        edge_grads = gradients[edge_positions[:, 0], edge_positions[:, 1]]
        
        # 找到梯度最小的 edges（Remove对性能影响最小的 edges）
        _, indices_to_remove = torch.topk(edge_grads, k=min(num_edges_to_remove, len(edge_grads)), largest=False)
        
        # Create new adjacency matrix
        new_adj = adj_matrix.clone().detach()
        
        for idx in indices_to_remove:
            i, j = edge_positions[idx]
            # Remove无向 edges（对称Remove）
            new_adj[i, j] = 0
            new_adj[j, i] = 0
        
        return new_adj
    
    def reset(self) -> None:
        """Reset model state"""
        self.train_mask = None
        self.val_mask = None
        self.labels = None
