"""
Training Strategy Implementation

Implements fixed reweighting and dynamic reweighting training strategies according to MODEL.md document.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Dict, Any, Optional, Callable
import copy
import math
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from .main_model import LearnableGraphSummarization


class TrainingStrategy(ABC):
    """Training strategy base class"""
    
    @abstractmethod
    def compute_weights(self, gradients: List[torch.Tensor], **kwargs) -> List[float]:
        """
        Compute weights for each step.
        
        Args:
            gradients: 各步骤的Gradient list
            
        Returns:
            Weight list
        """
        pass


class FixedReweightingStrategy(TrainingStrategy):
    """Fixed reweighting strategy"""
    
    def __init__(self, strategy_type: str = 'uniform'):
        """
        Initialize fixed reweighting strategy.
        
        Args:
            strategy_type: Strategy type
                - 'uniform': All weights are 1
                - 'cosine': 0.5 + 0.5*cos(k/N_step)
        """
        self.strategy_type = strategy_type
    
    def compute_weights(self, gradients: List[torch.Tensor], num_steps: int, **kwargs) -> List[float]:
        """Compute fixed weights"""
        if self.strategy_type == 'uniform':
            return [1.0] * num_steps
        elif self.strategy_type == 'cosine':
            weights = []
            for k in range(1, num_steps + 1):
                weight = 0.5 + 0.5 * math.cos(k / num_steps * math.pi)
                weights.append(weight)
            return weights
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")


class FrankWolfeSolver:
    """Frank-Wolfe solver"""
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
    
    def solve(self, gradients: List[torch.Tensor]) -> List[float]:
        """
        Solve optimal weights using Frank-Wolfe algorithm.
        
        Args:
            gradients: Gradient list
            
        Returns:
            最优Weight list
        """
        N = len(gradients)
        if N == 0:
            return []
        
        # Initialize weights
        eta = [1.0 / N] * N
        
        # Flatten gradients to vectors
        grad_vectors = []
        for g in gradients:
            if isinstance(g, torch.Tensor):
                grad_vectors.append(g.detach().flatten())
            else:
                # If scalar, convert to tensor
                grad_vectors.append(torch.tensor([float(g)]))
        
        # Ensure all gradient vectors have the same length
        max_len = max(len(gv) for gv in grad_vectors)
        for i in range(len(grad_vectors)):
            if len(grad_vectors[i]) < max_len:
                # Pad with zeros
                padding = torch.zeros(max_len - len(grad_vectors[i]))
                grad_vectors[i] = torch.cat([grad_vectors[i], padding])
        
        for t in range(self.max_iterations):
            # Compute current direction
            d = torch.zeros_like(grad_vectors[0])
            for k, gk in enumerate(grad_vectors):
                d += eta[k] * gk
            
            # Find minimum inner product
            min_dot = float('inf')
            k_star = 0
            for k, gk in enumerate(grad_vectors):
                dot_product = torch.dot(d, gk).item()
                if dot_product < min_dot:
                    min_dot = dot_product
                    k_star = k
            
            # Build vertex vector
            v = [0.0] * N
            v[k_star] = 1.0
            
            # Line search
            eta_tensor = torch.tensor(eta)
            v_tensor = torch.tensor(v)
            diff = v_tensor - eta_tensor
            
            # Compute optimal step size
            numerator = torch.dot(d, eta_tensor * d.sum() - sum(eta[j] * grad_vectors[j] for j in range(N)))
            denominator = torch.dot(diff, sum(eta[j] * grad_vectors[j] for j in range(N)) - d.sum() * v_tensor)
            
            if abs(denominator) > 1e-8:
                gamma = min(1.0, max(0.0, (numerator / denominator).item()))
            else:
                gamma = 0.0
            
            # Update weights
            for k in range(N):
                eta[k] = (1 - gamma) * eta[k] + gamma * v[k]
        
        return eta


class UGDSolver:
    """UGD solver"""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
    
    def solve(self, gradients: List[torch.Tensor]) -> List[float]:
        """
        Solve optimal weights using UGD algorithm.
        
        Args:
            gradients: Gradient list
            
        Returns:
            最优Weight list
        """
        N = len(gradients)
        if N == 0:
            return []
        
        # Initialize weights
        eta = [1.0 / N] * N
        
        # Flatten gradients to vectors
        grad_vectors = []
        for g in gradients:
            if isinstance(g, torch.Tensor):
                grad_vectors.append(g.detach().flatten())
            else:
                grad_vectors.append(torch.tensor([float(g)]))
        
        # Ensure all gradient vectors have the same length
        max_len = max(len(gv) for gv in grad_vectors)
        for i in range(len(grad_vectors)):
            if len(grad_vectors[i]) < max_len:
                padding = torch.zeros(max_len - len(grad_vectors[i]))
                grad_vectors[i] = torch.cat([grad_vectors[i], padding])
        
        for t in range(self.max_iterations):
            # Compute current direction
            d = torch.zeros_like(grad_vectors[0])
            for k, gk in enumerate(grad_vectors):
                d += eta[k] * gk
            
            # Update weights
            for k in range(N):
                dot_product = torch.dot(d, grad_vectors[k]).item()
                eta[k] -= self.learning_rate * dot_product
            
            # Project to simplex
            eta = self._project_to_simplex(eta)
        
        return eta
    
    def _project_to_simplex(self, weights: List[float]) -> List[float]:
        """Project to simplex"""
        weights = np.array(weights)
        n = len(weights)
        
        # Ensure non-negative
        weights = np.maximum(weights, 0)
        
        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            weights = np.ones(n) / n
        
        return weights.tolist()


class DynamicReweightingStrategy(TrainingStrategy):
    """Dynamic reweighting strategy"""
    
    def __init__(self, solver_type: str = 'frank_wolfe'):
        """
        Initialize dynamic reweighting strategy.
        
        Args:
            solver_type: Solver type
                - 'frank_wolfe': Frank-Wolfe算法
                - 'ugd': UGD算法
        """
        self.solver_type = solver_type
        
        if solver_type == 'frank_wolfe':
            self.solver = FrankWolfeSolver()
        elif solver_type == 'ugd':
            self.solver = UGDSolver()
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
    
    def compute_weights(self, gradients: List[torch.Tensor], **kwargs) -> List[float]:
        """Compute dynamic weights using solver"""
        return self.solver.solve(gradients)


class GraphSummarizationTrainer:
    """Graph summarization model trainer"""
    
    def __init__(self,
                 model: LearnableGraphSummarization,
                 strategy: TrainingStrategy,
                 downstream_model_factory: Callable = None,
                 optimizer: torch.optim.Optimizer = None,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: Graph summarization model
            strategy: Training strategy
            downstream_model_factory: Factory function to create downstream model
            optimizer: Optimizer
            device: Computing device
        """
        self.model = model
        self.strategy = strategy
        self.downstream_model_factory = downstream_model_factory
        self.device = device
        
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer

    def train_epoch_fixed_reweighting(self, 
                                     graph: Data,
                                     train_labels: torch.Tensor,
                                     train_mask: torch.Tensor,
                                     val_mask: torch.Tensor,
                                     num_steps: int = 10,
                                     downstream_epochs: int = 100) -> Dict[str, Any]:
        """Training epoch for fixed weight strategy"""
        return self._train_epoch_impl(
            graph, train_labels, train_mask, val_mask, 
            num_steps, downstream_epochs, use_dynamic_weights=False
        )
    
    def train_epoch_dynamic_reweighting(self, 
                                       graph: Data,
                                       train_labels: torch.Tensor,
                                       train_mask: torch.Tensor,
                                       val_mask: torch.Tensor,
                                       num_steps: int = 10,
                                       downstream_epochs: int = 100) -> Dict[str, Any]:
        """Training epoch for dynamic weight strategy"""
        return self._train_epoch_impl(
            graph, train_labels, train_mask, val_mask, 
            num_steps, downstream_epochs, use_dynamic_weights=True
        )
    
    def _train_epoch_impl(self, 
                         graph: Data,
                         train_labels: torch.Tensor,
                         train_mask: torch.Tensor,
                         val_mask: torch.Tensor,
                         num_steps: int = 10,
                         downstream_epochs: int = 100,
                         use_dynamic_weights: bool = False) -> Dict[str, Any]:
        """
        Training implementation supporting fixed and dynamic weight strategies.
        
        Args:
            graph: Training graph
            train_labels: Training labels
            train_mask: Training mask
            val_mask: Validation mask
            num_steps: Number of steps
            downstream_epochs: Number of epochs for downstream task model training
            use_dynamic_weights: Whether to use dynamic weights
            
        Returns:
            Training information dictionary
        """
        self.model.train()
        
        if use_dynamic_weights:
            return self._train_epoch_dynamic(
                graph, train_labels, train_mask, val_mask, 
                num_steps, downstream_epochs
            )
        else:
            return self._train_epoch_fixed(
                graph, train_labels, train_mask, val_mask, 
                num_steps, downstream_epochs
            )
    
    def _train_epoch_fixed(self, graph, train_labels, train_mask, val_mask, num_steps, downstream_epochs):
        """Fixed weight training implementation"""
        self.optimizer.zero_grad()
        
        step_losses = []
        downstream_models = []
        
        # For each step k from 1 to N
        for k in range(1, num_steps + 1):
            current_graph, downstream_model = self._generate_step_k_graph_and_train_downstream(
                graph, train_labels, train_mask, val_mask, k, num_steps, downstream_epochs
            )
            downstream_models.append(downstream_model)
            
            # Compute task loss L^k(φ) = Eval(h^{(k)}, G_k, Y_train)
            task_loss = downstream_model.evaluate(
                graph=current_graph,
                test_mask=train_mask,
                labels=train_labels
            )
            step_loss = torch.tensor(task_loss, device=self.device, requires_grad=True)
            step_losses.append(step_loss)
        
        # Compute fixed weights η^k
        weights = self.strategy.compute_weights([loss.item() for loss in step_losses], num_steps=num_steps)
        
        # Accumulate weighted loss: L_total = Σ η^k * L^k(φ)
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for w, loss in zip(weights, step_losses):
            total_loss = total_loss + w * loss
        
        # Compute gradient and update
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'step_losses': [loss.item() for loss in step_losses],
            'weights': weights,
            'downstream_models': downstream_models
        }
    
    def _train_epoch_dynamic(self, graph, train_labels, train_mask, val_mask, num_steps, downstream_epochs):
        """Dynamic weight training implementation, following dynamic reweighting pseudocode"""
        
        # Initialize gradient accumulator G ← 0
        step_losses = []
        step_gradients = []
        downstream_models = []
        
        # For k=1 to N: 计算每个步骤的损失和梯度
        for k in range(1, num_steps + 1):
            self.optimizer.zero_grad()  # Zero gradients, prepare for single-step gradient computation
            
            current_graph, downstream_model = self._generate_step_k_graph_and_train_downstream(
                graph, train_labels, train_mask, val_mask, k, num_steps, downstream_epochs
            )
            downstream_models.append(downstream_model)
            
            # Compute task loss L^k(φ) = Eval(h^{(k)}, G_k, Y_train)  
            task_loss = downstream_model.evaluate(
                graph=current_graph,
                test_mask=train_mask,
                labels=train_labels
            )
            step_loss = torch.tensor(task_loss, device=self.device, requires_grad=True)
            step_losses.append(step_loss.item())
            
            # Compute gradient g_k = ∇_φ L^k(φ)
            step_loss.backward()
            
            # Collect gradients
            step_gradient = []
            for param in self.model.parameters():
                if param.grad is not None:
                    step_gradient.append(param.grad.clone().flatten())
                else:
                    step_gradient.append(torch.zeros_like(param).flatten())
            
            step_gradient = torch.cat(step_gradient)
            step_gradients.append(step_gradient)
        
        # {η^k} ← Ψ({g_k}): 使用求解器计算权重
        weights = self.strategy.compute_weights(step_gradients)
        
        # Aggregate update direction G ← Σ η^k * g_k
        self.optimizer.zero_grad()
        
        aggregated_gradient = torch.zeros_like(step_gradients[0])
        for w, grad in zip(weights, step_gradients):
            aggregated_gradient += w * grad
        
        # Assign aggregated gradients to model parameters
        param_idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.grad = aggregated_gradient[param_idx:param_idx + param_size].view_as(param)
            param_idx += param_size
        
        # Update parameters φ ← Optimizer(φ, G)
        self.optimizer.step()
        
        # Compute total loss for logging
        total_loss = sum(w * loss for w, loss in zip(weights, step_losses))
        
        return {
            'total_loss': total_loss,
            'step_losses': step_losses,
            'weights': weights,
            'downstream_models': downstream_models
        }
    
    def _generate_step_k_graph_and_train_downstream(self, graph, train_labels, train_mask, val_mask, k, num_steps, downstream_epochs):
        """Generate step k simplified graph and train downstream model"""
        # Generate simplified graph G_k using g_φ
        current_graph = copy.deepcopy(graph)
        
        # Apply k steps of simplification
        for step in range(k):
            if current_graph.edge_index.size(1) == 0:
                break
                
            # A_k ← g_φ(G_{k-1}, k-1): Get edge removal decisions
            edge_scores = self.model.forward(
                current_graph.x, 
                current_graph.edge_index, 
                step + 1
            )
            
            # Determine edges to keep
            keep_ratio = 1.0 - ((step + 1) / num_steps)
            num_keep = max(0, int(current_graph.edge_index.size(1) * keep_ratio))
            
            if num_keep == 0:
                # Empty graph
                current_graph = Data(
                    x=graph.x,
                    edge_index=torch.zeros((2, 0), dtype=torch.long, device=graph.edge_index.device),
                    y=graph.y,
                    num_nodes=graph.x.size(0)
                )
                break
            else:
                # G_k ← (A - Σ A_i, X): Apply edge removals
                _, top_indices = torch.topk(edge_scores, num_keep, largest=True)
                kept_edge_index = current_graph.edge_index[:, top_indices]
                
                current_graph = Data(
                    x=graph.x,
                    edge_index=kept_edge_index,
                    y=graph.y,
                    num_nodes=graph.x.size(0)
                )
        
        # Initialize downstream model h^{(k)} with random weights
        if self.downstream_model_factory is not None:
            downstream_model = self.downstream_model_factory()
        else:
            # Default GCN downstream model
            from ..models import GCNDownstreamModel
            downstream_model = GCNDownstreamModel(
                input_dim=current_graph.x.size(1),
                device=self.device
            )
        
        # Train h^{(k)} on (G_k, Y_train) until convergence
        downstream_model.train_model(
            graph=current_graph,
            train_mask=train_mask,
            val_mask=val_mask,
            labels=train_labels,
            epochs=downstream_epochs
        )
        
        return current_graph, downstream_model
    
    def train(self,
              graph: Data,
              train_labels: torch.Tensor,
              train_mask: torch.Tensor,
              val_mask: torch.Tensor,
              num_epochs: int = 100,
              num_steps: int = 10,
              downstream_epochs: int = 50) -> List[Dict[str, Any]]:
        """
        Complete training process, implemented according to MODEL.md pseudocode.
        
        Args:
            graph: Training graph
            train_labels: Training labels
            train_mask: Training mask
            val_mask: Validation mask
            num_epochs: Number of epochs for graph summarization network training
            num_steps: Number of steps
            downstream_epochs: Number of epochs for each downstream task model
            
        Returns:
            Training history
        """
        history = []
        
        print(f"开始Training graph总结网络 ({num_epochs} epochs)")
        print(f"Each epoch will train {num_steps} downstream task models, each model trained for {downstream_epochs} epochs")
        
        # Create progress bar
        epoch_pbar = tqdm(range(num_epochs), desc="Training graph总结网络", unit="epoch")
        
        for epoch in epoch_pbar:
            # 根据Strategy type选择训练方法
            if isinstance(self.strategy, DynamicReweightingStrategy):
                epoch_info = self.train_epoch_dynamic_reweighting(
                    graph, train_labels, train_mask, val_mask, 
                    num_steps, downstream_epochs
                )
            else:
                epoch_info = self.train_epoch_fixed_reweighting(
                    graph, train_labels, train_mask, val_mask, 
                    num_steps, downstream_epochs
                )
            
            epoch_info['epoch'] = epoch
            history.append(epoch_info)
            
            # Update progress bar description
            strategy_type = "Dynamic" if isinstance(self.strategy, DynamicReweightingStrategy) else "Fixed"
            epoch_pbar.set_postfix({
                'Loss': f'{epoch_info["total_loss"]:.4f}',
                'Strategy': strategy_type,
                'Weights': f'{epoch_info["weights"][0]:.3f}...{epoch_info["weights"][-1]:.3f}'
            })
            
            # Detailed output every 10 epochs
            if epoch % 10 == 0:
                tqdm.write(f"Epoch {epoch} ({strategy_type}): Total Loss = {epoch_info['total_loss']:.4f}")
                tqdm.write(f"  Step losses: {[f'{x:.4f}' for x in epoch_info['step_losses']]}")
                tqdm.write(f"  Weights: {[f'{x:.3f}' for x in epoch_info['weights']]}")
        
        epoch_pbar.close()
        print("Graph summarization network training completed")
        return history