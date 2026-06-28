"""
Model Base Class Definitions

Defines abstract base classes for Graph Pruning and Downstream Task models.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from abc import ABC, abstractmethod
from typing import List, Optional


class GraphSummarizationModel(ABC):
    """
    Abstract base class for Graph Summarization models.
    
    Defines the interface that all graph summarization models should implement.
    Models should be able to take an input graph and produce a series of 
    simplified graphs with progressively fewer edges.
    """
    
    @abstractmethod
    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        Generate a series of simplified graphs from the original graph.
        
        Args:
            original_graph: Input graph to be simplified
            num_steps: Number of simplification steps to perform
            
        Returns:
            List[Data]: List of simplified graphs with progressively fewer edges
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset model state for new graph processing."""
        pass


class DownstreamModel(ABC):
    """
    Abstract base class for downstream task models.
    
    These models perform specific tasks (like node classification) on graphs
    produced by graph summarization models.
    """
    
    @abstractmethod
    def train_model(self, 
                    graph: Data,
                    train_mask: torch.Tensor, 
                    val_mask: torch.Tensor, 
                    labels: torch.Tensor, 
                    epochs: int = 100) -> None:
        """
        Train the downstream model on the provided graph.
        
        Args:
            graph: Input graph data
            train_mask: Boolean mask for training nodes
            val_mask: Boolean mask for validation nodes
            labels: Node labels for training
            epochs: Number of training epochs
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                 graph: Data, 
                 test_mask: torch.Tensor, 
                 labels: torch.Tensor) -> float:
        """
        Evaluate the model on the test set.
        
        Args:
            graph: Input graph data
            test_mask: Boolean mask for test nodes
            labels: True labels for evaluation
            
        Returns:
            float: Evaluation loss/metric
        """
        pass
    
    @abstractmethod
    def predict(self, graph: Data) -> torch.Tensor:
        """
        Generate predictions for all nodes in the graph.

        Args:
            graph: Input graph data

        Returns:
            torch.Tensor: Prediction logits for all nodes
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset model parameters."""
        pass