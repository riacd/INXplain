"""
Training Strategy Variants

Uses training strategies as ablation experiment variants for LearnableGraphSummarization model.
According to MODEL.md, training strategy is part of the model, not an independent component.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Union
import math
import numpy as np
import torch
from torch_geometric.data import Data

from .main_model import LearnableGraphSummarization
from .training_strategies import (
    FixedReweightingStrategy, 
    DynamicReweightingStrategy,
    GraphSummarizationTrainer
)
from .gradient_based import GradientBasedGraphSummarization
from .gradient_based_undirected import GradientBasedUndirectedGraphSummarization, JointSubsetGradientBase
from .base import GraphSummarizationModel


class LearnableGraphSummarization_FixedUniform(LearnableGraphSummarization):
    """
    Model variant using fixed uniform weight training strategy.
    
    Training strategy: All step weights are 1 (uniform weighting)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = FixedReweightingStrategy('uniform')
        self.variant_name = "fixed_uniform"
    
    def get_training_strategy(self):
        """Return training strategy used by this variant"""
        return self.training_strategy


class LearnableGraphSummarization_FixedCosine(LearnableGraphSummarization):
    """
    Model variant using fixed cosine weight training strategy.
    
    Training strategy: weight = 0.5 + 0.5*cos(k/N_step)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = FixedReweightingStrategy('cosine')
        self.variant_name = "fixed_cosine"
    
    def get_training_strategy(self):
        """Return training strategy used by this variant"""
        return self.training_strategy


class LearnableGraphSummarization_DynamicFW(LearnableGraphSummarization):
    """
    Model variant using dynamic Frank-Wolfe weight training strategy.
    
    Training strategy: Dynamically compute weights using Frank-Wolfe algorithm
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_strategy = DynamicReweightingStrategy('frank_wolfe')
        self.variant_name = "dynamic_frank_wolfe"
    
    def get_training_strategy(self):
        """Return training strategy used by this variant"""
        return self.training_strategy


class LearnableGraphSummarization_DynamicUGD(LearnableGraphSummarization):
    """
    Model variant using dynamic UGD weight training strategy.
    
    Training strategy: Dynamically compute weights using UGD algorithm
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.training_strategy = DynamicReweightingStrategy('ugd')
        self.variant_name = "dynamic_ugd"
    
    def get_training_strategy(self):
        """Return training strategy used by this variant"""
        return self.training_strategy


class TrainableGraphSummarizationModel:
    """
    Trainable graph summarization model wrapper.
    
    Combines model with its corresponding training strategy, provides unified training interface.
    """
    
    def __init__(self, model: Union[LearnableGraphSummarization, GradientBasedGraphSummarization, GradientBasedUndirectedGraphSummarization, JointSubsetGradientBase], downstream_model_factory=None):
        """
        Initialize trainable model.

        Args:
            model: Graph summarization model
            downstream_model_factory: Factory function to create downstream task model, uses default GCN if None
        """
        self.model = model
        self.downstream_model_factory = downstream_model_factory

        # Select different training methods based on model type
        if isinstance(model, (GradientBasedGraphSummarization, GradientBasedUndirectedGraphSummarization, JointSubsetGradientBase)):
            # Gradient-based models use special training method
            self.training_strategy = None
            self.trainer = None
        elif isinstance(model, LearnableGraphSummarization):
            # Learnable models use standard training strategy
            if hasattr(model, 'training_strategy'):
                self.training_strategy = model.training_strategy
            else:
                # Default to fixed uniform weight strategy
                self.training_strategy = FixedReweightingStrategy('uniform')
            
            # Create trainer
            self.trainer = GraphSummarizationTrainer(
                model=self.model,
                strategy=self.training_strategy,
                downstream_model_factory=downstream_model_factory
            )
        else:
            # Other models do not support training yet
            self.training_strategy = None
            self.trainer = None
    
    def train(self,
              graph: Data,
              train_labels: torch.Tensor,
              train_mask: torch.Tensor,
              val_mask: torch.Tensor,
              *args, **kwargs):
        """Train model"""
        if isinstance(self.model, (GradientBasedGraphSummarization, GradientBasedUndirectedGraphSummarization, JointSubsetGradientBase)):
            # Gradient-based models need to set training data
            self.model.train_mask = train_mask.to(self.model.device)
            self.model.val_mask = val_mask.to(self.model.device)
            self.model.labels = train_labels.to(self.model.device)
            self.model.is_trained = True

            # Return empty training history (gradient-based models do not need traditional training)
            return {'loss_history': [], 'val_loss_history': []}
        elif self.trainer is not None:
            # Learnable models use standard trainer
            return self.trainer.train(graph, train_labels, train_mask, val_mask, *args, **kwargs)
        else:
            # Models that do not support training
            print("Warning: This model does not support training")
            return {'loss_history': [], 'val_loss_history': []}
    
    def summarize(self, *args, **kwargs):
        """Generate graph summary (delegate to underlying model)"""
        return self.model.summarize(*args, **kwargs)
    
    def get_variant_info(self) -> Dict[str, Any]:
        """Get variant information"""
        return {
            'model_type': type(self.model).__name__,
            'training_strategy': type(self.training_strategy).__name__,
            'variant_name': getattr(self.model, 'variant_name', 'unknown'),
            'node_encoder': getattr(self.model, 'node_encoder_type', 'gin'),
            'hidden_dim': getattr(self.model, 'hidden_dim', 256),
            'use_step_embedding': getattr(self.model, 'use_step_embedding', True),
            'use_edge_diff': getattr(self.model, 'use_edge_diff', True)
        }
