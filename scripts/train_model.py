#!/usr/bin/env python3
"""
Training script for the Main Graph Summarization model.

This script implements the training procedure described in MODEL.md,
including the multi-task learning approach with reweighting strategies.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

# Add GS package to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from GS.datasets import DatasetLoader
from GS.models import MainGraphSummarizationModel, GCNDownstreamModel
from GS.metrics import InformationMetric
from torch_geometric.data import Data
import copy


class GraphSummarizationTrainer:
    """Trainer class for the Main Graph Summarization model."""
    
    def __init__(self, 
                 model: MainGraphSummarizationModel,
                 downstream_model,
                 device: torch.device,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 reweighting_strategy: str = 'uniform'):
        """
        Initialize trainer.
        
        Args:
            model: The graph summarization model to train
            downstream_model: Downstream task model for computing information metric
            device: Torch device
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            reweighting_strategy: Strategy for multi-task weighting ('uniform' or 'cosine')
        """
        self.model = model
        self.downstream_model = downstream_model
        self.device = device
        self.reweighting_strategy = reweighting_strategy
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
    def compute_reweighting(self, num_steps: int) -> List[float]:
        """
        Compute reweighting factors for multi-task learning.
        
        Args:
            num_steps: Number of summarization steps
            
        Returns:
            List of weights for each step
        """
        if self.reweighting_strategy == 'uniform':
            # All steps weighted equally
            return [1.0] * num_steps
        elif self.reweighting_strategy == 'cosine':
            # Cosine weighting: 0.5 + 0.5 * cos(k/N_step * pi)
            weights = []
            for k in range(num_steps):
                weight = 0.5 + 0.5 * np.cos(k / num_steps * np.pi)
                weights.append(weight)
            return weights
        else:
            raise ValueError(f"Unknown reweighting strategy: {self.reweighting_strategy}")
    
    def generate_edge_labels(self, 
                            original_graph: Data,
                            summary_graphs: List[Data],
                            step: int) -> torch.Tensor:
        """
        Generate edge removal labels for training.
        
        For each edge in the current graph, determine if it should be removed
        in the next step based on the ground truth summary graphs.
        
        Args:
            original_graph: Original input graph
            summary_graphs: List of target summary graphs
            step: Current step index
            
        Returns:
            Binary labels for each edge (1 = remove, 0 = keep)
        """
        current_edges = summary_graphs[step].edge_index
        next_edges = summary_graphs[step + 1].edge_index if step + 1 < len(summary_graphs) else torch.empty((2, 0))
        
        # Create edge sets for comparison
        current_edge_set = set()
        for i in range(current_edges.size(1)):
            edge = (int(current_edges[0, i]), int(current_edges[1, i]))
            current_edge_set.add(edge)
        
        next_edge_set = set()
        for i in range(next_edges.size(1)):
            edge = (int(next_edges[0, i]), int(next_edges[1, i]))
            next_edge_set.add(edge)
        
        # Generate labels
        labels = []
        for i in range(current_edges.size(1)):
            edge = (int(current_edges[0, i]), int(current_edges[1, i]))
            # Label is 1 if edge should be removed (not in next graph)
            label = 1.0 if edge not in next_edge_set else 0.0
            labels.append(label)
        
        return torch.tensor(labels, dtype=torch.float32, device=self.device)
    
    def train_step(self,
                  original_graph: Data,
                  num_steps: int,
                  train_mask: torch.Tensor,
                  val_mask: torch.Tensor,
                  test_mask: torch.Tensor) -> float:
        """
        Perform one training step using the multi-task learning approach.
        
        Args:
            original_graph: Input graph
            num_steps: Number of summarization steps
            train_mask: Training node mask
            val_mask: Validation node mask
            test_mask: Test node mask
            
        Returns:
            Total loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get reweighting factors
        weights = self.compute_reweighting(num_steps)
        
        # Generate target summary graphs (using random for now as ground truth)
        # In practice, this would come from a pre-computed optimal summarization
        from GS.models import RandomGraphSummarizationModel
        target_model = RandomGraphSummarizationModel(device=self.device)
        target_summary_graphs = target_model.summarize(original_graph, num_steps)
        
        # Initialize total loss
        total_loss = 0.0
        current_graph = copy.deepcopy(original_graph).to(self.device)
        
        # Autoregressive generation with loss computation
        for step in range(num_steps):
            if current_graph.edge_index.size(1) == 0:
                break
            
            # Forward pass to get edge logits
            edge_logits = self.model.forward(current_graph, step)
            
            if len(edge_logits) == 0:
                break
            
            # Generate edge labels based on target
            edge_labels = self.generate_edge_labels(
                original_graph, target_summary_graphs, step
            )
            
            # Compute loss for this step
            if len(edge_labels) > 0 and len(edge_logits) > 0:
                step_loss = self.criterion(edge_logits, edge_labels)
                
                # Apply reweighting
                weighted_loss = weights[step] * step_loss
                total_loss += weighted_loss
            
            # Apply the model's decision to get next graph
            with torch.no_grad():
                edge_probs = torch.sigmoid(edge_logits)
                total_edges = current_graph.edge_index.size(1)
                edges_to_remove = max(1, total_edges // (num_steps - step))
                
                _, remove_indices = torch.topk(edge_probs, min(edges_to_remove, len(edge_probs)))
                
                keep_mask = torch.ones(total_edges, dtype=torch.bool, device=self.device)
                keep_mask[remove_indices] = False
                
                new_edge_index = current_graph.edge_index[:, keep_mask]
                current_graph = Data(
                    x=current_graph.x,
                    edge_index=new_edge_index,
                    y=current_graph.y,
                    num_nodes=current_graph.num_nodes
                ).to(self.device)
        
        # Add information metric loss (optional, for end-to-end training)
        if self.downstream_model is not None:
            info_metric = InformationMetric(self.downstream_model, self.device)
            # Use the final simplified graph for downstream task loss
            self.downstream_model.reset()
            info_loss = info_metric.compute(
                current_graph, train_mask, val_mask, test_mask, 
                original_graph.y, epochs=50  # Fewer epochs during training
            )
            total_loss += 0.1 * info_loss  # Weight the downstream task loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return float(total_loss)
    
    def train(self,
             dataset_name: str,
             num_epochs: int = 100,
             num_steps: int = 10,
             save_dir: str = './ckpt',
             verbose: bool = True) -> Dict[str, Any]:
        """
        Train the model on a dataset.
        
        Args:
            dataset_name: Name of the dataset to train on
            num_epochs: Number of training epochs
            num_steps: Number of summarization steps
            save_dir: Directory to save checkpoints
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Load dataset
        loader = DatasetLoader('./data')
        original_graph, train_mask, val_mask, test_mask = loader.load_dataset(dataset_name)
        original_graph = loader.preprocess_for_summarization(original_graph)
        original_graph = original_graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device) 
        test_mask = test_mask.to(self.device)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training step
            train_loss = self.train_step(
                original_graph, num_steps, 
                train_mask, val_mask, test_mask
            )
            history['train_loss'].append(train_loss)
            history['epoch'].append(epoch)
            
            # Validation (simplified - just use training loss as proxy)
            val_loss = train_loss  # In practice, compute on validation set
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'config': {
                        'input_dim': self.model.input_dim,
                        'hidden_dim': self.model.hidden_dim,
                        'step_embedding_dim': self.model.step_embedding_dim,
                        'num_gin_layers': self.model.num_gin_layers,
                        'dropout': self.model.dropout
                    }
                }
                checkpoint_path = os.path.join(save_dir, f'best_model_{dataset_name}.pt')
                torch.save(checkpoint, checkpoint_path)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs} - Loss: {train_loss:.4f} - LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        # Save final model
        final_checkpoint = {
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
            'history': history,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'step_embedding_dim': self.model.step_embedding_dim,
                'num_gin_layers': self.model.num_gin_layers,
                'dropout': self.model.dropout
            }
        }
        final_path = os.path.join(save_dir, f'final_model_{dataset_name}.pt')
        torch.save(final_checkpoint, final_path)
        
        if verbose:
            print(f"Training completed. Best loss: {best_loss:.4f}")
            print(f"Models saved to {save_dir}")
        
        return history


def main():
    parser = argparse.ArgumentParser(description='Train Graph Summarization Model')
    parser.add_argument('--dataset', type=str, default='Cora',
                       choices=['Cora', 'CiteSeer', 'PubMed'],
                       help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of summarization steps')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--reweighting', type=str, default='uniform',
                       choices=['uniform', 'cosine'],
                       help='Reweighting strategy for multi-task learning')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--save-dir', type=str, default='./ckpt',
                       help='Directory to save checkpoints')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    if not args.quiet:
        print(f"Training Graph Summarization Model")
        print(f"Dataset: {args.dataset}")
        print(f"Device: {device}")
        print(f"Epochs: {args.epochs}")
        print(f"Steps: {args.steps}")
        print(f"Reweighting: {args.reweighting}")
        print("-" * 50)
    
    # Get dataset info to determine input dimension
    loader = DatasetLoader('./data')
    data, _, _, _ = loader.load_dataset(args.dataset)
    input_dim = data.x.size(1)
    
    # Create model
    model = MainGraphSummarizationModel(
        input_dim=input_dim,
        hidden_dim=256,
        step_embedding_dim=32,
        num_gin_layers=3,
        dropout=0.2,
        device=device
    )
    
    # Create downstream model for information metric
    downstream_model = GCNDownstreamModel(
        input_dim=input_dim,
        hidden_dim=64,
        device=device
    )
    
    # Create trainer
    trainer = GraphSummarizationTrainer(
        model=model,
        downstream_model=downstream_model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        reweighting_strategy=args.reweighting
    )
    
    # Train model
    try:
        history = trainer.train(
            dataset_name=args.dataset,
            num_epochs=args.epochs,
            num_steps=args.steps,
            save_dir=args.save_dir,
            verbose=not args.quiet
        )
        
        # Save training history
        history_path = os.path.join(args.save_dir, f'history_{args.dataset}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        if not args.quiet:
            print(f"\nTraining completed successfully!")
            print(f"Checkpoints saved to {args.save_dir}")
            print(f"Training history saved to {history_path}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()