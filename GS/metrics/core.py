"""
Metrics module for graph summarization evaluation.

Contains implementations of Complexity Metric and Information Metric
for evaluating graph summarization quality.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Optional, Tuple
import numpy as np
import math
import gc
import random
from abc import ABC, abstractmethod


class ComplexityMetric:
    """
    Complexity Metric implementation using normalized L0 norm of adjacency matrix.

    As defined in the paper, the complexity metric C(A_k) measures
    the compression degree of a graph using the normalized L0 norm:
    C(A_k) = ||A_k||_0 / ||A_0||_0
    """
    
    @staticmethod
    def compute(graph: Data, original_graph: Data) -> float:
        """
        Compute the normalized complexity metric for a single graph.

        Args:
            graph: PyTorch Geometric Data object containing the simplified graph
            original_graph: PyTorch Geometric Data object containing the original graph

        Returns:
            float: Normalized L0 norm (number of edges relative to original)
        """
        if original_graph.edge_index.size(1) == 0:
            return 0.0

        # For sparse graphs, the L0 norm is simply the number of edges
        # Since edge_index stores edges in COO format, each column is an edge
        graph_edges = float(graph.edge_index.size(1))
        original_edges = float(original_graph.edge_index.size(1))

        return graph_edges / original_edges
    
    @staticmethod
    def compute_list(summary_graph_list: List[Data], original_graph: Data) -> List[float]:
        """
        Compute normalized complexity metrics for a list of graphs.

        Args:
            summary_graph_list: List of simplified graphs
            original_graph: Original graph for normalization

        Returns:
            List[float]: Normalized complexity metrics for each graph
        """
        return [ComplexityMetric.compute(graph, original_graph) for graph in summary_graph_list]


class InformationMetric:
    """
    Information Metric implementation using downstream task loss.

    Measures how much task-relevant information is preserved
    in the simplified graph by evaluating downstream task performance.

    Supports two normalization schemes:
    (A) Additive Normalization: I^add(G_k) = (I(G_k;Y) - I(G_N;Y)) / (I(G_0;Y) - I(G_N;Y))
    (B) Log-ratio Normalization: I^log(G_k) = log(L(G_k)/L(G_N)) / log(L(G_0)/L(G_N))
    """

    def __init__(self, downstream_model, device: Optional[torch.device] = None, random_seed: int = 42):
        """
        Initialize Information Metric calculator.

        Args:
            downstream_model: Model implementing DownstreamTaskModel interface
            device: torch device for computation
            random_seed: Random seed for reproducibility
        """
        self.downstream_model = downstream_model
        self.device = device if device is not None else torch.device('cpu')
        self.random_seed = random_seed

    def _set_random_seed(self) -> None:
        """Use paired initialization and training randomness for every graph."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)

    @staticmethod
    def normalize_losses(test_losses: List[float], normalization: str) -> List[float]:
        """Normalize precomputed losses without retraining downstream models."""
        if normalization not in {'additive', 'log_ratio'}:
            raise ValueError(f"Unknown information normalization: {normalization}")
        if not test_losses:
            return []

        l_g0 = test_losses[0]
        l_gn = test_losses[-1]
        if normalization == 'additive':
            denominator = l_gn - l_g0
            if abs(denominator) <= 1e-10:
                raise ValueError(
                    "Cannot normalize information: original and empty graphs have "
                    "indistinguishable test losses"
                )
            return [(l_gn - loss) / denominator for loss in test_losses]

        if l_g0 <= 0 or l_gn <= 0 or any(loss <= 0 for loss in test_losses):
            raise ValueError("Log-ratio information requires strictly positive losses")
        denominator = math.log(l_g0 / l_gn)
        if abs(denominator) <= 1e-10:
            raise ValueError(
                "Cannot normalize information: original and empty graphs have "
                "indistinguishable test losses"
            )
        return [math.log(loss / l_gn) / denominator for loss in test_losses]
    
    def compute(self,
                graph: Data,
                train_mask: torch.Tensor,
                val_mask: torch.Tensor,
                test_mask: torch.Tensor,
                labels: torch.Tensor,
                epochs: int = 200) -> float:
        """
        Compute information metric for a single graph.

        Memory-optimized version with careful tensor management.

        Args:
            graph: Input graph for evaluation
            train_mask: Training node mask
            val_mask: Validation node mask
            test_mask: Test node mask
            labels: Node labels
            epochs: Number of training epochs

        Returns:
            float: Test loss after training (information metric)
        """
        # Move data to device and ensure float32 with memory optimization
        with torch.no_grad():
            # Clone to avoid modifying original data
            graph_copy = Data(
                x=graph.x.clone().float().to(self.device),
                edge_index=graph.edge_index.clone().to(self.device),
                y=graph.y.clone().to(self.device) if hasattr(graph, 'y') and graph.y is not None else None
            )

            train_mask = train_mask.clone().to(self.device)
            val_mask = val_mask.clone().to(self.device)
            test_mask = test_mask.clone().to(self.device)
            labels = labels.clone().to(self.device)

        # Train the downstream model
        self.downstream_model.train_model(
            graph_copy, train_mask, val_mask, labels, epochs=epochs
        )

        # Evaluate on test set
        test_loss = self.downstream_model.evaluate(
            graph_copy, test_mask, labels
        )

        # Clean up intermediate tensors
        del graph_copy, train_mask, val_mask, test_mask, labels

        return float(test_loss)

    def compute_metrics(self,
                        graph: Data,
                        train_mask: torch.Tensor,
                        val_mask: torch.Tensor,
                        test_mask: torch.Tensor,
                        labels: torch.Tensor,
                        epochs: int = 200) -> Tuple[float, float]:
        """Train once and return both test loss and test accuracy."""
        with torch.no_grad():
            graph_copy = Data(
                x=graph.x.clone().float().to(self.device),
                edge_index=graph.edge_index.clone().to(self.device),
                y=graph.y.clone().to(self.device)
                if hasattr(graph, 'y') and graph.y is not None else None,
            )
            train_mask_copy = train_mask.clone().to(self.device)
            val_mask_copy = val_mask.clone().to(self.device)
            test_mask_copy = test_mask.clone().to(self.device)
            labels_copy = labels.clone().to(self.device)

        self.downstream_model.train_model(
            graph_copy,
            train_mask_copy,
            val_mask_copy,
            labels_copy,
            epochs=epochs,
        )
        test_loss = self.downstream_model.evaluate(
            graph_copy, test_mask_copy, labels_copy
        )
        with torch.no_grad():
            predictions = self.downstream_model.predict(graph_copy)
            accuracy = (
                predictions[test_mask_copy].argmax(dim=1)
                == labels_copy[test_mask_copy]
            ).float().mean().item()

        del graph_copy, train_mask_copy, val_mask_copy, test_mask_copy, labels_copy
        return float(test_loss), float(accuracy)

    def evaluate_list(self,
                      summary_graph_list: List[Data],
                      train_mask: torch.Tensor,
                      val_mask: torch.Tensor,
                      test_mask: torch.Tensor,
                      labels: torch.Tensor,
                      epochs: int = 200) -> Tuple[List[float], List[float]]:
        """Evaluate all graph steps with paired randomness and one fit per step."""
        test_losses = []
        accuracies = []
        for i, graph in enumerate(summary_graph_list):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Every step receives the same initialization and dropout RNG stream.
            self._set_random_seed()
            self.downstream_model.reset()
            try:
                test_loss, accuracy = self.compute_metrics(
                    graph, train_mask, val_mask, test_mask, labels, epochs
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"CUDA OOM while evaluating graph step {i}; refusing to "
                        "change epochs for only one step"
                    ) from exc
                raise

            test_losses.append(test_loss)
            accuracies.append(accuracy)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return test_losses, accuracies
    
    def compute_list(self,
                     summary_graph_list: List[Data],
                     train_mask: torch.Tensor,
                     val_mask: torch.Tensor,
                     test_mask: torch.Tensor,
                     labels: torch.Tensor,
                     epochs: int = 200,
                     normalization: str = 'additive') -> List[float]:
        """
        Compute information metrics for a list of graphs using specified normalization.

        Memory-optimized version that processes graphs sequentially and cleans up
        CUDA memory after each computation.

        Args:
            summary_graph_list: List of simplified graphs (G_0, G_1, ..., G_N)
            train_mask: Training node mask
            val_mask: Validation node mask
            test_mask: Test node mask
            labels: Node labels
            epochs: Training epochs per graph
            normalization: 'log_ratio' (default) or 'additive'

        Returns:
            List[float]: Information metrics for each graph using specified normalization
        """
        test_losses, _ = self.evaluate_list(
            summary_graph_list,
            train_mask,
            val_mask,
            test_mask,
            labels,
            epochs,
        )
        return self.normalize_losses(test_losses, normalization)


class AccuracyMetric:
    """
    Accuracy Metric implementation for downstream task evaluation.

    Computes the accuracy of the trained downstream model on the test set
    for each simplified graph in the summarization sequence.
    """

    def __init__(self, downstream_model, device: Optional[torch.device] = None, random_seed: int = 42):
        """
        Initialize Accuracy Metric calculator.

        Args:
            downstream_model: Model implementing DownstreamTaskModel interface
            device: torch device for computation
            random_seed: Random seed for reproducibility
        """
        self.downstream_model = downstream_model
        self.device = device if device is not None else torch.device('cpu')
        self.random_seed = random_seed

    def compute(self,
                graph: Data,
                train_mask: torch.Tensor,
                val_mask: torch.Tensor,
                test_mask: torch.Tensor,
                labels: torch.Tensor,
                epochs: int = 200) -> float:
        """
        Compute accuracy metric for a single graph.

        Args:
            graph: Input graph for evaluation
            train_mask: Training node mask
            val_mask: Validation node mask
            test_mask: Test node mask
            labels: Node labels
            epochs: Number of training epochs

        Returns:
            float: Test accuracy after training
        """
        # Move data to device and ensure proper types
        with torch.no_grad():
            # Clone to avoid modifying original data
            graph_copy = Data(
                x=graph.x.clone().float().to(self.device),
                edge_index=graph.edge_index.clone().to(self.device),
                y=graph.y.clone().to(self.device) if hasattr(graph, 'y') and graph.y is not None else None
            )

            train_mask = train_mask.clone().to(self.device)
            val_mask = val_mask.clone().to(self.device)
            test_mask = test_mask.clone().to(self.device)
            labels = labels.clone().to(self.device)

        # Train the downstream model
        self.downstream_model.train_model(
            graph_copy, train_mask, val_mask, labels, epochs=epochs
        )

        # Get predictions on test set
        with torch.no_grad():
            predictions = self.downstream_model.predict(graph_copy)
            test_predictions = predictions[test_mask]
            test_labels = labels[test_mask]

            # Calculate accuracy
            correct = (test_predictions.argmax(dim=1) == test_labels).float()
            accuracy = correct.mean().item()

        # Clean up intermediate tensors
        del graph_copy, train_mask, val_mask, test_mask, labels

        return float(accuracy)

    def compute_list(self,
                     summary_graph_list: List[Data],
                     train_mask: torch.Tensor,
                     val_mask: torch.Tensor,
                     test_mask: torch.Tensor,
                     labels: torch.Tensor,
                     epochs: int = 200) -> List[float]:
        """
        Compute accuracy metrics for a list of graphs.

        Args:
            summary_graph_list: List of simplified graphs (G_0, G_1, ..., G_N)
            train_mask: Training node mask
            val_mask: Validation node mask
            test_mask: Test node mask
            labels: Node labels
            epochs: Training epochs per graph

        Returns:
            List[float]: Accuracy values for each graph
        """
        evaluator = InformationMetric(
            self.downstream_model, self.device, random_seed=self.random_seed
        )
        _, accuracies = evaluator.evaluate_list(
            summary_graph_list,
            train_mask,
            val_mask,
            test_mask,
            labels,
            epochs,
        )
        return accuracies


class ICAnalysis:
    """
    Information-Complexity (IC) analysis for graph summarization.

    Implements IC curve plotting, AUC-IC calculation, and information threshold point
    calculation as described in the updated requirements.
    """

    @staticmethod
    def compute_ic_auc(complexity_metrics: List[float],
                       information_metrics: List[float]) -> float:
        """
        Compute AUC-IC (Area under the Information-Complexity curve) metric.

        Args:
            complexity_metrics: List of complexity values (X-axis)
            information_metrics: List of information values (Y-axis)

        Returns:
            float: Area under the IC curve
        """
        if len(complexity_metrics) != len(information_metrics):
            raise ValueError("Complexity and information metrics must have same length")

        if len(complexity_metrics) < 2:
            return 0.0

        # Sort by complexity (X-axis) for proper curve calculation
        sorted_pairs = sorted(zip(complexity_metrics, information_metrics))
        x_vals = [pair[0] for pair in sorted_pairs]
        y_vals = [pair[1] for pair in sorted_pairs]

        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(x_vals)):
            dx = x_vals[i] - x_vals[i-1]
            avg_y = (y_vals[i] + y_vals[i-1]) / 2
            auc += dx * avg_y

        return auc

    @staticmethod
    def compute_information_threshold_point(complexity_metrics: List[float],
                                          information_metrics: List[float],
                                          threshold: float = 0.8) -> Optional[float]:
        """
        Compute the information threshold point: minimum complexity achieving
        information retention threshold τ.

        Args:
            complexity_metrics: List of complexity values (X-axis)
            information_metrics: List of information values (Y-axis)
            threshold: Information retention threshold τ (default: 0.8)

        Returns:
            Optional[float]: Minimum complexity achieving threshold, or None if not found
        """
        if len(complexity_metrics) != len(information_metrics):
            raise ValueError("Complexity and information metrics must have same length")

        if len(complexity_metrics) == 0:
            return None

        # Sort by complexity (X-axis) for proper curve calculation
        sorted_pairs = sorted(zip(complexity_metrics, information_metrics))

        # Find the minimum complexity where information >= threshold
        for complexity, information in sorted_pairs:
            if information >= threshold:
                return complexity

        return None  # Threshold not achieved

    # Keep the old method name for backward compatibility
    @staticmethod
    def compute_snr_auc(complexity_metrics: List[float],
                        information_metrics: List[float]) -> float:
        """
        Legacy method name. Use compute_ic_auc instead.
        """
        return ICAnalysis.compute_ic_auc(complexity_metrics, information_metrics)

    @staticmethod
    def plot_ic_curve(complexity_metrics: List[float],
                      information_metrics: List[float],
                      title: str = "IC Curve",
                      save_path: Optional[str] = None,
                      normalization: str = "log_ratio") -> None:
        """
        Plot Information-Complexity (IC) curve for visualization.

        Args:
            complexity_metrics: X-axis values
            information_metrics: Y-axis values
            title: Plot title
            save_path: Optional path to save the plot
            normalization: Type of normalization used ('log_ratio' or 'additive')
        """
        try:
            import matplotlib.pyplot as plt

            # Sort by complexity for proper curve
            sorted_pairs = sorted(zip(complexity_metrics, information_metrics))
            x_vals = [pair[0] for pair in sorted_pairs]
            y_vals = [pair[1] for pair in sorted_pairs]

            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, 'b-', linewidth=2, marker='o')
            plt.fill_between(x_vals, y_vals, alpha=0.3)
            plt.xlabel('Complexity Metric (Normalized Edge Count)')

            if normalization == 'additive':
                plt.ylabel('Information Metric (Additive Normalization)')
            else:
                plt.ylabel('Information Metric (Log-ratio Normalization)')

            plt.title(title)
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except ImportError:
            print("matplotlib not available for plotting")

    # Keep the old method name for backward compatibility
    @staticmethod
    def plot_snr_curve(complexity_metrics: List[float],
                       information_metrics: List[float],
                       title: str = "IC Curve",
                       save_path: Optional[str] = None) -> None:
        """
        Legacy method name. Use plot_ic_curve instead.
        """
        ICAnalysis.plot_ic_curve(complexity_metrics, information_metrics, title, save_path)


# Backward compatibility alias
SNRAnalysis = ICAnalysis
