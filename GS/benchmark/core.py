"""
Benchmark framework for graph summarization evaluation.

This module implements the complete benchmarking pipeline for evaluating
graph summarization models using complexity and information metrics.
"""

import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import numpy as np

from ..metrics import ComplexityMetric, InformationMetric, AccuracyMetric, ICAnalysis
from ..models import GraphSummarizationModel, DownstreamModel
from ..datasets import DatasetLoader


class Benchmark:
    """
    Main benchmark class for evaluating graph summarization models.
    
    Provides a complete pipeline for:
    1. Loading datasets
    2. Running graph summarization
    3. Computing complexity and information metrics
    4. Calculating IC-AUC scores
    5. Generating reports and visualizations
    """
    
    def __init__(self, 
                 data_dir: str = './data',
                 results_dir: str = './results',
                 device: Optional[torch.device] = None,
                 random_seed: int = 42):
        """
        Initialize benchmark.
        
        Args:
            data_dir: Directory to store datasets
            results_dir: Directory to store results
            device: Torch device for computation
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.device = device if device is not None else torch.device('cpu')
        self.random_seed = random_seed
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize dataset loader
        self.dataset_loader = DatasetLoader(data_dir)
        
        # Initialize metrics calculators
        self.complexity_metric = ComplexityMetric()
        self.information_metric = None  # Will be initialized with downstream model
        
        # Results storage
        self.results = {}
        
    def register_models(self,
                       graph_summarization_models: Dict[str, GraphSummarizationModel],
                       downstream_task_models: Dict[str, DownstreamModel]) -> None:
        """
        Register models for benchmarking.
        
        Args:
            graph_summarization_models: Dict mapping model names to models
            downstream_task_models: Dict mapping model names to models
        """
        self.gs_models = graph_summarization_models
        self.dt_models = downstream_task_models
        
        print(f"Registered {len(self.gs_models)} graph summarization models:")
        for name in self.gs_models.keys():
            print(f"  - {name}")
            
        print(f"Registered {len(self.dt_models)} downstream task models:")
        for name in self.dt_models.keys():
            print(f"  - {name}")
    
    def run_single_experiment(self,
                             dataset_name: str,
                             gs_model_name: str,
                             dt_model_name: str,
                             num_summarization_steps: int = 10,
                             training_epochs: int = 200,
                             verbose: bool = True,
                             experiment_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single benchmark experiment.
        
        Args:
            dataset_name: Name of dataset to use
            gs_model_name: Name of graph summarization model
            dt_model_name: Name of downstream task model
            num_summarization_steps: Number of graph simplification steps
            training_epochs: Epochs for downstream model training
            verbose: Whether to print progress
            experiment_seed: Optional seed for this experiment (defaults to self.random_seed)
            
        Returns:
            Dict with experiment results
        """
        if verbose:
            print(f"\nRunning experiment: {dataset_name} + {gs_model_name} + {dt_model_name}")
        
        # Set random seed for this experiment
        if experiment_seed is None:
            experiment_seed = self.random_seed
        torch.manual_seed(experiment_seed)
        np.random.seed(experiment_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(experiment_seed)
            torch.cuda.manual_seed_all(experiment_seed)
        
        # Load dataset
        original_graph, train_mask, val_mask, test_mask = self.dataset_loader.load_dataset(dataset_name)
        original_graph = original_graph.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        
        # Preprocess for summarization
        original_graph = self.dataset_loader.preprocess_for_summarization(original_graph)
        
        # Get models
        gs_model = self.gs_models[gs_model_name]
        dt_model = self.dt_models[dt_model_name]
        
        # Reset models for fresh experiment with consistent seed
        gs_model.reset()
        # Reset downstream model with fixed seed for consistent initialization
        torch.manual_seed(experiment_seed)
        np.random.seed(experiment_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(experiment_seed)
            torch.cuda.manual_seed_all(experiment_seed)
        dt_model.reset()
        
        # Train model if it's trainable (check for training capability)
        needs_training = (
            hasattr(gs_model, 'train_model') or  # Direct training method
            hasattr(gs_model, 'train') or        # Training through wrapper
            'Learnable' in gs_model_name          # Learnable models need training
        )
        
        if needs_training:
            if verbose:
                print(f"  Training {gs_model_name} model (50 epochs)...")
            
            # Set consistent seed for training
            torch.manual_seed(experiment_seed + 1000)  # Different seed for training model
            np.random.seed(experiment_seed + 1000)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(experiment_seed + 1000)
                torch.cuda.manual_seed_all(experiment_seed + 1000)
            
            # Handle different training interfaces
            if hasattr(gs_model, 'train_model'):
                # Direct training method (legacy MainGS)
                from ..models import GCNDownstreamModel
                training_dt_model = GCNDownstreamModel(
                    input_dim=original_graph.x.size(1),
                    device=self.device
                )
                gs_model.train_model(
                    original_graph=original_graph,
                    train_mask=train_mask,
                    val_mask=val_mask, 
                    labels=original_graph.y,
                    downstream_model=training_dt_model,
                    num_steps=num_summarization_steps,
                    epochs=50,  # Reduced for benchmark efficiency
                    verbose=False
                )
            elif hasattr(gs_model, 'train'):
                # Training through TrainableGraphSummarizationModel wrapper
                # Ensure training uses the same downstream task model type
                print(f"📋 Training will use the same downstream task model: {dt_model_name}")
                gs_model.train(
                    graph=original_graph,
                    train_labels=original_graph.y,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    num_epochs=50,
                    num_steps=num_summarization_steps
                )
            else:
                if verbose:
                    print(f"  Warning: {gs_model_name} is marked as trainable but no training method found")
        
        # Generate simplified graphs
        if verbose:
            print(f"  Generating {num_summarization_steps} simplified graphs...")
        summary_graph_list = gs_model.summarize(original_graph, num_summarization_steps)
        
        # Move all graphs to device
        summary_graph_list = [graph.to(self.device) for graph in summary_graph_list]
        
        # Compute complexity metrics
        if verbose:
            print("  Computing complexity metrics...")
        complexity_metrics = self.complexity_metric.compute_list(summary_graph_list, original_graph)
        
        # Compute information metrics
        if verbose:
            print(f"  Computing information metrics (training {training_epochs} epochs per graph)...")

        # Initialize information metric with downstream model
        info_metric = InformationMetric(dt_model, self.device, random_seed=experiment_seed)
        information_metrics = info_metric.compute_list(
            summary_graph_list,
            train_mask,
            val_mask,
            test_mask,
            original_graph.y,
            epochs=training_epochs
        )

        # Compute accuracy metrics
        if verbose:
            print(f"  Computing accuracy metrics (training {training_epochs} epochs per graph)...")

        # Initialize accuracy metric with another instance of downstream model
        # Create a fresh copy of the downstream model for accuracy computation
        accuracy_dt_model = type(dt_model)(dt_model.input_dim, device=self.device)
        accuracy_metric = AccuracyMetric(accuracy_dt_model, self.device, random_seed=experiment_seed)
        accuracy_metrics = accuracy_metric.compute_list(
            summary_graph_list,
            train_mask,
            val_mask,
            test_mask,
            original_graph.y,
            epochs=training_epochs
        )
        
        # Compute IC-AUC
        if verbose:
            print("  Computing IC-AUC...")
        snr_auc = ICAnalysis.compute_ic_auc(complexity_metrics, information_metrics)
        
        # Compile results
        experiment_results = {
            'dataset': dataset_name,
            'gs_model': gs_model_name,
            'dt_model': dt_model_name,
            'num_steps': num_summarization_steps,
            'training_epochs': training_epochs,
            'complexity_metrics': complexity_metrics,
            'information_metrics': information_metrics,
            'accuracy_metrics': accuracy_metrics,
            'snr_auc': snr_auc,
            'original_edges': int(original_graph.edge_index.size(1)),
            'original_nodes': int(original_graph.num_nodes),
            'timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            print(f"  Results: IC-AUC = {snr_auc:.4f}")
            print(f"  Original graph: {original_graph.num_nodes} nodes, {original_graph.edge_index.size(1)} edges")
            print(f"  Complexity range: [{min(complexity_metrics):.1f}, {max(complexity_metrics):.1f}]")
            print(f"  Information range: [{min(information_metrics):.4f}, {max(information_metrics):.4f}]")
            print(f"  Accuracy range: [{min(accuracy_metrics):.4f}, {max(accuracy_metrics):.4f}]")
        
        return experiment_results
    
    def run_full_benchmark(self,
                          datasets: Optional[List[str]] = None,
                          gs_models: Optional[List[str]] = None,
                          dt_models: Optional[List[str]] = None,
                          num_summarization_steps: int = 10,
                          training_epochs: int = 200,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        Run full benchmark across multiple datasets and models.
        
        Args:
            datasets: List of dataset names (default: all supported)
            gs_models: List of GS model names (default: all registered)
            dt_models: List of DT model names (default: all registered)
            num_summarization_steps: Number of simplification steps
            training_epochs: Training epochs per experiment
            save_results: Whether to save results to file
            
        Returns:
            Dict with all experiment results
        """
        # Use default values if not specified
        if datasets is None:
            datasets = ['Cora', 'CiteSeer', 'PubMed']
        if gs_models is None:
            gs_models = list(self.gs_models.keys())
        if dt_models is None:
            dt_models = list(self.dt_models.keys())
        
        print(f"Starting full benchmark:")
        print(f"  Datasets: {datasets}")
        print(f"  GS Models: {gs_models}")
        print(f"  DT Models: {dt_models}")
        print(f"  Total experiments: {len(datasets) * len(gs_models) * len(dt_models)}")
        
        all_results = {
            'benchmark_config': {
                'datasets': datasets,
                'gs_models': gs_models,
                'dt_models': dt_models,
                'num_summarization_steps': num_summarization_steps,
                'training_epochs': training_epochs,
                'device': str(self.device)
            },
            'experiments': []
        }
        
        experiment_count = 0
        total_experiments = len(datasets) * len(gs_models) * len(dt_models)
        
        # Run all combinations
        for dataset in datasets:
            for gs_model in gs_models:
                for dt_model in dt_models:
                    experiment_count += 1
                    print(f"\n[{experiment_count}/{total_experiments}] Running experiment...")
                    
                    try:
                        result = self.run_single_experiment(
                            dataset, gs_model, dt_model,
                            num_summarization_steps, training_epochs
                        )
                        all_results['experiments'].append(result)
                        
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        # Add failed experiment record
                        all_results['experiments'].append({
                            'dataset': dataset,
                            'gs_model': gs_model,
                            'dt_model': dt_model,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
        
        # Store results
        self.results = all_results
        
        # Save results if requested
        if save_results:
            self.save_results()
            
        # Generate summary
        self.print_summary()
        
        return all_results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save benchmark results to JSON file.
        
        Args:
            filename: Optional filename (default: auto-generated)
            
        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"Results saved to: {filepath}")
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load benchmark results from JSON file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Dict with loaded results
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)
            
        print(f"Results loaded from: {filepath}")
        return self.results
    
    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        if not self.results or 'experiments' not in self.results:
            print("No results to summarize.")
            return
            
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        experiments = self.results['experiments']
        successful_experiments = [exp for exp in experiments if 'snr_auc' in exp]
        failed_experiments = [exp for exp in experiments if 'error' in exp]
        
        print(f"Total experiments: {len(experiments)}")
        print(f"Successful: {len(successful_experiments)}")
        print(f"Failed: {len(failed_experiments)}")
        
        if successful_experiments:
            print("\nIC-AUC Results by Dataset:")
            print("-" * 40)
            
            # Group by dataset
            by_dataset = {}
            for exp in successful_experiments:
                dataset = exp['dataset']
                if dataset not in by_dataset:
                    by_dataset[dataset] = []
                by_dataset[dataset].append(exp)
            
            # Print results for each dataset
            for dataset, exps in by_dataset.items():
                print(f"\n{dataset}:")
                for exp in sorted(exps, key=lambda x: -x['snr_auc']):
                    print(f"  {exp['gs_model']:15} + {exp['dt_model']:8} = {exp['snr_auc']:.4f}")
            
            # Overall best results
            print(f"\nOverall Best Results:")
            print("-" * 40)
            best_exps = sorted(successful_experiments, key=lambda x: -x['snr_auc'])[:5]
            for i, exp in enumerate(best_exps, 1):
                print(f"{i}. {exp['dataset']:8} + {exp['gs_model']:15} + {exp['dt_model']:8} = {exp['snr_auc']:.4f}")
        
        if failed_experiments:
            print(f"\nFailed Experiments:")
            print("-" * 40)
            for exp in failed_experiments:
                print(f"  {exp['dataset']} + {exp['gs_model']} + {exp['dt_model']}: {exp['error']}")
    
    def generate_plots(self, save_dir: Optional[str] = None) -> None:
        """
        Generate SNR curve plots for all experiments.
        
        Args:
            save_dir: Directory to save plots (default: results_dir/plots)
        """
        if save_dir is None:
            save_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results or 'experiments' not in self.results:
            print("No results available for plotting.")
            return
            
        successful_experiments = [exp for exp in self.results['experiments'] if 'snr_auc' in exp]
        
        for exp in successful_experiments:
            title = f"IC Curve: {exp['dataset']} + {exp['gs_model']} + {exp['dt_model']}"
            filename = f"ic_{exp['dataset']}_{exp['gs_model']}_{exp['dt_model']}.png"
            filepath = os.path.join(save_dir, filename)
            
            ICAnalysis.plot_ic_curve(
                exp['complexity_metrics'],
                exp['information_metrics'], 
                title=title,
                save_path=filepath
            )
        
        print(f"Plots saved to: {save_dir}")
    
    def export_ic_auc_table(self, filename: Optional[str] = None) -> str:
        """
        Export IC-AUC results in TSV format.
        
        Args:
            filename: Optional filename for TSV export
            
        Returns:
            str: Path to exported TSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ic_auc_results_{timestamp}.tsv"
        
        filepath = os.path.join(self.results_dir, filename)
        
        if not self.results or 'experiments' not in self.results:
            print("No results available for export.")
            return filepath
        
        successful_experiments = [exp for exp in self.results['experiments'] if 'snr_auc' in exp]
        
        with open(filepath, 'w') as f:
            # Write header
            f.write("Dataset\tGS_Model\tDT_Model\tSNR_AUC\tOriginal_Nodes\tOriginal_Edges\tNum_Steps\tTimestamp\n")
            
            # Write data rows
            for exp in successful_experiments:
                f.write(f"{exp['dataset']}\t{exp['gs_model']}\t{exp['dt_model']}\t{exp['snr_auc']:.6f}\t"
                       f"{exp['original_nodes']}\t{exp['original_edges']}\t{exp['num_steps']}\t{exp['timestamp']}\n")
        
        print(f"IC-AUC results exported to: {filepath}")
        return filepath

    def export_snr_auc_table(self, filename: Optional[str] = None) -> str:
        """Compatibility wrapper. Use export_ic_auc_table instead."""
        return self.export_ic_auc_table(filename)

    def export_accuracy_table(self, filename: Optional[str] = None) -> str:
        """
        Export accuracy results in TSV format.

        Args:
            filename: Optional filename for TSV export

        Returns:
            str: Path to exported TSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_results_{timestamp}.tsv"

        filepath = os.path.join(self.results_dir, filename)

        if not self.results or 'experiments' not in self.results:
            print("No results available for export.")
            return filepath

        successful_experiments = [exp for exp in self.results['experiments'] if 'accuracy_metrics' in exp]

        with open(filepath, 'w') as f:
            # Write header
            f.write("Dataset\tGS_Model\tDT_Model\tStep\tAccuracy\tComplexity\tOriginal_Nodes\tOriginal_Edges\tTimestamp\n")

            # Write data rows
            for exp in successful_experiments:
                accuracy_metrics = exp.get('accuracy_metrics', [])
                complexity_metrics = exp.get('complexity_metrics', [])

                for step, (accuracy, complexity) in enumerate(zip(accuracy_metrics, complexity_metrics)):
                    f.write(f"{exp['dataset']}\t{exp['gs_model']}\t{exp['dt_model']}\t{step}\t"
                           f"{accuracy:.6f}\t{complexity:.6f}\t"
                           f"{exp['original_nodes']}\t{exp['original_edges']}\t{exp['timestamp']}\n")

        print(f"Accuracy results exported to: {filepath}")
        return filepath
