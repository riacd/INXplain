# INexplainer

Official PyTorch implementation of **"Information-Guided Structure Discovery from Interaction Networks"** (KDD 2026).

## Overview

IGPrune is a framework for graph pruning that progressively removes edges while preserving task-relevant information.

## Installation

```bash
# Create environment
conda create -n IGPrune python=3.9
conda activate IGPrune

# Install dependencies
pip install torch torch-geometric numpy matplotlib pandas networkx scikit-learn
```

## Quick Start

```python
from GS.datasets import DatasetLoader
from GS.models import model_registry
from GS.benchmark import UnifiedBenchmark

# Load dataset
loader = DatasetLoader('./data')
data, train_mask, val_mask, test_mask = loader.load_dataset('Cora')

# Create model
model = model_registry.create_model(
    'learnable_graph_summarization',
    input_dim=data.x.size(1),
    device='cuda'
)

# Run benchmark
benchmark = UnifiedBenchmark(device='cuda')
results = benchmark.test_model(
    model_name='learnable_graph_summarization',
    dataset_name='Cora',
    task_type='original',
    downstream_model='gcn'
)
```

## Project Structure

```
IGPrune/
├── GS/                     # Main package
│   ├── models/            # Graph pruning models
│   ├── datasets/          # Dataset loaders
│   ├── benchmark/         # Benchmark framework
│   └── metrics/           # Evaluation metrics
├── baselines/             # Baseline implementations
├── scripts/               # Experiment scripts
├── data/                  # Datasets
└── results/               # Experiment results
```

## Models
- **IGPrune**: Baseline using validation loss gradients
- **NetworKit Methods**: Random, degree-based, centrality-based pruning
- **PRI-Graphs**: Probabilistic graph simplification

## Datasets

**Citation Networks**: Cora, CiteSeer, PubMed
**Social Networks**: Karate Club
**Biological Networks**: SO_relation (ME/MT)

## Running Experiments

# Run benchmark
python scripts/run_unified_benchmark.py --model learnable_graph_summarization --dataset Cora

# Run comprehensive experiments
python scripts/run_multi_dataset_repeated_experiments.py --datasets Cora CiteSeer PubMed
```

## Evaluation Metrics

- **Complexity Metric**: Normalized edge count
- **Information Metric**: Task-relevant information preservation
- **IC-AUC**: Area under Information-Complexity curve
- **Accuracy**: Downstream task performance
