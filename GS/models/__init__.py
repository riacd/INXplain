"""
Models Module

Contains implementations of Neural-Enhanced graph summarization models and related components.

Module Structure:
- base: Abstract base class definitions
- neural_enhanced_gradient: Neural-Enhanced graph summarization model and its variants
- gradient_based: Basic gradient-based model
- downstream: Downstream task model implementations
- registry: Unified model registration mechanism

Ablation Experiment Variant Categories:
1. Fusion weight variants (high fusion, low fusion, etc.)
2. Learning strategy variants (residual learning, direct fusion, etc.)
3. Computational precision variants (fast computation, exact computation, etc.)
"""

# Base classes
from .base import GraphSummarizationModel, DownstreamModel

# Neural-Enhanced graph summarization model and its variants
from .neural_enhanced_gradient import (
    NeuralEnhancedGradientModel,
    NeuralEnhancedGradientModel_HighFusion,
    NeuralEnhancedGradientModel_LowFusion,
    NeuralEnhancedGradientModel_NoResidual,
    NeuralEnhancedGradientModel_SlowGradient,
    TrainableNeuralEnhancedGradientModel,
    EdgeImportanceRefiner
)

# Basic gradient-based model
from .gradient_based import GradientBasedGraphSummarization

# Downstream task models
from .downstream import (
    GCNDownstreamModel,
    GATDownstreamModel,
    GraphSAGEDownstreamModel,
    H2GCNDownstreamModel,
    GCNIIDownstreamModel,
    create_downstream_model,
    normalize_downstream_model_name,
    GCNModel,
    GATModel,
    GraphSAGEModel,
    H2GCNPyTorchModel,
    GCNIIModel,
)

# Gradient-based undirected variants
from .gradient_based_undirected import (
    GradientBasedUndirectedGraphSummarization,
    JointSubsetBestGradientSummarization,
    JointSubsetEdgeScoreGradientSummarization,
    JointSubsetStabilityAwareEdgeScoreGradientSummarization,
    JointSubsetProductImportanceGradientSummarization,
    JointSubsetModelStableGradientSummarization,
)

# Model registration mechanism
from .registry import (
    model_registry,
    register_model,
    get_model_class,
    create_model,
    list_all_models
)

# Automatically register all models
try:
    # Register main model variants
    from .register_main_models import register_all_main_models
    register_all_main_models()
except ImportError:
    pass  # Fail silently if registration fails

# Register baseline models
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from baselines.register_baselines import register_all_baselines
    register_all_baselines()
except ImportError:
    pass  # Fail silently if baseline module is unavailable

__all__ = [
    # Base classes
    'GraphSummarizationModel',
    'DownstreamModel',

    # Neural-Enhanced graph summarization model and its variants
    'NeuralEnhancedGradientModel',
    'NeuralEnhancedGradientModel_HighFusion',
    'NeuralEnhancedGradientModel_LowFusion',
    'NeuralEnhancedGradientModel_NoResidual',
    'NeuralEnhancedGradientModel_SlowGradient',
    'TrainableNeuralEnhancedGradientModel',
    'EdgeImportanceRefiner',

    # Basic gradient-based model
    'GradientBasedGraphSummarization',
    'GradientBasedUndirectedGraphSummarization',
    'JointSubsetBestGradientSummarization',
    'JointSubsetEdgeScoreGradientSummarization',
    'JointSubsetStabilityAwareEdgeScoreGradientSummarization',
    'JointSubsetProductImportanceGradientSummarization',
    'JointSubsetModelStableGradientSummarization',

    # Downstream task models
    'GCNDownstreamModel',
    'GATDownstreamModel',
    'GraphSAGEDownstreamModel',
    'H2GCNDownstreamModel',
    'GCNIIDownstreamModel',
    'create_downstream_model',
    'normalize_downstream_model_name',
    'GCNModel',
    'GATModel',
    'GraphSAGEModel',
    'H2GCNPyTorchModel',
    'GCNIIModel',

    # Model registration mechanism
    'model_registry',
    'register_model',
    'get_model_class',
    'create_model',
    'list_all_models'
]
