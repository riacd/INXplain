"""
IGPrune: Information-Guided Graph Pruning Library

A comprehensive Python package for graph pruning tasks, providing:
1. Dataset loading and preprocessing
2. Graph pruning model implementations
3. Downstream task models
4. Benchmark testing framework
5. Performance metrics

Module Structure:
- datasets: Dataset processing module
- models: Model implementation module
- benchmark: Benchmark testing module
- metrics: Performance metrics module
- utils: Utility functions module
"""

__version__ = "1.0.0"

# Import main components
from .datasets import DatasetLoader
from .models import (
    GraphSummarizationModel,
    DownstreamModel,
    NeuralEnhancedGradientModel,
    GradientBasedGraphSummarization,
    GCNDownstreamModel,
    GATDownstreamModel
)
from .benchmark import Benchmark, UnifiedBenchmark
from .metrics import ComplexityMetric, InformationMetric, ICAnalysis, SNRAnalysis

__all__ = [
    # 数据集
    'DatasetLoader',

    # 模型基类
    'GraphSummarizationModel',
    'DownstreamModel',

    # 图总结模型
    'NeuralEnhancedGradientModel',
    'GradientBasedGraphSummarization',

    # 下游任务模型
    'GCNDownstreamModel',
    'GATDownstreamModel',

    # 基准测试
    'Benchmark',
    'UnifiedBenchmark',

    # 度量指标
    'ComplexityMetric',
    'InformationMetric',
    'ICAnalysis',
    'SNRAnalysis'
]
