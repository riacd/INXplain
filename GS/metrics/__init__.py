"""
Metrics Module

Provides various metrics for evaluating graph pruning model performance.

Contains:
- ComplexityMetric: Compute graph complexity (edge count)
- InformationMetric: Compute information preservation metrics (supports dual normalization)
- AccuracyMetric: Compute downstream task accuracy
- ICAnalysis: Compute IC-AUC and information threshold analysis metrics
- SNRAnalysis: compatibility alias for older scripts
"""

from .core import ComplexityMetric, InformationMetric, AccuracyMetric, ICAnalysis, SNRAnalysis

__all__ = ['ComplexityMetric', 'InformationMetric', 'AccuracyMetric', 'ICAnalysis', 'SNRAnalysis']
