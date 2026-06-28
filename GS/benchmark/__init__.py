"""
Benchmark Testing Module

Provides standardized benchmark framework for evaluating different graph pruning models.

Features:
1. Standardized testing workflow
2. Multi-dataset support
3. Multiple downstream task support
4. Result visualization and saving
"""

from .core import Benchmark
from .unified import UnifiedBenchmark

__all__ = ['Benchmark', 'UnifiedBenchmark']