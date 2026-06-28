"""
Baseline实现模块

包含各种第三方baseline方法的实现：
- PRI-Graphs: 基于概率的图稀疏化方法
- NetworKit: 基于NetworKit库的多种稀疏化方法
- SparRL: 基于强化学习的稀疏化方法
"""

from .pri_graphs_model import PRIGraphsModel
from .networkit_wrapper import NetworKitSparsifier

__all__ = [
    'PRIGraphsModel',
    'NetworKitSparsifier'
]