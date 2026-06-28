"""
Baselines模块

包含用于比较的各种基准模型实现，按照以下结构组织：

- core/: 定义统一的baseline接口和基类
- implementations/: 具体的baseline方法实现
  - PRI-Graphs: 概率图稀疏化方法
  - NetworKit: 基于NetworKit库的稀疏化方法 
  - SparRL: 强化学习稀疏化方法
- tests/: 测试脚本和工具

所有baseline模型都应该遵循core中定义的统一接口。
"""

from .implementations import PRIGraphsModel, NetworKitSparsifier
from .core import BaselineModel

__all__ = [
    'BaselineModel',
    'PRIGraphsModel', 
    'NetworKitSparsifier'
]