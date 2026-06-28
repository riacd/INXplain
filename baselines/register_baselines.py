"""
Baseline模型注册模块

将所有baseline模型注册到GS模型注册表中，实现统一管理。
"""

import sys
import os
from pathlib import Path

# Add GS package to path
sys.path.append(str(Path(__file__).parent.parent))

from GS.models.registry import model_registry


def register_all_baselines():
    """注册所有baseline模型到全局注册表"""
    
    # 注册NetworKit方法
    try:
        from .implementations.networkit_wrapper import NetworKitSparsifier
        
        # 为每种NetworKit方法创建独立的模型类
        for method in NetworKitSparsifier.SUPPORTED_METHODS:
            method_info = {
                'forest_fire': {
                    'name': 'Edge Forest Fire (EFF)',
                    'description': 'Forest Fire sparsification based on random walks'
                },
                'local_degree': {
                    'name': 'Local Degree (LD)', 
                    'description': 'Determines maximum parameter value for edge retention'
                },
                'local_similarity': {
                    'name': 'Local Similarity (LS)',
                    'description': 'Sparsification based on local node similarity'
                },
                'random_edge': {
                    'name': 'Random Edge (RE)',
                    'description': 'Randomly delete edges'
                },
                'random_node_edge': {
                    'name': 'Random Node Edge (RN)',
                    'description': 'Randomly delete edges based on nodes'
                },
                'scan': {
                    'name': 'SCAN Structural Similarity',
                    'description': 'Structural Clustering Algorithm for Networks'
                },
                'simmelian': {
                    'name': 'Simmelian Overlap (SO)',
                    'description': 'Calculates minimum parameter for edge retention'
                }
            }.get(method, {'name': method, 'description': f'NetworKit {method} method'})
            
            # 创建特定方法的模型类（避免闭包问题）
            def create_model_class(method_name):
                class SpecificNetworKitModel(NetworKitSparsifier):
                    def __init__(self, **kwargs):
                        super().__init__(method=method_name, **kwargs)
                return SpecificNetworKitModel
            
            SpecificNetworKitModel = create_model_class(method)
            
            model_name = f"networkit_{method}"
            model_registry.register_model(
                model_name,
                SpecificNetworKitModel,
                category="baseline",
                description=method_info['description'],
                paper_url="https://networkit.github.io/",
                source="NetworKit Library",
                method=method
            )
            
        print(f"✓ 已注册 {len(NetworKitSparsifier.SUPPORTED_METHODS)} 个NetworKit方法")
        
    except ImportError as e:
        print(f"Warning: Could not register NetworKit baselines: {e}")
    
    # 注册PRI-Graphs方法
    try:
        from .implementations.pri_graphs_model import PRIGraphsModel
        
        model_registry.register_model(
            "pri_graphs",
            PRIGraphsModel,
            category="baseline",
            description="概率图稀疏化方法，基于随机游走和边重要性评分",
            paper_url="https://github.com/SJYuCNEL/PRI-Graphs",
            source="PRI-Graphs"
        )
        
        print("✓ 已注册 PRI-Graphs 方法")
        
    except ImportError as e:
        print(f"Warning: Could not register PRI-Graphs: {e}")

    # 注册优化的Gradient-Based方法
    try:
        # 引入原始和优化版本的gradient-based模型
        from GS.models.gradient_based import GradientBasedGraphSummarization
        from GS.models.gradient_based_undirected import GradientBasedUndirectedGraphSummarization

        # 注册原始gradient-based方法（有向边删除版本）
        model_registry.register_model(
            "gradient_based_original",
            GradientBasedGraphSummarization,
            category="baseline",
            description="原始基于梯度的图简化方法（开发模型2）- 有向边删除版本",
            source="Development Model 2"
        )

        # 注册无向图版本的gradient-based方法（额外变体）
        model_registry.register_model(
            "gradient_based_undirected",
            GradientBasedUndirectedGraphSummarization,
            category="baseline",
            description="基于梯度的无向图简化方法 - 正确处理无向边对称删除",
            source="Development Model 2 - Undirected"
        )

        # 注意：gradient_based 已在 registry.py 中注册为默认的无向图版本

        print("✓ 已注册 2 个Gradient-Based方法变体")

    except ImportError as e:
        print(f"Warning: Could not register Gradient-Based methods: {e}")


def list_baseline_models():
    """列出所有已注册的baseline模型"""
    baselines = model_registry.list_baseline_models()
    print(f"\n已注册的Baseline模型 ({len(baselines)} 个):")
    for name in baselines:
        info = model_registry.get_model_info(name)
        print(f"  - {name}: {info['description']}")
    return baselines


if __name__ == '__main__':
    # 注册所有baseline模型
    register_all_baselines()
    
    # 显示注册结果
    list_baseline_models()
    
    # 显示所有模型
    all_models = model_registry.list_models()
    dev_models = model_registry.list_development_models()
    
    print(f"\n总计模型: {len(all_models)} 个")
    print(f"  - 开发模型: {len(dev_models)} 个")
    print(f"  - 基准模型: {len(all_models) - len(dev_models)} 个")