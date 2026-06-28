"""
Model Registration Mechanism

Provides unified model registration and management functionality, supporting unified management of development models and baseline models.
"""

from typing import Dict, Type, List, Any
from .base import GraphSummarizationModel
import importlib


class ModelRegistry:
    """
    Unified model registry, manages all Graph Summarization models.
    """
    
    def __init__(self):
        self._models: Dict[str, Type[GraphSummarizationModel]] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_models()
    
    def register_model(self,
                      name: str,
                      model_class: Type[GraphSummarizationModel],
                      category: str = "custom",
                      description: str = "",
                      paper_url: str = "",
                      **kwargs) -> None:
        """
        Register Graph Summarization model

        Args:
            name: Model name (unique identifier)
            model_class: Model class (must inherit from GraphSummarizationModel)
            category: Model category ("development", "baseline", "custom")
            description: Model description
            paper_url: Related paper URL
            **kwargs: Other metadata
        """
        if not issubclass(model_class, GraphSummarizationModel):
            raise ValueError(f"Model {name} must inherit from GraphSummarizationModel")
        
        if name in self._models:
            raise ValueError(f"Model {name} already registered")
        
        self._models[name] = model_class
        self._model_info[name] = {
            "class": model_class,
            "category": category,
            "description": description,
            "paper_url": paper_url,
            **kwargs
        }
    
    def get_model_class(self, name: str) -> Type[GraphSummarizationModel]:
        """Get model class"""
        if name not in self._models:
            raise ValueError(f"Model {name} not found. Available: {list(self._models.keys())}")
        return self._models[name]

    def create_model(self, name: str, **kwargs) -> GraphSummarizationModel:
        """Create model instance"""
        model_class = self.get_model_class(name)
        return model_class(**kwargs)

    def list_models(self, category: str = None) -> List[str]:
        """List all model names"""
        if category is None:
            return list(self._models.keys())
        return [name for name, info in self._model_info.items()
                if info.get("category") == category]

    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get model information"""
        if name not in self._model_info:
            raise ValueError(f"Model {name} not found")
        return self._model_info[name].copy()

    def list_development_models(self) -> List[str]:
        """List development models"""
        return self.list_models("development")

    def list_baseline_models(self) -> List[str]:
        """List baseline models"""
        return self.list_models("baseline")
    
    def _register_builtin_models(self):
        """Register built-in models"""
        # Old learnable models have been replaced by neural_enhanced_gradient series
        # If you need to use old models, import directly from main_model

        try:
            # Register gradient-based model - use undirected graph version as default
            from .gradient_based_undirected import (
                GradientBasedUndirectedGraphSummarization,
                JointSubsetBestGradientSummarization,
                JointSubsetEdgeScoreGradientSummarization,
                JointSubsetStabilityAwareEdgeScoreGradientSummarization,
                JointSubsetProductImportanceGradientSummarization,
                JointSubsetModelStableGradientSummarization,
            )

            self.register_model(
                "gradient_based",
                GradientBasedUndirectedGraphSummarization,
                category="development",
                description="INXplain - Gradient-based undirected graph simplification model (Development Model 2)"
            )

            self.register_model(
                "gradient_based_joint_subset_best",
                JointSubsetBestGradientSummarization,
                category="development",
                description="INXplain joint-subset variant - delete the sampled subset with minimum validation loss impact"
            )

            self.register_model(
                "gradient_based_joint_edge_score",
                JointSubsetEdgeScoreGradientSummarization,
                category="development",
                description="INXplain joint-subset variant - aggregate sampled subset losses to edge scores"
            )

            self.register_model(
                "gradient_based_joint_edge_score_stable",
                JointSubsetStabilityAwareEdgeScoreGradientSummarization,
                category="development",
                description="INXplain joint-subset variant - edge scores with stability penalty over sampled subsets"
            )

            self.register_model(
                "gradient_based_joint_product_importance",
                JointSubsetProductImportanceGradientSummarization,
                category="development",
                description="INXplain joint-subset variant - product aggregation of GCN/GAT/GraphSAGE importance scores"
            )

            self.register_model(
                "gradient_based_joint_model_stable",
                JointSubsetModelStableGradientSummarization,
                category="development",
                description="INXplain(stable) - rank-mean edge scoring over GCN/GAT/GraphSAGE joint-subset deletion scores"
            )

        except ImportError as e:
            print(f"Warning: Could not register gradient-based model: {e}")

        # Neural-Enhanced models are registered via register_main_models.py
        # Avoid duplicate registration


# Global model registry instance
model_registry = ModelRegistry()


def register_model(name: str, model_class: Type[GraphSummarizationModel], **kwargs):
    """Convenience function: Register model to global registry"""
    model_registry.register_model(name, model_class, **kwargs)


def get_model_class(name: str) -> Type[GraphSummarizationModel]:
    """Convenience function: Get model class"""
    return model_registry.get_model_class(name)


def create_model(name: str, **kwargs) -> GraphSummarizationModel:
    """Convenience function: Create model instance"""
    return model_registry.create_model(name, **kwargs)


def list_all_models() -> List[str]:
    """Convenience function: List all models"""
    return model_registry.list_models()
