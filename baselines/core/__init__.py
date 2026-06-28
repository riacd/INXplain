"""
Baseline模型核心接口

定义所有baseline模型的统一接口和基类。
"""

from abc import ABC, abstractmethod
from typing import List
from torch_geometric.data import Data


class BaselineModel(ABC):
    """
    Baseline模型基类
    
    所有baseline模型都应该继承此类并实现summarize方法。
    """
    
    @abstractmethod
    def summarize(self, original_graph: Data, num_steps: int = 10) -> List[Data]:
        """
        生成图总结序列
        
        Args:
            original_graph: 输入原始图
            num_steps: 总结步数
            
        Returns:
            包含不同简化程度图的列表
        """
        pass
    
    @abstractmethod 
    def get_method_name(self) -> str:
        """返回方法名称"""
        pass
    
    @abstractmethod
    def get_method_description(self) -> str:
        """返回方法描述"""
        pass