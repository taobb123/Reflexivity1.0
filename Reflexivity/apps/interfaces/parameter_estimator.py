"""
参数估计器接口
用于计算反身性模型参数（α, β, γ）
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from .optimizer import IOptimizer


class IParameterEstimator(ABC):
    """参数估计器接口"""
    
    @abstractmethod
    def estimate(self,
                 price_data: np.ndarray,
                 fundamental_data: np.ndarray,
                 **kwargs) -> Dict[str, Any]:
        """
        估计反身性模型参数
        
        Args:
            price_data: 价格序列
            fundamental_data: 基本面序列
            **kwargs: 其他估计参数
            
        Returns:
            估计结果字典，包含：
            - parameters: {'alpha': float, 'gamma': float, 'beta': float}
            - lambda: λ值
            - stability: 稳定性类型
            - fitness: 拟合效果指标
            - predicted_data: 预测数据
        """
        pass
    
    @abstractmethod
    def set_optimizer(self, optimizer: IOptimizer) -> None:
        """
        设置优化器（组合模式）
        
        Args:
            optimizer: 优化器实例
        """
        pass
    
    @abstractmethod
    def get_estimation_method(self) -> str:
        """
        获取估计方法名称
        
        Returns:
            估计方法字符串
        """
        pass
