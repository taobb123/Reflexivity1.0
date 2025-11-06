"""
优化器接口
用于参数优化的细粒度接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Tuple, Optional
import numpy as np


class IOptimizer(ABC):
    """优化器接口"""
    
    @abstractmethod
    def optimize(self, 
                 objective_function: Callable,
                 initial_params: np.ndarray,
                 bounds: Optional[Tuple] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        执行优化操作
        
        Args:
            objective_function: 目标函数
            initial_params: 初始参数
            bounds: 参数边界
            **kwargs: 其他优化参数
            
        Returns:
            优化结果字典，包含：
            - optimal_params: 最优参数
            - optimal_value: 最优值
            - success: 是否成功
            - iterations: 迭代次数
            - message: 优化消息
        """
        pass
    
    @abstractmethod
    def get_optimizer_type(self) -> str:
        """
        获取优化器类型（如 'differential_evolution', 'L-BFGS-B', 'gradient_descent'）
        
        Returns:
            优化器类型字符串
        """
        pass
