"""
拟合器接口
用于图表拟合的细粒度接口
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class IFitter(ABC):
    """拟合器接口"""
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        执行拟合操作
        
        Args:
            x: 自变量序列
            y: 因变量序列
            
        Returns:
            拟合结果字典，包含：
            - parameters: 拟合参数
            - residuals: 残差
            - r_squared: 决定系数
            - fit_quality: 拟合质量评估
        """
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        使用拟合参数进行预测
        
        Args:
            x: 自变量序列
            parameters: 拟合参数
            
        Returns:
            预测值序列
        """
        pass
    
    @abstractmethod
    def get_fit_type(self) -> str:
        """
        获取拟合类型（如 'linear', 'polynomial', 'nonlinear'）
        
        Returns:
            拟合类型字符串
        """
        pass
