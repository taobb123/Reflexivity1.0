"""
图表拟合器接口
用于两个图表（价格图 vs 基本面图）的拟合
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from .fitter import IFitter


class IChartFitter(ABC):
    """图表拟合器接口"""
    
    @abstractmethod
    def fit_charts(self, 
                   price_data: np.ndarray,
                   fundamental_data: np.ndarray,
                   **kwargs) -> Dict[str, Any]:
        """
        拟合价格图表和基本面图表
        
        Args:
            price_data: 价格序列
            fundamental_data: 基本面序列
            **kwargs: 其他拟合参数
            
        Returns:
            拟合结果字典，包含：
            - price_fit: 价格拟合结果
            - fundamental_fit: 基本面拟合结果
            - correlation: 相关性
            - divergence: 背离程度
            - fit_metrics: 拟合指标
        """
        pass
    
    @abstractmethod
    def set_fitter(self, fitter: IFitter) -> None:
        """
        设置底层拟合器（组合模式）
        
        Args:
            fitter: 拟合器实例
        """
        pass
    
    @abstractmethod
    def get_fit_method(self) -> str:
        """
        获取拟合方法名称
        
        Returns:
            拟合方法字符串
        """
        pass
