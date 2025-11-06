"""
图表可视化器接口
用于生成拟合对比图表
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class IChartVisualizer(ABC):
    """图表可视化器接口"""
    
    @abstractmethod
    def visualize_fit(self,
                      price_data: np.ndarray,
                      fundamental_data: np.ndarray,
                      fit_results: Dict[str, Any],
                      **kwargs) -> Any:
        """
        可视化拟合结果
        
        Args:
            price_data: 价格序列
            fundamental_data: 基本面序列
            fit_results: 拟合结果
            **kwargs: 其他可视化参数
            
        Returns:
            图表对象（matplotlib figure 或 base64 字符串）
        """
        pass
    
    @abstractmethod
    def visualize_comparison(self,
                            actual_data: Dict[str, np.ndarray],
                            predicted_data: Dict[str, np.ndarray],
                            **kwargs) -> Any:
        """
        可视化对比图（实际 vs 预测）
        
        Args:
            actual_data: 实际数据字典 {'price': ..., 'fundamental': ...}
            predicted_data: 预测数据字典 {'price': ..., 'fundamental': ...}
            **kwargs: 其他参数
            
        Returns:
            图表对象
        """
        pass
    
    @abstractmethod
    def save_chart(self, chart: Any, save_path: str, **kwargs) -> str:
        """
        保存图表
        
        Args:
            chart: 图表对象
            save_path: 保存路径
            **kwargs: 其他参数
            
        Returns:
            保存路径
        """
        pass
    
    @abstractmethod
    def chart_to_base64(self, chart: Any, **kwargs) -> str:
        """
        将图表转换为 base64 字符串
        
        Args:
            chart: 图表对象
            **kwargs: 其他参数
            
        Returns:
            base64 编码的字符串
        """
        pass
