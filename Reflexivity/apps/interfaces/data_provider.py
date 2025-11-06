"""
数据提供者接口
统一数据输入格式，支持多种数据源
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class IDataProvider(ABC):
    """数据提供者接口，统一数据输入格式"""
    
    @abstractmethod
    def get_price_data(self, stock_code: str, **kwargs) -> np.ndarray:
        """
        获取价格序列数据
        
        Args:
            stock_code: 股票代码
            **kwargs: 其他参数（如时间范围、数据源等）
            
        Returns:
            价格序列 numpy 数组
        """
        pass
    
    @abstractmethod
    def get_fundamental_data(self, stock_code: str, **kwargs) -> np.ndarray:
        """
        获取基本面序列数据
        
        Args:
            stock_code: 股票代码
            **kwargs: 其他参数
            
        Returns:
            基本面序列 numpy 数组
        """
        pass
    
    @abstractmethod
    def get_dataframe(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """
        获取完整的数据框（包含价格和基本面）
        
        Args:
            stock_code: 股票代码
            **kwargs: 其他参数
            
        Returns:
            包含 'P_t' 和 'F_t' 列的 DataFrame
        """
        pass
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """
        获取数据信息（如数据源、时间范围等）
        
        Returns:
            数据信息字典
        """
        pass
