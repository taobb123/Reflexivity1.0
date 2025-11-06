"""
数据提供者实现
支持多种数据源（tushare、akshare等）
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from apps.interfaces.data_provider import IDataProvider
from tools.data_fetchers.data_fetcher_unified import UnifiedDataFetcher


class UnifiedDataProvider(IDataProvider):
    """统一数据提供者，封装现有的 UnifiedDataFetcher"""
    
    def __init__(self, tushare_token: Optional[str] = None, 
                 preferred_sources: Optional[list] = None):
        """
        初始化数据提供者
        
        Args:
            tushare_token: Tushare token
            preferred_sources: 优先使用的数据源列表
        """
        self.fetcher = UnifiedDataFetcher(
            tushare_token=tushare_token,
            preferred_sources=preferred_sources
        )
        self._data_info = {}
    
    def get_price_data(self, stock_code: str, **kwargs) -> np.ndarray:
        """获取价格序列数据"""
        df = self.get_dataframe(stock_code, **kwargs)
        return df['P_t'].values
    
    def get_fundamental_data(self, stock_code: str, **kwargs) -> np.ndarray:
        """获取基本面序列数据"""
        df = self.get_dataframe(stock_code, **kwargs)
        return df['F_t'].values
    
    def get_dataframe(self, stock_code: str, **kwargs) -> pd.DataFrame:
        """获取完整的数据框"""
        lookback_weeks = kwargs.get('lookback_weeks', 120)
        
        # UnifiedDataFetcher.fetch_complete_data 返回 (DataFrame, stock_code, sources_info)
        df, actual_stock_code, sources_info = self.fetcher.fetch_complete_data(
            stock_name=stock_code,
            lookback_weeks=lookback_weeks
        )
        
        # 确保 DataFrame 包含 'P_t' 和 'F_t' 列
        if 'P_t' not in df.columns or 'F_t' not in df.columns:
            raise ValueError("获取的数据缺少 'P_t' 或 'F_t' 列")
        
        # 更新数据信息
        self._data_info = {
            'stock_code': actual_stock_code,
            'lookback_weeks': lookback_weeks,
            'data_points': len(df),
            'sources': sources_info
        }
        
        return df
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        return self._data_info.copy()


class DataFrameDataProvider(IDataProvider):
    """从DataFrame直接提供数据的数据提供者"""
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化
        
        Args:
            df: 包含 'P_t' 和 'F_t' 列的 DataFrame
        """
        if 'P_t' not in df.columns or 'F_t' not in df.columns:
            raise ValueError("DataFrame 必须包含 'P_t' 和 'F_t' 列")
        self.df = df
        self._data_info = {
            'data_points': len(df),
            'source': 'DataFrame'
        }
    
    def get_price_data(self, stock_code: str = None, **kwargs) -> np.ndarray:
        """获取价格序列数据"""
        return self.df['P_t'].values
    
    def get_fundamental_data(self, stock_code: str = None, **kwargs) -> np.ndarray:
        """获取基本面序列数据"""
        return self.df['F_t'].values
    
    def get_dataframe(self, stock_code: str = None, **kwargs) -> pd.DataFrame:
        """获取完整的数据框"""
        return self.df.copy()
    
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        return self._data_info.copy()
