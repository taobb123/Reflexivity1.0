"""
配置设置模块
"""

import os
from typing import Optional, List


class Config:
    """配置类"""
    
    # 数据源配置
    DEFAULT_PREFERRED_SOURCES = ['akshare', 'tushare', 'pandas_datareader', 'yfinance']
    
    # API Keys (从环境变量读取)
    TUSHARE_TOKEN: Optional[str] = os.getenv('TUSHARE_TOKEN')
    ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv('ALPHA_VANTAGE_API_KEY')
    QUANDL_API_KEY: Optional[str] = os.getenv('QUANDL_API_KEY')
    
    # 默认参数
    DEFAULT_LOOKBACK_WEEKS = 120
    DEFAULT_STOCK = "平安银行"
    DEFAULT_OUTPUT_DIR = "results"
    
    # 参数估计配置
    DEFAULT_ESTIMATION_METHOD = 'differential_evolution'
    PARAMETER_BOUNDS = {
        'alpha': (0.0, 2.0),
        'gamma': (0.0, 5.0),
        'beta': (0.0, 2.0)
    }
    
    # 均线周期
    MA_PERIODS = [5, 10, 20, 60]
    
    @classmethod
    def get_preferred_sources(cls, custom: Optional[List[str]] = None) -> List[str]:
        """获取优先数据源列表"""
        return custom or cls.DEFAULT_PREFERRED_SOURCES
    
    @classmethod
    def get_tushare_token(cls, custom: Optional[str] = None) -> Optional[str]:
        """获取Tushare Token"""
        return custom or cls.TUSHARE_TOKEN

