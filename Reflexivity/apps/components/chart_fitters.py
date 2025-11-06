"""
图表拟合器实现
用于两个图表（价格图 vs 基本面图）的拟合
"""
import numpy as np
from typing import Dict, Any, Optional
from scipy import stats

from apps.interfaces.chart_fitter import IChartFitter
from apps.interfaces.fitter import IFitter
from apps.components.fitters import LinearFitter


class PriceFundamentalChartFitter(IChartFitter):
    """价格与基本面图表拟合器"""
    
    def __init__(self, fitter: Optional[IFitter] = None):
        """
        初始化
        
        Args:
            fitter: 底层拟合器，如果为None则使用默认的线性拟合器
        """
        self.fitter = fitter or LinearFitter()
    
    def fit_charts(self, 
                   price_data: np.ndarray,
                   fundamental_data: np.ndarray,
                   **kwargs) -> Dict[str, Any]:
        """
        拟合价格图表和基本面图表
        
        这里使用价格序列作为自变量，基本面序列作为因变量进行拟合
        """
        # 确保数据长度一致
        min_len = min(len(price_data), len(fundamental_data))
        price_data = price_data[:min_len]
        fundamental_data = fundamental_data[:min_len]
        
        # 使用底层拟合器进行拟合
        fit_result = self.fitter.fit(price_data, fundamental_data)
        
        # 计算相关性
        correlation, p_value = stats.pearsonr(price_data, fundamental_data)
        
        # 计算背离程度
        divergence = price_data - fundamental_data
        mean_divergence = np.mean(divergence)
        std_divergence = np.std(divergence)
        
        # 计算预测的基本面
        predicted_fundamental = self.fitter.predict(price_data, fit_result['parameters'])
        
        return {
            'price_fit': {
                'data': price_data.tolist(),
                'type': 'actual'
            },
            'fundamental_fit': {
                'data': fundamental_data.tolist(),
                'predicted': predicted_fundamental.tolist(),
                'fit_result': fit_result,
                'type': 'actual'
            },
            'correlation': {
                'value': float(correlation),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'divergence': {
                'mean': float(mean_divergence),
                'std': float(std_divergence),
                'max': float(np.max(divergence)),
                'min': float(np.min(divergence)),
                'data': divergence.tolist()
            },
            'fit_metrics': {
                'r_squared': fit_result['r_squared'],
                'fit_quality': fit_result['fit_quality'],
                'fit_type': self.fitter.get_fit_type()
            }
        }
    
    def set_fitter(self, fitter: IFitter) -> None:
        """设置底层拟合器"""
        self.fitter = fitter
    
    def get_fit_method(self) -> str:
        """获取拟合方法名称"""
        return f"Price-Fundamental Chart Fitting (using {self.fitter.get_fit_type()})"


class BidirectionalChartFitter(IChartFitter):
    """双向图表拟合器（同时拟合价格->基本面和基本面->价格）"""
    
    def __init__(self, fitter: Optional[IFitter] = None):
        """
        初始化
        
        Args:
            fitter: 底层拟合器
        """
        self.fitter = fitter or LinearFitter()
    
    def fit_charts(self, 
                   price_data: np.ndarray,
                   fundamental_data: np.ndarray,
                   **kwargs) -> Dict[str, Any]:
        """双向拟合"""
        # 确保数据长度一致
        min_len = min(len(price_data), len(fundamental_data))
        price_data = price_data[:min_len]
        fundamental_data = fundamental_data[:min_len]
        
        # 拟合1：价格 -> 基本面
        fit_price_to_fundamental = self.fitter.fit(price_data, fundamental_data)
        pred_fundamental = self.fitter.predict(price_data, fit_price_to_fundamental['parameters'])
        
        # 拟合2：基本面 -> 价格
        fit_fundamental_to_price = self.fitter.fit(fundamental_data, price_data)
        pred_price = self.fitter.predict(fundamental_data, fit_fundamental_to_price['parameters'])
        
        # 相关性
        correlation, p_value = stats.pearsonr(price_data, fundamental_data)
        
        # 背离
        divergence = price_data - fundamental_data
        mean_divergence = np.mean(divergence)
        std_divergence = np.std(divergence)
        
        return {
            'price_fit': {
                'data': price_data.tolist(),
                'predicted_from_fundamental': pred_price.tolist(),
                'fit_result': fit_fundamental_to_price,
                'type': 'bidirectional'
            },
            'fundamental_fit': {
                'data': fundamental_data.tolist(),
                'predicted_from_price': pred_fundamental.tolist(),
                'fit_result': fit_price_to_fundamental,
                'type': 'bidirectional'
            },
            'correlation': {
                'value': float(correlation),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'divergence': {
                'mean': float(mean_divergence),
                'std': float(std_divergence),
                'max': float(np.max(divergence)),
                'min': float(np.min(divergence)),
                'data': divergence.tolist()
            },
            'fit_metrics': {
                'price_to_fundamental_r2': fit_price_to_fundamental['r_squared'],
                'fundamental_to_price_r2': fit_fundamental_to_price['r_squared'],
                'fit_quality': min(fit_price_to_fundamental['fit_quality'], 
                                  fit_fundamental_to_price['fit_quality']),
                'fit_type': self.fitter.get_fit_type()
            }
        }
    
    def set_fitter(self, fitter: IFitter) -> None:
        """设置底层拟合器"""
        self.fitter = fitter
    
    def get_fit_method(self) -> str:
        """获取拟合方法名称"""
        return f"Bidirectional Chart Fitting (using {self.fitter.get_fit_type()})"
