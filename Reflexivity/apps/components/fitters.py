"""
拟合器实现
支持多种拟合方法（线性、多项式、非线性等）
"""
import numpy as np
from typing import Dict, Any, Tuple
from scipy import stats

# sklearn 是可选的，用于多项式拟合
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from apps.interfaces.fitter import IFitter


class LinearFitter(IFitter):
    """线性拟合器"""
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """执行线性拟合"""
        # 使用 scipy 的线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # 预测值
        y_pred = slope * x + intercept
        
        # 残差
        residuals = y - y_pred
        
        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'parameters': {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_value': float(r_value),
                'p_value': float(p_value),
                'std_err': float(std_err)
            },
            'residuals': residuals.tolist(),
            'r_squared': float(r_squared),
            'fit_quality': self._assess_quality(r_squared)
        }
    
    def predict(self, x: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """使用拟合参数进行预测"""
        slope = parameters['slope']
        intercept = parameters['intercept']
        return slope * x + intercept
    
    def get_fit_type(self) -> str:
        """获取拟合类型"""
        return 'linear'
    
    def _assess_quality(self, r_squared: float) -> str:
        """评估拟合质量"""
        if r_squared >= 0.9:
            return 'excellent'
        elif r_squared >= 0.7:
            return 'good'
        elif r_squared >= 0.5:
            return 'fair'
        else:
            return 'poor'


class PolynomialFitter(IFitter):
    """多项式拟合器"""
    
    def __init__(self, degree: int = 2):
        """
        初始化
        
        Args:
            degree: 多项式次数
        """
        if not HAS_SKLEARN:
            raise ImportError("PolynomialFitter 需要 scikit-learn。请安装: pip install scikit-learn")
        self.degree = degree
        self.model = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """执行多项式拟合"""
        if not HAS_SKLEARN:
            raise ImportError("PolynomialFitter 需要 scikit-learn。请安装: pip install scikit-learn")
        
        # 使用 sklearn 的多项式回归
        x_reshaped = x.reshape(-1, 1)
        
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree)),
            ('linear', LinearRegression())
        ])
        
        model.fit(x_reshaped, y)
        self.model = model
        
        # 预测值
        y_pred = model.predict(x_reshaped)
        
        # 残差
        residuals = y - y_pred
        
        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 获取系数
        coefficients = model.named_steps['linear'].coef_
        intercept = model.named_steps['linear'].intercept_
        
        return {
            'parameters': {
                'coefficients': coefficients.tolist(),
                'intercept': float(intercept),
                'degree': self.degree
            },
            'residuals': residuals.tolist(),
            'r_squared': float(r_squared),
            'fit_quality': self._assess_quality(r_squared)
        }
    
    def predict(self, x: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """使用拟合参数进行预测"""
        if not HAS_SKLEARN:
            raise ImportError("PolynomialFitter 需要 scikit-learn。请安装: pip install scikit-learn")
        if self.model is None:
            raise ValueError("模型未拟合，请先调用 fit 方法")
        
        x_reshaped = x.reshape(-1, 1)
        return self.model.predict(x_reshaped)
    
    def get_fit_type(self) -> str:
        """获取拟合类型"""
        return f'polynomial_{self.degree}'


class NonlinearFitter(IFitter):
    """非线性拟合器（使用最小二乘法）"""
    
    def __init__(self, func_type: str = 'exponential'):
        """
        初始化
        
        Args:
            func_type: 函数类型 ('exponential', 'logarithmic', 'power')
        """
        self.func_type = func_type
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """执行非线性拟合"""
        if self.func_type == 'exponential':
            return self._fit_exponential(x, y)
        elif self.func_type == 'logarithmic':
            return self._fit_logarithmic(x, y)
        elif self.func_type == 'power':
            return self._fit_power(x, y)
        else:
            raise ValueError(f"不支持的函数类型: {self.func_type}")
    
    def _fit_exponential(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """指数拟合 y = a * exp(b * x)"""
        # 转换为线性：log(y) = log(a) + b * x
        y_log = np.log(np.maximum(y, 1e-10))  # 避免 log(0)
        slope, intercept, r_value, _, _ = stats.linregress(x, y_log)
        
        a = np.exp(intercept)
        b = slope
        
        y_pred = a * np.exp(b * x)
        residuals = y - y_pred
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'parameters': {
                'a': float(a),
                'b': float(b),
                'r_value': float(r_value)
            },
            'residuals': residuals.tolist(),
            'r_squared': float(r_squared),
            'fit_quality': self._assess_quality(r_squared)
        }
    
    def _fit_logarithmic(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """对数拟合 y = a + b * log(x)"""
        x_log = np.log(np.maximum(x, 1e-10))
        slope, intercept, r_value, _, _ = stats.linregress(x_log, y)
        
        a = intercept
        b = slope
        
        y_pred = a + b * np.log(x)
        residuals = y - y_pred
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'parameters': {
                'a': float(a),
                'b': float(b),
                'r_value': float(r_value)
            },
            'residuals': residuals.tolist(),
            'r_squared': float(r_squared),
            'fit_quality': self._assess_quality(r_squared)
        }
    
    def _fit_power(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """幂函数拟合 y = a * x^b"""
        # 转换为线性：log(y) = log(a) + b * log(x)
        x_log = np.log(np.maximum(x, 1e-10))
        y_log = np.log(np.maximum(y, 1e-10))
        
        slope, intercept, r_value, _, _ = stats.linregress(x_log, y_log)
        
        a = np.exp(intercept)
        b = slope
        
        y_pred = a * (x ** b)
        residuals = y - y_pred
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'parameters': {
                'a': float(a),
                'b': float(b),
                'r_value': float(r_value)
            },
            'residuals': residuals.tolist(),
            'r_squared': float(r_squared),
            'fit_quality': self._assess_quality(r_squared)
        }
    
    def predict(self, x: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """使用拟合参数进行预测"""
        if self.func_type == 'exponential':
            a = parameters['a']
            b = parameters['b']
            return a * np.exp(b * x)
        elif self.func_type == 'logarithmic':
            a = parameters['a']
            b = parameters['b']
            return a + b * np.log(x)
        elif self.func_type == 'power':
            a = parameters['a']
            b = parameters['b']
            return a * (x ** b)
        else:
            raise ValueError(f"不支持的函数类型: {self.func_type}")
    
    def get_fit_type(self) -> str:
        """获取拟合类型"""
        return f'nonlinear_{self.func_type}'
    
    def _assess_quality(self, r_squared: float) -> str:
        """评估拟合质量"""
        if r_squared >= 0.9:
            return 'excellent'
        elif r_squared >= 0.7:
            return 'good'
        elif r_squared >= 0.5:
            return 'fair'
        else:
            return 'poor'
