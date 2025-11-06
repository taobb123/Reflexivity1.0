"""
组件实现模块
包含所有接口的具体实现
"""

from .data_providers import UnifiedDataProvider, DataFrameDataProvider
from .fitters import LinearFitter, NonlinearFitter

# PolynomialFitter 需要 sklearn，可选导入
try:
    from .fitters import PolynomialFitter
    HAS_POLYNOMIAL_FITTER = True
except ImportError:
    HAS_POLYNOMIAL_FITTER = False
    PolynomialFitter = None

from .optimizers import DifferentialEvolutionOptimizer, LBFGSOptimizer, GradientDescentOptimizer
from .chart_fitters import PriceFundamentalChartFitter, BidirectionalChartFitter
from .parameter_estimators import ReflexivityParameterEstimator
from .stage_detectors import ComprehensiveStageDetector
from .conclusion_generators import ChineseConclusionGenerator
from .chart_visualizers import MatplotlibChartVisualizer

__all__ = [
    # 数据提供者
    'UnifiedDataProvider',
    'DataFrameDataProvider',
    # 拟合器
    'LinearFitter',
    'NonlinearFitter',
    # 优化器
    'DifferentialEvolutionOptimizer',
    'LBFGSOptimizer',
    'GradientDescentOptimizer',
    # 图表拟合器
    'PriceFundamentalChartFitter',
    'BidirectionalChartFitter',
    # 参数估计器
    'ReflexivityParameterEstimator',
    # 阶段检测器
    'ComprehensiveStageDetector',
    # 结论生成器
    'ChineseConclusionGenerator',
    # 图表可视化器
    'MatplotlibChartVisualizer',
]

# 如果 PolynomialFitter 可用，添加到 __all__
if HAS_POLYNOMIAL_FITTER:
    __all__.insert(2, 'PolynomialFitter')
