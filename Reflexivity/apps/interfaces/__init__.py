"""
接口层模块
定义所有接口，遵循接口编程原则
"""

from .data_provider import IDataProvider
from .fitter import IFitter
from .optimizer import IOptimizer
from .chart_fitter import IChartFitter
from .parameter_estimator import IParameterEstimator
from .stage_detector import IStageDetector
from .conclusion_generator import IConclusionGenerator
from .chart_visualizer import IChartVisualizer

__all__ = [
    'IDataProvider',
    'IFitter',
    'IOptimizer',
    'IChartFitter',
    'IParameterEstimator',
    'IStageDetector',
    'IConclusionGenerator',
    'IChartVisualizer',
]
