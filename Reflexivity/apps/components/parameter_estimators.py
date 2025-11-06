"""
参数估计器实现
基于拟合结果计算反身性模型参数（α, β, γ）
封装现有的 ParameterEstimator
"""
import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from apps.interfaces.parameter_estimator import IParameterEstimator
from apps.interfaces.optimizer import IOptimizer
from core.parameter_estimator import ParameterEstimator as CoreParameterEstimator


class ReflexivityParameterEstimator(IParameterEstimator):
    """反身性参数估计器，封装现有的 ParameterEstimator"""
    
    def __init__(self, optimizer: Optional[IOptimizer] = None):
        """
        初始化
        
        Args:
            optimizer: 优化器，如果为None则使用默认的差分进化优化器
        """
        self.optimizer = optimizer
        self._core_estimator = None
    
    def estimate(self,
                 price_data: np.ndarray,
                 fundamental_data: np.ndarray,
                 **kwargs) -> Dict[str, Any]:
        """估计反身性模型参数"""
        # 创建核心估计器
        self._core_estimator = CoreParameterEstimator(
            P_t=price_data,
            F_t=fundamental_data,
            noise_std=kwargs.get('noise_std', 1.0)
        )
        
        # 确定优化方法
        method = kwargs.get('method', 'differential_evolution')
        if self.optimizer:
            # 如果提供了自定义优化器，需要适配
            method = self._get_method_from_optimizer()
        
        # 执行估计
        results = self._core_estimator.estimate_parameters(method=method)
        
        # 转换为标准格式
        return {
            'parameters': results['parameters'],
            'lambda': results['lambda'],
            'stability': results['stability'],
            'fitness': results['fitness'],
            'predicted_data': {
                'price': results['predicted_P'],
                'fundamental': results['predicted_F']
            }
        }
    
    def set_optimizer(self, optimizer: IOptimizer) -> None:
        """设置优化器"""
        self.optimizer = optimizer
    
    def get_estimation_method(self) -> str:
        """获取估计方法名称"""
        if self.optimizer:
            return f"Reflexivity Estimation (using {self.optimizer.get_optimizer_type()})"
        return "Reflexivity Estimation (using default differential_evolution)"
    
    def _get_method_from_optimizer(self) -> str:
        """从优化器获取方法名称"""
        if self.optimizer:
            opt_type = self.optimizer.get_optimizer_type()
            if opt_type == 'differential_evolution':
                return 'differential_evolution'
            elif opt_type == 'L-BFGS-B':
                return 'minimize'
        return 'differential_evolution'
