"""
优化器实现
支持多种优化算法
"""
import numpy as np
from typing import Dict, Any, Callable, Tuple, Optional
from scipy.optimize import minimize, differential_evolution

from apps.interfaces.optimizer import IOptimizer


class DifferentialEvolutionOptimizer(IOptimizer):
    """差分进化优化器"""
    
    def optimize(self, 
                 objective_function: Callable,
                 initial_params: np.ndarray,
                 bounds: Optional[Tuple] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行差分进化优化"""
        if bounds is None:
            # 默认边界
            bounds = [(0, 2) for _ in initial_params]
        
        result = differential_evolution(
            objective_function,
            bounds=bounds,
            seed=kwargs.get('seed', 42),
            maxiter=kwargs.get('maxiter', 100),
            popsize=kwargs.get('popsize', 15),
            tol=kwargs.get('tol', 1e-6),
            mutation=kwargs.get('mutation', (0.5, 1)),
            recombination=kwargs.get('recombination', 0.7)
        )
        
        return {
            'optimal_params': result.x,
            'optimal_value': float(result.fun),
            'success': result.success,
            'iterations': result.nit,
            'message': result.message,
            'optimizer_type': 'differential_evolution'
        }
    
    def get_optimizer_type(self) -> str:
        """获取优化器类型"""
        return 'differential_evolution'


class LBFGSOptimizer(IOptimizer):
    """L-BFGS-B 优化器（局部优化）"""
    
    def optimize(self, 
                 objective_function: Callable,
                 initial_params: np.ndarray,
                 bounds: Optional[Tuple] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行 L-BFGS-B 优化"""
        if bounds is None:
            # 默认边界
            bounds = [(0, 2) for _ in initial_params]
        
        result = minimize(
            objective_function,
            x0=initial_params,
            bounds=bounds,
            method='L-BFGS-B',
            options=kwargs.get('options', {})
        )
        
        return {
            'optimal_params': result.x,
            'optimal_value': float(result.fun),
            'success': result.success,
            'iterations': result.nit if hasattr(result, 'nit') else 0,
            'message': result.message,
            'optimizer_type': 'L-BFGS-B'
        }
    
    def get_optimizer_type(self) -> str:
        """获取优化器类型"""
        return 'L-BFGS-B'


class GradientDescentOptimizer(IOptimizer):
    """梯度下降优化器"""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000):
        """
        初始化
        
        Args:
            learning_rate: 学习率
            max_iterations: 最大迭代次数
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
    
    def optimize(self, 
                 objective_function: Callable,
                 initial_params: np.ndarray,
                 bounds: Optional[Tuple] = None,
                 **kwargs) -> Dict[str, Any]:
        """执行梯度下降优化"""
        from scipy.optimize import approx_fprime
        
        params = initial_params.copy()
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        max_iter = kwargs.get('max_iterations', self.max_iterations)
        tol = kwargs.get('tol', 1e-6)
        
        for iteration in range(max_iter):
            # 计算梯度（数值近似）
            grad = approx_fprime(params, objective_function, epsilon=1e-8)
            
            # 更新参数
            new_params = params - learning_rate * grad
            
            # 应用边界约束
            if bounds:
                for i in range(len(new_params)):
                    new_params[i] = np.clip(new_params[i], bounds[i][0], bounds[i][1])
            
            # 检查收敛
            if np.linalg.norm(new_params - params) < tol:
                break
            
            params = new_params
        
        optimal_value = objective_function(params)
        
        return {
            'optimal_params': params,
            'optimal_value': float(optimal_value),
            'success': iteration < max_iter - 1,
            'iterations': iteration + 1,
            'message': '梯度下降优化完成',
            'optimizer_type': 'gradient_descent'
        }
    
    def get_optimizer_type(self) -> str:
        """获取优化器类型"""
        return 'gradient_descent'
