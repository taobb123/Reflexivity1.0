"""
反身性（Reflexivity）模型实现

基于索罗斯《金融炼金术》第一章的理论，将反身性概念转化为可运算、可仿真的动态模型。

第1节：概念框架
反身性 = 市场参与者的认知（对"基本面"的看法）影响价格，
价格反过来又改变现实基本面——形成双向反馈回路（认知 → 价格 → 基本面 → 认知）。
"""

import numpy as np
from typing import Tuple, Optional


class ReflexivityModel:
    """
    第2节：最小可行模型（两个变量、两条反馈）
    
    变量：
    - P_t: 市场价格（或价格指数）
    - F_t: 被市场关注的"基本面"变量
    
    参数：
    - alpha (α): 市场如何根据当前价格调整对基本面的预期（价格在形成"认知"中的权重）≥0（可>1表示极端反身性）
    - gamma (γ): 价格向市场预期调整的速度（价格冲击的吸收速率）≥0
    - beta (β): 价格对基本面的影响强度（价格变化如何实质性地改变基本面）≥0
    """
    
    def __init__(self, alpha: float, gamma: float, beta: float, 
                 P0: float = 100.0, F0: float = 100.0, 
                 noise_std: float = 0.0):
        """
        初始化反身性模型
        
        Args:
            alpha: 价格在认知中的权重 [0,1]
            gamma: 价格调整速度 ≥0
            beta: 价格对基本面的影响强度 ≥0
            P0: 初始价格
            F0: 初始基本面
            noise_std: 噪声标准差（信息冲击）
        """
        if alpha < 0:
            raise ValueError("alpha must be >= 0 (can be > 1 for extreme reflexivity)")
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if beta < 0:
            raise ValueError("beta must be >= 0")
            
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.P0 = P0
        self.F0 = F0
        self.noise_std = noise_std
        
        # 存储历史数据
        self.P_history = []
        self.F_history = []
        self.E_history = []
        self.x_history = []  # 差异 x_t = P_t - F_t
        
    def compute_lambda(self) -> float:
        """
        第3节：计算系统的特征值 λ
        
        λ = 1 + γ(α-1) - β
        
        Returns:
            lambda值，用于稳定性分析
        """
        return 1 + self.gamma * (self.alpha - 1) - self.beta
    
    def analyze_stability(self) -> Tuple[str, dict]:
        """
        第3节：分析系统稳定性
        
        Returns:
            (稳定性描述, 分析结果字典)
        """
        lam = self.compute_lambda()
        
        if abs(lam) < 1:
            stability = "稳定收敛"
            description = "差异 x_t 收敛到 0，价格和基本面会最终一致（反身性自我修正）"
        elif abs(lam) > 1:
            if lam < -1:
                stability = "振荡发散"
                description = "会出现振荡且振幅发散（价格—基本面交替过度反应）"
            else:
                stability = "单调发散"
                description = "差异发散，价格与基本面越走越远，可能形成泡沫或崩溃（强烈的持续反身性）"
        else:  # |λ| = 1
            stability = "临界状态"
            description = "系统处于临界状态，差异既不收敛也不发散"
        
        return stability, {
            'lambda': lam,
            'abs_lambda': abs(lam),
            'stability_type': stability,
            'description': description
        }
    
    def compute_expectation(self, P: float, F: float) -> float:
        """
        第2节：计算市场对基本面的"认知"（或预期）
        
        E_t = α * P_t + (1-α) * F_t
        
        Args:
            P: 当前价格
            F: 当前基本面
            
        Returns:
            市场预期 E_t
        """
        return self.alpha * P + (1 - self.alpha) * F
    
    def step(self, P: float, F: float, noise: Optional[float] = None) -> Tuple[float, float]:
        """
        第2节：执行一步迭代
        
        规则：
        1. E_t = α * P_t + (1-α) * F_t  （市场认知）
        2. P_{t+1} = P_t + γ * (E_t - P_t) + ε_t  （价格调整）
        3. F_{t+1} = F_t + β * (P_t - F_t)  （基本面调整）
        
        Args:
            P: 当前价格
            F: 当前基本面
            noise: 可选的噪声项，如果为None则从N(0, noise_std^2)采样
            
        Returns:
            (P_next, F_next): 下一时刻的价格和基本面
        """
        # 1. 计算市场认知
        E = self.compute_expectation(P, F)
        
        # 2. 价格根据认知调整
        if noise is None:
            noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0
        P_next = P + self.gamma * (E - P) + noise
        
        # 3. 基本面被价格影响
        F_next = F + self.beta * (P - F)
        
        return P_next, F_next
    
    def simulate(self, T: int, reset: bool = True) -> dict:
        """
        运行完整仿真
        
        Args:
            T: 仿真步数
            reset: 是否重置历史数据
            
        Returns:
            包含所有历史数据的字典
        """
        if reset:
            self.P_history = [self.P0]
            self.F_history = [self.F0]
            self.E_history = [self.compute_expectation(self.P0, self.F0)]
            self.x_history = [self.P0 - self.F0]
        
        P = self.P0
        F = self.F0
        
        for t in range(T):
            P, F = self.step(P, F)
            
            E = self.compute_expectation(P, F)
            x = P - F
            
            self.P_history.append(P)
            self.F_history.append(F)
            self.E_history.append(E)
            self.x_history.append(x)
        
        return {
            'P': np.array(self.P_history),
            'F': np.array(self.F_history),
            'E': np.array(self.E_history),
            'x': np.array(self.x_history),
            't': np.arange(len(self.P_history))
        }
    
    def get_model_info(self) -> dict:
        """
        获取模型信息摘要
        
        Returns:
            模型参数和稳定性分析的字典
        """
        stability_type, stability_info = self.analyze_stability()
        
        return {
            'parameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'beta': self.beta,
                'P0': self.P0,
                'F0': self.F0,
                'noise_std': self.noise_std
            },
            'lambda': self.compute_lambda(),
            'stability': stability_type,
            'stability_info': stability_info
        }


def calculate_lambda_detail(alpha: float, gamma: float, beta: float) -> dict:
    """
    第4节：参数直觉与例子 - 详细计算步骤
    
    逐步计算 λ = 1 + γ(α-1) - β 的每一步
    
    Args:
        alpha: α参数
        gamma: γ参数
        beta: β参数
        
    Returns:
        包含计算步骤和结果的字典
    """
    # 步骤1: 计算 (α-1)
    alpha_minus_1 = alpha - 1
    
    # 步骤2: 计算 γ(α-1)
    gamma_times_alpha_minus_1 = gamma * alpha_minus_1
    
    # 步骤3: 计算 λ
    lambda_val = 1 + gamma_times_alpha_minus_1 - beta
    
    return {
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta,
        'steps': {
            'alpha_minus_1': alpha_minus_1,
            'gamma_times_alpha_minus_1': gamma_times_alpha_minus_1,
            'lambda': lambda_val
        },
        'lambda': lambda_val,
        'abs_lambda': abs(lambda_val),
        'stability': '稳定' if abs(lambda_val) < 1 else ('振荡发散' if lambda_val < -1 else '单调发散' if lambda_val > 1 else '临界')
    }


def example_calculation_1() -> dict:
    """
    第4节：示例计算1 - 稳定情况
    α=0.8, γ=0.5, β=0.1
    """
    return calculate_lambda_detail(0.8, 0.5, 0.1)


def example_calculation_2() -> dict:
    """
    第4节：示例计算2 - 尝试制造泡沫
    α=0.95, γ=0.8, β=0.05
    """
    return calculate_lambda_detail(0.95, 0.8, 0.05)


def example_calculation_3() -> dict:
    """
    第4节：示例计算3 - 制造明显泡沫（极端参数）
    α=1.2, γ=0.8, β=0.05
    （注意：α>1表示过度把价格当"基本面"信号）
    """
    return calculate_lambda_detail(1.2, 0.8, 0.05)

