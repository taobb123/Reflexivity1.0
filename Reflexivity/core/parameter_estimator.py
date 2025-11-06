"""
参数反推模块
使用优化法从真实价格数据反推模型参数 α、β、γ
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Dict, Optional
from .reflexivity_model import ReflexivityModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ParameterEstimator:
    """参数估计器"""
    
    def __init__(self, P_t: np.ndarray, F_t: np.ndarray, 
                 noise_std: float = 1.0):
        """
        初始化参数估计器
        
        Args:
            P_t: 真实价格序列
            F_t: 真实基本面序列（如EPS）
            noise_std: 噪声标准差（可以从残差中估计，这里作为输入）
        """
        if len(P_t) != len(F_t):
            raise ValueError("价格序列和基本面序列长度必须相同")
        
        self.P_t = np.array(P_t)
        self.F_t = np.array(F_t)
        self.noise_std = noise_std
        self.T = len(P_t)
        
        # 归一化（避免数值问题）
        self.P_mean = np.mean(P_t)
        self.P_std = np.std(P_t) if np.std(P_t) > 0 else 1.0
        self.F_mean = np.mean(F_t)
        self.F_std = np.std(F_t) if np.std(F_t) > 0 else 1.0
        
        self.P_normalized = (P_t - self.P_mean) / self.P_std
        self.F_normalized = (F_t - self.F_mean) / self.F_std
        
        print(f"✓ 参数估计器初始化: {self.T} 个数据点")
    
    def simulate_model(self, alpha: float, gamma: float, beta: float,
                      P0: Optional[float] = None, 
                      F0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用给定参数运行模型仿真
        
        Args:
            alpha: α参数
            gamma: γ参数
            beta: β参数
            P0: 初始价格（归一化），如果为None则使用实际第一个值
            F0: 初始基本面（归一化），如果为None则使用实际第一个值
            
        Returns:
            (预测价格序列, 预测基本面序列) - 已归一化
        """
        if P0 is None:
            P0 = self.P_normalized[0]
        if F0 is None:
            F0 = self.F_normalized[0]
        
        P_pred = np.zeros(self.T)
        F_pred = np.zeros(self.T)
        
        P_pred[0] = P0
        F_pred[0] = F0
        
        # 迭代计算（不使用噪声，因为我们拟合的是趋势）
        for t in range(self.T - 1):
            # 市场认知
            E_t = alpha * P_pred[t] + (1 - alpha) * F_pred[t]
            
            # 价格调整（不加噪声，因为我们拟合的是确定性部分）
            P_pred[t+1] = P_pred[t] + gamma * (E_t - P_pred[t])
            
            # 基本面调整
            F_pred[t+1] = F_pred[t] + beta * (P_pred[t] - F_pred[t])
        
        return P_pred, F_pred
    
    def objective_function(self, params: np.ndarray) -> float:
        """
        优化目标函数：最小化预测误差
        
        Args:
            params: [alpha, gamma, beta]
            
        Returns:
            均方误差（MSE）
        """
        alpha, gamma, beta = params
        
        # 参数边界检查
        if alpha < 0 or alpha > 2:
            return 1e10
        if gamma < 0 or gamma > 5:
            return 1e10
        if beta < 0 or beta > 2:
            return 1e10
        
        try:
            # 运行模型
            P_pred, _ = self.simulate_model(alpha, gamma, beta)
            
            # 计算均方误差
            mse = np.mean((self.P_normalized - P_pred) ** 2)
            
            # 添加正则化项（鼓励参数在合理范围内）
            regularization = 0.01 * (
                (alpha - 1.0) ** 2 + 
                (gamma - 0.5) ** 2 + 
                (beta - 0.1) ** 2
            )
            
            return mse + regularization
            
        except Exception as e:
            return 1e10
    
    def estimate_parameters(self, 
                           method: str = 'differential_evolution',
                           bounds: Optional[Tuple] = None,
                           initial_guess: Optional[np.ndarray] = None) -> Dict:
        """
        估计模型参数
        
        Args:
            method: 优化方法
                - 'differential_evolution': 差分进化算法（全局优化，推荐）
                - 'minimize': 局部优化（更快但可能陷入局部最优）
            bounds: 参数边界 [(alpha_min, alpha_max), (gamma_min, gamma_max), (beta_min, beta_max)]
            initial_guess: 初始猜测值 [alpha, gamma, beta]
            
        Returns:
            包含估计参数的字典
        """
        if bounds is None:
            # 默认边界
            bounds = [
                (0.0, 2.0),   # alpha
                (0.0, 5.0),   # gamma
                (0.0, 2.0)    # beta
            ]
        
        print(f"\n🔍 开始参数估计 (方法: {method})...")
        print(f"参数边界: α∈[{bounds[0]}], γ∈[{bounds[1]}], β∈[{bounds[2]}]")
        
        if method == 'differential_evolution':
            # 差分进化算法（全局优化）
            result = differential_evolution(
                self.objective_function,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7
            )
            
            alpha_est, gamma_est, beta_est = result.x
            
        elif method == 'minimize':
            # 局部优化
            if initial_guess is None:
                initial_guess = np.array([0.8, 0.5, 0.1])
            
            result = minimize(
                self.objective_function,
                x0=initial_guess,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            alpha_est, gamma_est, beta_est = result.x
            
        else:
            raise ValueError(f"未知的优化方法: {method}")
        
        # 计算拟合效果
        P_pred, F_pred = self.simulate_model(alpha_est, gamma_est, beta_est)
        
        # 反归一化
        P_pred_denorm = P_pred * self.P_std + self.P_mean
        P_actual = self.P_t
        
        mse = np.mean((P_actual - P_pred_denorm) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(P_actual - P_pred_denorm))
        
        # R²
        ss_res = np.sum((P_actual - P_pred_denorm) ** 2)
        ss_tot = np.sum((P_actual - np.mean(P_actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 计算λ值
        lambda_val = 1 + gamma_est * (alpha_est - 1) - beta_est
        
        # 稳定性分析
        if abs(lambda_val) < 1:
            stability = "稳定收敛"
        elif abs(lambda_val) > 1:
            if lambda_val < -1:
                stability = "振荡发散"
            else:
                stability = "单调发散"
        else:
            stability = "临界状态"
        
        results = {
            'parameters': {
                'alpha': float(alpha_est),
                'gamma': float(gamma_est),
                'beta': float(beta_est)
            },
            'lambda': float(lambda_val),
            'stability': stability,
            'fitness': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r_squared': float(r_squared)
            },
            'predicted_P': P_pred_denorm.tolist(),
            'predicted_F': (F_pred * self.F_std + self.F_mean).tolist()
        }
        
        print(f"\n✓ 参数估计完成!")
        print(f"  估计参数: α={alpha_est:.4f}, γ={gamma_est:.4f}, β={beta_est:.4f}")
        print(f"  λ={lambda_val:.4f}, 稳定性: {stability}")
        print(f"  拟合效果: RMSE={rmse:.4f}, R²={r_squared:.4f}")
        
        return results
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None) -> str:
        """
        绘制估计结果对比图
        
        Args:
            results: estimate_parameters返回的结果
            save_path: 保存路径（如果为None则返回base64编码）
            
        Returns:
            如果save_path为None，返回base64编码的图片
        """
        P_pred = np.array(results['predicted_P'])
        F_pred = np.array(results['predicted_F'])
        
        params = results['parameters']
        fitness = results['fitness']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('参数反推结果对比', fontsize=16, fontweight='bold')
        
        t = np.arange(self.T)
        
        # 子图1：价格对比
        ax1 = axes[0, 0]
        ax1.plot(t, self.P_t, 'b-', label='真实价格', linewidth=2, alpha=0.8)
        ax1.plot(t, P_pred, 'r--', label='模型预测', linewidth=2, alpha=0.8)
        ax1.set_xlabel('时间（周）', fontsize=12)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title(f'价格拟合对比 (R²={fitness["r_squared"]:.4f}, RMSE={fitness["rmse"]:.2f})', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：基本面对比
        ax2 = axes[0, 1]
        ax2.plot(t, self.F_t, 'g-', label='真实基本面', linewidth=2, alpha=0.8)
        ax2.plot(t, F_pred, 'm--', label='模型预测', linewidth=2, alpha=0.8)
        ax2.set_xlabel('时间（周）', fontsize=12)
        ax2.set_ylabel('基本面（EPS）', fontsize=12)
        ax2.set_title('基本面拟合对比', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 子图3：残差分析
        ax3 = axes[1, 0]
        residuals = self.P_t - P_pred
        ax3.plot(t, residuals, 'purple', linewidth=1.5, alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.fill_between(t, -fitness['rmse'], fitness['rmse'], alpha=0.2, color='gray')
        ax3.set_xlabel('时间（周）', fontsize=12)
        ax3.set_ylabel('残差 (真实 - 预测)', fontsize=12)
        ax3.set_title(f'残差分析 (MAE={fitness["mae"]:.2f})', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 子图4：参数信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        info_text = f"""
估计参数：
  α (价格在认知中的权重) = {params['alpha']:.6f}
  γ (价格调整速度) = {params['gamma']:.6f}
  β (价格对基本面的影响) = {params['beta']:.6f}

系统特征：
  λ = {results['lambda']:.6f}
  |λ| = {abs(results['lambda']):.6f}
  稳定性: {results['stability']}

拟合效果：
  R² (决定系数) = {fitness['r_squared']:.6f}
  RMSE (均方根误差) = {fitness['rmse']:.4f}
  MAE (平均绝对误差) = {fitness['mae']:.4f}
  MSE (均方误差) = {fitness['mse']:.4f}

数据信息：
  数据点数量 = {self.T}
  价格均值 = {self.P_mean:.2f}
  价格标准差 = {self.P_std:.2f}
"""
        ax4.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            # 转换为base64
            import io
            import base64
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode()
            plt.close()
            return img_base64


def estimate_from_stock_data(df: pd.DataFrame,
                             method: str = 'differential_evolution') -> Dict:
    """
    从股票数据DataFrame估计参数
    
    Args:
        df: 包含'P_t'和'F_t'列的DataFrame
        method: 优化方法
        
    Returns:
        估计结果字典
    """
    P_t = df['P_t'].values
    F_t = df['F_t'].values
    
    estimator = ParameterEstimator(P_t, F_t)
    results = estimator.estimate_parameters(method=method)
    chart_base64 = estimator.plot_results(results)
    results['chart_base64'] = chart_base64
    
    return results

