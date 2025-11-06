"""
价格与市场背离分析模块
核心功能：分析股票价格与基本面的背离程度
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .reflexivity_model import ReflexivityModel
from .parameter_estimator import ParameterEstimator


class DivergenceAnalyzer:
    """背离分析器"""
    
    def __init__(self, P_t: np.ndarray, F_t: np.ndarray):
        """
        初始化背离分析器
        
        Args:
            P_t: 价格序列
            F_t: 基本面序列
        """
        if len(P_t) != len(F_t):
            raise ValueError("价格序列和基本面序列长度必须相同")
        
        self.P_t = np.array(P_t)
        self.F_t = np.array(F_t)
        self.T = len(P_t)
        
        # 计算背离
        self.divergence = self.P_t - self.F_t
        self.divergence_ratio = self.divergence / (self.F_t + 1e-10)  # 避免除零
    
    def calculate_divergence_metrics(self) -> Dict:
        """
        计算背离指标
        
        Returns:
            包含各种背离指标的字典
        """
        # 基本统计
        mean_div = np.mean(self.divergence)
        std_div = np.std(self.divergence)
        max_div = np.max(self.divergence)
        min_div = np.min(self.divergence)
        
        # 背离比率
        mean_ratio = np.mean(self.divergence_ratio)
        std_ratio = np.std(self.divergence_ratio)
        
        # 背离方向
        positive_divergence = np.sum(self.divergence > 0) / self.T
        negative_divergence = np.sum(self.divergence < 0) / self.T
        
        # 极端背离（超过2个标准差）
        extreme_positive = np.sum(self.divergence > (mean_div + 2 * std_div))
        extreme_negative = np.sum(self.divergence < (mean_div - 2 * std_div))
        
        # 相关性
        correlation = np.corrcoef(self.P_t, self.F_t)[0, 1]
        
        return {
            'mean_divergence': float(mean_div),
            'std_divergence': float(std_div),
            'max_divergence': float(max_div),
            'min_divergence': float(min_div),
            'mean_ratio': float(mean_ratio),
            'std_ratio': float(std_ratio),
            'positive_ratio': float(positive_divergence),
            'negative_ratio': float(negative_divergence),
            'extreme_positive_count': int(extreme_positive),
            'extreme_negative_count': int(extreme_negative),
            'correlation': float(correlation),
            'divergence_trend': self._analyze_trend()
        }
    
    def _analyze_trend(self) -> str:
        """分析背离趋势"""
        # 计算背离的变化趋势
        div_diff = np.diff(self.divergence)
        mean_diff = np.mean(div_diff)
        
        if mean_diff > 0.1:
            return "背离扩大（价格与基本面差距增大）"
        elif mean_diff < -0.1:
            return "背离缩小（价格与基本面差距减小）"
        else:
            return "背离稳定（价格与基本面差距相对稳定）"
    
    def detect_divergence_periods(self, threshold: float = 2.0) -> pd.DataFrame:
        """
        检测背离周期
        
        Args:
            threshold: 背离阈值（标准差倍数）
        
        Returns:
            包含背离周期的DataFrame
        """
        mean_div = np.mean(self.divergence)
        std_div = np.std(self.divergence)
        
        # 检测极端背离
        upper_threshold = mean_div + threshold * std_div
        lower_threshold = mean_div - threshold * std_div
        
        periods = []
        in_period = False
        period_start = None
        
        for i, div in enumerate(self.divergence):
            if div > upper_threshold:
                if not in_period:
                    period_start = i
                    in_period = True
                    period_type = "高估"
            elif div < lower_threshold:
                if not in_period:
                    period_start = i
                    in_period = True
                    period_type = "低估"
            else:
                if in_period:
                    periods.append({
                        'type': period_type,
                        'start': period_start,
                        'end': i - 1,
                        'duration': i - period_start,
                        'max_divergence': np.max(self.divergence[period_start:i]) if period_type == "高估" 
                                        else np.min(self.divergence[period_start:i])
                    })
                    in_period = False
        
        if in_period:
            periods.append({
                'type': period_type,
                'start': period_start,
                'end': self.T - 1,
                'duration': self.T - period_start,
                'max_divergence': np.max(self.divergence[period_start:]) if period_type == "高估"
                                else np.min(self.divergence[period_start:])
            })
        
        return pd.DataFrame(periods)
    
    def analyze_with_reflexivity(self) -> Dict:
        """
        结合反身性模型进行背离分析
        
        Returns:
            包含反身性参数和背离分析的字典
        """
        # 参数估计
        estimator = ParameterEstimator(self.P_t, self.F_t)
        results = estimator.estimate_parameters()
        
        # 背离指标
        divergence_metrics = self.calculate_divergence_metrics()
        
        # 稳定性分析
        model = ReflexivityModel(
            alpha=results['parameters']['alpha'],
            gamma=results['parameters']['gamma'],
            beta=results['parameters']['beta']
        )
        stability_type, stability_info = model.analyze_stability()
        
        return {
            'reflexivity_parameters': results['parameters'],
            'stability': stability_type,
            'stability_info': stability_info,
            'lambda': results['lambda'],
            'divergence_metrics': divergence_metrics,
            'fitness': results['fitness'],
            'interpretation': self._generate_interpretation(results, divergence_metrics, stability_type)
        }
    
    def _generate_interpretation(self, results: Dict, metrics: Dict, stability: str) -> str:
        """生成分析解释"""
        alpha = results['parameters']['alpha']
        beta = results['parameters']['beta']
        lambda_val = results['lambda']
        correlation = metrics['correlation']
        
        interpretation = []
        
        # 反身性强度
        if alpha > 1:
            interpretation.append(f"⚠️ 极端反身性 (α={alpha:.2f}>1): 市场过度依赖价格信号")
        elif alpha > 0.8:
            interpretation.append(f"强反身性 (α={alpha:.2f}): 价格对市场认知影响较大")
        else:
            interpretation.append(f"弱反身性 (α={alpha:.2f}): 价格对市场认知影响较小")
        
        # 价格对基本面的影响
        if beta > 0.5:
            interpretation.append(f"价格对基本面影响强 (β={beta:.2f}): 价格变化会显著改变基本面")
        else:
            interpretation.append(f"价格对基本面影响弱 (β={beta:.2f}): 价格变化对基本面影响有限")
        
        # 稳定性
        if abs(lambda_val) < 1:
            interpretation.append(f"✓ 系统稳定 (λ={lambda_val:.2f}): 价格和基本面会收敛")
        elif abs(lambda_val) > 1:
            interpretation.append(f"⚠️ 系统不稳定 (λ={lambda_val:.2f}): 可能形成泡沫或崩溃")
        
        # 背离程度
        if abs(metrics['mean_divergence']) > metrics['std_divergence']:
            interpretation.append(f"存在明显背离: 平均背离={metrics['mean_divergence']:.2f}")
        
        # 相关性
        if correlation < 0.5:
            interpretation.append(f"⚠️ 价格与基本面相关性低 (r={correlation:.2f}): 可能存在信息不对称")
        
        return "\n".join(interpretation)


def analyze_stock_divergence(df: pd.DataFrame) -> Dict:
    """
    从DataFrame分析股票背离
    
    Args:
        df: 包含'P_t'和'F_t'列的DataFrame
    
    Returns:
        背离分析结果
    """
    analyzer = DivergenceAnalyzer(df['P_t'].values, df['F_t'].values)
    return analyzer.analyze_with_reflexivity()




