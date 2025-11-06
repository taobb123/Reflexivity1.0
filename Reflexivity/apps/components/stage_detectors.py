"""
阶段检测器实现
判断当前反身性处于哪个阶段
基于 λ 值、参数阈值、趋势变化，多指标综合
"""
import numpy as np
from typing import Dict, Any, List
from scipy import stats

from apps.interfaces.stage_detector import IStageDetector


class ComprehensiveStageDetector(IStageDetector):
    """综合阶段检测器，多指标综合分析"""
    
    # 阶段定义
    STAGES = {
        '稳定收敛': {
            'lambda_range': (-1, 1),
            'alpha_range': (0, 1),
            'trend_stable': True,
            'divergence_trend': 'converging'
        },
        '临界状态': {
            'lambda_range': (0.95, 1.05),
            'alpha_range': (0.8, 1.2),
            'trend_stable': True,
            'divergence_trend': 'stable'
        },
        '泡沫形成': {
            'lambda_range': (1, 1.5),
            'alpha_range': (0.8, 1.5),
            'trend_stable': False,
            'divergence_trend': 'expanding'
        },
        '泡沫破灭': {
            'lambda_range': (0.5, 1),
            'alpha_range': (0.5, 1),
            'trend_stable': False,
            'divergence_trend': 'contracting'
        },
        '崩溃': {
            'lambda_range': (-2, -1),
            'alpha_range': (0, 2),
            'trend_stable': False,
            'divergence_trend': 'oscillating'
        }
    }
    
    def detect_stage(self,
                     parameters: Dict[str, float],
                     price_data: np.ndarray,
                     fundamental_data: np.ndarray,
                     lambda_value: float,
                     **kwargs) -> Dict[str, Any]:
        """检测当前反身性阶段"""
        alpha = parameters.get('alpha', 0)
        gamma = parameters.get('gamma', 0)
        beta = parameters.get('beta', 0)
        
        # 计算各项指标
        divergence = price_data - fundamental_data
        divergence_trend = self._analyze_divergence_trend(divergence)
        price_trend = self._analyze_trend(price_data)
        fundamental_trend = self._analyze_trend(fundamental_data)
        volatility = self._calculate_volatility(price_data)
        
        # 计算各阶段的匹配分数
        stage_scores = {}
        for stage_name, criteria in self.STAGES.items():
            score = self._calculate_stage_score(
                lambda_value, alpha, divergence_trend,
                price_trend, fundamental_trend, volatility,
                criteria
            )
            stage_scores[stage_name] = score
        
        # 选择得分最高的阶段
        best_stage = max(stage_scores, key=stage_scores.get)
        confidence = stage_scores[best_stage]
        
        # 风险等级评估
        risk_level = self._assess_risk_level(lambda_value, alpha, divergence_trend)
        
        # 生成描述
        description = self._generate_stage_description(
            best_stage, lambda_value, alpha, beta, gamma,
            divergence_trend, price_trend, fundamental_trend
        )
        
        return {
            'stage': best_stage,
            'confidence': float(confidence),
            'indicators': {
                'lambda': float(lambda_value),
                'alpha': float(alpha),
                'gamma': float(gamma),
                'beta': float(beta),
                'divergence_trend': divergence_trend,
                'price_trend': price_trend,
                'fundamental_trend': fundamental_trend,
                'volatility': float(volatility),
                'abs_lambda': float(abs(lambda_value))
            },
            'description': description,
            'risk_level': risk_level,
            'stage_scores': {k: float(v) for k, v in stage_scores.items()}
        }
    
    def get_available_stages(self) -> List[str]:
        """获取所有可用的阶段名称"""
        return list(self.STAGES.keys())
    
    def get_stage_criteria(self) -> Dict[str, Dict[str, Any]]:
        """获取各阶段的判断标准"""
        return self.STAGES.copy()
    
    def _calculate_stage_score(self, lambda_val, alpha, divergence_trend,
                              price_trend, fundamental_trend, volatility,
                              criteria) -> float:
        """计算阶段匹配分数"""
        score = 0.0
        
        # Lambda 值匹配
        lambda_min, lambda_max = criteria['lambda_range']
        if lambda_min <= lambda_val <= lambda_max:
            lambda_score = 1.0 - abs(lambda_val - (lambda_min + lambda_max) / 2) / (lambda_max - lambda_min)
            score += lambda_score * 0.4
        
        # Alpha 值匹配
        alpha_min, alpha_max = criteria['alpha_range']
        if alpha_min <= alpha <= alpha_max:
            alpha_score = 1.0 - abs(alpha - (alpha_min + alpha_max) / 2) / (alpha_max - alpha_min)
            score += alpha_score * 0.3
        
        # 背离趋势匹配
        if criteria['divergence_trend'] == divergence_trend:
            score += 0.2
        
        # 趋势稳定性匹配
        if criteria['trend_stable'] == (price_trend == 'stable' and fundamental_trend == 'stable'):
            score += 0.1
        
        return score
    
    def _analyze_divergence_trend(self, divergence: np.ndarray) -> str:
        """分析背离趋势"""
        # 计算背离的变化率
        if len(divergence) < 2:
            return 'stable'
        
        diff = np.diff(divergence)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        if mean_diff > std_diff:
            return 'expanding'  # 背离扩大
        elif mean_diff < -std_diff:
            return 'converging'  # 背离缩小
        elif abs(mean_diff) < std_diff * 0.5:
            return 'stable'  # 稳定
        else:
            # 检查是否有振荡
            if np.std(diff) > abs(mean_diff) * 2:
                return 'oscillating'  # 振荡
            return 'stable'
    
    def _analyze_trend(self, data: np.ndarray) -> str:
        """分析趋势"""
        if len(data) < 2:
            return 'stable'
        
        # 线性回归斜率
        x = np.arange(len(data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
        
        # 归一化斜率
        normalized_slope = slope / (np.mean(data) + 1e-10)
        
        if abs(normalized_slope) < 0.01:
            return 'stable'
        elif normalized_slope > 0.05:
            return 'increasing'
        elif normalized_slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_volatility(self, data: np.ndarray) -> float:
        """计算波动率"""
        if len(data) < 2:
            return 0.0
        
        returns = np.diff(data) / (data[:-1] + 1e-10)
        return float(np.std(returns))
    
    def _assess_risk_level(self, lambda_val: float, alpha: float, 
                          divergence_trend: str) -> str:
        """评估风险等级"""
        abs_lambda = abs(lambda_val)
        
        if abs_lambda < 0.8:
            return '低风险'
        elif abs_lambda < 1.0:
            if divergence_trend == 'converging':
                return '低风险'
            else:
                return '中风险'
        elif abs_lambda < 1.2:
            return '中高风险'
        elif abs_lambda >= 1.2:
            return '高风险'
        else:
            return '中等风险'
    
    def _generate_stage_description(self, stage: str, lambda_val: float,
                                   alpha: float, beta: float, gamma: float,
                                   divergence_trend: str, price_trend: str,
                                   fundamental_trend: str) -> str:
        """生成阶段描述"""
        descriptions = {
            '稳定收敛': f"系统处于稳定收敛状态。λ值 ({lambda_val:.4f}) 小于1，"
                       f"价格与基本面的背离正在缩小 ({divergence_trend})。"
                       f"反身性参数 α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f} 表明系统具有自我修正能力。",
            
            '临界状态': f"系统处于临界状态。λ值 ({lambda_val:.4f}) 接近1，"
                       f"系统行为对参数变化非常敏感。反身性参数 α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}。"
                       f"需要密切关注市场动态。",
            
            '泡沫形成': f"系统呈现泡沫形成特征。λ值 ({lambda_val:.4f}) 大于1，"
                       f"价格与基本面的背离正在扩大 ({divergence_trend})。"
                       f"反身性参数 α={alpha:.4f} 表明市场过度依赖价格信号，"
                       f"存在价格自我强化的风险。",
            
            '泡沫破灭': f"系统处于泡沫破灭阶段。λ值 ({lambda_val:.4f}) 从高位回落，"
                       f"价格与基本面的背离正在缩小 ({divergence_trend})。"
                       f"反身性参数 α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f} 表明系统正在调整。",
            
            '崩溃': f"系统处于崩溃状态。λ值 ({lambda_val:.4f}) 为负且绝对值较大，"
                   f"系统出现振荡发散。价格与基本面的背离呈现振荡特征 ({divergence_trend})。"
                   f"反身性参数 α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f} 表明系统失控。"
        }
        
        return descriptions.get(stage, f"未知阶段，λ值: {lambda_val:.4f}")
