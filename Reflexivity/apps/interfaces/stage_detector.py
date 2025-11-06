"""
阶段检测器接口
用于判断当前反身性处于哪个阶段
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class IStageDetector(ABC):
    """阶段检测器接口"""
    
    @abstractmethod
    def detect_stage(self,
                     parameters: Dict[str, float],
                     price_data: np.ndarray,
                     fundamental_data: np.ndarray,
                     lambda_value: float,
                     **kwargs) -> Dict[str, Any]:
        """
        检测当前反身性阶段
        
        Args:
            parameters: 反身性参数 {'alpha': float, 'gamma': float, 'beta': float}
            price_data: 价格序列
            fundamental_data: 基本面序列
            lambda_value: λ值
            **kwargs: 其他检测参数
            
        Returns:
            阶段检测结果字典，包含：
            - stage: 阶段名称（如 '稳定收敛', '临界状态', '泡沫形成', '泡沫破灭', '崩溃'）
            - confidence: 置信度 [0, 1]
            - indicators: 各项指标
            - description: 阶段描述
            - risk_level: 风险等级
        """
        pass
    
    @abstractmethod
    def get_available_stages(self) -> List[str]:
        """
        获取所有可用的阶段名称
        
        Returns:
            阶段名称列表
        """
        pass
    
    @abstractmethod
    def get_stage_criteria(self) -> Dict[str, Dict[str, Any]]:
        """
        获取各阶段的判断标准
        
        Returns:
            阶段标准字典，格式：
            {
                'stage_name': {
                    'lambda_range': (min, max),
                    'alpha_range': (min, max),
                    'trend_indicators': [...],
                    ...
                }
            }
        """
        pass
