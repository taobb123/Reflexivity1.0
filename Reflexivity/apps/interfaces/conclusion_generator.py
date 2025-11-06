"""
结论生成器接口
根据拟合参数和阶段生成分析结论（中文文本描述）
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class IConclusionGenerator(ABC):
    """结论生成器接口"""
    
    @abstractmethod
    def generate(self,
                 parameters: Dict[str, float],
                 stage_result: Dict[str, Any],
                 fit_results: Dict[str, Any],
                 **kwargs) -> str:
        """
        生成分析结论（中文文本描述）
        
        Args:
            parameters: 反身性参数
            stage_result: 阶段检测结果
            fit_results: 拟合结果
            **kwargs: 其他参数
            
        Returns:
            中文结论文本
        """
        pass
    
    @abstractmethod
    def generate_summary(self, all_results: Dict[str, Any]) -> str:
        """
        生成总结性结论
        
        Args:
            all_results: 所有分析结果
            
        Returns:
            总结文本
        """
        pass
    
    @abstractmethod
    def generate_detailed(self, all_results: Dict[str, Any]) -> str:
        """
        生成详细结论
        
        Args:
            all_results: 所有分析结果
            
        Returns:
            详细文本
        """
        pass
