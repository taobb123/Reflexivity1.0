"""
反身性分析器主协调器
通过组合方式整合所有组件，实现完整的反身性分析流程
"""
import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.interfaces.data_provider import IDataProvider
from apps.interfaces.chart_fitter import IChartFitter
from apps.interfaces.parameter_estimator import IParameterEstimator
from apps.interfaces.stage_detector import IStageDetector
from apps.interfaces.conclusion_generator import IConclusionGenerator
from apps.interfaces.chart_visualizer import IChartVisualizer

# 默认实现
from apps.components.data_providers import UnifiedDataProvider, DataFrameDataProvider
from apps.components.chart_fitters import PriceFundamentalChartFitter
from apps.components.parameter_estimators import ReflexivityParameterEstimator
from apps.components.stage_detectors import ComprehensiveStageDetector
from apps.components.conclusion_generators import ChineseConclusionGenerator
from apps.components.chart_visualizers import MatplotlibChartVisualizer


class ReflexivityAnalyzer:
    """
    反身性分析器主协调器
    使用组合模式，通过接口组合各个组件
    """
    
    def __init__(self,
                 data_provider: Optional[IDataProvider] = None,
                 chart_fitter: Optional[IChartFitter] = None,
                 parameter_estimator: Optional[IParameterEstimator] = None,
                 stage_detector: Optional[IStageDetector] = None,
                 conclusion_generator: Optional[IConclusionGenerator] = None,
                 chart_visualizer: Optional[IChartVisualizer] = None):
        """
        初始化反身性分析器
        
        Args:
            data_provider: 数据提供者（如果为None则使用默认实现）
            chart_fitter: 图表拟合器（如果为None则使用默认实现）
            parameter_estimator: 参数估计器（如果为None则使用默认实现）
            stage_detector: 阶段检测器（如果为None则使用默认实现）
            conclusion_generator: 结论生成器（如果为None则使用默认实现）
            chart_visualizer: 图表可视化器（如果为None则使用默认实现）
        """
        # 使用组合模式，通过接口组合组件
        self.data_provider = data_provider or UnifiedDataProvider()
        self.chart_fitter = chart_fitter or PriceFundamentalChartFitter()
        self.parameter_estimator = parameter_estimator or ReflexivityParameterEstimator()
        self.stage_detector = stage_detector or ComprehensiveStageDetector()
        self.conclusion_generator = conclusion_generator or ChineseConclusionGenerator()
        self.chart_visualizer = chart_visualizer or MatplotlibChartVisualizer()
    
    def analyze(self,
                stock_code: str,
                **kwargs) -> Dict[str, Any]:
        """
        执行完整的反身性分析流程
        
        Args:
            stock_code: 股票代码
            **kwargs: 其他参数
                - lookback_weeks: 回溯周数
                - tushare_token: Tushare token
                - preferred_sources: 优先数据源
                - estimation_method: 估计方法
                - save_charts: 是否保存图表
                - chart_save_path: 图表保存路径
        
        Returns:
            完整分析结果字典，包含：
            - data_info: 数据信息
            - fit_results: 拟合结果
            - parameter_results: 参数估计结果
            - stage_results: 阶段检测结果
            - conclusion: 分析结论
            - charts: 图表（base64或路径）
        """
        # 1. 获取数据
        print("=" * 60)
        print("开始反身性分析...")
        print("=" * 60)
        print("\n[1/6] 获取数据...")
        
        df = self.data_provider.get_dataframe(stock_code, **kwargs)
        price_data = df['P_t'].values
        fundamental_data = df['F_t'].values
        data_info = self.data_provider.get_data_info()
        
        print(f"✓ 数据获取完成: {len(price_data)} 个数据点")
        
        # 2. 图表拟合
        print("\n[2/6] 执行图表拟合...")
        fit_results = self.chart_fitter.fit_charts(
            price_data=price_data,
            fundamental_data=fundamental_data,
            **kwargs
        )
        print(f"✓ 图表拟合完成: R² = {fit_results['fit_metrics']['r_squared']:.4f}")
        
        # 3. 参数估计
        print("\n[3/6] 估计反身性参数...")
        parameter_results = self.parameter_estimator.estimate(
            price_data=price_data,
            fundamental_data=fundamental_data,
            **kwargs
        )
        print(f"✓ 参数估计完成: α={parameter_results['parameters']['alpha']:.4f}, "
              f"γ={parameter_results['parameters']['gamma']:.4f}, "
              f"β={parameter_results['parameters']['beta']:.4f}, "
              f"λ={parameter_results['lambda']:.4f}")
        
        # 4. 阶段检测
        print("\n[4/6] 检测反身性阶段...")
        stage_results = self.stage_detector.detect_stage(
            parameters=parameter_results['parameters'],
            price_data=price_data,
            fundamental_data=fundamental_data,
            lambda_value=parameter_results['lambda'],
            **kwargs
        )
        print(f"✓ 阶段检测完成: {stage_results['stage']} (置信度: {stage_results['confidence']:.2%})")
        
        # 5. 生成结论
        print("\n[5/6] 生成分析结论...")
        conclusion = self.conclusion_generator.generate(
            parameters=parameter_results['parameters'],
            stage_result=stage_results,
            fit_results=fit_results,
            **kwargs
        )
        print("✓ 结论生成完成")
        
        # 6. 生成图表
        print("\n[6/6] 生成可视化图表...")
        charts = {}
        
        # 拟合图表
        fit_chart = self.chart_visualizer.visualize_fit(
            price_data=price_data,
            fundamental_data=fundamental_data,
            fit_results=fit_results,
            **kwargs
        )
        
        # 对比图表（如果有预测数据）
        if 'predicted_data' in parameter_results:
            comparison_chart = self.chart_visualizer.visualize_comparison(
                actual_data={
                    'price': price_data,
                    'fundamental': fundamental_data
                },
                predicted_data=parameter_results['predicted_data'],
                **kwargs
            )
            charts['comparison'] = self.chart_visualizer.chart_to_base64(comparison_chart)
        
        charts['fit'] = self.chart_visualizer.chart_to_base64(fit_chart)
        
        # 保存图表（如果需要）
        if kwargs.get('save_charts', False):
            save_path = kwargs.get('chart_save_path', 'results')
            import os
            os.makedirs(save_path, exist_ok=True)
            
            chart_path = os.path.join(save_path, f'{stock_code}_fit_chart.png')
            self.chart_visualizer.save_chart(fit_chart, chart_path)
            charts['fit_path'] = chart_path
            
            if 'comparison' in charts:
                comparison_path = os.path.join(save_path, f'{stock_code}_comparison_chart.png')
                comparison_chart = self.chart_visualizer.visualize_comparison(
                    actual_data={
                        'price': price_data,
                        'fundamental': fundamental_data
                    },
                    predicted_data=parameter_results['predicted_data'],
                    **kwargs
                )
                self.chart_visualizer.save_chart(comparison_chart, comparison_path)
                charts['comparison_path'] = comparison_path
        
        print("✓ 图表生成完成")
        
        # 组装结果
        results = {
            'data_info': data_info,
            'fit_results': fit_results,
            'parameter_results': parameter_results,
            'stage_results': stage_results,
            'conclusion': conclusion,
            'charts': charts
        }
        
        print("\n" + "=" * 60)
        print("反身性分析完成！")
        print("=" * 60)
        
        return results
    
    def analyze_from_dataframe(self,
                               df,
                               **kwargs) -> Dict[str, Any]:
        """
        从 DataFrame 执行分析（使用 DataFrameDataProvider）
        
        Args:
            df: 包含 'P_t' 和 'F_t' 列的 DataFrame
            **kwargs: 其他参数
        
        Returns:
            分析结果字典
        """
        # 创建 DataFrame 数据提供者
        df_provider = DataFrameDataProvider(df)
        
        # 临时替换数据提供者
        original_provider = self.data_provider
        self.data_provider = df_provider
        
        try:
            results = self.analyze(stock_code='', **kwargs)
            return results
        finally:
            # 恢复原始数据提供者
            self.data_provider = original_provider
    
    def get_component_info(self) -> Dict[str, str]:
        """
        获取各组件信息（用于调试和展示）
        
        Returns:
            组件信息字典
        """
        return {
            'data_provider': type(self.data_provider).__name__,
            'chart_fitter': type(self.chart_fitter).__name__,
            'parameter_estimator': type(self.parameter_estimator).__name__,
            'stage_detector': type(self.stage_detector).__name__,
            'conclusion_generator': type(self.conclusion_generator).__name__,
            'chart_visualizer': type(self.chart_visualizer).__name__,
        }

