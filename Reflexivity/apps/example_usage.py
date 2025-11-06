"""
反身性分析器使用示例
展示如何使用接口化设计进行股票反身性分析
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.reflexivity_analyzer import ReflexivityAnalyzer
from apps.components.data_providers import UnifiedDataProvider, DataFrameDataProvider
from apps.components.chart_fitters import PriceFundamentalChartFitter, BidirectionalChartFitter
from apps.components.fitters import LinearFitter, PolynomialFitter
from apps.components.optimizers import DifferentialEvolutionOptimizer
from apps.components.parameter_estimators import ReflexivityParameterEstimator
from apps.components.stage_detectors import ComprehensiveStageDetector
from apps.components.conclusion_generators import ChineseConclusionGenerator
from apps.components.chart_visualizers import MatplotlibChartVisualizer


def example_1_basic_usage():
    """示例1：基本使用（使用默认组件）"""
    print("\n" + "=" * 60)
    print("示例1：基本使用（使用默认组件）")
    print("=" * 60)
    
    # 创建分析器（使用所有默认组件）
    analyzer = ReflexivityAnalyzer()
    
    # 执行分析
    results = analyzer.analyze(
        stock_code="平安银行",
        lookback_weeks=120,
        save_charts=True,
        chart_save_path="results"
    )
    
    # 打印结论
    print("\n分析结论：")
    print(results['conclusion'])
    
    return results


def example_2_custom_components():
    """示例2：自定义组件（展示接口编程和组合模式的灵活性）"""
    print("\n" + "=" * 60)
    print("示例2：自定义组件（展示接口编程和组合模式的灵活性）")
    print("=" * 60)
    
    # 创建自定义组件
    # 使用多项式拟合器
    polynomial_fitter = PolynomialFitter(degree=2)
    custom_chart_fitter = PriceFundamentalChartFitter(fitter=polynomial_fitter)
    
    # 使用差分进化优化器
    custom_optimizer = DifferentialEvolutionOptimizer()
    custom_parameter_estimator = ReflexivityParameterEstimator(optimizer=custom_optimizer)
    
    # 创建自定义分析器（组合这些组件）
    analyzer = ReflexivityAnalyzer(
        chart_fitter=custom_chart_fitter,
        parameter_estimator=custom_parameter_estimator
    )
    
    # 执行分析
    results = analyzer.analyze(
        stock_code="平安银行",
        lookback_weeks=120
    )
    
    # 显示组件信息
    print("\n使用的组件：")
    for component, impl in analyzer.get_component_info().items():
        print(f"  {component}: {impl}")
    
    # 打印结论
    print("\n分析结论：")
    print(results['conclusion'])
    
    return results


def example_3_from_dataframe():
    """示例3：从DataFrame分析（展示数据接口的灵活性）"""
    print("\n" + "=" * 60)
    print("示例3：从DataFrame分析（展示数据接口的灵活性）")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    
    # 创建示例数据
    np.random.seed(42)
    T = 100
    t = np.arange(T)
    
    # 生成模拟的价格和基本面数据
    P_t = 100 + 0.5 * t + np.random.randn(T) * 2
    F_t = 100 + 0.3 * t + np.random.randn(T) * 1.5
    
    df = pd.DataFrame({
        'P_t': P_t,
        'F_t': F_t
    })
    
    # 使用DataFrame数据提供者
    analyzer = ReflexivityAnalyzer()
    
    # 从DataFrame分析
    results = analyzer.analyze_from_dataframe(df)
    
    # 打印结论
    print("\n分析结论：")
    print(results['conclusion'])
    
    return results


def example_4_custom_data_source():
    """示例4：自定义数据源（展示数据提供者接口的灵活性）"""
    print("\n" + "=" * 60)
    print("示例4：自定义数据源（展示数据提供者接口的灵活性）")
    print("=" * 60)
    
    # 创建自定义数据提供者（可以指定token和数据源）
    data_provider = UnifiedDataProvider(
        tushare_token=None,  # 可以设置你的token
        preferred_sources=['akshare']  # 优先使用akshare
    )
    
    # 创建分析器
    analyzer = ReflexivityAnalyzer(data_provider=data_provider)
    
    # 执行分析
    results = analyzer.analyze(
        stock_code="平安银行",
        lookback_weeks=120
    )
    
    # 显示数据信息
    print("\n数据信息：")
    for key, value in results['data_info'].items():
        print(f"  {key}: {value}")
    
    return results


def example_5_stage_detection():
    """示例5：阶段检测详细分析"""
    print("\n" + "=" * 60)
    print("示例5：阶段检测详细分析")
    print("=" * 60)
    
    analyzer = ReflexivityAnalyzer()
    results = analyzer.analyze(
        stock_code="平安银行",
        lookback_weeks=120
    )
    
    # 显示阶段检测结果
    stage_results = results['stage_results']
    print(f"\n检测到的阶段: {stage_results['stage']}")
    print(f"置信度: {stage_results['confidence']:.2%}")
    print(f"风险等级: {stage_results['risk_level']}")
    print(f"\n阶段描述:\n{stage_results['description']}")
    
    print("\n各项指标：")
    for key, value in stage_results['indicators'].items():
        print(f"  {key}: {value}")
    
    print("\n各阶段匹配分数：")
    for stage, score in sorted(stage_results['stage_scores'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"  {stage}: {score:.2%}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("反身性分析器使用示例")
    print("=" * 60)
    print("\n本示例展示了如何使用接口化设计的反身性分析系统")
    print("系统特点：")
    print("  - 使用接口编程，易于扩展")
    print("  - 使用对象组合，灵活配置")
    print("  - 支持多种数据源")
    print("  - 支持多种拟合方法")
    print("  - 支持多种优化算法")
    print("=" * 60)
    
    # 运行示例（根据需要取消注释）
    # example_1_basic_usage()  # 需要真实股票数据
    # example_2_custom_components()  # 需要真实股票数据
    example_3_from_dataframe()  # 使用模拟数据，无需真实数据源
    # example_4_custom_data_source()  # 需要真实股票数据
    # example_5_stage_detection()  # 需要真实股票数据
    
    print("\n提示：取消注释上面的示例函数来运行不同的示例")

