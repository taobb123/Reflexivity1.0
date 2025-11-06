"""
交互式功能展示脚本
展示系统中各个功能的使用方法
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_1_data_provider():
    """演示：数据提供者功能"""
    print("\n" + "=" * 60)
    print("功能演示 1: 数据提供者")
    print("=" * 60)
    
    from apps.components.data_providers import DataFrameDataProvider
    import pandas as pd
    import numpy as np
    
    # 创建模拟数据
    np.random.seed(42)
    T = 50
    prices = 100 + 0.5 * np.arange(T) + np.random.randn(T) * 2
    fundamentals = 100 + 0.3 * np.arange(T) + np.random.randn(T) * 1.5
    
    df = pd.DataFrame({
        'P_t': prices,
        'F_t': fundamentals
    })
    
    # 使用 DataFrame 数据提供者
    provider = DataFrameDataProvider(df)
    
    print("\n✓ 从 DataFrame 创建数据提供者")
    print(f"✓ 数据点数量: {len(df)}")
    
    # 获取数据
    price_data = provider.get_price_data()
    fundamental_data = provider.get_fundamental_data()
    
    print(f"✓ 价格数据范围: [{price_data.min():.2f}, {price_data.max():.2f}]")
    print(f"✓ 基本面数据范围: [{fundamental_data.min():.2f}, {fundamental_data.max():.2f}]")
    
    # 获取信息
    info = provider.get_data_info()
    print(f"✓ 数据信息: {info}")
    
    return provider, price_data, fundamental_data


def demo_2_chart_fitting(price_data, fundamental_data):
    """演示：图表拟合功能"""
    print("\n" + "=" * 60)
    print("功能演示 2: 图表拟合")
    print("=" * 60)
    
    from apps.components.chart_fitters import PriceFundamentalChartFitter
    from apps.components.fitters import LinearFitter
    
    # 使用线性拟合器
    print("\n使用线性拟合器...")
    linear_fitter = LinearFitter()
    chart_fitter = PriceFundamentalChartFitter(fitter=linear_fitter)
    
    results = chart_fitter.fit_charts(price_data, fundamental_data)
    
    print(f"✓ 拟合完成")
    print(f"  - 拟合方法: {results['fit_metrics']['fit_type']}")
    print(f"  - R² (决定系数): {results['fit_metrics']['r_squared']:.4f}")
    print(f"  - 拟合质量: {results['fit_metrics']['fit_quality']}")
    print(f"  - 相关性: {results['correlation']['value']:.4f}")
    print(f"  - 相关性显著性: {'是' if results['correlation']['significant'] else '否'}")
    print(f"  - 平均背离: {results['divergence']['mean']:.4f}")
    print(f"  - 背离标准差: {results['divergence']['std']:.4f}")
    
    return results


def demo_3_parameter_estimation(price_data, fundamental_data):
    """演示：参数估计功能"""
    print("\n" + "=" * 60)
    print("功能演示 3: 参数估计")
    print("=" * 60)
    
    from apps.components.parameter_estimators import ReflexivityParameterEstimator
    
    print("\n估计反身性模型参数...")
    estimator = ReflexivityParameterEstimator()
    
    results = estimator.estimate(
        price_data=price_data,
        fundamental_data=fundamental_data,
        method='differential_evolution'
    )
    
    params = results['parameters']
    print(f"\n✓ 参数估计完成")
    print(f"  - α (价格在认知中的权重): {params['alpha']:.4f}")
    print(f"  - γ (价格调整速度): {params['gamma']:.4f}")
    print(f"  - β (价格对基本面的影响): {params['beta']:.4f}")
    print(f"  - λ (系统特征值): {results['lambda']:.4f}")
    print(f"  - |λ|: {abs(results['lambda']):.4f}")
    print(f"  - 稳定性: {results['stability']}")
    
    fitness = results['fitness']
    print(f"\n拟合效果:")
    print(f"  - R²: {fitness['r_squared']:.4f}")
    print(f"  - RMSE: {fitness['rmse']:.4f}")
    print(f"  - MAE: {fitness['mae']:.4f}")
    
    return results


def demo_4_stage_detection(price_data, fundamental_data, param_results):
    """演示：阶段检测功能"""
    print("\n" + "=" * 60)
    print("功能演示 4: 阶段检测")
    print("=" * 60)
    
    from apps.components.stage_detectors import ComprehensiveStageDetector
    
    print("\n检测反身性阶段...")
    detector = ComprehensiveStageDetector()
    
    # 获取所有可用阶段
    available_stages = detector.get_available_stages()
    print(f"\n✓ 可用阶段: {', '.join(available_stages)}")
    
    # 执行阶段检测
    stage_results = detector.detect_stage(
        parameters=param_results['parameters'],
        price_data=price_data,
        fundamental_data=fundamental_data,
        lambda_value=param_results['lambda']
    )
    
    print(f"\n✓ 阶段检测完成")
    print(f"  - 检测到的阶段: {stage_results['stage']}")
    print(f"  - 置信度: {stage_results['confidence']:.2%}")
    print(f"  - 风险等级: {stage_results['risk_level']}")
    print(f"\n阶段描述:")
    print(f"  {stage_results['description']}")
    
    print(f"\n各项指标:")
    for key, value in stage_results['indicators'].items():
        print(f"  - {key}: {value}")
    
    print(f"\n各阶段匹配分数（按分数排序）:")
    for stage, score in sorted(stage_results['stage_scores'].items(), 
                              key=lambda x: x[1], reverse=True):
        print(f"  - {stage}: {score:.2%}")
    
    return stage_results


def demo_5_conclusion_generation(param_results, stage_results, fit_results):
    """演示：结论生成功能"""
    print("\n" + "=" * 60)
    print("功能演示 5: 结论生成")
    print("=" * 60)
    
    from apps.components.conclusion_generators import ChineseConclusionGenerator
    
    print("\n生成分析结论...")
    generator = ChineseConclusionGenerator()
    
    conclusion = generator.generate(
        parameters=param_results['parameters'],
        stage_result=stage_results,
        fit_results=fit_results
    )
    
    print("\n✓ 结论生成完成")
    print("\n" + "=" * 60)
    print("分析结论:")
    print("=" * 60)
    print(conclusion)
    
    return conclusion


def demo_6_visualization(price_data, fundamental_data, fit_results):
    """演示：图表可视化功能"""
    print("\n" + "=" * 60)
    print("功能演示 6: 图表可视化")
    print("=" * 60)
    
    from apps.components.chart_visualizers import MatplotlibChartVisualizer
    
    print("\n生成可视化图表...")
    visualizer = MatplotlibChartVisualizer()
    
    # 生成拟合图表
    fig = visualizer.visualize_fit(
        price_data=price_data,
        fundamental_data=fundamental_data,
        fit_results=fit_results
    )
    
    # 保存图表
    save_path = 'demo_fit_chart.png'
    visualizer.save_chart(fig, save_path)
    print(f"✓ 图表已保存至: {save_path}")
    
    # 转换为 base64
    chart_base64 = visualizer.chart_to_base64(fig)
    print(f"✓ 图表已转换为 base64 (长度: {len(chart_base64)} 字符)")
    
    return save_path


def demo_7_complete_analysis():
    """演示：完整分析流程（使用主协调器）"""
    print("\n" + "=" * 60)
    print("功能演示 7: 完整分析流程（使用主协调器）")
    print("=" * 60)
    
    from apps.reflexivity_analyzer import ReflexivityAnalyzer
    from apps.components.data_providers import DataFrameDataProvider
    import pandas as pd
    import numpy as np
    
    # 创建模拟数据
    np.random.seed(42)
    T = 50
    prices = 100 + 0.5 * np.arange(T) + np.random.randn(T) * 2
    fundamentals = 100 + 0.3 * np.arange(T) + np.random.randn(T) * 1.5
    
    df = pd.DataFrame({
        'P_t': prices,
        'F_t': fundamentals
    })
    
    # 使用主协调器
    print("\n使用 ReflexivityAnalyzer 进行完整分析...")
    analyzer = ReflexivityAnalyzer()
    
    results = analyzer.analyze_from_dataframe(df)
    
    print("\n✓ 完整分析完成")
    print(f"\n组件信息:")
    for component, impl in analyzer.get_component_info().items():
        print(f"  - {component}: {impl}")
    
    print(f"\n分析结果摘要:")
    print(f"  - 阶段: {results['stage_results']['stage']}")
    print(f"  - 置信度: {results['stage_results']['confidence']:.2%}")
    params = results['parameter_results']['parameters']
    print(f"  - 参数: α={params['alpha']:.4f}, γ={params['gamma']:.4f}, β={params['beta']:.4f}")
    print(f"  - λ: {results['parameter_results']['lambda']:.4f}")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("股票反身性分析系统 - 功能演示")
    print("=" * 60)
    print("\n本脚本将演示系统的各个核心功能")
    print("所有演示使用模拟数据，无需真实股票数据源")
    print("=" * 60)
    
    # 演示1: 数据提供者
    provider, price_data, fundamental_data = demo_1_data_provider()
    
    # 演示2: 图表拟合
    fit_results = demo_2_chart_fitting(price_data, fundamental_data)
    
    # 演示3: 参数估计
    param_results = demo_3_parameter_estimation(price_data, fundamental_data)
    
    # 演示4: 阶段检测
    stage_results = demo_4_stage_detection(price_data, fundamental_data, param_results)
    
    # 演示5: 结论生成
    conclusion = demo_5_conclusion_generation(param_results, stage_results, fit_results)
    
    # 演示6: 图表可视化
    chart_path = demo_6_visualization(price_data, fundamental_data, fit_results)
    
    # 演示7: 完整分析流程
    complete_results = demo_7_complete_analysis()
    
    print("\n" + "=" * 60)
    print("所有功能演示完成！")
    print("=" * 60)
    print(f"\n✓ 图表已保存: {chart_path}")
    print("\n提示:")
    print("  - 查看详细使用指南: apps/USAGE_GUIDE.md")
    print("  - 查看快速参考: apps/QUICK_REFERENCE.md")
    print("  - 查看使用示例: apps/example_usage.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n演示已取消")
    except Exception as e:
        print(f"\n\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


