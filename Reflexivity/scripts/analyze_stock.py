"""
股票参数反推分析脚本
完整流程：获取数据 -> 估计参数 -> 可视化结果
"""

import os
import sys
import pandas as pd
import numpy as np
from tools.data_fetchers.data_fetcher import TushareDataFetcher
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.parameter_estimator import estimate_from_stock_data, ParameterEstimator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def analyze_stock(stock_name: str = "平安银行",
                 lookback_weeks: int = 120,
                 token: str = None,
                 save_dir: str = "results") -> None:
    """
    分析股票并反推参数
    
    Args:
        stock_name: 股票名称
        lookback_weeks: 回溯周数（固定窗口）
        token: Tushare Token，如果为None则从环境变量读取
        save_dir: 结果保存目录
    """
    print("="*60)
    print("  股票参数反推分析")
    print("="*60)
    
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 步骤1: 获取数据
        print("\n【步骤1】获取股票数据...")
        print("-" * 60)
        
        fetcher = TushareDataFetcher(token=token)
        df, ts_code = fetcher.fetch_complete_data(stock_name, lookback_weeks)
        
        # 保存原始数据
        data_file = os.path.join(save_dir, f"{stock_name}_data.csv")
        df.to_csv(data_file, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存至: {data_file}")
        
        # 数据预览
        print(f"\n数据统计:")
        print(f"  时间范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"  数据点数: {len(df)}")
        print(f"  价格范围: [{df['P_t'].min():.2f}, {df['P_t'].max():.2f}]")
        print(f"  EPS范围: [{df['F_t'].min():.4f}, {df['F_t'].max():.4f}]")
        
        # 检查数据质量
        if len(df) < 20:
            print("\n⚠️ 警告: 数据点太少，可能影响估计精度")
        
        # 步骤2: 参数估计
        print("\n【步骤2】参数反推估计...")
        print("-" * 60)
        
        results = estimate_from_stock_data(df, method='differential_evolution')
        
        # 保存结果
        results_file = os.path.join(save_dir, f"{stock_name}_results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"  股票: {stock_name} ({ts_code})\n")
            f.write(f"  回溯周数: {lookback_weeks}\n")
            f.write("="*60 + "\n\n")
            
            f.write("估计参数:\n")
            f.write(f"  α (价格在认知中的权重) = {results['parameters']['alpha']:.6f}\n")
            f.write(f"  γ (价格调整速度) = {results['parameters']['gamma']:.6f}\n")
            f.write(f"  β (价格对基本面的影响) = {results['parameters']['beta']:.6f}\n\n")
            
            f.write(f"系统特征值: λ = {results['lambda']:.6f}\n")
            f.write(f"稳定性: {results['stability']}\n\n")
            
            f.write("拟合效果:\n")
            f.write(f"  R² = {results['fitness']['r_squared']:.6f}\n")
            f.write(f"  RMSE = {results['fitness']['rmse']:.4f}\n")
            f.write(f"  MAE = {results['fitness']['mae']:.4f}\n")
            f.write(f"  MSE = {results['fitness']['mse']:.4f}\n")
        
        print(f"✓ 结果已保存至: {results_file}")
        
        # 步骤3: 可视化
        print("\n【步骤3】生成可视化图表...")
        print("-" * 60)
        
        chart_file = os.path.join(save_dir, f"{stock_name}_chart.png")
        estimator = ParameterEstimator(
            df['P_t'].values, 
            df['F_t'].values
        )
        estimator.plot_results(results, save_path=chart_file)
        print(f"✓ 图表已保存至: {chart_file}")
        
        # 步骤4: 绘制历史价格曲线
        print("\n【步骤4】生成历史价格曲线...")
        print("-" * 60)
        
        price_chart_file = os.path.join(save_dir, f"{stock_name}_price_history.png")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['date'], df['P_t'], 'b-', linewidth=2, label='收盘价')
        ax.fill_between(df['date'], df['P_t'], alpha=0.3)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('价格', fontsize=12)
        ax.set_title(f'{stock_name} ({ts_code}) 历史价格曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(price_chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 价格曲线已保存至: {price_chart_file}")
        
        # 总结
        print("\n" + "="*60)
        print("  分析完成!")
        print("="*60)
        print(f"\n结果摘要:")
        print(f"  股票: {stock_name} ({ts_code})")
        print(f"  估计参数: α={results['parameters']['alpha']:.4f}, "
              f"γ={results['parameters']['gamma']:.4f}, "
              f"β={results['parameters']['beta']:.4f}")
        print(f"  特征值: λ={results['lambda']:.4f} ({results['stability']})")
        print(f"  拟合效果: R²={results['fitness']['r_squared']:.4f}, "
              f"RMSE={results['fitness']['rmse']:.2f}")
        print(f"\n所有结果已保存至目录: {save_dir}/")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='股票参数反推分析')
    parser.add_argument('--stock', type=str, default='平安银行',
                       help='股票名称（默认: 平安银行）')
    parser.add_argument('--weeks', type=int, default=120,
                       help='回溯周数（默认: 120）')
    parser.add_argument('--token', type=str, default=None,
                       help='Tushare Token（默认: 从环境变量读取）')
    parser.add_argument('--output', type=str, default='results',
                       help='结果保存目录（默认: results）')
    
    args = parser.parse_args()
    
    analyze_stock(
        stock_name=args.stock,
        lookback_weeks=args.weeks,
        token=args.token,
        save_dir=args.output
    )


if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    if len(sys.argv) == 1:
        print("\n使用示例:")
        print("  python analyze_stock.py --stock 平安银行 --weeks 120")
        print("\n或使用命令行参数:")
        print("  python analyze_stock.py --help")
        print("\n开始使用默认参数分析...\n")
        
        # 检查Token
        token = os.getenv('TUSHARE_TOKEN')
        if token is None:
            print("⚠️ 警告: 未设置TUSHARE_TOKEN环境变量")
            print("请设置环境变量或使用 --token 参数")
            print("参考文档: tushare_guide.md\n")
        
        analyze_stock()
    else:
        main()

