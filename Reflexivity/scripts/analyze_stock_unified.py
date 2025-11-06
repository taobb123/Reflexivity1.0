"""
股票参数反推分析脚本 - 使用统一多数据源获取器
支持自动回退机制,可以混合使用多个金融API
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.parameter_estimator import estimate_from_stock_data, ParameterEstimator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def analyze_with_unified_fetcher(stock_name: str = "平安银行",
                                 lookback_weeks: int = 120,
                                 save_dir: str = "results",
                                 tushare_token: str = None,
                                 preferred_sources: list = None) -> None:
    """
    使用统一多数据源获取器进行分析(A股专用)
    
    Args:
        stock_name: 股票名称或代码
        lookback_weeks: 回溯周数
        save_dir: 结果保存目录
        tushare_token: Tushare Token
        preferred_sources: 优先使用的数据源列表,如['tushare', 'yfinance', 'pandas_datareader', 'akshare']
    """
    print("="*60)
    print("  股票参数反推分析 (统一多数据源)")
    print("="*60)
    
    try:
        from tools.data_fetchers.data_fetcher_unified import UnifiedDataFetcher
        
        print("\n【步骤1】初始化数据获取器...")
        print("-" * 60)
        
        # 从环境变量读取API Key(如果未提供)
        if tushare_token is None:
            tushare_token = os.getenv('TUSHARE_TOKEN')
        
        # 初始化统一数据获取器(A股专用)
        fetcher = UnifiedDataFetcher(
            tushare_token=tushare_token,
            preferred_sources=preferred_sources
        )
        
        print(f"\n可用数据源: {list(fetcher.fetchers.keys())}")
        
        print("\n【步骤2】获取股票数据...")
        print("-" * 60)
        
        # 获取数据(自动回退)
        df, stock_code, sources_info = fetcher.fetch_complete_data(
            stock_name, lookback_weeks=lookback_weeks
        )
        
        # 保存原始数据
        os.makedirs(save_dir, exist_ok=True)
        data_file = os.path.join(save_dir, f"{stock_name}_unified_data.csv")
        df.to_csv(data_file, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存至: {data_file}")
        
        # 数据预览
        print(f"\n数据统计:")
        print(f"  时间范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"  数据点数: {len(df)}")
        print(f"  价格范围: [{df['P_t'].min():.2f}, {df['P_t'].max():.2f}]")
        print(f"  基本面范围: [{df['F_t'].min():.4f}, {df['F_t'].max():.4f}]")
        print(f"  价格数据源: {sources_info['price_source']}")
        print(f"  财务数据源: {sources_info['finance_source']}")
        if 'MA5' in df.columns:
            print(f"  包含均线数据: MA5, MA10, MA20, MA60")
        
        # 检查数据质量
        if len(df) < 20:
            print("\n⚠️ 警告: 数据点太少,可能影响估计精度")
        
        # 继续参数估计
        run_parameter_estimation(df, stock_name, stock_code, 
                               sources_info, save_dir)
        
    except ImportError as e:
        print(f"❌ 导入失败: {str(e)}")
        print("\n请安装必要的依赖:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()


def run_parameter_estimation(df: pd.DataFrame,
                             stock_name: str,
                             stock_code: str,
                             sources_info: dict,
                             save_dir: str) -> None:
    """运行参数估计"""
    print(f"\n【步骤3】参数反推估计...")
    print("-" * 60)
    
    # 参数估计
    results = estimate_from_stock_data(df, method='differential_evolution')
    
    # 生成数据源标识
    data_source = f"{sources_info['price_source']}_{sources_info['finance_source']}"
    
    # 保存结果
    results_file = os.path.join(save_dir, f"{stock_name}_unified_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"  股票: {stock_name} ({stock_code})\n")
        f.write(f"  数据源: 统一多数据源 (价格: {sources_info['price_source']}, "
               f"财务: {sources_info['finance_source']})\n")
        f.write("="*60 + "\n\n")
        
        f.write("估计参数:\n")
        f.write(f"  α = {results['parameters']['alpha']:.6f}\n")
        f.write(f"  γ = {results['parameters']['gamma']:.6f}\n")
        f.write(f"  β = {results['parameters']['beta']:.6f}\n\n")
        
        f.write(f"系统特征值: λ = {results['lambda']:.6f}\n")
        f.write(f"稳定性: {results['stability']}\n\n")
        
        f.write("拟合效果:\n")
        f.write(f"  R² = {results['fitness']['r_squared']:.6f}\n")
        f.write(f"  RMSE = {results['fitness']['rmse']:.4f}\n")
        f.write(f"  MAE = {results['fitness']['mae']:.4f}\n")
    
    print(f"✓ 结果已保存至: {results_file}")
    
    # 可视化
    print(f"\n【步骤4】生成可视化图表...")
    print("-" * 60)
    
    chart_file = os.path.join(save_dir, f"{stock_name}_unified_chart.png")
    estimator = ParameterEstimator(df['P_t'].values, df['F_t'].values)
    estimator.plot_results(results, save_path=chart_file)
    print(f"✓ 图表已保存至: {chart_file}")
    
    # 历史价格曲线
    price_chart_file = os.path.join(save_dir, f"{stock_name}_unified_price_history.png")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['date'], df['P_t'], 'b-', linewidth=2, label='收盘价')
    ax.fill_between(df['date'], df['P_t'], alpha=0.3)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('价格', fontsize=12)
    ax.set_title(f'{stock_name} ({stock_code}) 历史价格曲线 - 统一数据源', 
                fontsize=14, fontweight='bold')
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
    print(f"  股票: {stock_name} ({stock_code})")
    print(f"  价格数据源: {sources_info['price_source']}")
    print(f"  财务数据源: {sources_info['finance_source']}")
    print(f"  估计参数: α={results['parameters']['alpha']:.4f}, "
          f"γ={results['parameters']['gamma']:.4f}, "
          f"β={results['parameters']['beta']:.4f}")
    print(f"  特征值: λ={results['lambda']:.4f} ({results['stability']})")
    print(f"  拟合效果: R²={results['fitness']['r_squared']:.4f}, "
          f"RMSE={results['fitness']['rmse']:.2f}")
    print(f"\n所有结果已保存至目录: {save_dir}/")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='股票参数反推分析(统一多数据源,支持自动回退)'
    )
    parser.add_argument('--stock', type=str, default='平安银行',
                       help='股票名称或代码(默认: 平安银行)')
    parser.add_argument('--weeks', type=int, default=120,
                       help='回溯周数(默认: 120)')
    parser.add_argument('--tushare-token', type=str, default=None,
                       help='Tushare Token(也可设置环境变量TUSHARE_TOKEN)')
    parser.add_argument('--preferred-sources', type=str, nargs='+',
                       default=None,
                       help='优先使用的数据源列表,如: tushare yfinance pandas_datareader akshare')
    parser.add_argument('--output', type=str, default='results',
                       help='结果保存目录(默认: results)')
    
    args = parser.parse_args()
    
    # 解析优先数据源
    preferred = args.preferred_sources
    
    analyze_with_unified_fetcher(
        stock_name=args.stock,
        lookback_weeks=args.weeks,
        save_dir=args.output,
        tushare_token=args.tushare_token,
        preferred_sources=preferred
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 默认使用统一数据获取器
        print("="*60)
        print("  股票参数反推分析(统一多数据源)")
        print("="*60)
        print("\n使用示例:")
        print("  python analyze_stock_unified.py --stock 平安银行 --weeks 120")
        print("  python analyze_stock_unified.py --stock 平安银行 --preferred-sources tushare akshare")
        print("  python analyze_stock_unified.py --stock 000001 --tushare-token your_token")
        print("\n提示:")
        print("  - 系统会自动回退到可用的数据源")
        print("  - 支持的数据源: Tushare, yfinance, pandas-datareader, AKShare")
        print("  - 所有数据源都支持A股数据获取")
        print("  - 包含均线数据: MA5, MA10, MA20, MA60")
        print("  - 可以通过环境变量设置Tushare Token,无需命令行参数")
        print("\n开始使用默认参数分析...\n")
        analyze_with_unified_fetcher()
    else:
        main()

