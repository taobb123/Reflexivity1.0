"""
股票参数反推分析脚本 - 支持多种数据源
优先使用AKShare（免费），可选Tushare（需要Token）
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


def analyze_with_akshare(stock_name: str = "贵州茅台",
                        lookback_weeks: int = 120,
                        save_dir: str = "results") -> None:
    """使用AKShare获取数据并分析"""
    print("="*60)
    print("  使用AKShare数据源")
    print("="*60)
    
    try:
        from tools.data_fetchers.data_fetcher_akshare import AKShareDataFetcher
        
        print("\n【步骤1】获取股票数据（AKShare）...")
        print("-" * 60)
        
        fetcher = AKShareDataFetcher()
        df, ts_code = fetcher.fetch_complete_data(stock_name, lookback_weeks)
        
        # 保存原始数据
        os.makedirs(save_dir, exist_ok=True)
        data_file = os.path.join(save_dir, f"{stock_name}_akshare_data.csv")
        df.to_csv(data_file, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存至: {data_file}")
        
        # 继续参数估计
        run_parameter_estimation(df, stock_name, ts_code, "AKShare", save_dir)
        
    except ImportError:
        print("❌ AKShare未安装，请运行: pip install akshare")
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()


def analyze_with_tushare(stock_name: str = "300274",
                        lookback_weeks: int = 120,
                        token: str = None,
                        save_dir: str = "results") -> None:
    """使用Tushare获取数据并分析"""
    print("="*60)
    print("  使用Tushare数据源")
    print("="*60)
    
    try:
        from tools.data_fetchers.data_fetcher import TushareDataFetcher
        
        print("\n【步骤1】获取股票数据（Tushare）...")
        print("-" * 60)
        
        fetcher = TushareDataFetcher(token=token)
        df, ts_code = fetcher.fetch_complete_data(stock_name, lookback_weeks)
        
        # 保存原始数据
        os.makedirs(save_dir, exist_ok=True)
        data_file = os.path.join(save_dir, f"{stock_name}_tushare_data.csv")
        df.to_csv(data_file, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存至: {data_file}")
        
        # 继续参数估计
        run_parameter_estimation(df, stock_name, ts_code, "Tushare", save_dir)
        
    except ImportError:
        print("❌ Tushare未安装，请运行: pip install tushare")
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()


def analyze_with_hybrid(stock_name: str = "平安银行",
                       lookback_weeks: int = 120,
                       token: str = None,
                       save_dir: str = "results") -> None:
    """使用混合模式（AKShare+baostock）获取数据并分析"""
    print("="*60)
    print("  使用混合数据源 (AKShare + baostock)")
    print("  - 价格数据: AKShare (免费)")
    print("  - 财务数据: AKShare (优先), baostock (备用)")
    print("="*60)
    
    try:
        from tools.data_fetchers.data_fetcher_hybrid import HybridDataFetcher
        
        print("\n【步骤1】获取股票数据（混合模式）...")
        print("-" * 60)
        
        fetcher = HybridDataFetcher()
        df, stock_code = fetcher.fetch_complete_data(stock_name, lookback_weeks)
        
        # 保存原始数据
        os.makedirs(save_dir, exist_ok=True)
        data_file = os.path.join(save_dir, f"{stock_name}_hybrid_data.csv")
        df.to_csv(data_file, index=False, encoding='utf-8-sig')
        print(f"✓ 数据已保存至: {data_file}")
        
        # 继续参数估计
        run_parameter_estimation(df, stock_name, stock_code, "Hybrid", save_dir)
        
    except ImportError as e:
        print(f"❌ 导入失败: {str(e)}")
        print("请确保已安装: pip install akshare tushare")
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()


def run_parameter_estimation(df: pd.DataFrame,
                             stock_name: str,
                             stock_code: str,
                             data_source: str,
                             save_dir: str) -> None:
    """运行参数估计"""
    print(f"\n【步骤2】参数反推估计...")
    print("-" * 60)
    
    # 数据预览
    print(f"\n数据统计:")
    print(f"  时间范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"  数据点数: {len(df)}")
    print(f"  价格范围: [{df['P_t'].min():.2f}, {df['P_t'].max():.2f}]")
    print(f"  基本面范围: [{df['F_t'].min():.4f}, {df['F_t'].max():.4f}]")
    
    if len(df) < 20:
        print("\n⚠️ 警告: 数据点太少，可能影响估计精度")
    
    # 参数估计
    results = estimate_from_stock_data(df, method='differential_evolution')
    
    # 保存结果
    results_file = os.path.join(save_dir, f"{stock_name}_{data_source.lower()}_results.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"  股票: {stock_name} ({stock_code})\n")
        f.write(f"  数据源: {data_source}\n")
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
    print(f"\n【步骤3】生成可视化图表...")
    print("-" * 60)
    
    chart_file = os.path.join(save_dir, f"{stock_name}_{data_source.lower()}_chart.png")
    estimator = ParameterEstimator(df['P_t'].values, df['F_t'].values)
    estimator.plot_results(results, save_path=chart_file)
    print(f"✓ 图表已保存至: {chart_file}")
    
    # 历史价格曲线
    price_chart_file = os.path.join(save_dir, f"{stock_name}_{data_source.lower()}_price_history.png")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['date'], df['P_t'], 'b-', linewidth=2, label='收盘价')
    ax.fill_between(df['date'], df['P_t'], alpha=0.3)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('价格', fontsize=12)
    ax.set_title(f'{stock_name} ({stock_code}) 历史价格曲线 - {data_source}', 
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
    print(f"  数据源: {data_source}")
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
    
    parser = argparse.ArgumentParser(description='股票参数反推分析（支持多数据源）')
    parser.add_argument('--stock', type=str, default='平安银行',
                       help='股票名称（默认: 平安银行）')
    parser.add_argument('--weeks', type=int, default=120,
                       help='回溯周数（默认: 120）')
    parser.add_argument('--source', type=str, choices=['akshare', 'tushare', 'hybrid', 'auto'],
                       default='auto',
                       help='数据源: akshare(免费), tushare(需Token), hybrid(AKShare+baostock), auto(自动选择)')
    parser.add_argument('--token', type=str, default=None,
                       help='Tushare Token（如果使用Tushare，混合模式不需要）')
    parser.add_argument('--output', type=str, default='results',
                       help='结果保存目录（默认: results）')
    
    args = parser.parse_args()
    
    # 自动选择数据源
    if args.source == 'auto':
        # 优先使用AKShare（免费）
        print("🔍 自动选择数据源: AKShare（免费）")
        analyze_with_akshare(args.stock, args.weeks, args.output)
    elif args.source == 'akshare':
        analyze_with_akshare(args.stock, args.weeks, args.output)
    elif args.source == 'tushare':
        if args.token is None:
            args.token = os.getenv('TUSHARE_TOKEN')
            if args.token is None:
                print("❌ 错误: 使用Tushare需要Token")
                print("请设置环境变量 TUSHARE_TOKEN 或使用 --token 参数")
                return
        analyze_with_tushare(args.stock, args.weeks, args.token, args.output)
    elif args.source == 'hybrid':
        # 混合模式不需要Token（使用AKShare和baostock，都是免费的）
        analyze_with_hybrid(args.stock, args.weeks, None, args.output)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 默认使用AKShare
        print("="*60)
        print("  股票参数反推分析（默认使用AKShare）")
        print("="*60)
        print("\n使用示例:")
        print("  python analyze_stock_multi_source.py --stock 平安银行 --weeks 120")
        print("  python analyze_stock_multi_source.py --source tushare --token your_token")
        print("  python analyze_stock_multi_source.py --source hybrid")
        print("\n开始使用默认参数分析...\n")
        analyze_with_akshare()
    else:
        main()

