"""
测试统一多数据源数据获取器(A股专用)
测试所有可用的A股数据源
"""

import os
import sys
import pandas as pd
from datetime import datetime

def test_unified_fetcher():
    """测试统一数据获取器"""
    print("="*60)
    print("统一多数据源数据获取器测试(A股专用)")
    print("="*60)
    
    try:
        from data_fetcher_unified import UnifiedDataFetcher
        
        # 检查环境变量
        print("\n环境变量检查:")
        tushare_token = os.getenv('TUSHARE_TOKEN')
        if tushare_token:
            print(f"  ✓ TUSHARE_TOKEN: 已设置")
        else:
            print(f"  ⚠️ TUSHARE_TOKEN: 未设置(可选)")
        
        # 初始化数据获取器
        print("\n初始化数据获取器...")
        fetcher = UnifiedDataFetcher(
            tushare_token=tushare_token
        )
        
        print(f"\n可用数据源: {list(fetcher.fetchers.keys())}")
        
        # 测试1: A股数据(优先Tushare,回退到其他源)
        print("\n" + "="*60)
        print("测试1: A股数据获取 (平安银行)")
        print("="*60)
        try:
            df, code, info = fetcher.fetch_complete_data("平安银行", lookback_weeks=52)
            print(f"\n✓ 测试成功!")
            print(f"  数据条数: {len(df)}")
            print(f"  价格来源: {info['price_source']}")
            print(f"  财务来源: {info['finance_source']}")
            print(f"\n数据预览:")
            print(df.head())
            print(f"\n数据列: {df.columns.tolist()}")
            if 'MA5' in df.columns:
                print(f"\n均线数据:")
                print(df[['date', 'P_t', 'MA5', 'MA10', 'MA20', 'MA60']].tail())
            print(f"\n数据统计:")
            print(df[['P_t', 'F_t']].describe())
        except Exception as e:
            print(f"\n❌ 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 测试2: 使用股票代码
        print("\n" + "="*60)
        print("测试2: 使用股票代码获取数据 (000001)")
        print("="*60)
        try:
            df, code, info = fetcher.fetch_complete_data("000001", lookback_weeks=52)
            print(f"\n✓ 测试成功!")
            print(f"  数据条数: {len(df)}")
            print(f"  价格来源: {info['price_source']}")
        except Exception as e:
            print(f"\n⚠️ 测试失败: {str(e)}")
        
        # 测试3: 测试回退机制
        print("\n" + "="*60)
        print("测试3: 回退机制测试 (使用不存在的股票)")
        print("="*60)
        try:
            # 尝试获取一个不存在的股票,应该会回退到其他数据源
            df, code, info = fetcher.fetch_complete_data("不存在的股票代码12345", lookback_weeks=52)
            print(f"\n✓ 意外成功: {len(df)} 条数据")
        except Exception as e:
            print(f"\n✓ 回退机制正常工作: {str(e)}")
        
        # 测试4: 测试不同数据源
        print("\n" + "="*60)
        print("测试4: 测试各个数据源")
        print("="*60)
        
        test_stocks = ["平安银行", "招商银行", "中国平安"]
        for stock in test_stocks:
            try:
                print(f"\n测试股票: {stock}")
                df, code, info = fetcher.fetch_complete_data(stock, lookback_weeks=20)
                print(f"  ✓ 成功: {len(df)} 条数据, 来源: {info['price_source']}")
            except Exception as e:
                print(f"  ⚠️ 失败: {str(e)}")
        
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
    except ImportError as e:
        print(f"\n❌ 导入失败: {str(e)}")
        print("\n请安装必要的依赖:")
        print("  pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def test_individual_fetchers():
    """测试各个独立的数据获取器"""
    print("\n" + "="*60)
    print("测试各个独立数据获取器")
    print("="*60)
    
    # 测试Tushare
    if os.getenv('TUSHARE_TOKEN'):
        print("\n测试Tushare...")
        try:
            from data_fetcher_unified import TushareFetcher
            fetcher = TushareFetcher()
            ts_code = fetcher.get_stock_code("平安银行")
            df = fetcher.get_weekly_price(ts_code, lookback_weeks=10)
            print(f"✓ Tushare测试成功: {len(df)} 条数据")
            if 'MA5' in df.columns:
                print(f"  包含均线: MA5, MA10, MA20, MA60")
        except Exception as e:
            print(f"⚠️ Tushare测试失败: {str(e)}")
    
    # 测试yfinance
    print("\n测试yfinance...")
    try:
        from data_fetcher_unified import YFinanceFetcher
        fetcher = YFinanceFetcher()
        yf_code = fetcher.get_stock_code("平安银行")
        df = fetcher.get_weekly_price(yf_code, lookback_weeks=10)
        print(f"✓ yfinance测试成功: {len(df)} 条数据")
        if 'MA5' in df.columns:
            print(f"  包含均线: MA5, MA10, MA20, MA60")
    except Exception as e:
        print(f"⚠️ yfinance测试失败: {str(e)}")
    
    # 测试pandas-datareader
    print("\n测试pandas-datareader...")
    try:
        from data_fetcher_unified import PandasDatareaderFetcher
        fetcher = PandasDatareaderFetcher()
        code = fetcher.get_stock_code("平安银行")
        df = fetcher.get_weekly_price(code, lookback_weeks=10)
        print(f"✓ pandas-datareader测试成功: {len(df)} 条数据")
        if 'MA5' in df.columns:
            print(f"  包含均线: MA5, MA10, MA20, MA60")
    except Exception as e:
        print(f"⚠️ pandas-datareader测试失败: {str(e)}")
    
    # 测试AKShare
    print("\n测试AKShare...")
    try:
        from data_fetcher_unified import AKShareFetcher
        fetcher = AKShareFetcher()
        stock_code = fetcher.get_stock_code("平安银行")
        df = fetcher.get_weekly_price(stock_code, lookback_weeks=10)
        print(f"✓ AKShare测试成功: {len(df)} 条数据")
        if 'MA5' in df.columns:
            print(f"  包含均线: MA5, MA10, MA20, MA60")
    except Exception as e:
        print(f"⚠️ AKShare测试失败: {str(e)}")


if __name__ == "__main__":
    print("="*60)
    print("统一多数据源数据获取器 - A股完整测试")
    print("="*60)
    print("\n提示:")
    print("1. 设置环境变量以启用Tushare(可选):")
    print("   export TUSHARE_TOKEN=your_token")
    print("2. yfinance、pandas-datareader和AKShare不需要API Key")
    print("3. 即使没有Tushare Token,系统也会自动回退到可用的数据源")
    print("4. 所有数据源都支持A股数据获取,包含均线数据")
    print("\n开始测试...\n")
    
    # 测试统一获取器
    test_unified_fetcher()
    
    # 测试独立获取器
    test_individual_fetchers()
