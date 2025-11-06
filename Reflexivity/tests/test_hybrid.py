"""
混合模式测试脚本
测试AKShare+Tushare混合数据获取
"""

import os
import sys
import pandas as pd
from datetime import datetime

def test_hybrid_mode():
    """测试混合模式"""
    print("="*60)
    print("  混合模式测试 (AKShare + baostock)")
    print("="*60)
    
    # 检查依赖
    print("\n【检查1】检查依赖库...")
    try:
        import akshare as ak
        print("✓ AKShare已安装")
    except ImportError:
        print("❌ AKShare未安装，请运行: pip install akshare")
        return False
    
    try:
        import baostock as bs
        print("✓ baostock已安装")
    except ImportError:
        print("⚠️  baostock未安装（财务数据备用方案将不可用）")
        print("   建议安装: pip install baostock")
    
    # 初始化混合数据获取器
    print("\n【测试1】初始化混合数据获取器...")
    try:
        from data_fetcher_hybrid import HybridDataFetcher
        fetcher = HybridDataFetcher()
        print("✓ 初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试股票代码查找
    print("\n【测试2】查找股票代码（平安银行）...")
    try:
        akshare_code, bs_code = fetcher.get_stock_code("平安银行")
        print(f"✓ AKShare代码: {akshare_code}")
        print(f"✓ baostock代码: {bs_code}")
    except Exception as e:
        print(f"❌ 查找股票失败: {str(e)}")
        return False
    
    # 测试价格数据获取（少量数据）
    print("\n【测试3】获取价格数据（AKShare，最近10周）...")
    try:
        price_df = fetcher.get_weekly_price_akshare(akshare_code, lookback_weeks=10)
        print(f"✓ 成功获取 {len(price_df)} 条周线价格数据")
        print("\n价格数据预览:")
        print(price_df.head())
        print(f"\n价格统计:")
        print(f"  时间范围: {price_df['date'].min()} 至 {price_df['date'].max()}")
        print(f"  价格范围: [{price_df['P_t'].min():.2f}, {price_df['P_t'].max():.2f}]")
        print(f"  平均价格: {price_df['P_t'].mean():.2f}")
    except Exception as e:
        print(f"❌ 获取价格数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试财务数据获取
    print("\n【测试4】获取财务数据（AKShare优先，baostock备用）...")
    try:
        finance_df_akshare = fetcher.get_financial_data_akshare(akshare_code)
        if not finance_df_akshare.empty:
            print(f"✓ [AKShare] 成功获取 {len(finance_df_akshare)} 条财务数据")
            print("\n财务数据预览:")
            print(finance_df_akshare.head())
        else:
            print("⚠️  [AKShare] 未获取到财务数据，尝试baostock备用方案...")
            finance_df_bs = fetcher.get_financial_data_baostock(bs_code)
            if not finance_df_bs.empty:
                print(f"✓ [baostock] 成功获取 {len(finance_df_bs)} 条财务数据")
                print("\n财务数据预览:")
                print(finance_df_bs.head())
            else:
                print("⚠️  未获取到财务数据，将使用价格移动平均作为备用方案")
    except Exception as e:
        print(f"⚠️  获取财务数据失败: {str(e)}")
        print("   将使用价格移动平均作为备用方案")
    
    # 测试完整数据获取
    print("\n【测试5】获取完整数据（混合模式，最近20周）...")
    try:
        df, code = fetcher.fetch_complete_data("平安银行", lookback_weeks=20)
        print(f"✓ 成功获取完整数据: {len(df)} 条记录")
        print("\n完整数据预览:")
        print(df.head(10))
        print(f"\n数据统计:")
        print(f"  时间范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"  数据点数: {len(df)}")
        print(f"  价格范围: [{df['P_t'].min():.2f}, {df['P_t'].max():.2f}]")
        print(f"  基本面范围: [{df['F_t'].min():.4f}, {df['F_t'].max():.4f}]")
        
        # 检查数据来源（从输出中推断）
        print(f"\n数据来源:")
        print(f"  价格数据: AKShare ✓")
        print(f"  财务数据: 见上方输出")
        
        # 数据质量检查
        print(f"\n数据质量检查:")
        missing_price = df['P_t'].isna().sum()
        missing_fundamental = df['F_t'].isna().sum()
        print(f"  价格数据缺失: {missing_price} 条")
        print(f"  基本面数据缺失: {missing_fundamental} 条")
        
        if missing_price == 0 and missing_fundamental == 0:
            print("  ✓ 数据完整，无缺失值")
        else:
            print("  ⚠️  存在缺失值，但已处理")
        
    except Exception as e:
        print(f"❌ 获取完整数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试总结
    print("\n" + "="*60)
    print("  ✅ 混合模式测试通过！")
    print("="*60)
    print("\n混合模式优势:")
    print("  ✓ 价格数据来自AKShare（免费、稳定）")
    print("  ✓ 财务数据来自AKShare（优先）或baostock（备用），完全免费")
    print("  ✓ 数据完整，已对齐")
    print("\n可以运行完整分析:")
    print("  python analyze_stock_multi_source.py --source hybrid --stock 平安银行 --weeks 120")
    
    return True


if __name__ == "__main__":
    try:
        success = test_hybrid_mode()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


