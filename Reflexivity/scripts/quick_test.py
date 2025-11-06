"""
快速测试 - 验证Token并测试数据获取
支持多种Token输入方式
"""

import os
import sys

def quick_test():
    """快速测试"""
    print("="*60)
    print("  快速测试 - Tushare连接")
    print("="*60)
    
    # 获取Token
    token = os.getenv('TUSHARE_TOKEN')
    
    if token:
        print(f"✓ 从环境变量读取Token: {token[:10]}...")
    elif len(sys.argv) > 1:
        token = sys.argv[1]
        print(f"✓ 从命令行参数读取Token: {token[:10]}...")
    else:
        print("\n❌ 未找到Token")
        print("\n使用方法:")
        print("  方式1: python quick_test.py your_token_here")
        print("  方式2: 先设置环境变量 $env:TUSHARE_TOKEN='your_token'")
        print("  方式3: 直接运行 python analyze_stock.py --token your_token_here")
        return
    
    try:
        from data_fetcher import TushareDataFetcher
        
        print("\n【测试1】初始化...")
        fetcher = TushareDataFetcher(token=token)
        
        print("\n【测试2】查找股票...")
        code = fetcher.get_stock_code("平安银行")
        print(f"✓ 找到: {code}")
        
        print("\n【测试3】获取周线数据（最近10周）...")
        price_df = fetcher.get_weekly_price(code, lookback_weeks=10)
        print(f"✓ 获取成功: {len(price_df)} 条数据")
        print("\n数据示例:")
        print(price_df[['trade_date', 'close']].tail(5))
        
        print("\n【测试4】获取财务数据...")
        try:
            finance_df = fetcher.get_financial_data(code)
            print(f"✓ 获取成功: {len(finance_df)} 条数据")
            if 'eps' in finance_df.columns:
                print("\nEPS数据示例:")
                print(finance_df[['end_date', 'eps']].tail(3))
        except Exception as e:
            print(f"⚠️  财务数据获取失败: {str(e)}")
            print("    (可能是积分不足，但不影响价格数据获取)")
        
        print("\n" + "="*60)
        print("  ✅ 测试通过！可以开始分析了")
        print("="*60)
        print("\n运行完整分析:")
        print(f"  python analyze_stock.py --stock 平安银行 --weeks 120 --token {token[:10]}...")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        print("\n可能原因:")
        print("  1. Token无效或已过期")
        print("  2. 积分不足（需要≥120积分）")
        print("  3. 网络连接问题")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()

