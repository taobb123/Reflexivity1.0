"""
快速测试Tushare连接
"""

import os
from data_fetcher import TushareDataFetcher

def test_tushare_connection():
    """测试Tushare连接"""
    print("="*60)
    print("  Tushare连接测试")
    print("="*60)
    
    try:
        # 获取Token
        token = os.getenv('TUSHARE_TOKEN')
        if token is None:
            print("❌ 错误: 未找到TUSHARE_TOKEN环境变量")
            print("请设置环境变量: $env:TUSHARE_TOKEN='your_token'")
            return False
        
        print(f"✓ 找到Token: {token[:10]}...{token[-10:]}")
        
        # 初始化数据获取器
        print("\n【步骤1】初始化Tushare客户端...")
        fetcher = TushareDataFetcher(token=token)
        
        # 测试1：查找股票代码
        print("\n【步骤2】测试股票代码查找...")
        try:
            code = fetcher.get_stock_code("平安银行")
            print(f"✓ 成功找到股票代码: {code}")
        except Exception as e:
            print(f"❌ 查找股票失败: {str(e)}")
            return False
        
        # 测试2：获取少量数据
        print("\n【步骤3】测试数据获取（少量数据）...")
        try:
            price_df = fetcher.get_weekly_price(code, lookback_weeks=10)
            print(f"✓ 成功获取 {len(price_df)} 条周线数据")
            print("\n数据预览:")
            print(price_df.head())
        except Exception as e:
            print(f"❌ 获取价格数据失败: {str(e)}")
            return False
        
        # 测试3：获取财务数据
        print("\n【步骤4】测试财务数据获取...")
        try:
            finance_df = fetcher.get_financial_data(code)
            print(f"✓ 成功获取 {len(finance_df)} 条财务数据")
            print("\n财务数据预览:")
            print(finance_df.head())
        except Exception as e:
            print(f"❌ 获取财务数据失败: {str(e)}")
            print("提示: 可能是积分不足或数据不存在")
            return False
        
        print("\n" + "="*60)
        print("  ✅ 所有测试通过！Tushare连接正常")
        print("="*60)
        print("\n可以开始运行完整分析了:")
        print("  python analyze_stock.py --stock 平安银行 --weeks 120")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tushare_connection()

