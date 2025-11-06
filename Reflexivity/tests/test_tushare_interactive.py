"""
交互式Tushare连接测试
如果环境变量未设置，可以手动输入Token
"""

import os
import sys
from data_fetcher import TushareDataFetcher

def test_tushare_connection():
    """测试Tushare连接"""
    print("="*60)
    print("  Tushare连接测试")
    print("="*60)
    
    # 获取Token
    token = os.getenv('TUSHARE_TOKEN')
    
    if token is None:
        print("\n⚠️  未找到TUSHARE_TOKEN环境变量")
        print("请输入你的Tushare Token (或按Ctrl+C取消):")
        token = input("Token: ").strip()
        
        if not token:
            print("❌ Token为空，退出测试")
            return False
    else:
        print(f"✓ 从环境变量读取Token: {token[:10]}...{token[-10:]}")
    
    try:
        # 初始化数据获取器
        print("\n【步骤1】初始化Tushare客户端...")
        fetcher = TushareDataFetcher(token=token)
        
        # 测试1：查找股票代码
        print("\n【步骤2】测试股票代码查找（平安银行）...")
        try:
            code = fetcher.get_stock_code("平安银行")
            print(f"✓ 成功找到股票代码: {code}")
        except Exception as e:
            print(f"❌ 查找股票失败: {str(e)}")
            return False
        
        # 测试2：获取少量数据
        print("\n【步骤3】测试数据获取（最近10周）...")
        try:
            price_df = fetcher.get_weekly_price(code, lookback_weeks=10)
            print(f"✓ 成功获取 {len(price_df)} 条周线数据")
            print("\n数据预览（前5条）:")
            print(price_df[['trade_date', 'close', 'vol']].head())
        except Exception as e:
            print(f"❌ 获取价格数据失败: {str(e)}")
            print("\n可能原因：")
            print("  1. Token无效")
            print("  2. 积分不足（需要≥120积分）")
            print("  3. 网络问题")
            return False
        
        # 测试3：获取财务数据
        print("\n【步骤4】测试财务数据获取...")
        try:
            finance_df = fetcher.get_financial_data(code)
            print(f"✓ 成功获取 {len(finance_df)} 条财务数据")
            print("\n财务数据预览（前5条）:")
            if 'eps' in finance_df.columns:
                print(finance_df[['end_date', 'eps']].head())
            else:
                print(finance_df.head())
        except Exception as e:
            print(f"⚠️  获取财务数据失败: {str(e)}")
            print("提示: 可能是积分不足或数据不存在，但不影响价格数据获取")
        
        # 测试4：尝试数据对齐
        print("\n【步骤5】测试数据对齐...")
        try:
            price_df = fetcher.get_weekly_price(code, lookback_weeks=20)
            finance_df = fetcher.get_financial_data(code)
            
            if len(finance_df) > 0:
                aligned_df = fetcher.align_price_and_fundamental(price_df, finance_df)
                print(f"✓ 数据对齐成功: {len(aligned_df)} 条记录")
                print("\n对齐后数据预览（前5条）:")
                print(aligned_df[['date', 'P_t', 'F_t']].head())
            else:
                print("⚠️  跳过对齐测试（财务数据为空）")
        except Exception as e:
            print(f"⚠️  数据对齐测试失败: {str(e)}")
        
        print("\n" + "="*60)
        print("  ✅ 连接测试完成！")
        print("="*60)
        
        # 询问是否运行完整分析
        print("\n是否运行完整分析？(y/n): ", end='')
        choice = input().strip().lower()
        
        if choice == 'y':
            print("\n开始运行完整分析...")
            print("-" * 60)
            from analyze_stock import analyze_stock
            analyze_stock(stock_name="平安银行", lookback_weeks=120, token=token)
        else:
            print("\n可以稍后运行:")
            print("  python analyze_stock.py --stock 平安银行 --weeks 120")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n用户取消操作")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_tushare_connection()

