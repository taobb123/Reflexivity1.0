"""
测试 yfinance 和 AKShare 获取股票财务数据
比较两者的功能和易用性
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def test_yfinance():
    """测试 yfinance 获取数据"""
    print("="*60)
    print("  测试 yfinance")
    print("="*60)
    
    try:
        import yfinance as yf
        print("✓ yfinance 导入成功\n")
        
        # 测试股票：平安银行（深交所：000001.SZ）
        # yfinance使用格式：股票代码.交易所，深交所用.SZ，上交所用.SS
        ticker_symbol = "000001.SZ"  # 平安银行
        stock_name = "平安银行"
        
        print(f"【测试1】获取股票基本信息: {stock_name} ({ticker_symbol})")
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            print(f"✓ 股票名称: {info.get('longName', 'N/A')}")
            print(f"  行业: {info.get('industry', 'N/A')}")
            print(f"  市值: {info.get('marketCap', 'N/A')}")
        except Exception as e:
            print(f"⚠️  获取基本信息失败: {str(e)}")
        
        print(f"\n【测试2】获取历史价格数据（最近1年）")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            hist = ticker.history(start=start_date, end=end_date, interval="1wk")
            
            if hist.empty:
                print("⚠️  未获取到数据，尝试其他方式...")
                # 尝试不指定日期
                hist = ticker.history(period="1y", interval="1wk")
            
            if not hist.empty:
                print(f"✓ 成功获取 {len(hist)} 条周线数据")
                print("\n数据预览（最近5条）:")
                print(hist[['Close', 'Volume']].tail())
                
                # 保存为P_t格式
                price_data = hist.reset_index()
                price_data = price_data.rename(columns={
                    'Date': 'date',
                    'Close': 'P_t',
                    'Volume': 'vol'
                })
                print("\n整理后的数据格式:")
                print(price_data[['date', 'P_t']].head())
                return price_data
            else:
                print("❌ 未获取到价格数据")
                return None
                
        except Exception as e:
            print(f"❌ 获取价格数据失败: {str(e)}")
            return None
        
        print(f"\n【测试3】获取财务数据（EPS等）")
        try:
            # 获取财务报表
            financials = ticker.financials
            income_statement = ticker.income_stmt
            balance_sheet = ticker.balance_sheet
            
            print(f"✓ 财务数据获取成功")
            print(f"  财务报表维度: {financials.shape if not financials.empty else 'N/A'}")
            print(f"  利润表维度: {income_statement.shape if not income_statement.empty else 'N/A'}")
            
            # 尝试获取EPS（每股收益）
            if not income_statement.empty:
                # yfinance的财务报表是倒序的（最新在最后）
                print("\n利润表最近一期数据:")
                print(income_statement.iloc[-1].head(10))
                
                # 查找EPS相关指标
                eps_row = income_statement[income_statement.index.str.contains('EPS', case=False, na=False)]
                if not eps_row.empty:
                    print("\n✓ 找到EPS数据:")
                    print(eps_row.iloc[-1])
            
        except Exception as e:
            print(f"⚠️  获取财务数据失败: {str(e)}")
            # yfinance可能不支持某些中国股票的详细财务数据
        
        return None
        
    except ImportError:
        print("❌ yfinance 未安装")
        print("安装命令: pip install yfinance")
        return None
    except Exception as e:
        print(f"❌ yfinance 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_akshare():
    """测试 AKShare 获取数据"""
    print("\n" + "="*60)
    print("  测试 AKShare")
    print("="*60)
    
    try:
        import akshare as ak
        print("✓ AKShare 导入成功\n")
        
        stock_name = "平安银行"
        stock_code = "000001"  # 深交所
        
        print(f"【测试1】获取股票基本信息: {stock_name}")
        # 直接使用股票代码，避免网络请求问题
        print(f"  使用代码: {stock_code}")
        
        print(f"\n【测试2】获取历史价格数据（周线）")
        try:
            # 获取周线数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
            # AKShare获取周线数据
            # period参数: "daily"(日), "weekly"(周), "monthly"(月)
            weekly_data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="weekly",  # 修正：使用英文
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if not weekly_data.empty:
                print(f"✓ 成功获取 {len(weekly_data)} 条周线数据")
                print("\n数据列名:", weekly_data.columns.tolist())
                print("\n数据预览（最近5条）:")
                
                # AKShare可能返回不同的列名，尝试多种可能
                date_col = None
                close_col = None
                vol_col = None
                
                for col in weekly_data.columns:
                    if '日期' in col or 'date' in col.lower() or 'Date' in col:
                        date_col = col
                    if '收盘' in col or 'close' in col.lower() or 'Close' in col:
                        close_col = col
                    if '成交量' in col or 'volume' in col.lower() or 'Volume' in col:
                        vol_col = col
                
                if date_col and close_col:
                    print(weekly_data[[date_col, close_col]].tail())
                    
                    # 整理数据格式
                    price_data = weekly_data.rename(columns={
                        date_col: 'date',
                        close_col: 'P_t'
                    })
                    if vol_col:
                        price_data = price_data.rename(columns={vol_col: 'vol'})
                    else:
                        price_data['vol'] = 0
                    
                    price_data['date'] = pd.to_datetime(price_data['date'])
                    price_data = price_data[['date', 'P_t', 'vol']].sort_values('date')
                    
                    print("\n整理后的数据格式:")
                    print(price_data[['date', 'P_t']].head())
                    return price_data
                else:
                    print("⚠️  无法识别数据列名")
                    print(weekly_data.head())
                    return None
            else:
                print("❌ 未获取到价格数据")
                return None
                
        except Exception as e:
            print(f"❌ 获取价格数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        print(f"\n【测试3】获取财务数据（EPS等）")
        try:
            # 获取财务指标
            fina_indicator = ak.stock_financial_analysis_indicator(symbol=stock_code)
            
            if not fina_indicator.empty:
                print(f"✓ 成功获取财务指标: {len(fina_indicator)} 条记录")
                print("\n财务指标预览:")
                print(fina_indicator[['报告日期', '每股收益', '净资产收益率', '销售净利率']].tail())
                
                # 整理为时间序列
                fina_data = fina_indicator.copy()
                fina_data['报告日期'] = pd.to_datetime(fina_data['报告日期'])
                fina_data = fina_data.rename(columns={
                    '报告日期': 'date',
                    '每股收益': 'eps',
                    '净资产收益率': 'roe',
                    '销售净利率': 'netprofit_margin'
                })
                fina_data = fina_data.sort_values('date')
                
                print("\n整理后的财务数据:")
                print(fina_data[['date', 'eps', 'roe']].tail())
                return fina_data
            else:
                print("⚠️  财务数据为空")
                return None
                
        except Exception as e:
            print(f"⚠️  获取财务数据失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
    except ImportError:
        print("❌ AKShare 未安装")
        print("安装命令: pip install akshare")
        return None
    except Exception as e:
        print(f"❌ AKShare 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def compare_data_sources():
    """对比不同数据源"""
    print("\n" + "="*60)
    print("  数据源对比总结")
    print("="*60)
    
    comparison = {
        'Tushare': {
            '优点': ['数据质量高', '接口规范', '有官方文档'],
            '缺点': ['需要注册和Token', '需要积分（≥120）', '可能有调用频率限制'],
            '适用': '专业用户，需要高质量数据'
        },
        'yfinance': {
            '优点': ['无需注册', '免费', '全球股票支持', '使用简单'],
            '缺点': ['中国A股数据可能不完整', '财务数据格式可能不一致', '网络要求高'],
            '适用': '快速测试，美股港股等'
        },
        'AKShare': {
            '优点': ['专为中国市场设计', '免费', '数据全面', 'A股数据完整'],
            '缺点': ['接口可能变化', '需要维护', '文档可能更新不及时'],
            '适用': 'A股市场分析，需要本地数据'
        }
    }
    
    for source, info in comparison.items():
        print(f"\n【{source}】")
        print(f"  优点: {', '.join(info['优点'])}")
        print(f"  缺点: {', '.join(info['缺点'])}")
        print(f"  适用: {info['适用']}")


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("  股票数据源测试")
    print("  测试 yfinance 和 AKShare")
    print("="*60 + "\n")
    
    # 测试yfinance
    yf_data = test_yfinance()
    
    # 测试AKShare
    ak_data = test_akshare()
    
    # 对比总结
    compare_data_sources()
    
    print("\n" + "="*60)
    print("  测试完成")
    print("="*60)
    
    print("\n建议:")
    if ak_data is not None:
        print("✓ AKShare 适合A股数据获取，推荐用于本项目的参数反推")
    if yf_data is not None:
        print("✓ yfinance 可以获取价格数据，但财务数据可能不完整")
    
    print("\n下一步:")
    print("  1. 根据测试结果选择合适的数据源")
    print("  2. 创建对应的数据获取模块")
    print("  3. 集成到参数反推分析流程中")


if __name__ == "__main__":
    main()

