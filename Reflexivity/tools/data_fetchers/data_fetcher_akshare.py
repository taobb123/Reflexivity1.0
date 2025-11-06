"""
基于 AKShare 的数据获取模块
用于获取A股股票价格和财务数据
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告: AKShare未安装，请运行: pip install akshare")


class AKShareDataFetcher:
    """AKShare数据获取器"""
    
    def __init__(self):
        """初始化AKShare数据获取器"""
        if not AKSHARE_AVAILABLE:
            raise ImportError("AKShare未安装，请运行: pip install akshare")
        print("✓ AKShare数据获取器初始化成功")
    
    def get_stock_code(self, stock_name: str = "平安银行") -> str:
        """
        根据股票名称查找股票代码
        
        Args:
            stock_name: 股票名称，如"平安银行"
            
        Returns:
            股票代码，如"000001"
        """
        # 常见股票代码映射（避免网络请求）
        stock_map = {
            "平安银行": "000001",
            "万科A": "000002",
            "国农科技": "000004",
            "招商银行": "600036",
            "中国平安": "601318",
            "工商银行": "601398",
            "建设银行": "601939",
            "阳光电源": "300274",
            "300274": "300274",  # 支持直接使用代码
        }
        
        if stock_name in stock_map:
            code = stock_map[stock_name]
            print(f"✓ 找到股票: {stock_name} ({code})")
            return code
        
        # 如果不在映射中，尝试网络查询（可能失败）
        try:
            realtime = ak.stock_zh_a_spot_em()
            stock_info = realtime[realtime['名称'].str.contains(stock_name, na=False)]
            
            if not stock_info.empty:
                code = stock_info.iloc[0]['代码']
                print(f"✓ 找到股票: {stock_name} ({code})")
                return code
        except Exception as e:
            print(f"⚠️  网络查询失败: {str(e)}")
        
        raise ValueError(f"未找到股票: {stock_name}")
    
    def get_weekly_price(self, stock_code: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        lookback_weeks: int = 120) -> pd.DataFrame:
        """
        获取周线价格数据
        
        Args:
            stock_code: 股票代码，如"000001"
            start_date: 开始日期，格式"YYYYMMDD"，如果为None则自动计算
            end_date: 结束日期，格式"YYYYMMDD"，如果为None则为今天
            lookback_weeks: 回溯周数（如果start_date为None）
            
        Returns:
            DataFrame包含: date, P_t, vol
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if start_date is None:
            # 计算回溯日期
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - timedelta(weeks=lookback_weeks)
            start_date = start_dt.strftime('%Y%m%d')
        
        print(f"📊 获取周线数据: {start_date} 至 {end_date}")
        
        try:
            # 获取周线数据
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="weekly",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                raise ValueError(f"未获取到数据")
            
            # 整理数据
            # 自动识别列名
            date_col = None
            close_col = None
            vol_col = None
            
            for col in df.columns:
                if '日期' in col or 'date' in col.lower():
                    date_col = col
                if '收盘' in col or 'close' in col.lower():
                    close_col = col
                if '成交量' in col or 'volume' in col.lower():
                    vol_col = col
            
            if not date_col or not close_col:
                raise ValueError("无法识别数据列名")
            
            # 重命名
            result = df.rename(columns={
                date_col: 'date',
                close_col: 'P_t'
            })
            
            if vol_col:
                result = result.rename(columns={vol_col: 'vol'})
            else:
                result['vol'] = 0
            
            # 转换日期格式
            result['date'] = pd.to_datetime(result['date'])
            
            # 选择需要的列并排序
            result = result[['date', 'P_t', 'vol']].sort_values('date')
            result = result.reset_index(drop=True)
            
            print(f"✓ 成功获取 {len(result)} 条周线数据")
            return result
            
        except Exception as e:
            raise Exception(f"获取周线数据失败: {str(e)}")
    
    def get_financial_data(self, stock_code: str) -> pd.DataFrame:
        """
        获取财务指标数据（EPS等）
        
        注意：AKShare的财务数据接口可能变化，这里尝试多种方式
        
        Args:
            stock_code: 股票代码
            
        Returns:
            DataFrame包含财务指标
        """
        print(f"📈 获取财务数据...")
        
        # 方法1：财务分析指标
        try:
            fina_indicator = ak.stock_financial_analysis_indicator(symbol=stock_code)
            if not fina_indicator.empty and '每股收益' in fina_indicator.columns:
                print(f"✓ 成功获取财务指标: {len(fina_indicator)} 条记录")
                
                # 整理数据
                result = fina_indicator.copy()
                result['报告日期'] = pd.to_datetime(result['报告日期'])
                result = result.rename(columns={
                    '报告日期': 'date',
                    '每股收益': 'eps'
                })
                
                # 只保留需要的列
                if '净资产收益率' in result.columns:
                    result = result.rename(columns={'净资产收益率': 'roe'})
                if '销售净利率' in result.columns:
                    result = result.rename(columns={'销售净利率': 'netprofit_margin'})
                
                result = result.sort_values('date')
                result = result.reset_index(drop=True)
                
                return result[['date', 'eps']].dropna()
        except Exception as e:
            print(f"⚠️  方法1失败: {str(e)}")
        
        # 方法2：财务报表数据
        try:
            # 获取利润表
            income = ak.stock_profit_sheet_by_quarterly_em(symbol=stock_code)
            if not income.empty:
                print(f"⚠️  从利润表获取数据需要进一步处理")
                # 这里需要从利润表中计算EPS，暂时跳过
        except Exception as e:
            print(f"⚠️  方法2失败: {str(e)}")
        
        print(f"⚠️  无法获取完整财务数据，将使用价格数据估算基本面")
        return pd.DataFrame()  # 返回空DataFrame
    
    def align_price_and_fundamental(self,
                                   price_df: pd.DataFrame,
                                   finance_df: pd.DataFrame) -> pd.DataFrame:
        """
        对齐价格数据和财务数据（周频）
        
        Args:
            price_df: 价格DataFrame
            finance_df: 财务DataFrame
            
        Returns:
            合并后的DataFrame
        """
        if finance_df.empty:
            # 如果没有财务数据，使用价格平滑作为基本面代理
            print("⚠️  无财务数据，使用价格移动平均作为基本面代理")
            price_df = price_df.copy()
            # 使用长期移动平均作为基本面代理（例如52周）
            price_df['F_t'] = price_df['P_t'].rolling(window=min(52, len(price_df)), 
                                                      min_periods=1).mean()
            return price_df[['date', 'P_t', 'F_t']].dropna()
        
        # 合并财务数据
        result = price_df.merge(
            finance_df[['date', 'eps']],
            on='date',
            how='left'
        )
        
        # 前向填充EPS
        result['eps'] = result['eps'].fillna(method='ffill')
        
        # 如果没有EPS，使用价格移动平均
        if result['eps'].isna().all():
            print("⚠️  EPS数据缺失，使用价格移动平均作为基本面代理")
            result['F_t'] = result['P_t'].rolling(window=min(52, len(result)), 
                                                  min_periods=1).mean()
        else:
            result = result.rename(columns={'eps': 'F_t'})
        
        # 删除缺失值
        result = result.dropna(subset=['P_t', 'F_t'])
        result = result.sort_values('date')
        result = result.reset_index(drop=True)
        
        print(f"✓ 数据对齐完成: {len(result)} 条记录")
        return result[['date', 'P_t', 'F_t']]
    
    def fetch_complete_data(self,
                           stock_name: str = "平安银行",
                           lookback_weeks: int = 120) -> Tuple[pd.DataFrame, str]:
        """
        获取完整的股票数据（价格+基本面）
        
        Args:
            stock_name: 股票名称
            lookback_weeks: 回溯周数
            
        Returns:
            (合并后的DataFrame, 股票代码)
        """
        # 获取股票代码
        stock_code = self.get_stock_code(stock_name)
        
        # 获取价格数据
        price_df = self.get_weekly_price(stock_code, lookback_weeks=lookback_weeks)
        
        # 获取财务数据（可能为空）
        finance_df = self.get_financial_data(stock_code)
        
        # 对齐数据
        aligned_df = self.align_price_and_fundamental(price_df, finance_df)
        
        return aligned_df, stock_code


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("AKShare数据获取测试")
    print("="*60)
    
    try:
        fetcher = AKShareDataFetcher()
        df, code = fetcher.fetch_complete_data("平安银行", lookback_weeks=120)
        
        print("\n数据预览：")
        print(df.head(10))
        print(f"\n数据范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"数据条数: {len(df)}")
        print(f"\n价格统计:")
        print(df['P_t'].describe())
        print(f"\n基本面统计:")
        print(df['F_t'].describe())
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

