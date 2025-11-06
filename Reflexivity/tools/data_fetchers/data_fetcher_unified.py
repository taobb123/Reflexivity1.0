"""
统一多数据源数据获取模块 - A股专用
支持多个金融API,具有智能回退机制
支持的数据源:
- Tushare (A股数据,需要Token)
- yfinance (A股数据,免费)
- pandas-datareader (A股数据,免费)
- AKShare (A股数据,免费,作为备用)

主要功能:
- 获取价格数据(周线)
- 计算均线数据(MA5, MA10, MA20, MA60等)
- 获取财务数据(EPS等)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# ============ 数据源可用性检查 ============

# Tushare
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False

# yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# pandas-datareader
try:
    import pandas_datareader.data as web
    PANDAS_DATAREADER_AVAILABLE = True
except ImportError:
    PANDAS_DATAREADER_AVAILABLE = False

# AKShare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


# ============ 辅助函数 ============

def _convert_stock_code_to_yfinance(stock_code: str) -> str:
    """
    将A股代码转换为yfinance格式
    深交所: 000001 -> 000001.SZ
    上交所: 600036 -> 600036.SS
    """
    if '.' in stock_code:
        return stock_code
    
    if stock_code.startswith('6'):
        return f"{stock_code}.SS"  # 上交所
    elif stock_code.startswith(('0', '3')):
        return f"{stock_code}.SZ"  # 深交所
    else:
        return f"{stock_code}.SZ"  # 默认深交所


def _calculate_ma(data: pd.Series, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """
    计算移动平均线
    
    Args:
        data: 价格数据序列
        periods: 均线周期列表
    
    Returns:
        包含各周期均线的DataFrame
    """
    result = pd.DataFrame()
    for period in periods:
        result[f'MA{period}'] = data.rolling(window=period, min_periods=1).mean()
    return result


# ============ Tushare数据获取器 ============

class TushareFetcher:
    """Tushare数据获取器"""
    
    def __init__(self, token: Optional[str] = None):
        if not TUSHARE_AVAILABLE:
            raise ImportError("Tushare未安装: pip install tushare")
        
        if token is None:
            token = os.getenv('TUSHARE_TOKEN')
            if token is None:
                raise ValueError("Tushare需要Token,请设置环境变量TUSHARE_TOKEN")
        
        ts.set_token(token)
        self.pro = ts.pro_api()
        print("✓ Tushare数据获取器初始化成功")
    
    def get_stock_code(self, stock_name: str) -> str:
        """获取股票代码"""
        df = self.pro.stock_basic(exchange='', list_status='L', 
                                  fields='ts_code,symbol,name')
        match = df[df['name'].str.contains(stock_name, na=False)]
        if match.empty:
            raise ValueError(f"未找到股票: {stock_name}")
        code = match.iloc[0]['ts_code']
        name = match.iloc[0]['name']
        print(f"✓ 找到股票: {name} ({code})")
        return code
    
    def get_weekly_price(self, ts_code: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        lookback_weeks: int = 120) -> pd.DataFrame:
        """获取周线价格数据,包含均线"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - timedelta(weeks=lookback_weeks)
            start_date = start_dt.strftime('%Y%m%d')
        
        df = self.pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df.empty:
            raise ValueError("未获取到数据")
        
        df = df.sort_values('trade_date')
        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.rename(columns={'close': 'P_t', 'vol': 'vol'})
        
        # 计算均线
        ma_data = _calculate_ma(df['P_t'], periods=[5, 10, 20, 60])
        for col in ma_data.columns:
            df[col] = ma_data[col]
        
        result = df[['date', 'P_t', 'vol', 'MA5', 'MA10', 'MA20', 'MA60']]
        return result.reset_index(drop=True)
    
    def get_financial_data(self, ts_code: str) -> pd.DataFrame:
        """获取财务数据"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=5*365)
        start_date = start_dt.strftime('%Y%m%d')
        
        try:
            df = self.pro.fina_indicator(ts_code=ts_code, start_date=start_date,
                                        end_date=end_date, fields='end_date,eps')
            if df.empty:
                return pd.DataFrame()
            
            df['date'] = pd.to_datetime(df['end_date'], format='%Y%m%d')
            return df[['date', 'eps']].sort_values('date').reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ Tushare获取财务数据失败: {str(e)}")
            return pd.DataFrame()


# ============ yfinance数据获取器 ============

class YFinanceFetcher:
    """yfinance数据获取器"""
    
    def __init__(self):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance未安装: pip install yfinance")
        print("✓ yfinance数据获取器初始化成功")
    
    def get_stock_code(self, stock_name: str) -> str:
        """获取股票代码(yfinance格式)"""
        # 常见股票代码映射
        stock_map = {
            "平安银行": "000001",
            "万科A": "000002",
            "招商银行": "600036",
            "中国平安": "601318",
            "工商银行": "601398",
            "建设银行": "601939",
            "阳光电源": "300274",
        }
        
        if stock_name in stock_map:
            code = stock_map[stock_name]
        elif stock_name.isdigit() and len(stock_name) == 6:
            code = stock_name
        else:
            # 尝试从AKShare获取代码
            try:
                if AKSHARE_AVAILABLE:
                    realtime = ak.stock_zh_a_spot_em()
                    stock_info = realtime[realtime['名称'].str.contains(stock_name, na=False)]
                    if not stock_info.empty:
                        code = stock_info.iloc[0]['代码']
                    else:
                        raise ValueError(f"未找到股票: {stock_name}")
                else:
                    raise ValueError(f"未找到股票: {stock_name}")
            except:
                raise ValueError(f"未找到股票: {stock_name}")
        
        yf_code = _convert_stock_code_to_yfinance(code)
        print(f"✓ 找到股票: {stock_name} (yfinance: {yf_code})")
        return yf_code
    
    def get_weekly_price(self, yf_code: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        lookback_weeks: int = 120) -> pd.DataFrame:
        """获取周线价格数据,包含均线"""
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = pd.to_datetime(end_date)
        
        if start_date is None:
            start_dt = end_dt - timedelta(weeks=lookback_weeks)
        else:
            start_dt = pd.to_datetime(start_date)
        
        try:
            ticker = yf.Ticker(yf_code)
            # 获取日线数据,然后转换为周线
            df = ticker.history(start=start_dt, end=end_dt)
            
            if df.empty:
                raise ValueError("未获取到数据")
            
            # 转换为周线(取每周最后一个交易日)
            df_weekly = df.resample('W').last()
            
            df_weekly['date'] = df_weekly.index
            df_weekly = df_weekly.rename(columns={'Close': 'P_t', 'Volume': 'vol'})
            
            # 计算均线
            ma_data = _calculate_ma(df_weekly['P_t'], periods=[5, 10, 20, 60])
            for col in ma_data.columns:
                df_weekly[col] = ma_data[col]
            
            result = df_weekly[['date', 'P_t', 'vol', 'MA5', 'MA10', 'MA20', 'MA60']]
            result = result.sort_values('date').reset_index(drop=True)
            
            return result
        except Exception as e:
            raise Exception(f"yfinance获取数据失败: {str(e)}")
    
    def get_financial_data(self, yf_code: str) -> pd.DataFrame:
        """获取财务数据"""
        try:
            ticker = yf.Ticker(yf_code)
            info = ticker.info
            
            # 尝试获取EPS
            eps = info.get('trailingEps') or info.get('forwardEps')
            if eps:
                # 获取季度财务数据
                financials = ticker.quarterly_financials
                if not financials.empty:
                    # 尝试从财务数据中提取EPS
                    if 'Diluted EPS' in financials.index:
                        eps_data = financials.loc['Diluted EPS']
                        dates = pd.to_datetime(eps_data.index)
                        result = pd.DataFrame({
                            'date': dates,
                            'eps': eps_data.values
                        })
                        result = result.sort_values('date').reset_index(drop=True)
                        return result
            
            return pd.DataFrame()
        except Exception as e:
            print(f"⚠️ yfinance获取财务数据失败: {str(e)}")
            return pd.DataFrame()


# ============ pandas-datareader数据获取器 ============

class PandasDatareaderFetcher:
    """pandas-datareader数据获取器"""
    
    def __init__(self):
        if not PANDAS_DATAREADER_AVAILABLE:
            raise ImportError("pandas-datareader未安装: pip install pandas-datareader")
        print("✓ pandas-datareader数据获取器初始化成功")
    
    def get_stock_code(self, stock_name: str) -> str:
        """获取股票代码(pandas-datareader格式,使用yahoo)"""
        # 常见股票代码映射
        stock_map = {
            "平安银行": "000001",
            "万科A": "000002",
            "招商银行": "600036",
            "中国平安": "601318",
            "工商银行": "601398",
            "建设银行": "601939",
            "阳光电源": "300274",
        }
        
        if stock_name in stock_map:
            code = stock_map[stock_name]
        elif stock_name.isdigit() and len(stock_name) == 6:
            code = stock_name
        else:
            # 尝试从AKShare获取代码
            try:
                if AKSHARE_AVAILABLE:
                    realtime = ak.stock_zh_a_spot_em()
                    stock_info = realtime[realtime['名称'].str.contains(stock_name, na=False)]
                    if not stock_info.empty:
                        code = stock_info.iloc[0]['代码']
                    else:
                        raise ValueError(f"未找到股票: {stock_name}")
                else:
                    raise ValueError(f"未找到股票: {stock_name}")
            except:
                raise ValueError(f"未找到股票: {stock_name}")
        
        # pandas-datareader使用yahoo格式
        yahoo_code = _convert_stock_code_to_yfinance(code)
        print(f"✓ 找到股票: {stock_name} (pandas-datareader: {yahoo_code})")
        return yahoo_code
    
    def get_weekly_price(self, symbol: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        lookback_weeks: int = 120) -> pd.DataFrame:
        """获取周线价格数据,包含均线"""
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = pd.to_datetime(end_date)
        
        if start_date is None:
            start_dt = end_dt - timedelta(weeks=lookback_weeks)
        else:
            start_dt = pd.to_datetime(start_date)
        
        try:
            # 使用yahoo数据源
            df = web.DataReader(symbol, 'yahoo', start_dt, end_dt)
            
            if df.empty:
                raise ValueError("未获取到数据")
            
            # 转换为周线
            df_weekly = df.resample('W').last()
            
            df_weekly['date'] = df_weekly.index
            df_weekly = df_weekly.rename(columns={'Close': 'P_t', 'Volume': 'vol'})
            
            # 计算均线
            ma_data = _calculate_ma(df_weekly['P_t'], periods=[5, 10, 20, 60])
            for col in ma_data.columns:
                df_weekly[col] = ma_data[col]
            
            result = df_weekly[['date', 'P_t', 'vol', 'MA5', 'MA10', 'MA20', 'MA60']]
            result = result.sort_values('date').reset_index(drop=True)
            
            return result
        except Exception as e:
            raise Exception(f"pandas-datareader获取数据失败: {str(e)}")
    
    def get_financial_data(self, symbol: str) -> pd.DataFrame:
        """获取财务数据"""
        # pandas-datareader对A股财务数据支持有限
        # 返回空,使用价格移动平均作为备用
        return pd.DataFrame()


# ============ AKShare数据获取器 ============

class AKShareFetcher:
    """AKShare数据获取器"""
    
    def __init__(self):
        if not AKSHARE_AVAILABLE:
            raise ImportError("AKShare未安装: pip install akshare")
        print("✓ AKShare数据获取器初始化成功")
    
    def get_stock_code(self, stock_name: str) -> str:
        """获取股票代码"""
        stock_map = {
            "平安银行": "000001",
            "万科A": "000002",
            "招商银行": "600036",
            "中国平安": "601318",
            "工商银行": "601398",
            "建设银行": "601939",
            "阳光电源": "300274",
        }
        
        if stock_name in stock_map:
            code = stock_map[stock_name]
            print(f"✓ 找到股票: {stock_name} ({code})")
            return code
        
        if stock_name.isdigit() and len(stock_name) == 6:
            return stock_name
        
        try:
            realtime = ak.stock_zh_a_spot_em()
            stock_info = realtime[realtime['名称'].str.contains(stock_name, na=False)]
            if not stock_info.empty:
                code = stock_info.iloc[0]['代码']
                print(f"✓ 找到股票: {stock_name} ({code})")
                return code
        except:
            pass
        
        raise ValueError(f"未找到股票: {stock_name}")
    
    def get_weekly_price(self, stock_code: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        lookback_weeks: int = 120) -> pd.DataFrame:
        """获取周线价格数据,包含均线"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - timedelta(weeks=lookback_weeks)
            start_date = start_dt.strftime('%Y%m%d')
        
        df = ak.stock_zh_a_hist(symbol=stock_code, period="weekly",
                                start_date=start_date, end_date=end_date, adjust="qfq")
        
        if df.empty:
            raise ValueError("未获取到数据")
        
        # 识别列名
        date_col = [c for c in df.columns if '日期' in c or 'date' in c.lower()][0]
        close_col = [c for c in df.columns if '收盘' in c or 'close' in c.lower()][0]
        vol_col = [c for c in df.columns if '成交量' in c or 'volume' in c.lower()]
        
        result = df.rename(columns={date_col: 'date', close_col: 'P_t'})
        if vol_col:
            result = result.rename(columns={vol_col[0]: 'vol'})
        else:
            result['vol'] = 0
        
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date')
        
        # 计算均线
        ma_data = _calculate_ma(result['P_t'], periods=[5, 10, 20, 60])
        for col in ma_data.columns:
            result[col] = ma_data[col]
        
        return result[['date', 'P_t', 'vol', 'MA5', 'MA10', 'MA20', 'MA60']].reset_index(drop=True)
    
    def get_financial_data(self, stock_code: str) -> pd.DataFrame:
        """获取财务数据"""
        try:
            fina_indicator = ak.stock_financial_analysis_indicator(symbol=stock_code)
            if not fina_indicator.empty and '每股收益' in fina_indicator.columns:
                result = fina_indicator.copy()
                result['date'] = pd.to_datetime(result['报告日期'])
                result = result.rename(columns={'每股收益': 'eps'})
                return result[['date', 'eps']].dropna().sort_values('date').reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ AKShare获取财务数据失败: {str(e)}")
        
        return pd.DataFrame()


# ============ 统一数据获取器(带回退机制) ============

class UnifiedDataFetcher:
    """统一多数据源数据获取器,支持智能回退 - A股专用"""
    
    def __init__(self, 
                 tushare_token: Optional[str] = None,
                 preferred_sources: Optional[List[str]] = None):
        """
        初始化统一数据获取器
        
        Args:
            tushare_token: Tushare Token
            preferred_sources: 优先使用的数据源列表,如[ 'akshare', 'tushare', 'pandas_datareader''yfinance']
        """
        self.fetchers = {}
        self.preferred_sources = preferred_sources or ['akshare', 'tushare', 'pandas_datareader', 'yfinance']
        
        # 初始化各个数据获取器
        if TUSHARE_AVAILABLE:
            try:
                self.fetchers['tushare'] = TushareFetcher(token=tushare_token)
            except Exception as e:
                print(f"⚠️ Tushare初始化失败: {str(e)}")
        
        if YFINANCE_AVAILABLE:
            try:
                self.fetchers['yfinance'] = YFinanceFetcher()
            except Exception as e:
                print(f"⚠️ yfinance初始化失败: {str(e)}")
        
        if PANDAS_DATAREADER_AVAILABLE:
            try:
                self.fetchers['pandas_datareader'] = PandasDatareaderFetcher()
            except Exception as e:
                print(f"⚠️ pandas-datareader初始化失败: {str(e)}")
        
        if AKSHARE_AVAILABLE:
            try:
                self.fetchers['akshare'] = AKShareFetcher()
            except Exception as e:
                print(f"⚠️ AKShare初始化失败: {str(e)}")
        
        if not self.fetchers:
            raise ValueError("没有可用的数据源,请至少安装一个数据源库")
    
    def get_weekly_price_with_fallback(self,
                                       stock_name: str,
                                       start_date: Optional[str] = None,
                                       end_date: Optional[str] = None,
                                       lookback_weeks: int = 120) -> Tuple[pd.DataFrame, str]:
        """
        获取周线价格数据(包含均线),带智能回退
        
        Returns:
            (DataFrame, 使用的数据源名称)
        """
        sources = self.preferred_sources
        
        # 尝试每个数据源
        last_error = None
        for source in sources:
            if source not in self.fetchers:
                continue
            
            print(f"🔄 尝试使用 {source} 获取数据...")
            try:
                fetcher = self.fetchers[source]
                
                # 获取股票代码
                if hasattr(fetcher, 'get_stock_code'):
                    code = fetcher.get_stock_code(stock_name)
                else:
                    code = stock_name
                
                # 获取价格数据
                df = fetcher.get_weekly_price(code, start_date, end_date, lookback_weeks)
                
                if not df.empty:
                    print(f"✓ 成功使用 {source} 获取 {len(df)} 条数据")
                    return df, source
                    
            except Exception as e:
                print(f"⚠️ {source} 获取失败: {str(e)}")
                last_error = e
                continue
        
        # 所有数据源都失败
        raise Exception(f"所有数据源都无法获取数据。最后错误: {str(last_error)}")
    
    def get_financial_data_with_fallback(self,
                                        stock_name: str,
                                        price_source: str) -> Tuple[pd.DataFrame, str]:
        """
        获取财务数据,带智能回退
        
        Args:
            stock_name: 股票名称
            price_source: 价格数据使用的数据源
        
        Returns:
            (DataFrame, 使用的数据源名称)
        """
        # 优先使用价格数据相同的源
        sources = [price_source] + [s for s in self.preferred_sources if s != price_source]
        
        for source in sources:
            if source not in self.fetchers:
                continue
            
            print(f"🔄 尝试使用 {source} 获取财务数据...")
            try:
                fetcher = self.fetchers[source]
                
                # 获取股票代码
                if hasattr(fetcher, 'get_stock_code'):
                    code = fetcher.get_stock_code(stock_name)
                else:
                    code = stock_name
                
                # 获取财务数据
                df = fetcher.get_financial_data(code)
                
                if not df.empty:
                    print(f"✓ 成功使用 {source} 获取财务数据")
                    return df, source
                    
            except Exception as e:
                print(f"⚠️ {source} 获取财务数据失败: {str(e)}")
                continue
        
        # 所有数据源都失败,返回空DataFrame
        print("⚠️ 无法获取财务数据,将使用价格移动平均作为备用")
        return pd.DataFrame(), 'price_ma'
    
    def align_price_and_fundamental(self,
                                   price_df: pd.DataFrame,
                                   finance_df: pd.DataFrame) -> pd.DataFrame:
        """对齐价格数据和财务数据"""
        if finance_df.empty:
            # 使用价格移动平均作为基本面代理
            price_df = price_df.copy()
            price_df['F_t'] = price_df['P_t'].rolling(
                window=min(52, len(price_df)), min_periods=1
            ).mean()
            return price_df.dropna()
        
        # 合并财务数据
        result = price_df.copy()
        result['F_t'] = None
        
        for idx, row in result.iterrows():
            available_finance = finance_df[finance_df['date'] <= row['date']]
            if not available_finance.empty:
                # 检查是否有'eps'列
                if 'eps' in available_finance.columns:
                    result.at[idx, 'F_t'] = available_finance.iloc[-1]['eps']
                else:
                    # 尝试使用第一个数值列
                    numeric_cols = available_finance.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        result.at[idx, 'F_t'] = available_finance.iloc[-1][numeric_cols[0]]
        
        result['F_t'] = result['F_t'].ffill()
        
        if result['F_t'].isna().all():
            result['F_t'] = result['P_t'].rolling(
                window=min(52, len(result)), min_periods=1
            ).mean()
        
        return result.dropna(subset=['P_t', 'F_t']).sort_values('date').reset_index(drop=True)
    
    def fetch_complete_data(self,
                           stock_name: str,
                           lookback_weeks: int = 120) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
        """
        获取完整的股票数据(价格+基本面+均线),带智能回退
        
        Args:
            stock_name: 股票名称或代码
            lookback_weeks: 回溯周数
        
        Returns:
            (DataFrame, 股票代码, 数据源信息字典)
        """
        print(f"\n{'='*60}")
        print(f"获取A股数据: {stock_name}")
        print(f"{'='*60}\n")
        
        # 获取价格数据(包含均线)
        price_df, price_source = self.get_weekly_price_with_fallback(
            stock_name, lookback_weeks=lookback_weeks
        )
        
        # 获取财务数据
        finance_df, finance_source = self.get_financial_data_with_fallback(
            stock_name, price_source
        )
        
        # 对齐数据
        aligned_df = self.align_price_and_fundamental(price_df, finance_df)
        
        sources_info = {
            'price_source': price_source,
            'finance_source': finance_source
        }
        
        print(f"\n✓ 数据获取完成:")
        print(f"  价格数据: {price_source} ({len(price_df)} 条)")
        print(f"  财务数据: {finance_source}")
        print(f"  对齐后数据: {len(aligned_df)} 条")
        print(f"  包含均线: MA5, MA10, MA20, MA60")
        
        return aligned_df, stock_name, sources_info


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("统一多数据源数据获取器测试(A股专用)")
    print("="*60)
    
    try:
        # 初始化
        fetcher = UnifiedDataFetcher()
        
        # 测试A股数据获取
        print("\n测试A股数据获取...")
        df, code, info = fetcher.fetch_complete_data("平安银行", lookback_weeks=120)
        
        print("\n数据预览:")
        print(df.head(10))
        print(f"\n数据范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"数据条数: {len(df)}")
        print(f"\n数据源信息: {info}")
        print(f"\n包含的列: {df.columns.tolist()}")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
