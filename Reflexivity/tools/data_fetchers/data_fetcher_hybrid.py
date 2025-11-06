"""
混合数据获取模块 - 结合AKShare和baostock的优势
- AKShare: 获取价格数据和财务指标（免费、稳定）
- baostock: 作为财务数据的备用方案（免费、稳定）
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

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False
    print("警告: baostock未安装，请运行: pip install baostock")


class HybridDataFetcher:
    """混合数据获取器 - 结合AKShare和baostock"""
    
    def __init__(self):
        """
        初始化混合数据获取器
        - 价格数据: AKShare
        - 财务数据: AKShare（优先），baostock（备用）
        """
        if not AKSHARE_AVAILABLE:
            raise ImportError("AKShare未安装，请运行: pip install akshare")
        
        # 初始化AKShare（用于价格和财务数据）
        print("✓ AKShare数据获取器初始化成功（用于价格和财务数据）")
        
        # 初始化baostock（用于财务数据备用）
        if BAOSTOCK_AVAILABLE:
            print("✓ baostock已安装（作为财务数据备用方案）")
            self.bs_logged_in = False
        else:
            print("⚠️ 警告: baostock未安装，财务数据将仅使用AKShare")
            self.bs_logged_in = False
    
    def get_stock_code(self, stock_name: str = "平安银行") -> Tuple[str, str]:
        """
        获取股票代码（AKShare和baostock格式）
        
        Args:
            stock_name: 股票名称
            
        Returns:
            (akshare_code, baostock_code) 如 ("000001", "sz.000001")
        """
        # 使用AKShare获取股票代码
        stock_map = {
            "平安银行": "000001",
            "万科A": "000002",
            "国农科技": "000004",
            "招商银行": "600036",
            "中国平安": "601318",
            "工商银行": "601398",
            "建设银行": "601939",
            "阳光电源": "300274",
        }
        
        akshare_code = None
        if stock_name in stock_map:
            akshare_code = stock_map[stock_name]
        else:
            # 尝试网络查询
            try:
                realtime = ak.stock_zh_a_spot_em()
                stock_info = realtime[realtime['名称'].str.contains(stock_name, na=False)]
                if not stock_info.empty:
                    akshare_code = stock_info.iloc[0]['代码']
            except Exception as e:
                print(f"⚠️  AKShare查询失败: {str(e)}")
        
        if not akshare_code:
            raise ValueError(f"未找到股票: {stock_name}")
        
        # 转换为baostock格式
        if akshare_code.startswith('6'):
            baostock_code = f"sh.{akshare_code}"
        elif akshare_code.startswith(('0', '3')):
            baostock_code = f"sz.{akshare_code}"
        else:
            baostock_code = f"sz.{akshare_code}"  # 默认深圳
        
        print(f"✓ 找到股票: {stock_name} (AKShare: {akshare_code}, baostock: {baostock_code})")
        return akshare_code, baostock_code
    
    def get_weekly_price_akshare(self, stock_code: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 lookback_weeks: int = 120) -> pd.DataFrame:
        """
        使用AKShare获取周线价格数据
        
        Args:
            stock_code: AKShare格式的股票代码，如"000001"
            start_date: 开始日期，格式"YYYYMMDD"
            end_date: 结束日期，格式"YYYYMMDD"
            lookback_weeks: 回溯周数
            
        Returns:
            DataFrame包含: date, P_t, vol
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if start_date is None:
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - timedelta(weeks=lookback_weeks)
            start_date = start_dt.strftime('%Y%m%d')
        
        print(f"📊 [AKShare] 获取周线价格数据: {start_date} 至 {end_date}")
        
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="weekly",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                raise ValueError(f"未获取到数据")
            
            # 识别列名
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
            
            # 整理数据
            result = df.rename(columns={
                date_col: 'date',
                close_col: 'P_t'
            })
            
            if vol_col:
                result = result.rename(columns={vol_col: 'vol'})
            else:
                result['vol'] = 0
            
            result['date'] = pd.to_datetime(result['date'])
            result = result[['date', 'P_t', 'vol']].sort_values('date')
            result = result.reset_index(drop=True)
            
            print(f"✓ [AKShare] 成功获取 {len(result)} 条周线价格数据")
            return result
            
        except Exception as e:
            raise Exception(f"AKShare获取价格数据失败: {str(e)}")
    
    def _login_baostock(self) -> bool:
        """登录baostock"""
        if not BAOSTOCK_AVAILABLE:
            return False
        
        if self.bs_logged_in:
            return True
        
        try:
            lg = bs.login()
            if lg.error_code == '0':
                self.bs_logged_in = True
                return True
            else:
                print(f"⚠️  baostock登录失败: {lg.error_msg}")
                return False
        except Exception as e:
            print(f"⚠️  baostock登录异常: {str(e)}")
            return False
    
    def _logout_baostock(self):
        """登出baostock"""
        if BAOSTOCK_AVAILABLE and self.bs_logged_in:
            try:
                bs.logout()
                self.bs_logged_in = False
            except:
                pass
    
    def get_financial_data_akshare(self, stock_code: str) -> pd.DataFrame:
        """
        使用AKShare获取财务数据（EPS等）
        
        Args:
            stock_code: AKShare格式的股票代码，如"000001"
            
        Returns:
            DataFrame包含: date, eps
        """
        print(f"📈 [AKShare] 获取财务数据...")
        
        # 方法1：财务分析指标
        try:
            fina_indicator = ak.stock_financial_analysis_indicator(symbol=stock_code)
            if not fina_indicator.empty and '每股收益' in fina_indicator.columns:
                print(f"✓ [AKShare] 成功获取财务指标: {len(fina_indicator)} 条记录")
                
                # 整理数据
                result = fina_indicator.copy()
                result['报告日期'] = pd.to_datetime(result['报告日期'])
                result = result.rename(columns={
                    '报告日期': 'date',
                    '每股收益': 'eps'
                })
                
                result = result.sort_values('date')
                result = result.reset_index(drop=True)
                
                return result[['date', 'eps']].dropna()
        except Exception as e:
            print(f"⚠️  [AKShare] 方法1失败: {str(e)}")
        
        # 方法2：尝试其他接口
        try:
            # 尝试获取利润表数据计算EPS
            income = ak.stock_profit_sheet_by_quarterly_em(symbol=stock_code)
            if not income.empty and '报告日期' in income.columns:
                print(f"⚠️  [AKShare] 方法2需要进一步处理")
        except Exception as e:
            print(f"⚠️  [AKShare] 方法2失败: {str(e)}")
        
        print(f"⚠️  [AKShare] 无法获取财务数据，尝试备用方案...")
        return pd.DataFrame()
    
    def get_financial_data_baostock(self, bs_code: str) -> pd.DataFrame:
        """
        使用baostock获取财务数据（EPS等）作为备用方案
        
        Args:
            bs_code: baostock格式的股票代码，如"sz.000001"
            
        Returns:
            DataFrame包含: date, eps
        """
        if not BAOSTOCK_AVAILABLE:
            return pd.DataFrame()
        
        print(f"📈 [baostock] 获取财务数据（备用方案）...")
        
        # 登录baostock
        if not self._login_baostock():
            return pd.DataFrame()
        
        try:
            # 获取最近几年的财务数据
            current_year = datetime.now().year
            years = range(current_year - 5, current_year + 1)
            quarters = [1, 2, 3, 4]
            
            all_data = []
            for year in years:
                for quarter in quarters:
                    try:
                        # 查询盈利能力数据
                        rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                        if rs.error_code == '0':
                            data_list = []
                            while rs.next():
                                data_list.append(rs.get_row_data())
                            
                            if data_list:
                                df = pd.DataFrame(data_list, columns=rs.fields)
                                # baostock的字段名通常是 'pubDate' 和 'epsTTM' 或 'eps'
                                # 需要根据实际字段名调整
                                date_col = None
                                eps_col = None
                                
                                for col in df.columns:
                                    col_lower = col.lower()
                                    if 'date' in col_lower or 'pub' in col_lower:
                                        date_col = col
                                    if 'eps' in col_lower:
                                        eps_col = col
                                
                                if date_col and eps_col:
                                    df_clean = df[[date_col, eps_col]].copy()
                                    df_clean = df_clean.rename(columns={
                                        date_col: 'date',
                                        eps_col: 'eps'
                                    })
                                    # 转换EPS为数值类型
                                    df_clean['eps'] = pd.to_numeric(df_clean['eps'], errors='coerce')
                                    df_clean = df_clean.dropna(subset=['eps'])
                                    if not df_clean.empty:
                                        all_data.append(df_clean)
                    except Exception as e:
                        continue
            
            if not all_data:
                print("⚠️  [baostock] 未获取到财务数据")
                return pd.DataFrame()
            
            # 合并所有数据
            result = pd.concat(all_data, ignore_index=True)
            result['date'] = pd.to_datetime(result['date'], errors='coerce')
            result = result.dropna(subset=['date', 'eps'])
            result = result.sort_values('date')
            result = result.reset_index(drop=True)
            
            if not result.empty:
                print(f"✓ [baostock] 成功获取 {len(result)} 条财务数据")
                return result[['date', 'eps']]
            else:
                print("⚠️  [baostock] 数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"⚠️  [baostock] 获取财务数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_financial_data(self, stock_code: str, bs_code: str) -> Tuple[pd.DataFrame, str]:
        """
        获取财务数据，优先使用AKShare，失败则使用baostock
        
        Args:
            stock_code: AKShare格式的股票代码
            bs_code: baostock格式的股票代码
            
        Returns:
            (DataFrame包含: date, eps, 数据来源标识)
        """
        # 优先使用AKShare
        finance_df = self.get_financial_data_akshare(stock_code)
        source = "AKShare"
        
        # 如果AKShare失败，使用baostock作为备用
        if finance_df.empty:
            finance_df = self.get_financial_data_baostock(bs_code)
            source = "baostock" if not finance_df.empty else "price_ma"
        
        return finance_df, source
    
    def align_price_and_fundamental(self,
                                   price_df: pd.DataFrame,
                                   finance_df: pd.DataFrame,
                                   finance_source: str = "unknown") -> pd.DataFrame:
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
            price_df['F_t'] = price_df['P_t'].rolling(
                window=min(52, len(price_df)), 
                min_periods=1
            ).mean()
            result = price_df[['date', 'P_t', 'F_t']].dropna()
            print(f"✓ 数据对齐完成: {len(result)} 条记录")
            print(f"   价格数据来源: AKShare")
            print(f"   财务数据来源: 价格移动平均（备用）")
            return result
        
        # 将财务数据转换为周频（使用最近一期的财务数据）
        result = price_df.copy()
        
        # 为每个价格日期匹配最近的财务数据
        result['F_t'] = None
        for idx, row in result.iterrows():
            # 找到该日期之前最近的财务数据
            available_finance = finance_df[finance_df['date'] <= row['date']]
            if not available_finance.empty:
                result.at[idx, 'F_t'] = available_finance.iloc[-1]['eps']
        
        # 前向填充
        result['F_t'] = result['F_t'].fillna(method='ffill')
        
        # 如果还是没有，使用价格移动平均
        if result['F_t'].isna().all():
            print("⚠️  EPS数据缺失，使用价格移动平均作为基本面代理")
            result['F_t'] = result['P_t'].rolling(
                window=min(52, len(result)), 
                min_periods=1
            ).mean()
        
        # 删除缺失值
        result = result.dropna(subset=['P_t', 'F_t'])
        result = result.sort_values('date')
        result = result.reset_index(drop=True)
        
        print(f"✓ 数据对齐完成: {len(result)} 条记录")
        print(f"   价格数据来源: AKShare")
        if finance_source == "AKShare":
            print(f"   财务数据来源: AKShare")
        elif finance_source == "baostock":
            print(f"   财务数据来源: baostock（备用）")
        else:
            print(f"   财务数据来源: 价格移动平均（备用）")
        return result[['date', 'P_t', 'F_t']]
    
    def fetch_complete_data(self,
                           stock_name: str = "平安银行",
                           lookback_weeks: int = 120) -> Tuple[pd.DataFrame, str]:
        """
        获取完整的股票数据（价格+基本面）
        - 价格数据: AKShare（免费、稳定）
        - 财务数据: AKShare（优先），baostock（备用）
        
        Args:
            stock_name: 股票名称
            lookback_weeks: 回溯周数
            
        Returns:
            (合并后的DataFrame, 股票代码)
        """
        try:
            # 获取股票代码
            akshare_code, bs_code = self.get_stock_code(stock_name)
            
            # 使用AKShare获取价格数据
            price_df = self.get_weekly_price_akshare(
                akshare_code, 
                lookback_weeks=lookback_weeks
            )
            
            # 获取财务数据（AKShare优先，baostock备用）
            finance_df, finance_source = self.get_financial_data(akshare_code, bs_code)
            
            # 对齐数据
            aligned_df = self.align_price_and_fundamental(price_df, finance_df, finance_source)
            
            return aligned_df, akshare_code
        finally:
            # 确保登出baostock
            self._logout_baostock()


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("混合数据获取测试 (AKShare + baostock)")
    print("="*60)
    
    try:
        fetcher = HybridDataFetcher()
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


