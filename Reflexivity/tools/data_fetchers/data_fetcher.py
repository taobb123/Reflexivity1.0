"""
Tushare数据获取模块
用于获取股票价格和基本面数据
"""

import tushare as ts
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import os
from datetime import datetime, timedelta


class TushareDataFetcher:
    """Tushare数据获取器"""
    
    def __init__(self, token: Optional[str] = None):
        """
        初始化Tushare客户端
        
        Args:
            token: Tushare Token，如果为None则从环境变量读取
        """
        if token is None:
            token = os.getenv('TUSHARE_TOKEN')
            if token is None:
                raise ValueError(
                    "Tushare Token未设置！\n"
                    "请设置环境变量 TUSHARE_TOKEN 或传递给构造函数\n"
                    "或在代码中调用: ts.set_token('your_token_here')"
                )
        
        ts.set_token(token)
        self.pro = ts.pro_api()
        print("✓ Tushare客户端初始化成功")
    
    def get_stock_code(self, stock_name: str = "平安银行") -> str:
        """
        根据股票名称查找股票代码
        
        Args:
            stock_name: 股票名称，如"平安银行"
            
        Returns:
            Tushare格式的股票代码，如"000001.SZ"
        """
        # 获取股票基本信息
        df = self.pro.stock_basic(exchange='', list_status='L', 
                                  fields='ts_code,symbol,name')
        
        # 查找匹配的股票
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
        """
        获取周线价格数据
        
        Args:
            ts_code: 股票代码，如"000001.SZ"
            start_date: 开始日期，格式"YYYYMMDD"，如果为None则自动计算
            end_date: 结束日期，格式"YYYYMMDD"，如果为None则为今天
            lookback_weeks: 回溯周数（如果start_date为None）
            
        Returns:
            DataFrame包含: trade_date, open, high, low, close, vol, amount
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if start_date is None:
            # 计算回溯日期（大约 lookback_weeks * 7 天前）
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - timedelta(weeks=lookback_weeks)
            start_date = start_dt.strftime('%Y%m%d')
        
        print(f"📊 获取周线数据: {start_date} 至 {end_date}")
        
        try:
            df = self.pro.weekly(ts_code=ts_code, 
                               start_date=start_date, 
                               end_date=end_date)
            
            if df.empty:
                raise ValueError(f"未获取到数据，请检查股票代码和日期范围")
            
            # 排序（从早到晚）
            df = df.sort_values('trade_date')
            df = df.reset_index(drop=True)
            
            print(f"✓ 成功获取 {len(df)} 条周线数据")
            return df
            
        except Exception as e:
            raise Exception(f"获取周线数据失败: {str(e)}\n"
                          f"可能原因：1) Token无效 2) 积分不足 3) 网络问题")
    
    def get_financial_data(self, ts_code: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        获取财务指标数据（EPS等）
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame包含财务指标
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        if start_date is None:
            # 默认获取最近5年数据
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            start_dt = end_dt - timedelta(days=5*365)
            start_date = start_dt.strftime('%Y%m%d')
        
        print(f"📈 获取财务数据: {start_date} 至 {end_date}")
        
        try:
            # 获取财务指标
            df = self.pro.fina_indicator(ts_code=ts_code,
                                        start_date=start_date,
                                        end_date=end_date,
                                        fields='end_date,eps,roe,roa,netprofit_margin')
            
            if df.empty:
                raise ValueError(f"未获取到财务数据")
            
            # 排序
            df = df.sort_values('end_date')
            df = df.reset_index(drop=True)
            
            print(f"✓ 成功获取 {len(df)} 条财务数据")
            return df
            
        except Exception as e:
            raise Exception(f"获取财务数据失败: {str(e)}")
    
    def align_price_and_fundamental(self, 
                                    price_df: pd.DataFrame,
                                    finance_df: pd.DataFrame) -> pd.DataFrame:
        """
        对齐价格数据和财务数据（周频）
        
        策略：将季度财务数据插值到每周，使用前向填充
        
        Args:
            price_df: 价格DataFrame
            finance_df: 财务DataFrame
            
        Returns:
            合并后的DataFrame，包含价格和基本面数据
        """
        # 转换日期格式
        price_df['date'] = pd.to_datetime(price_df['trade_date'], format='%Y%m%d')
        finance_df['date'] = pd.to_datetime(finance_df['end_date'], format='%Y%m%d')
        
        # 准备合并
        price_aligned = price_df[['date', 'close', 'vol']].copy()
        price_aligned = price_aligned.sort_values('date')
        
        # 创建完整的日期序列（周频）
        date_range = pd.date_range(
            start=price_aligned['date'].min(),
            end=price_aligned['date'].max(),
            freq='W'
        )
        
        # 创建基础DataFrame
        result = pd.DataFrame({'date': date_range})
        
        # 合并价格数据（使用最近的价格）
        result = result.merge(
            price_aligned,
            on='date',
            how='left'
        )
        # 前向填充价格
        result['close'] = result['close'].fillna(method='ffill')
        result['vol'] = result['vol'].fillna(method='ffill')
        
        # 合并财务数据
        finance_aligned = finance_df[['date', 'eps']].copy()
        finance_aligned = finance_aligned.sort_values('date')
        
        # 使用前向填充：每季度财务数据填充到下一季度
        result = result.merge(
            finance_aligned,
            on='date',
            how='left'
        )
        result['eps'] = result['eps'].fillna(method='ffill')
        
        # 删除缺失值行
        result = result.dropna(subset=['close', 'eps'])
        
        # 重命名和整理
        result = result.rename(columns={
            'close': 'P_t',
            'eps': 'F_t'
        })
        
        result = result.sort_values('date')
        result = result.reset_index(drop=True)
        
        print(f"✓ 数据对齐完成: {len(result)} 条记录")
        return result
    
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
        ts_code = self.get_stock_code(stock_name)
        
        # 获取价格数据
        price_df = self.get_weekly_price(ts_code, lookback_weeks=lookback_weeks)
        
        # 获取财务数据
        finance_df = self.get_financial_data(ts_code)
        
        # 对齐数据
        aligned_df = self.align_price_and_fundamental(price_df, finance_df)
        
        return aligned_df, ts_code


if __name__ == "__main__":
    # 测试代码
    print("="*60)
    print("Tushare数据获取测试")
    print("="*60)
    
    # 注意：需要先设置TUSHARE_TOKEN环境变量
    # 或修改下面的代码传入token
    
    try:
        fetcher = TushareDataFetcher()
        df, code = fetcher.fetch_complete_data("平安银行", lookback_weeks=120)
        
        print("\n数据预览：")
        print(df.head(10))
        print(f"\n数据范围: {df['date'].min()} 至 {df['date'].max()}")
        print(f"数据条数: {len(df)}")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("\n请检查：")
        print("1. 是否设置了TUSHARE_TOKEN环境变量？")
        print("2. Token是否有效？")
        print("3. 账户积分是否足够（需要≥120积分）？")
        print("\n参考文档: tushare_guide.md")

