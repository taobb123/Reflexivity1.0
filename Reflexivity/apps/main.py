"""
主程序入口
命令行工具，用于分析股票价格与市场背离
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.analyze_stock_unified import analyze_with_unified_fetcher
from config.settings import Config


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='股票价格与市场背离分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --stock 平安银行
  python main.py --stock 000001 --weeks 120
  python main.py --stock 平安银行 --preferred-sources tushare akshare
        """
    )
    
    parser.add_argument('--stock', type=str, default=Config.DEFAULT_STOCK,
                       help=f'股票名称或代码(默认: {Config.DEFAULT_STOCK})')
    parser.add_argument('--weeks', type=int, default=Config.DEFAULT_LOOKBACK_WEEKS,
                       help=f'回溯周数(默认: {Config.DEFAULT_LOOKBACK_WEEKS})')
    parser.add_argument('--tushare-token', type=str, default=None,
                       help='Tushare Token(也可设置环境变量TUSHARE_TOKEN)')
    parser.add_argument('--preferred-sources', type=str, nargs='+',
                       default=None,
                       help='优先使用的数据源列表')
    parser.add_argument('--output', type=str, default=Config.DEFAULT_OUTPUT_DIR,
                       help=f'结果保存目录(默认: {Config.DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # 使用配置
    preferred = Config.get_preferred_sources(args.preferred_sources)
    tushare_token = Config.get_tushare_token(args.tushare_token)
    
    print("="*60)
    print("  股票价格与市场背离分析")
    print("="*60)
    print(f"\n配置:")
    print(f"  股票: {args.stock}")
    print(f"  回溯周数: {args.weeks}")
    print(f"  输出目录: {args.output}")
    print(f"  优先数据源: {preferred}")
    print()
    
    analyze_with_unified_fetcher(
        stock_name=args.stock,
        lookback_weeks=args.weeks,
        save_dir=args.output,
        tushare_token=tushare_token,
        preferred_sources=preferred
    )


if __name__ == "__main__":
    main()
