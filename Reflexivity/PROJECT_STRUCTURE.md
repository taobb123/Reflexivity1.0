# 项目目录结构

## 目录组织

```
Reflexivity/
├── core/                          # 核心功能模块
│   ├── __init__.py
│   ├── reflexivity_model.py       # 反身性模型
│   ├── parameter_estimator.py     # 参数估计
│   └── divergence_analyzer.py     # 背离分析（核心功能）
│
├── tools/                          # 工具模块
│   ├── data_fetchers/             # 数据获取工具
│   │   ├── __init__.py
│   │   ├── data_fetcher_unified.py      # 统一数据获取器
│   │   ├── data_fetcher.py              # Tushare数据获取器
│   │   ├── data_fetcher_akshare.py      # AKShare数据获取器
│   │   └── data_fetcher_hybrid.py       # 混合数据获取器
│   │
│   └── visualization/             # 可视化工具
│       ├── __init__.py
│       ├── plot_reflexivity.py     # 反身性模型可视化
│       └── plot_demo.py            # 演示图表
│
├── config/                         # 配置模块
│   ├── __init__.py
│   └── settings.py                 # 配置设置
│
├── scripts/                        # 执行脚本
│   ├── analyze_stock.py            # 基础分析脚本
│   ├── analyze_stock_unified.py    # 统一数据源分析脚本
│   ├── analyze_stock_multi_source.py # 多数据源分析脚本
│   ├── stock_backtest.py           # 回测脚本
│   └── quick_test.py               # 快速测试脚本
│
├── tests/                          # 测试文件
│   ├── test_hybrid.py
│   ├── test_tushare_interactive.py
│   ├── test_tushare.py
│   ├── test_unified_fetcher.py
│   └── test_yfinance_akshare.py
│
├── apps/                           # 应用入口
│   ├── app.py                      # Flask Web应用
│   ├── run_web.py                  # Web应用启动脚本
│   └── main.py                     # 命令行主入口
│
├── docs/                           # 文档
│   ├── README.md
│   ├── QUICK_START.md
│   ├── STREAMLIT_GUIDE.md
│   ├── UNIFIED_DATA_FETCHER_GUIDE.md
│   └── ...
│
├── results/                        # 结果输出目录
├── templates/                      # Web模板
├── static/                         # 静态文件
│
├── requirements.txt                 # 依赖包
└── PROJECT_STRUCTURE.md           # 本文件
```

## 模块说明

### core/ - 核心功能
- **reflexivity_model.py**: 实现反身性理论模型，包含价格和基本面的动态关系
- **parameter_estimator.py**: 从真实数据反推模型参数
- **divergence_analyzer.py**: **核心功能** - 分析价格与市场背离

### tools/ - 工具模块
- **data_fetchers/**: 多数据源数据获取工具，支持Tushare、yfinance、pandas-datareader、AKShare
- **visualization/**: 数据可视化工具

### config/ - 配置
- **settings.py**: 集中管理配置参数（API Keys、默认参数等）

### scripts/ - 执行脚本
- 各种分析脚本，用于不同的使用场景

### apps/ - 应用入口
- **main.py**: 命令行主入口，推荐使用
- **app.py**: Flask Web应用
- **run_web.py**: Web应用启动脚本

## 使用方式

### 命令行使用（推荐）
```bash
python apps/main.py --stock 平安银行 --weeks 120
```

### 直接使用脚本
```bash
python scripts/analyze_stock_unified.py --stock 平安银行
```

### Python代码中使用
```python
from core.divergence_analyzer import analyze_stock_divergence
from tools.data_fetchers.data_fetcher_unified import UnifiedDataFetcher

# 获取数据
fetcher = UnifiedDataFetcher()
df, code, info = fetcher.fetch_complete_data("平安银行")

# 分析背离
results = analyze_stock_divergence(df)
```

## 导入路径

所有模块都使用相对导入，确保项目结构清晰：

- 核心模块: `from core.module import Class`
- 工具模块: `from tools.data_fetchers.module import Class`
- 配置: `from config.settings import Config`

