# 股票价格与市场背离分析工具

基于索罗斯反身性理论的股票价格与基本面背离分析工具。

## 项目结构

```
Reflexivity/
├── core/                    # 核心功能
│   ├── reflexivity_model.py      # 反身性模型
│   ├── parameter_estimator.py    # 参数估计
│   └── divergence_analyzer.py    # 背离分析（核心）
├── tools/                   # 工具模块
│   ├── data_fetchers/      # 数据获取工具
│   └── visualization/      # 可视化工具
├── config/                  # 配置
├── scripts/                 # 执行脚本
├── tests/                   # 测试
├── apps/                    # 应用入口
│   └── main.py             # 主入口（推荐）
└── docs/                    # 文档
```

详细结构见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

#### 方式1: 使用根目录启动脚本（最简单，推荐）

**Windows:**
```bash
run.bat --stock 平安银行 --weeks 120
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh --stock 平安银行 --weeks 120
```

**跨平台（Python脚本）:**
```bash
python run.py --stock 平安银行 --weeks 120
```

#### 方式2: 使用主程序入口
```bash
python apps/main.py --stock 平安银行 --weeks 120
```

#### 方式3: 直接使用脚本
```bash
python scripts/analyze_stock_unified.py --stock 平安银行
```

详细使用说明请查看 [START_HERE.md](START_HERE.md)

### 启动Web应用

**Windows:**
```bash
run_app.bat
```

**跨平台:**
```bash
python run_app.py
```

启动后访问: http://localhost:5000

### Python代码中使用

```python
from core.divergence_analyzer import analyze_stock_divergence
from tools.data_fetchers.data_fetcher_unified import UnifiedDataFetcher

# 获取数据
fetcher = UnifiedDataFetcher()
df, code, info = fetcher.fetch_complete_data("平安银行")

# 分析背离
results = analyze_stock_divergence(df)
print(results['interpretation'])
```

## 核心功能

### 1. 价格与市场背离分析

分析股票价格与基本面的背离程度，识别市场高估/低估情况。

### 2. 反身性参数估计

从真实数据反推反身性模型参数（α、γ、β），评估市场反身性强度。

### 3. 多数据源支持

支持Tushare、yfinance、pandas-datareader、AKShare，自动回退。

## 配置

### API Keys

```bash
# Windows PowerShell
$env:TUSHARE_TOKEN="your_token"

# Linux/Mac
export TUSHARE_TOKEN="your_token"
```

### 配置文件

编辑 `config/settings.py` 修改默认配置。

## 文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [数据源使用指南](docs/UNIFIED_DATA_FETCHER_GUIDE.md)
- [快速开始指南](docs/QUICK_START.md)

## 许可证

MIT License
