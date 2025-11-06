# 快速开始指南

## 启动方式

### 方式1: 使用启动脚本（推荐）

#### Windows:
```bash
run.bat --stock 平安银行 --weeks 120
```

#### Linux/Mac:
```bash
chmod +x run.sh
./run.sh --stock 平安银行 --weeks 120
```

#### 跨平台（Python脚本）:
```bash
python run.py --stock 平安银行 --weeks 120
```

### 方式2: 直接使用主程序
```bash
python apps/main.py --stock 平安银行 --weeks 120
```

## 常用命令

### 基本分析
```bash
python run.py --stock 平安银行
```

### 指定回溯周数
```bash
python run.py --stock 平安银行 --weeks 120
```

### 指定数据源优先级
```bash
python run.py --stock 平安银行 --preferred-sources tushare akshare
```

### 指定输出目录
```bash
python run.py --stock 平安银行 --output my_results
```

### 使用Tushare Token
```bash
python run.py --stock 平安银行 --tushare-token your_token
```

## 参数说明

- `--stock`: 股票名称或代码（默认: 平安银行）
- `--weeks`: 回溯周数（默认: 120）
- `--tushare-token`: Tushare Token（可选，也可设置环境变量）
- `--preferred-sources`: 优先使用的数据源列表
- `--output`: 结果保存目录（默认: results）

## 环境变量配置

```bash
# Windows PowerShell
$env:TUSHARE_TOKEN="your_token"

# Linux/Mac
export TUSHARE_TOKEN="your_token"
```

## 查看帮助

```bash
python run.py --help
```

## 启动Web应用

### Windows:
```bash
run_app.bat
```

### 跨平台:
```bash
python run_app.py
```

启动后访问: http://localhost:5000

## 示例输出

运行后会生成：
- 数据文件：`results/{股票名}_unified_data.csv`
- 结果文件：`results/{股票名}_unified_results.txt`
- 图表文件：`results/{股票名}_unified_chart.png`
- 价格曲线：`results/{股票名}_unified_price_history.png`

