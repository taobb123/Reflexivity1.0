# 快速参考卡片

## 🚀 最常用功能

### 1. 完整分析（一行代码）

```python
from apps.reflexivity_analyzer import ReflexivityAnalyzer

results = ReflexivityAnalyzer().analyze("平安银行", lookback_weeks=120)
print(results['conclusion'])
```

### 2. 从 DataFrame 分析

```python
import pandas as pd
from apps.reflexivity_analyzer import ReflexivityAnalyzer

df = pd.DataFrame({'P_t': prices, 'F_t': fundamentals})
results = ReflexivityAnalyzer().analyze_from_dataframe(df)
```

### 3. 获取特定结果

```python
results = analyzer.analyze("平安银行")

# 阶段
stage = results['stage_results']['stage']

# 参数
alpha = results['parameter_results']['parameters']['alpha']
lambda_val = results['parameter_results']['lambda']

# 结论
conclusion = results['conclusion']
```

---

## 📊 组件快速使用

### 数据获取

```python
from apps.components.data_providers import UnifiedDataProvider

provider = UnifiedDataProvider()
df = provider.get_dataframe("平安银行", lookback_weeks=120)
price = provider.get_price_data("平安银行")
fundamental = provider.get_fundamental_data("平安银行")
```

### 图表拟合

```python
from apps.components.chart_fitters import PriceFundamentalChartFitter

fitter = PriceFundamentalChartFitter()
results = fitter.fit_charts(price_data, fundamental_data)
r_squared = results['fit_metrics']['r_squared']
```

### 参数估计

```python
from apps.components.parameter_estimators import ReflexivityParameterEstimator

estimator = ReflexivityParameterEstimator()
results = estimator.estimate(price_data, fundamental_data)
alpha = results['parameters']['alpha']
lambda_val = results['lambda']
```

### 阶段检测

```python
from apps.components.stage_detectors import ComprehensiveStageDetector

detector = ComprehensiveStageDetector()
stage = detector.detect_stage(parameters, price_data, fundamental_data, lambda_val)
print(stage['stage'], stage['confidence'])
```

### 结论生成

```python
from apps.components.conclusion_generators import ChineseConclusionGenerator

generator = ChineseConclusionGenerator()
conclusion = generator.generate(parameters, stage_result, fit_results)
```

### 图表可视化

```python
from apps.components.chart_visualizers import MatplotlibChartVisualizer

visualizer = MatplotlibChartVisualizer()
fig = visualizer.visualize_fit(price_data, fundamental_data, fit_results)
visualizer.save_chart(fig, 'chart.png')
```

---

## 🔧 自定义配置

### 使用多项式拟合

```python
from apps.components.chart_fitters import PriceFundamentalChartFitter
from apps.components.fitters import PolynomialFitter

fitter = PriceFundamentalChartFitter(fitter=PolynomialFitter(degree=2))
```

### 使用自定义优化器

```python
from apps.components.parameter_estimators import ReflexivityParameterEstimator
from apps.components.optimizers import LBFGSOptimizer

estimator = ReflexivityParameterEstimator(optimizer=LBFGSOptimizer())
```

### 使用自定义数据源

```python
from apps.components.data_providers import UnifiedDataProvider

provider = UnifiedDataProvider(
    tushare_token="your_token",
    preferred_sources=['akshare']
)
```

---

## 🌐 Web API 快速调用

### Python

```python
import requests

response = requests.post('http://127.0.0.1:5000/api/analyze_stock', json={
    'stock_code': '平安银行',
    'lookback_weeks': 120
})
result = response.json()
```

### JavaScript

```javascript
fetch('/api/analyze_stock', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({stock_code: '平安银行', lookback_weeks: 120})
}).then(r => r.json()).then(console.log)
```

### curl

```bash
curl -X POST http://127.0.0.1:5000/api/analyze_stock \
  -H "Content-Type: application/json" \
  -d '{"stock_code":"平安银行","lookback_weeks":120}'
```

---

## 📈 阶段说明

| 阶段 | λ值范围 | 特征 |
|------|---------|------|
| 稳定收敛 | |λ| < 1 | 价格与基本面会收敛 |
| 临界状态 | |λ| ≈ 1 | 系统对参数变化敏感 |
| 泡沫形成 | λ > 1 | 背离扩大，可能形成泡沫 |
| 泡沫破灭 | λ 回落 | 背离开始缩小 |
| 崩溃 | λ < -1 | 振荡发散，可能崩溃 |

---

## 🔍 参数说明

| 参数 | 符号 | 含义 | 范围 |
|------|------|------|------|
| α (alpha) | α | 价格在认知中的权重 | ≥ 0（可>1） |
| γ (gamma) | γ | 价格调整速度 | ≥ 0 |
| β (beta) | β | 价格对基本面的影响强度 | ≥ 0 |
| λ (lambda) | λ | 系统特征值 | λ = 1 + γ(α-1) - β |

---

## 📝 常用检查清单

- [ ] 数据是否包含 'P_t' 和 'F_t' 列？
- [ ] 数据长度是否足够（建议 > 50 个数据点）？
- [ ] 是否安装了所有依赖（`pip install -r requirements.txt`）？
- [ ] Web API 是否已启动（`python apps/run_web.py`）？

---

## 📚 相关文档

- **详细使用指南**: `apps/USAGE_GUIDE.md`
- **系统架构说明**: `apps/README.md`
- **使用示例**: `apps/example_usage.py`
- **项目总览**: 项目根目录 `README.md`

