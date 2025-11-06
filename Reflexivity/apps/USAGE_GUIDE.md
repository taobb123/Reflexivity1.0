# 股票反身性分析系统 - 详细使用指南

## 目录

1. [快速开始](#快速开始)
2. [核心功能使用方法](#核心功能使用方法)
   - [数据获取](#1-数据获取)
   - [图表拟合](#2-图表拟合)
   - [参数估计](#3-参数估计)
   - [阶段检测](#4-阶段检测)
   - [结论生成](#5-结论生成)
   - [图表可视化](#6-图表可视化)
3. [完整分析流程](#完整分析流程)
4. [自定义组件](#自定义组件)
5. [Web API 使用](#web-api-使用)
6. [常见问题](#常见问题)

---

## 快速开始

### 最简单的使用方式

```python
from apps.reflexivity_analyzer import ReflexivityAnalyzer

# 创建分析器
analyzer = ReflexivityAnalyzer()

# 执行完整分析
results = analyzer.analyze(
    stock_code="平安银行",
    lookback_weeks=120
)

# 查看结果
print(results['conclusion'])  # 分析结论
print(results['stage_results']['stage'])  # 检测到的阶段
```

---

## 核心功能使用方法

### 1. 数据获取

#### 1.1 使用统一数据提供者（支持多种数据源）

```python
from apps.components.data_providers import UnifiedDataProvider

# 创建数据提供者
data_provider = UnifiedDataProvider(
    tushare_token="your_token",  # 可选
    preferred_sources=['akshare', 'tushare']  # 可选，指定优先数据源
)

# 获取价格数据
price_data = data_provider.get_price_data(
    stock_code="平安银行",
    lookback_weeks=120
)

# 获取基本面数据
fundamental_data = data_provider.get_fundamental_data(
    stock_code="平安银行",
    lookback_weeks=120
)

# 获取完整数据框
df = data_provider.get_dataframe(
    stock_code="平安银行",
    lookback_weeks=120
)

# 获取数据信息
info = data_provider.get_data_info()
print(f"数据点数量: {info['data_points']}")
print(f"数据源: {info['sources']}")
```

#### 1.2 从 DataFrame 提供数据

```python
import pandas as pd
import numpy as np
from apps.components.data_providers import DataFrameDataProvider

# 准备数据（必须包含 'P_t' 和 'F_t' 列）
df = pd.DataFrame({
    'P_t': price_array,
    'F_t': fundamental_array
})

# 创建 DataFrame 数据提供者
data_provider = DataFrameDataProvider(df)

# 获取数据
price_data = data_provider.get_price_data()
fundamental_data = data_provider.get_fundamental_data()
```

---

### 2. 图表拟合

#### 2.1 使用默认图表拟合器（线性拟合）

```python
from apps.components.chart_fitters import PriceFundamentalChartFitter
import numpy as np

# 创建图表拟合器（默认使用线性拟合）
chart_fitter = PriceFundamentalChartFitter()

# 执行拟合
results = chart_fitter.fit_charts(
    price_data=price_data,
    fundamental_data=fundamental_data
)

# 查看拟合结果
print(f"相关性: {results['correlation']['value']:.4f}")
print(f"R²: {results['fit_metrics']['r_squared']:.4f}")
print(f"拟合质量: {results['fit_metrics']['fit_quality']}")
print(f"平均背离: {results['divergence']['mean']:.4f}")
```

#### 2.2 使用自定义拟合器

```python
from apps.components.chart_fitters import PriceFundamentalChartFitter
from apps.components.fitters import PolynomialFitter, NonlinearFitter

# 使用多项式拟合器
poly_fitter = PolynomialFitter(degree=2)
chart_fitter = PriceFundamentalChartFitter(fitter=poly_fitter)
results = chart_fitter.fit_charts(price_data, fundamental_data)

# 使用非线性拟合器（指数）
exp_fitter = NonlinearFitter(func_type='exponential')
chart_fitter = PriceFundamentalChartFitter(fitter=exp_fitter)
results = chart_fitter.fit_charts(price_data, fundamental_data)

# 使用非线性拟合器（对数）
log_fitter = NonlinearFitter(func_type='logarithmic')
chart_fitter = PriceFundamentalChartFitter(fitter=log_fitter)
results = chart_fitter.fit_charts(price_data, fundamental_data)

# 使用非线性拟合器（幂函数）
power_fitter = NonlinearFitter(func_type='power')
chart_fitter = PriceFundamentalChartFitter(fitter=power_fitter)
results = chart_fitter.fit_charts(price_data, fundamental_data)
```

#### 2.3 使用双向图表拟合器

```python
from apps.components.chart_fitters import BidirectionalChartFitter

# 创建双向拟合器
bidirectional_fitter = BidirectionalChartFitter()

# 执行双向拟合
results = bidirectional_fitter.fit_charts(
    price_data=price_data,
    fundamental_data=fundamental_data
)

# 查看双向拟合结果
print(f"价格->基本面 R²: {results['fit_metrics']['price_to_fundamental_r2']:.4f}")
print(f"基本面->价格 R²: {results['fit_metrics']['fundamental_to_price_r2']:.4f}")
```

---

### 3. 参数估计

#### 3.1 使用默认参数估计器

```python
from apps.components.parameter_estimators import ReflexivityParameterEstimator
import numpy as np

# 创建参数估计器
estimator = ReflexivityParameterEstimator()

# 估计参数
results = estimator.estimate(
    price_data=price_data,
    fundamental_data=fundamental_data,
    method='differential_evolution'  # 或 'minimize'
)

# 查看估计结果
params = results['parameters']
print(f"α (价格在认知中的权重): {params['alpha']:.4f}")
print(f"γ (价格调整速度): {params['gamma']:.4f}")
print(f"β (价格对基本面的影响): {params['beta']:.4f}")
print(f"λ (系统特征值): {results['lambda']:.4f}")
print(f"稳定性: {results['stability']}")

# 查看拟合效果
fitness = results['fitness']
print(f"R²: {fitness['r_squared']:.4f}")
print(f"RMSE: {fitness['rmse']:.4f}")
print(f"MAE: {fitness['mae']:.4f}")
```

#### 3.2 使用自定义优化器

```python
from apps.components.parameter_estimators import ReflexivityParameterEstimator
from apps.components.optimizers import DifferentialEvolutionOptimizer, LBFGSOptimizer

# 使用差分进化优化器
de_optimizer = DifferentialEvolutionOptimizer()
estimator = ReflexivityParameterEstimator(optimizer=de_optimizer)
results = estimator.estimate(price_data, fundamental_data)

# 使用 L-BFGS-B 优化器（局部优化，更快）
lbfgs_optimizer = LBFGSOptimizer()
estimator = ReflexivityParameterEstimator(optimizer=lbfgs_optimizer)
results = estimator.estimate(price_data, fundamental_data)
```

---

### 4. 阶段检测

#### 4.1 使用默认阶段检测器

```python
from apps.components.stage_detectors import ComprehensiveStageDetector
import numpy as np

# 创建阶段检测器
detector = ComprehensiveStageDetector()

# 检测阶段（需要先估计参数）
from apps.components.parameter_estimators import ReflexivityParameterEstimator

estimator = ReflexivityParameterEstimator()
param_results = estimator.estimate(price_data, fundamental_data)

# 执行阶段检测
stage_results = detector.detect_stage(
    parameters=param_results['parameters'],
    price_data=price_data,
    fundamental_data=fundamental_data,
    lambda_value=param_results['lambda']
)

# 查看检测结果
print(f"检测到的阶段: {stage_results['stage']}")
print(f"置信度: {stage_results['confidence']:.2%}")
print(f"风险等级: {stage_results['risk_level']}")
print(f"\n阶段描述:\n{stage_results['description']}")

# 查看各项指标
indicators = stage_results['indicators']
print(f"\n各项指标:")
for key, value in indicators.items():
    print(f"  {key}: {value}")

# 查看各阶段匹配分数
print(f"\n各阶段匹配分数:")
for stage, score in sorted(stage_results['stage_scores'].items(), 
                          key=lambda x: x[1], reverse=True):
    print(f"  {stage}: {score:.2%}")
```

#### 4.2 获取所有可用阶段和判断标准

```python
from apps.components.stage_detectors import ComprehensiveStageDetector

detector = ComprehensiveStageDetector()

# 获取所有可用阶段
stages = detector.get_available_stages()
print("可用阶段:", stages)
# 输出: ['稳定收敛', '临界状态', '泡沫形成', '泡沫破灭', '崩溃']

# 获取各阶段的判断标准
criteria = detector.get_stage_criteria()
for stage_name, stage_criteria in criteria.items():
    print(f"\n{stage_name}:")
    print(f"  Lambda范围: {stage_criteria['lambda_range']}")
    print(f"  Alpha范围: {stage_criteria['alpha_range']}")
    print(f"  趋势稳定性: {stage_criteria['trend_stable']}")
    print(f"  背离趋势: {stage_criteria['divergence_trend']}")
```

---

### 5. 结论生成

#### 5.1 使用默认结论生成器

```python
from apps.components.conclusion_generators import ChineseConclusionGenerator

# 创建结论生成器
generator = ChineseConclusionGenerator()

# 生成结论（需要先完成拟合和阶段检测）
# ... 执行拟合和阶段检测 ...

conclusion = generator.generate(
    parameters=param_results['parameters'],
    stage_result=stage_results,
    fit_results=fit_results
)

print(conclusion)
```

#### 5.2 生成详细结论

```python
# 生成详细结论（包含更多指标）
all_results = {
    'parameters': param_results['parameters'],
    'stage_result': stage_results,
    'fit_results': fit_results
}

detailed_conclusion = generator.generate_detailed(all_results)
print(detailed_conclusion)
```

---

### 6. 图表可视化

#### 6.1 可视化拟合结果

```python
from apps.components.chart_visualizers import MatplotlibChartVisualizer

# 创建可视化器
visualizer = MatplotlibChartVisualizer()

# 生成拟合图表
fig = visualizer.visualize_fit(
    price_data=price_data,
    fundamental_data=fundamental_data,
    fit_results=fit_results
)

# 保存图表
visualizer.save_chart(fig, 'fit_chart.png', dpi=300)

# 或转换为 base64（用于 Web 显示）
chart_base64 = visualizer.chart_to_base64(fig)
```

#### 6.2 可视化对比图（实际 vs 预测）

```python
# 可视化实际数据与预测数据的对比
comparison_fig = visualizer.visualize_comparison(
    actual_data={
        'price': price_data,
        'fundamental': fundamental_data
    },
    predicted_data={
        'price': param_results['predicted_data']['price'],
        'fundamental': param_results['predicted_data']['fundamental']
    }
)

visualizer.save_chart(comparison_fig, 'comparison_chart.png')
```

---

## 完整分析流程

### 方式1：使用主协调器（推荐）

```python
from apps.reflexivity_analyzer import ReflexivityAnalyzer

# 创建分析器
analyzer = ReflexivityAnalyzer()

# 执行完整分析
results = analyzer.analyze(
    stock_code="平安银行",
    lookback_weeks=120,
    save_charts=True,
    chart_save_path="results"
)

# 查看所有结果
print("=" * 60)
print("数据信息:")
print(results['data_info'])

print("\n拟合结果:")
print(f"R²: {results['fit_results']['fit_metrics']['r_squared']:.4f}")

print("\n参数估计结果:")
params = results['parameter_results']['parameters']
print(f"α={params['alpha']:.4f}, γ={params['gamma']:.4f}, β={params['beta']:.4f}")
print(f"λ={results['parameter_results']['lambda']:.4f}")

print("\n阶段检测结果:")
print(f"阶段: {results['stage_results']['stage']}")
print(f"置信度: {results['stage_results']['confidence']:.2%}")

print("\n分析结论:")
print(results['conclusion'])
```

### 方式2：手动组合各个组件

```python
import numpy as np
from apps.components.data_providers import UnifiedDataProvider
from apps.components.chart_fitters import PriceFundamentalChartFitter
from apps.components.parameter_estimators import ReflexivityParameterEstimator
from apps.components.stage_detectors import ComprehensiveStageDetector
from apps.components.conclusion_generators import ChineseConclusionGenerator
from apps.components.chart_visualizers import MatplotlibChartVisualizer

# 1. 获取数据
data_provider = UnifiedDataProvider()
df = data_provider.get_dataframe("平安银行", lookback_weeks=120)
price_data = df['P_t'].values
fundamental_data = df['F_t'].values

# 2. 图表拟合
chart_fitter = PriceFundamentalChartFitter()
fit_results = chart_fitter.fit_charts(price_data, fundamental_data)

# 3. 参数估计
parameter_estimator = ReflexivityParameterEstimator()
param_results = parameter_estimator.estimate(price_data, fundamental_data)

# 4. 阶段检测
stage_detector = ComprehensiveStageDetector()
stage_results = stage_detector.detect_stage(
    parameters=param_results['parameters'],
    price_data=price_data,
    fundamental_data=fundamental_data,
    lambda_value=param_results['lambda']
)

# 5. 生成结论
conclusion_generator = ChineseConclusionGenerator()
conclusion = conclusion_generator.generate(
    parameters=param_results['parameters'],
    stage_result=stage_results,
    fit_results=fit_results
)

# 6. 可视化
visualizer = MatplotlibChartVisualizer()
fig = visualizer.visualize_fit(price_data, fundamental_data, fit_results)
visualizer.save_chart(fig, 'analysis_result.png')
```

---

## 自定义组件

### 创建自定义拟合器

```python
from apps.interfaces.fitter import IFitter
import numpy as np
from typing import Dict, Any

class CustomFitter(IFitter):
    """自定义拟合器示例"""
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        # 实现你的拟合逻辑
        # ...
        return {
            'parameters': {...},
            'residuals': [...],
            'r_squared': 0.95,
            'fit_quality': 'excellent'
        }
    
    def predict(self, x: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        # 实现预测逻辑
        # ...
        return predicted_values
    
    def get_fit_type(self) -> str:
        return 'custom'
```

### 创建自定义阶段检测器

```python
from apps.interfaces.stage_detector import IStageDetector
from typing import Dict, Any, List
import numpy as np

class CustomStageDetector(IStageDetector):
    """自定义阶段检测器示例"""
    
    def detect_stage(self, parameters, price_data, fundamental_data, 
                     lambda_value, **kwargs) -> Dict[str, Any]:
        # 实现你的阶段检测逻辑
        # ...
        return {
            'stage': '稳定收敛',
            'confidence': 0.95,
            'indicators': {...},
            'description': '...',
            'risk_level': '低风险'
        }
    
    def get_available_stages(self) -> List[str]:
        return ['稳定收敛', '临界状态', '泡沫形成']
    
    def get_stage_criteria(self) -> Dict[str, Dict[str, Any]]:
        return {...}
```

---

## Web API 使用

### 使用 curl 调用 API

```bash
# POST 请求
curl -X POST http://127.0.0.1:5000/api/analyze_stock \
  -H "Content-Type: application/json" \
  -d '{
    "stock_code": "平安银行",
    "lookback_weeks": 120,
    "preferred_sources": ["akshare"]
  }'
```

### 使用 Python requests

```python
import requests
import json

url = "http://127.0.0.1:5000/api/analyze_stock"
data = {
    "stock_code": "平安银行",
    "lookback_weeks": 120,
    "preferred_sources": ["akshare"]
}

response = requests.post(url, json=data)
result = response.json()

if result['success']:
    print(f"阶段: {result['data']['stage_results']['stage']}")
    print(f"结论: {result['data']['conclusion']}")
else:
    print(f"错误: {result['error']}")
```

### 使用 JavaScript (fetch)

```javascript
fetch('/api/analyze_stock', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        stock_code: '平安银行',
        lookback_weeks: 120,
        preferred_sources: ['akshare']
    })
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        console.log('阶段:', data.data.stage_results.stage);
        console.log('结论:', data.data.conclusion);
    } else {
        console.error('错误:', data.error);
    }
});
```

---

## 常见问题

### Q1: 如何选择拟合方法？

**A:** 根据数据特征选择：
- **线性拟合**：价格与基本面呈线性关系（默认，推荐）
- **多项式拟合**：存在非线性趋势
- **指数拟合**：增长/衰减趋势
- **对数拟合**：增长率递减
- **幂函数拟合**：幂律关系

### Q2: 如何选择优化器？

**A:** 
- **差分进化**（默认）：全局优化，适合复杂问题，但较慢
- **L-BFGS-B**：局部优化，速度快，需要好的初始值
- **梯度下降**：简单快速，但可能陷入局部最优

### Q3: 阶段检测的置信度如何理解？

**A:** 置信度表示检测结果的可信程度：
- > 0.8：高置信度，结果可靠
- 0.5-0.8：中等置信度
- < 0.5：低置信度，建议结合其他指标判断

### Q4: 如何提高拟合质量？

**A:** 
1. 增加数据量（更多历史数据）
2. 尝试不同的拟合方法
3. 检查数据质量（去除异常值）
4. 调整参数估计的边界条件

### Q5: 支持哪些数据源？

**A:** 目前支持：
- akshare（默认优先）
- tushare（需要 token）
- 其他数据源可通过实现 `IDataProvider` 接口添加

---

## 更多示例

详细示例代码请查看 `apps/example_usage.py`

---

## 技术支持

如有问题，请查看：
- `apps/README.md` - 系统架构说明
- `apps/example_usage.py` - 使用示例
- 项目根目录的 `README.md` - 项目总体说明

