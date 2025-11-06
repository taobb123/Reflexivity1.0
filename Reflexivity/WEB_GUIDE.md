# Web应用快速启动指南

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动服务器
```bash
python run_web.py
```

### 3. 打开浏览器
访问: **http://localhost:5000**

## 📋 功能说明

### 参数设置
- **α (alpha)**: 价格在认知中的权重 [0-2]
  - 越大表示市场越把价格当作基本面信号
  - >1 表示极端反身性
  
- **γ (gamma)**: 价格调整速度 [≥0]
  - 越大价格向市场预期调整越快
  
- **β (beta)**: 价格对基本面的影响 [≥0]
  - 越大价格对基本面影响越强

- **P₀**: 初始价格
- **F₀**: 初始基本面
- **噪声标准差**: 信息冲击的随机性
- **时间步数 T**: 仿真步数 [10-1000]

### 预设场景
点击预设按钮快速加载：
- **稳定收敛**: α=0.8, γ=0.5, β=0.1 (|λ|=0.8 < 1)
- **接近临界**: α=0.95, γ=0.8, β=0.05 (|λ|≈0.91)
- **泡沫发散**: α=1.2, γ=0.8, β=0.05 (|λ|>1)

### 结果展示
- 📊 **价格、基本面与市场预期图**: 展示三者随时间的变化
- 📈 **差异演化图**: 展示价格与基本面的差异
- 📉 **对数尺度差异图**: 观察收敛/发散速率
- 🔍 **稳定性分析**: 自动计算λ值和稳定性结论

## 🛠️ 技术栈
- **后端**: Flask
- **前端**: HTML + CSS + JavaScript
- **图表**: Matplotlib (base64编码嵌入)
- **模型**: 自定义反身性模型

## ⚙️ 配置
默认配置：
- 服务器地址: `0.0.0.0:5000`
- 调试模式: 开启
- 图表DPI: 100

如需修改，编辑 `app.py` 或 `run_web.py`

## 🐛 常见问题

### 1. 端口被占用
如果5000端口被占用，修改 `app.py` 中的端口号：
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # 改为其他端口
```

### 2. 中文显示问题
如果图表中文显示为方块，确保系统安装了中文字体：
- Windows: SimHei 或 Microsoft YaHei
- Mac: Arial Unicode MS
- Linux: 安装中文字体包

### 3. 依赖安装失败
确保使用Python 3.7+版本，并更新pip：
```bash
python --version
pip install --upgrade pip
pip install -r requirements.txt
```

## 📝 开发说明

### 文件结构
```
├── app.py              # Flask后端应用
├── run_web.py          # 启动脚本
├── templates/
│   └── index.html     # 前端页面
└── static/            # 静态资源目录（可选）
```

### API接口
- `GET /`: 主页面
- `POST /api/simulate`: 运行仿真
- `GET /api/presets`: 获取预设参数

### 自定义开发
1. 修改前端样式: 编辑 `templates/index.html`
2. 添加新功能: 修改 `app.py`
3. 扩展模型: 修改 `reflexivity_model.py`

