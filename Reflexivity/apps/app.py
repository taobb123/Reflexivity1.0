"""
反身性模型Web应用
Flask MVP核心实现
使用接口化设计，支持股票反身性分析
"""

from flask import Flask, render_template, request, jsonify
import base64
import io
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.reflexivity_model import ReflexivityModel
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 导入新的接口化分析器
from apps.reflexivity_analyzer import ReflexivityAnalyzer
from apps.components.data_providers import UnifiedDataProvider

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 获取项目根目录
project_root = Path(__file__).parent.parent

# 创建Flask应用，指定模板和静态文件目录
app = Flask(
    __name__,
    template_folder=str(project_root / 'templates'),
    static_folder=str(project_root / 'static')
)

# 创建全局分析器实例（可以配置）
analyzer = ReflexivityAnalyzer()


def make_json_serializable(obj):
    """
    将对象转换为 JSON 可序列化的格式
    处理 numpy 类型、pandas 类型等
    """
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        # 尝试转换为基本类型
        try:
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
        except (ValueError, AttributeError):
            pass
        return obj


def fig_to_base64(fig):
    """将matplotlib图表转换为base64编码的字符串"""
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return img_base64


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """仿真API接口"""
    try:
        # 获取参数
        data = request.json
        alpha = float(data.get('alpha', 0.8))
        gamma = float(data.get('gamma', 0.5))
        beta = float(data.get('beta', 0.1))
        P0 = float(data.get('P0', 100.0))
        F0 = float(data.get('F0', 100.0))
        noise_std = float(data.get('noise_std', 1.0))
        T = int(data.get('T', 100))
        
        # 参数验证
        if alpha < 0:
            return jsonify({'error': 'α必须大于等于0'}), 400
        if gamma < 0:
            return jsonify({'error': 'γ必须大于等于0'}), 400
        if beta < 0:
            return jsonify({'error': 'β必须大于等于0'}), 400
        if T <= 0 or T > 1000:
            return jsonify({'error': '时间步数必须在1-1000之间'}), 400
        
        # 创建模型
        model = ReflexivityModel(
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            P0=P0,
            F0=F0,
            noise_std=noise_std
        )
        
        # 运行仿真
        results = model.simulate(T)
        
        # 获取稳定性分析
        stability, stability_info = model.analyze_stability()
        model_info = model.get_model_info()
        
        # 生成图表
        chart_data = generate_charts(model, results, T)
        
        # 准备返回数据
        response_data = {
            'success': True,
            'data': {
                'results': {
                    'P': results['P'].tolist(),
                    'F': results['F'].tolist(),
                    'E': results['E'].tolist(),
                    'x': results['x'].tolist(),
                    't': results['t'].tolist()
                },
                'stability': {
                    'type': stability,
                    'lambda': float(model_info['lambda']),
                    'abs_lambda': float(abs(model_info['lambda'])),
                    'description': stability_info['description']
                },
                'parameters': {
                    'alpha': alpha,
                    'gamma': gamma,
                    'beta': beta,
                    'P0': P0,
                    'F0': F0,
                    'noise_std': noise_std,
                    'T': T
                },
                'charts': chart_data
            }
        }
        
        return jsonify(response_data)
        
    except ValueError as e:
        return jsonify({'error': f'参数错误: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'仿真失败: {str(e)}'}), 500


def generate_charts(model, results, T):
    """生成所有图表并返回base64编码"""
    charts = {}
    
    # 图表1：价格与基本面对比
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    t = results['t']
    P = results['P']
    F = results['F']
    E = results['E']
    
    ax1.plot(t, P, 'b-', label='价格 P_t', linewidth=2)
    ax1.plot(t, F, 'r--', label='基本面 F_t', linewidth=2)
    ax1.plot(t, E, 'g:', label='市场预期 E_t', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('时间步 t', fontsize=12)
    ax1.set_ylabel('数值', fontsize=12)
    ax1.set_title('价格、基本面与市场预期', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    charts['price_fundamental'] = fig_to_base64(fig1)
    
    # 图表2：差异演化
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    x = results['x']
    lambda_val = model.compute_lambda()
    stability, _ = model.analyze_stability()
    
    ax2.plot(t, x, 'purple', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('时间步 t', fontsize=12)
    ax2.set_ylabel('差异 x_t = P_t - F_t', fontsize=12)
    ax2.set_title(f'价格与基本面差异 (λ={lambda_val:.4f}, {stability})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    charts['difference'] = fig_to_base64(fig2)
    
    # 图表3：对数尺度差异
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    abs_x = np.abs(x)
    abs_x = np.where(abs_x < 1e-10, 1e-10, abs_x)
    ax3.plot(t, np.log10(abs_x), 'orange', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('时间步 t', fontsize=12)
    ax3.set_ylabel('log₁₀|差异|', fontsize=12)
    ax3.set_title('差异的对数尺度（观察收敛/发散）', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    charts['log_difference'] = fig_to_base64(fig3)
    
    return charts


@app.route('/api/presets', methods=['GET'])
def get_presets():
    """获取预设参数"""
    presets = {
        'stable': {
            'name': '稳定收敛',
            'alpha': 0.8,
            'gamma': 0.5,
            'beta': 0.1,
            'description': '系统稳定，差异收敛到0'
        },
        'critical': {
            'name': '接近临界',
            'alpha': 0.95,
            'gamma': 0.8,
            'beta': 0.05,
            'description': '接近临界状态，|λ|≈0.91'
        },
        'bubble': {
            'name': '泡沫发散',
            'alpha': 1.2,
            'gamma': 0.8,
            'beta': 0.05,
            'description': '系统发散，形成泡沫或崩溃'
        }
    }
    return jsonify(presets)


@app.route('/api/analyze_stock', methods=['GET', 'POST'])
def analyze_stock():
    """
    股票反身性分析API接口
    使用新的接口化设计
    
    GET: 返回 API 使用说明和测试页面
    POST: 执行股票反身性分析
    """
    if request.method == 'GET':
        # 返回使用说明和测试页面
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>股票反身性分析 API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .section {
            margin: 20px 0;
            padding: 15px;
            background: #f9f9f9;
            border-left: 4px solid #4CAF50;
        }
        .form-group {
            margin: 15px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        button {
            background: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 5px;
            display: none;
        }
        .error {
            background: #ffebee;
            color: #c62828;
        }
        pre {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        code {
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 股票反身性分析 API</h1>
        
        <div class="section">
            <h2>API 使用说明</h2>
            <p><strong>请求方法：</strong> POST</p>
            <p><strong>请求地址：</strong> <code>/api/analyze_stock</code></p>
            <p><strong>Content-Type：</strong> application/json</p>
            
            <h3>请求参数：</h3>
            <pre>{
    "stock_code": "平安银行",      // 必需：股票代码或名称
    "lookback_weeks": 120,         // 可选：回溯周数（默认120）
    "tushare_token": "your_token",  // 可选：Tushare token
    "preferred_sources": ["akshare"] // 可选：优先数据源
}</pre>
            
            <h3>响应示例：</h3>
            <pre>{
    "success": true,
    "data": {
        "stock_code": "平安银行",
        "data_info": {...},
        "fit_results": {...},
        "parameter_results": {...},
        "stage_results": {...},
        "conclusion": "分析结论...",
        "charts": {...}
    }
}</pre>
        </div>
        
        <div class="section">
            <h2>在线测试</h2>
            <form id="analyzeForm">
                <div class="form-group">
                    <label for="stock_code">股票代码/名称：</label>
                    <input type="text" id="stock_code" name="stock_code" 
                           value="平安银行" required>
                </div>
                
                <div class="form-group">
                    <label for="lookback_weeks">回溯周数：</label>
                    <input type="number" id="lookback_weeks" name="lookback_weeks" 
                           value="120" min="1" max="500">
                </div>
                
                <div class="form-group">
                    <label for="preferred_sources">优先数据源（可选）：</label>
                    <select id="preferred_sources" name="preferred_sources">
                        <option value="">默认</option>
                        <option value="akshare">akshare</option>
                        <option value="tushare">tushare</option>
                        <option value="akshare,tushare">akshare + tushare</option>
                    </select>
                </div>
                
                <button type="submit">🚀 开始分析</button>
            </form>
            
            <div id="result" class="result"></div>
        </div>
    </div>
    
    <script>
        document.getElementById('analyzeForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'result';
            resultDiv.innerHTML = '<p>⏳ 正在分析，请稍候...</p>';
            
            const formData = {
                stock_code: document.getElementById('stock_code').value,
                lookback_weeks: parseInt(document.getElementById('lookback_weeks').value),
                preferred_sources: document.getElementById('preferred_sources').value 
                    ? document.getElementById('preferred_sources').value.split(',') 
                    : null
            };
            
            try {
                const response = await fetch('/api/analyze_stock', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                // 检查响应状态
                if (!response.ok) {
                    // 尝试解析错误响应
                    let errorText = `HTTP ${response.status}: ${response.statusText}`;
                    try {
                        const errorData = await response.json();
                        errorText = errorData.error || errorText;
                    } catch (e) {
                        errorText = await response.text() || errorText;
                    }
                    throw new Error(errorText);
                }
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.className = 'result';
                    let html = '<h3>✅ 分析完成</h3>';
                    html += '<h4>📊 检测到的阶段：' + data.data.stage_results.stage + '</h4>';
                    html += '<p><strong>置信度：</strong>' + (data.data.stage_results.confidence * 100).toFixed(2) + '%</p>';
                    html += '<p><strong>风险等级：</strong>' + data.data.stage_results.risk_level + '</p>';
                    html += '<h4>📈 反身性参数：</h4>';
                    html += '<pre>' + JSON.stringify(data.data.parameter_results.parameters, null, 2) + '</pre>';
                    html += '<h4>💡 分析结论：</h4>';
                    html += '<pre style="white-space: pre-wrap;">' + data.data.conclusion + '</pre>';
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<h3>❌ 分析失败</h3><p>' + (data.error || '未知错误') + '</p>';
                }
            } catch (error) {
                resultDiv.className = 'result error';
                let errorMsg = error.message || '未知错误';
                if (errorMsg.includes('Failed to fetch') || errorMsg.includes('NetworkError')) {
                    errorMsg = '无法连接到服务器。请确保：<br>1. Flask 应用正在运行<br>2. 服务器地址正确（http://127.0.0.1:5000）<br>3. 没有防火墙阻止连接';
                }
                resultDiv.innerHTML = '<h3>❌ 请求失败</h3><p>' + errorMsg + '</p>';
                console.error('请求错误:', error);
            }
        });
    </script>
</body>
</html>
        """
        return html
    
    # POST 请求处理
    try:
        # 获取参数
        data = request.json or {}
        stock_code = data.get('stock_code', '')
        lookback_weeks = int(data.get('lookback_weeks', 120))
        tushare_token = data.get('tushare_token', None)
        preferred_sources = data.get('preferred_sources', None)
        
        if not stock_code:
            return jsonify({'error': '股票代码不能为空'}), 400
        
        # 创建数据提供者（如果提供了token或数据源）
        data_provider = None
        if tushare_token or preferred_sources:
            data_provider = UnifiedDataProvider(
                tushare_token=tushare_token,
                preferred_sources=preferred_sources
            )
            # 创建新的分析器实例
            current_analyzer = ReflexivityAnalyzer(data_provider=data_provider)
        else:
            current_analyzer = analyzer
        
        # 执行分析
        results = current_analyzer.analyze(
            stock_code=stock_code,
            lookback_weeks=lookback_weeks,
            save_charts=False  # Web API 不需要保存文件
        )
        
        # 准备返回数据
        response_data = {
            'success': True,
            'data': {
                'stock_code': stock_code,
                'data_info': results['data_info'],
                'fit_results': results['fit_results'],
                'parameter_results': results['parameter_results'],
                'stage_results': results['stage_results'],
                'conclusion': results['conclusion'],
                'charts': results['charts']
            }
        }
        
        # 确保所有数据都是 JSON 可序列化的
        response_data = make_json_serializable(response_data)
        
        return jsonify(response_data)
        
    except ValueError as e:
        return jsonify({'error': f'参数错误: {str(e)}'}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'分析失败: {str(e)}'}), 500


@app.route('/api/analyzer/info', methods=['GET'])
def get_analyzer_info():
    """获取分析器组件信息"""
    try:
        info = analyzer.get_component_info()
        return jsonify({'success': True, 'components': info})
    except Exception as e:
        return jsonify({'error': f'获取信息失败: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'ok',
        'message': '服务器运行正常',
        'endpoints': {
            'analyze_stock': '/api/analyze_stock',
            'simulate': '/api/simulate',
            'analyzer_info': '/api/analyzer/info'
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

