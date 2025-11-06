# Web 应用启动指南

## 🚀 快速启动

### 方法1：使用启动脚本（推荐）

```bash
# 在项目根目录
cd C:\Users\111\Desktop\Reflexivity
python apps/run_web.py
```

### 方法2：直接运行应用

```bash
# 在项目根目录
cd C:\Users\111\Desktop\Reflexivity
python apps/app.py
```

### 方法3：使用批处理文件（Windows）

```bash
# 双击运行
run_app.bat
```

## ✅ 验证应用是否运行

启动后，你应该看到类似这样的输出：

```
============================================================
  反身性模型Web应用
============================================================

正在启动服务器...
访问地址: http://localhost:5000
按 Ctrl+C 停止服务器

============================================================

 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

## 🌐 访问应用

### 1. 健康检查
在浏览器中访问：
```
http://127.0.0.1:5000/api/health
```
应该返回 JSON 数据，表示服务器运行正常。

### 2. 股票分析页面
在浏览器中访问：
```
http://127.0.0.1:5000/api/analyze_stock
```
应该显示分析界面。

### 3. 主页
在浏览器中访问：
```
http://127.0.0.1:5000/
```
应该显示主页（如果存在）。

## 🔍 测试连接

如果无法连接，运行测试脚本：

```bash
python apps/test_connection.py
```

这会测试所有端点并显示详细的连接状态。

## ⚠️ 常见问题

### 问题1：端口被占用

**错误信息：**
```
OSError: [WinError 10048] 通常每个套接字地址(协议/网络地址/端口)只允许使用一次。
```

**解决方案：**
1. 查找占用端口的进程：
   ```bash
   netstat -ano | findstr :5000
   ```
2. 结束进程（替换 PID 为实际进程ID）：
   ```bash
   taskkill /PID <进程ID> /F
   ```
3. 或修改端口号（在 `apps/app.py` 最后一行）：
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

### 问题2：无法导入模块

**错误信息：**
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案：**
```bash
pip install -r requirements.txt
```

### 问题3：浏览器显示"Failed to fetch"

**可能原因：**
1. Flask 应用未运行
2. 端口号不匹配
3. 防火墙阻止连接

**检查步骤：**
1. 确认 Flask 应用正在运行（查看终端输出）
2. 检查端口号是否正确
3. 尝试访问 `http://127.0.0.1:5000/api/health`
4. 运行测试脚本：`python apps/test_connection.py`

### 问题4：分析过程很慢

**说明：**
- 数据获取可能需要一些时间（特别是首次）
- 参数估计可能需要较长时间（取决于数据量）
- 这是正常的，请耐心等待

**优化建议：**
- 减少回溯周数（如从 120 改为 60）
- 使用更快的优化器（L-BFGS-B 而不是差分进化）

## 📝 使用说明

### 启动 Web 应用后：

1. **打开浏览器**，访问 `http://127.0.0.1:5000/api/analyze_stock`
2. **填写表单**：
   - 股票代码/名称（如"平安银行"）
   - 回溯周数（默认 120）
   - 优先数据源（可选）
3. **点击"开始分析"**按钮
4. **等待分析完成**（可能需要几分钟）
5. **查看结果**：
   - 检测到的阶段
   - 反身性参数
   - 详细分析结论

### 命令行分析 vs Web 分析

- **命令行分析**：使用 `python apps/main.py` 或 `python apps/example_usage.py`
  - 适合批量处理
  - 适合脚本自动化
  - 结果保存到文件

- **Web 分析**：使用 `python apps/run_web.py` 启动 Web 应用
  - 适合交互式使用
  - 可视化界面
  - 实时查看结果

**注意**：命令行分析和 Web 分析是两个独立的进程，不能混用。

## 🔧 调试技巧

### 查看详细日志

Flask 应用在调试模式下会显示详细日志。如果遇到错误，查看终端输出。

### 检查浏览器控制台

1. 按 `F12` 打开开发者工具
2. 查看 **Console** 标签页（JavaScript 错误）
3. 查看 **Network** 标签页（HTTP 请求状态）

### 测试 API 端点

使用 curl 或 Python requests 测试：

```python
import requests

# 测试健康检查
response = requests.get('http://127.0.0.1:5000/api/health')
print(response.json())

# 测试分析接口
response = requests.post('http://127.0.0.1:5000/api/analyze_stock', json={
    'stock_code': '平安银行',
    'lookback_weeks': 120
})
print(response.json())
```

## 📞 需要帮助？

如果问题仍然存在：
1. 检查终端输出的错误信息
2. 运行 `python apps/test_connection.py` 测试连接
3. 查看 `apps/README.md` 了解系统架构
4. 查看 `apps/USAGE_GUIDE.md` 了解详细使用方法

