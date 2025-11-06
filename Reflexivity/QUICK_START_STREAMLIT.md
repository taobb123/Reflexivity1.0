# Streamlit 应用快速启动指南

## 🚀 启动方式

### 方式1：使用 python -m streamlit（推荐）
如果直接使用 `streamlit` 命令提示找不到，可以使用：

```bash
python -m streamlit run stock_backtest.py
```

### 方式2：使用启动脚本
**Windows:**
```bash
run_streamlit.bat
```

**Linux/Mac:**
```bash
chmod +x run_streamlit.sh
./run_streamlit.sh
```

### 方式3：直接命令（如果已配置PATH）
```bash
streamlit run stock_backtest.py
```

## 📝 启动后

应用启动后会自动在浏览器中打开，通常地址是：
- **本地地址**: http://localhost:8501
- **网络地址**: http://你的IP:8501

如果没有自动打开，请手动在浏览器中访问上述地址。

## ⚙️ 如果遇到问题

### 问题1：'streamlit' 不是内部或外部命令
**解决方案**: 使用 `python -m streamlit run stock_backtest.py` 替代

### 问题2：端口被占用
**解决方案**: Streamlit 会自动选择下一个可用端口（8502, 8503等），或者手动指定：
```bash
python -m streamlit run stock_backtest.py --server.port 8502
```

### 问题3：模块导入错误
**解决方案**: 确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

## 🎯 使用步骤

1. **启动应用**（使用上述任一方式）
2. **在浏览器中打开**（通常是 http://localhost:8501）
3. **输入股票代码**（如"平安银行"或"000001"）
4. **选择时间范围**（建议120周）
5. **点击"开始分析"按钮**
6. **等待分析完成**（数据获取和参数反推需要一些时间）
7. **查看结果**（参数、图表、分析等）
8. **调整参数**（在侧边栏调整，点击"开始分析"重新运行）

## 💡 提示

- 首次获取数据可能需要10-30秒
- 参数反推需要10-60秒（取决于数据量和硬件）
- 数据会被缓存1小时，第二次访问会更快
- 可以随时调整参数并重新分析

