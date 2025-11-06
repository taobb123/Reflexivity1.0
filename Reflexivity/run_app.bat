@echo off
REM 启动Web应用批处理脚本
REM 启动Flask Web应用，用于股票价格与市场背离分析

echo ============================================================
echo   股票价格与市场背离分析 - Web应用
echo ============================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python
    pause
    exit /b 1
)

echo 正在启动Web服务器...
echo.
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止服务器
echo.
echo ============================================================
echo.

REM 切换到apps目录并启动应用
cd apps
python run_web.py

REM 如果程序退出，暂停以便查看输出
if errorlevel 1 (
    echo.
    echo [错误] Web应用启动失败
    pause
)




