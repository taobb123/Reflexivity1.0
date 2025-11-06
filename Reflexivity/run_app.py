#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动Web应用的Python脚本
跨平台启动Flask Web应用
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入并启动应用（不切换目录，保持项目根目录）
import sys
sys.path.insert(0, str(project_root / 'apps'))
from apps.app import app

if __name__ == '__main__':
    print("="*60)
    print("  股票价格与市场背离分析 - Web应用")
    print("="*60)
    print("\n正在启动Web服务器...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务器\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

