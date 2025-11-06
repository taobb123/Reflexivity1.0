#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票价格与市场背离分析工具 - 启动脚本
根目录启动脚本，方便使用
"""

import sys
import os
from pathlib import Path

# 确保项目根目录在路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入主程序
from apps.main import main

if __name__ == "__main__":
    main()




