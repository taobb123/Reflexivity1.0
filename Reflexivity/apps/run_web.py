"""
启动Web应用的便捷脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入应用（不切换目录，保持项目根目录）
from apps.app import app

if __name__ == '__main__':
    print("="*60)
    print("  反身性模型Web应用")
    print("="*60)
    print("\n正在启动服务器...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务器\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

