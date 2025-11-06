"""
快速演示脚本：直接运行生成所有可视化图表
"""

from .plot_reflexivity import (
    plot_single_simulation,
    plot_example_scenarios,
    plot_stability_region,
    plot_comparison
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.reflexivity_model import ReflexivityModel
import numpy as np

if __name__ == "__main__":
    print("="*60)
    print("反身性模型可视化演示")
    print("="*60)
    
    # 1. 单个仿真示例
    print("\n1. 生成单个仿真图表（稳定收敛场景）...")
    model1 = ReflexivityModel(alpha=0.8, gamma=0.5, beta=0.1, P0=100, F0=100, noise_std=1.0)
    plot_single_simulation(model1, T=100, title="稳定收敛场景")
    
    # 2. 场景对比
    print("\n2. 生成多个场景对比图...")
    plot_example_scenarios()
    
    # 3. 稳定性区域
    print("\n3. 生成稳定性区域分析图...")
    gamma_range = np.linspace(0, 2, 100)
    plot_stability_region(gamma_range, beta=0.1, save_path=None)
    
    # 4. 参数对比
    print("\n4. 生成参数对比图...")
    models = [
        ReflexivityModel(alpha=0.8, gamma=0.5, beta=0.1, P0=100, F0=100, noise_std=1.0),
        ReflexivityModel(alpha=0.95, gamma=0.8, beta=0.05, P0=100, F0=100, noise_std=1.0),
        ReflexivityModel(alpha=1.2, gamma=0.8, beta=0.05, P0=100, F0=100, noise_std=1.0),
    ]
    labels = ["稳定收敛 (α=0.8)", "接近临界 (α=0.95)", "泡沫发散 (α=1.2)"]
    plot_comparison(models, T=100, labels=labels, title="参数对比：不同α值的影响")
    
    print("\n" + "="*60)
    print("所有图表已生成完成！")
    print("="*60)

