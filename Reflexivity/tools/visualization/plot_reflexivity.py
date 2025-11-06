"""
反身性模型可视化工具
用于绘制价格、基本面、预期等变量的时间序列图
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.reflexivity_model import ReflexivityModel


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_single_simulation(model: ReflexivityModel, T: int = 100, 
                          title: Optional[str] = None, 
                          save_path: Optional[str] = None,
                          show: bool = True) -> None:
    """
    绘制单次仿真的完整图表
    
    Args:
        model: 反身性模型实例
        T: 仿真步数
        title: 图表标题
        save_path: 保存路径（如果提供则保存图片）
        show: 是否显示图表
    """
    # 运行仿真
    results = model.simulate(T)
    
    # 获取模型信息
    model_info = model.get_model_info()
    lambda_val = model_info['lambda']
    stability = model_info['stability']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title or f'反身性模型仿真 (α={model.alpha:.2f}, γ={model.gamma:.2f}, β={model.beta:.2f})', 
                 fontsize=14, fontweight='bold')
    
    t = results['t']
    P = results['P']
    F = results['F']
    E = results['E']
    x = results['x']
    
    # 子图1：价格与基本面对比
    ax1 = axes[0, 0]
    ax1.plot(t, P, 'b-', label='价格 P_t', linewidth=2)
    ax1.plot(t, F, 'r--', label='基本面 F_t', linewidth=2)
    ax1.plot(t, E, 'g:', label='市场预期 E_t', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('时间步 t')
    ax1.set_ylabel('数值')
    ax1.set_title('价格、基本面与市场预期')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：差异 x_t = P_t - F_t
    ax2 = axes[0, 1]
    ax2.plot(t, x, 'purple', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('时间步 t')
    ax2.set_ylabel('差异 x_t = P_t - F_t')
    ax2.set_title(f'价格与基本面差异 (λ={lambda_val:.4f}, {stability})')
    ax2.grid(True, alpha=0.3)
    
    # 子图3：差异的对数尺度（查看收敛/发散率）
    ax3 = axes[1, 0]
    abs_x = np.abs(x)
    # 避免log(0)
    abs_x = np.where(abs_x < 1e-10, 1e-10, abs_x)
    ax3.plot(t, np.log10(abs_x), 'orange', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('时间步 t')
    ax3.set_ylabel('log₁₀|差异|')
    ax3.set_title('差异的对数尺度（观察收敛/发散）')
    ax3.grid(True, alpha=0.3)
    
    # 子图4：稳定性分析信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    info_text = f"""
模型参数：
  α (价格在认知中的权重) = {model.alpha:.4f}
  γ (价格调整速度) = {model.gamma:.4f}
  β (价格对基本面的影响) = {model.beta:.4f}

系统特征值：
  λ = 1 + γ(α-1) - β
  λ = {lambda_val:.6f}
  |λ| = {abs(lambda_val):.6f}

稳定性分析：
  {stability}
  
{model_info['stability_info']['description']}

初始条件：
  P₀ = {model.P0:.2f}
  F₀ = {model.F0:.2f}
  噪声标准差 = {model.noise_std:.4f}

仿真参数：
  时间步数 = {T}
"""
    ax4.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(models: List[ReflexivityModel], T: int = 100,
                   labels: Optional[List[str]] = None,
                   title: Optional[str] = None,
                   save_path: Optional[str] = None,
                   show: bool = True) -> None:
    """
    对比多个模型参数的仿真结果
    
    Args:
        models: 模型实例列表
        T: 仿真步数
        labels: 每个模型的标签
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图表
    """
    if labels is None:
        labels = [f"模型{i+1}" for i in range(len(models))]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(title or '多参数对比：价格与基本面演化', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # 上子图：价格演化对比
    ax1 = axes[0]
    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        results = model.simulate(T)
        ax1.plot(results['t'], results['P'], color=color, label=f'{label} (价格)', 
                linewidth=2, alpha=0.8)
        ax1.plot(results['t'], results['F'], color=color, label=f'{label} (基本面)', 
                linewidth=2, linestyle='--', alpha=0.5)
    ax1.set_xlabel('时间步 t')
    ax1.set_ylabel('数值')
    ax1.set_title('价格与基本面时间序列对比')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 下子图：差异对比
    ax2 = axes[1]
    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        results = model.simulate(T)
        x = results['x']
        lambda_val = model.compute_lambda()
        ax2.plot(results['t'], x, color=color, 
                label=f'{label} (λ={lambda_val:.3f})', linewidth=2, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('时间步 t')
    ax2.set_ylabel('差异 x_t = P_t - F_t')
    ax2.set_title('价格与基本面差异对比')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_stability_region(gamma_range: np.ndarray, beta: float,
                         alpha_values: List[float] = [0.5, 0.8, 0.95, 1.0, 1.2],
                         save_path: Optional[str] = None,
                         show: bool = True) -> None:
    """
    绘制参数空间中的稳定性区域
    
    Args:
        gamma_range: gamma参数的范围
        beta: 固定的beta值
        alpha_values: 要绘制的alpha值列表
        save_path: 保存路径
        show: 是否显示图表
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for alpha in alpha_values:
        lambda_vals = 1 + gamma_range * (alpha - 1) - beta
        abs_lambda = np.abs(lambda_vals)
        
        # 稳定区域 (|λ| < 1)
        stable_mask = abs_lambda < 1
        # 发散区域 (|λ| > 1)
        unstable_mask = abs_lambda >= 1
        
        ax.plot(gamma_range[stable_mask], abs_lambda[stable_mask], 
               'o', label=f'α={alpha:.2f} (稳定)', markersize=4, alpha=0.7)
        ax.plot(gamma_range[unstable_mask], abs_lambda[unstable_mask], 
               'x', label=f'α={alpha:.2f} (发散)', markersize=4, alpha=0.7)
    
    # 临界线 |λ| = 1
    ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='临界线 |λ|=1')
    ax.set_xlabel('γ (价格调整速度)', fontsize=12)
    ax.set_ylabel('|λ| (特征值绝对值)', fontsize=12)
    ax.set_title(f'稳定性区域分析 (β={beta:.2f})', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"稳定性区域图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_example_scenarios(save_path: Optional[str] = None, show: bool = True) -> None:
    """
    绘制文档中的示例场景
    
    Args:
        save_path: 保存路径
        show: 是否显示图表
    """
    scenarios = [
        {
            'name': '稳定收敛',
            'model': ReflexivityModel(alpha=0.8, gamma=0.5, beta=0.1, P0=100, F0=100, noise_std=1.0),
            'color': 'blue'
        },
        {
            'name': '接近临界',
            'model': ReflexivityModel(alpha=0.95, gamma=0.8, beta=0.05, P0=100, F0=100, noise_std=1.0),
            'color': 'orange'
        },
        {
            'name': '泡沫（发散）',
            'model': ReflexivityModel(alpha=1.2, gamma=0.8, beta=0.05, P0=100, F0=100, noise_std=1.0),
            'color': 'red'
        }
    ]
    
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(15, 4*len(scenarios)))
    fig.suptitle('反身性模型：不同场景对比', fontsize=16, fontweight='bold')
    
    for idx, scenario in enumerate(scenarios):
        model = scenario['model']
        results = model.simulate(T=100)
        lambda_val = model.compute_lambda()
        stability, _ = model.analyze_stability()
        
        # 左图：价格与基本面
        ax1 = axes[idx, 0]
        ax1.plot(results['t'], results['P'], 'b-', label='价格 P_t', linewidth=2)
        ax1.plot(results['t'], results['F'], 'r--', label='基本面 F_t', linewidth=2)
        ax1.set_xlabel('时间步 t')
        ax1.set_ylabel('数值')
        ax1.set_title(f'{scenario["name"]} (α={model.alpha:.2f}, γ={model.gamma:.2f}, β={model.beta:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：差异
        ax2 = axes[idx, 1]
        ax2.plot(results['t'], results['x'], color=scenario['color'], linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('时间步 t')
        ax2.set_ylabel('差异 x_t = P_t - F_t')
        ax2.set_title(f'差异演化 (λ={lambda_val:.4f}, {stability})')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"场景对比图已保存至: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # 示例：绘制单个仿真
    print("生成单个仿真图表...")
    model1 = ReflexivityModel(alpha=0.8, gamma=0.5, beta=0.1, P0=100, F0=100, noise_std=1.0)
    plot_single_simulation(model1, T=100, title="示例：稳定收敛场景")
    
    # 示例：对比多个场景
    print("\n生成场景对比图...")
    plot_example_scenarios()
    
    # 示例：稳定性区域分析
    print("\n生成稳定性区域图...")
    gamma_range = np.linspace(0, 2, 100)
    plot_stability_region(gamma_range, beta=0.1)

