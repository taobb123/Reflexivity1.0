"""
图表可视化器实现
用于生成拟合对比图表
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any, Optional

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from apps.interfaces.chart_visualizer import IChartVisualizer


class MatplotlibChartVisualizer(IChartVisualizer):
    """基于 matplotlib 的图表可视化器"""
    
    def visualize_fit(self,
                      price_data: np.ndarray,
                      fundamental_data: np.ndarray,
                      fit_results: Dict[str, Any],
                      **kwargs) -> plt.Figure:
        """可视化拟合结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('反身性模型拟合分析', fontsize=16, fontweight='bold')
        
        t = np.arange(len(price_data))
        
        # 子图1：价格与基本面对比
        ax1 = axes[0, 0]
        ax1.plot(t, price_data, 'b-', label='实际价格', linewidth=2, alpha=0.8)
        ax1.plot(t, fundamental_data, 'r--', label='实际基本面', linewidth=2, alpha=0.8)
        
        # 添加拟合线（如果有）
        if 'fundamental_fit' in fit_results:
            fundamental_fit = fit_results['fundamental_fit']
            if 'predicted' in fundamental_fit:
                pred_fundamental = np.array(fundamental_fit['predicted'])
                ax1.plot(t, pred_fundamental, 'g:', label='拟合基本面', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel('时间', fontsize=12)
        ax1.set_ylabel('数值', fontsize=12)
        ax1.set_title('价格与基本面对比', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：背离分析
        ax2 = axes[0, 1]
        divergence = fit_results.get('divergence', {})
        if divergence and 'data' in divergence:
            div_data = np.array(divergence['data'])
            ax2.plot(t, div_data, 'purple', linewidth=2, alpha=0.8)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.fill_between(t, 0, div_data, where=(div_data > 0), 
                           alpha=0.3, color='red', label='正背离')
            ax2.fill_between(t, 0, div_data, where=(div_data < 0), 
                           alpha=0.3, color='green', label='负背离')
            ax2.set_xlabel('时间', fontsize=12)
            ax2.set_ylabel('背离值 (价格 - 基本面)', fontsize=12)
            ax2.set_title('价格与基本面背离分析', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        # 子图3：相关性散点图
        ax3 = axes[1, 0]
        ax3.scatter(price_data, fundamental_data, alpha=0.6, s=20)
        
        # 添加拟合线
        if 'fundamental_fit' in fit_results:
            fundamental_fit = fit_results['fundamental_fit']
            fit_result = fundamental_fit.get('fit_result', {})
            if fit_result and 'parameters' in fit_result:
                params = fit_result['parameters']
                if 'slope' in params and 'intercept' in params:
                    x_line = np.linspace(price_data.min(), price_data.max(), 100)
                    y_line = params['slope'] * x_line + params['intercept']
                    ax3.plot(x_line, y_line, 'r-', linewidth=2, label='拟合线')
        
        correlation = fit_results.get('correlation', {}).get('value', 0)
        ax3.set_xlabel('价格', fontsize=12)
        ax3.set_ylabel('基本面', fontsize=12)
        ax3.set_title(f'价格与基本面相关性 (r={correlation:.4f})', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 子图4：拟合质量指标
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        fit_metrics = fit_results.get('fit_metrics', {})
        r_squared = fit_metrics.get('r_squared', 0)
        fit_quality = fit_metrics.get('fit_quality', 'unknown')
        correlation_info = fit_results.get('correlation', {})
        
        info_text = f"""
拟合质量指标：
  R² (决定系数): {r_squared:.6f}
  拟合质量: {fit_quality}
  相关性: {correlation_info.get('value', 0):.6f}
  相关性显著性: {'是' if correlation_info.get('significant', False) else '否'}

背离统计：
  平均背离: {divergence.get('mean', 0):.4f}
  背离标准差: {divergence.get('std', 0):.4f}
  最大背离: {divergence.get('max', 0):.4f}
  最小背离: {divergence.get('min', 0):.4f}
"""
        ax4.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def visualize_comparison(self,
                            actual_data: Dict[str, np.ndarray],
                            predicted_data: Dict[str, np.ndarray],
                            **kwargs) -> plt.Figure:
        """可视化对比图（实际 vs 预测）"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('实际数据 vs 模型预测对比', fontsize=16, fontweight='bold')
        
        price_actual = actual_data.get('price', np.array([]))
        fundamental_actual = actual_data.get('fundamental', np.array([]))
        price_pred = predicted_data.get('price', np.array([]))
        fundamental_pred = predicted_data.get('fundamental', np.array([]))
        
        t = np.arange(len(price_actual))
        
        # 子图1：价格对比
        ax1 = axes[0, 0]
        if len(price_actual) > 0:
            ax1.plot(t, price_actual, 'b-', label='实际价格', linewidth=2, alpha=0.8)
        if len(price_pred) > 0:
            ax1.plot(t, price_pred, 'r--', label='预测价格', linewidth=2, alpha=0.8)
        ax1.set_xlabel('时间', fontsize=12)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title('价格对比', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：基本面对比
        ax2 = axes[0, 1]
        if len(fundamental_actual) > 0:
            ax2.plot(t, fundamental_actual, 'g-', label='实际基本面', linewidth=2, alpha=0.8)
        if len(fundamental_pred) > 0:
            ax2.plot(t, fundamental_pred, 'm--', label='预测基本面', linewidth=2, alpha=0.8)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('基本面', fontsize=12)
        ax2.set_title('基本面对比', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 子图3：价格残差
        ax3 = axes[1, 0]
        if len(price_actual) > 0 and len(price_pred) > 0:
            residuals = price_actual - price_pred
            ax3.plot(t, residuals, 'purple', linewidth=1.5, alpha=0.7)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            rmse = np.sqrt(np.mean(residuals ** 2))
            ax3.fill_between(t, -rmse, rmse, alpha=0.2, color='gray')
            ax3.set_xlabel('时间', fontsize=12)
            ax3.set_ylabel('残差 (实际 - 预测)', fontsize=12)
            ax3.set_title(f'价格残差分析 (RMSE={rmse:.4f})', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        # 子图4：基本面条差
        ax4 = axes[1, 1]
        if len(fundamental_actual) > 0 and len(fundamental_pred) > 0:
            residuals = fundamental_actual - fundamental_pred
            ax4.plot(t, residuals, 'orange', linewidth=1.5, alpha=0.7)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            rmse = np.sqrt(np.mean(residuals ** 2))
            ax4.fill_between(t, -rmse, rmse, alpha=0.2, color='gray')
            ax4.set_xlabel('时间', fontsize=12)
            ax4.set_ylabel('残差 (实际 - 预测)', fontsize=12)
            ax4.set_title(f'基本面残差分析 (RMSE={rmse:.4f})', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_chart(self, chart: plt.Figure, save_path: str, **kwargs) -> str:
        """保存图表"""
        dpi = kwargs.get('dpi', 300)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        chart.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(chart)
        return save_path
    
    def chart_to_base64(self, chart: plt.Figure, **kwargs) -> str:
        """将图表转换为 base64 字符串"""
        dpi = kwargs.get('dpi', 150)
        format = kwargs.get('format', 'png')
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        
        img = io.BytesIO()
        chart.savefig(img, format=format, dpi=dpi, bbox_inches=bbox_inches)
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close(chart)
        return img_base64

