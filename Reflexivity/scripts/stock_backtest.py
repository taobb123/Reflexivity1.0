"""
Streamlit股票反身性模型回测应用
支持股票数据获取、参数反推、实时回测和高级分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.data_fetchers.data_fetcher_hybrid import HybridDataFetcher
from core.parameter_estimator import ParameterEstimator, estimate_from_stock_data
from core.reflexivity_model import ReflexivityModel

# 设置页面配置
st.set_page_config(
    page_title="股票反身性模型回测",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # 缓存1小时
def fetch_stock_data(stock_name: str, lookback_weeks: int):
    """获取股票数据（带缓存）"""
    try:
        fetcher = HybridDataFetcher()
        df, code = fetcher.fetch_complete_data(stock_name, lookback_weeks)
        return df, code, None
    except Exception as e:
        return None, None, str(e)


@st.cache_data
def estimate_parameters(df: pd.DataFrame, method: str = 'differential_evolution'):
    """参数反推（带缓存）"""
    try:
        results = estimate_from_stock_data(df, method=method)
        return results, None
    except Exception as e:
        return None, str(e)


def run_model_simulation(alpha: float, gamma: float, beta: float, 
                        P0: float, F0: float, T: int, noise_std: float = 0.0):
    """运行模型仿真"""
    model = ReflexivityModel(alpha=alpha, gamma=gamma, beta=beta, 
                           P0=P0, F0=F0, noise_std=noise_std)
    results = model.simulate(T)
    stability_info = model.analyze_stability()
    return results, stability_info, model


def plot_price_comparison(df: pd.DataFrame, results_estimated: dict, 
                         results_manual: dict = None, stock_name: str = ""):
    """绘制价格对比图"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('价格对比', '基本面对比', '市场预期', '价格与基本面差异'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    dates = pd.to_datetime(df['date'])
    P_actual = df['P_t'].values
    F_actual = df['F_t'].values
    T = len(df)
    
    # 获取反推参数的预测结果
    estimator = ParameterEstimator(P_actual, F_actual)
    P_pred_est, F_pred_est = estimator.simulate_model(
        results_estimated['parameters']['alpha'],
        results_estimated['parameters']['gamma'],
        results_estimated['parameters']['beta']
    )
    P_pred_est_denorm = P_pred_est * estimator.P_std + estimator.P_mean
    F_pred_est_denorm = F_pred_est * estimator.F_std + estimator.F_mean
    
    # 子图1：价格对比
    fig.add_trace(
        go.Scatter(x=dates, y=P_actual, name='真实价格', 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=P_pred_est_denorm, name='反推参数预测', 
                  line=dict(color='red', width=2, dash='dash')),
        row=1, col=1
    )
    
    # 如果有手动参数，添加对比
    if results_manual:
        P_pred_manual, F_pred_manual = estimator.simulate_model(
            results_manual['alpha'],
            results_manual['gamma'],
            results_manual['beta']
        )
        P_pred_manual_denorm = P_pred_manual * estimator.P_std + estimator.P_mean
        F_pred_manual_denorm = F_pred_manual * estimator.F_std + estimator.F_mean
        
        fig.add_trace(
            go.Scatter(x=dates, y=P_pred_manual_denorm, name='手动参数预测', 
                      line=dict(color='green', width=2, dash='dot')),
            row=1, col=1
        )
        
        # 基本面对比
        fig.add_trace(
            go.Scatter(x=dates, y=F_actual, name='真实基本面', 
                      line=dict(color='purple', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dates, y=F_pred_manual_denorm, name='手动参数预测基本面', 
                      line=dict(color='orange', width=2, dash='dot')),
            row=1, col=2
        )
    
    # 基本面对比（反推参数）
    fig.add_trace(
        go.Scatter(x=dates, y=F_actual, name='真实基本面', 
                  line=dict(color='purple', width=2), showlegend=(not results_manual)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=dates, y=F_pred_est_denorm, name='反推参数预测基本面', 
                  line=dict(color='red', width=2, dash='dash'), showlegend=(not results_manual)),
        row=1, col=2
    )
    
    # 计算市场预期
    alpha_est = results_estimated['parameters']['alpha']
    E_est = alpha_est * P_pred_est_denorm + (1 - alpha_est) * F_pred_est_denorm
    
    # 子图3：市场预期
    fig.add_trace(
        go.Scatter(x=dates, y=E_est, name='市场预期 E_t', 
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=P_pred_est_denorm, name='价格', 
                  line=dict(color='blue', width=1, dash='dash')),
        row=2, col=1
    )
    
    # 子图4：价格与基本面差异
    x_actual = P_actual - F_actual
    x_pred_est = P_pred_est_denorm - F_pred_est_denorm
    fig.add_trace(
        go.Scatter(x=dates, y=x_actual, name='真实差异', 
                  line=dict(color='blue', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=dates, y=x_pred_est, name='预测差异', 
                  line=dict(color='red', width=2, dash='dash')),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
    
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=2)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="基本面", row=1, col=2)
    fig.update_yaxes(title_text="数值", row=2, col=1)
    fig.update_yaxes(title_text="差异", row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text=f"{stock_name} - 反身性模型回测分析",
        showlegend=True
    )
    
    return fig


def plot_sensitivity_analysis(estimator: ParameterEstimator, base_alpha: float, 
                             base_gamma: float, base_beta: float):
    """参数敏感性分析"""
    param_ranges = {
        'alpha': np.linspace(max(0, base_alpha - 0.3), base_alpha + 0.3, 20),
        'gamma': np.linspace(max(0, base_gamma - 0.5), base_gamma + 0.5, 20),
        'beta': np.linspace(max(0, base_beta - 0.2), base_beta + 0.2, 20)
    }
    
    # 计算每个参数的敏感性
    sensitivities = {}
    for param_name, param_range in param_ranges.items():
        mse_values = []
        for val in param_range:
            if param_name == 'alpha':
                P_pred, F_pred = estimator.simulate_model(val, base_gamma, base_beta)
            elif param_name == 'gamma':
                P_pred, F_pred = estimator.simulate_model(base_alpha, val, base_beta)
            else:  # beta
                P_pred, F_pred = estimator.simulate_model(base_alpha, base_gamma, val)
            
            P_pred_denorm = P_pred * estimator.P_std + estimator.P_mean
            mse = np.mean((estimator.P_t - P_pred_denorm) ** 2)
            mse_values.append(mse)
        
        sensitivities[param_name] = {
            'values': param_range,
            'mse': mse_values
        }
    
    # 绘制敏感性分析图
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('α 敏感性', 'γ 敏感性', 'β 敏感性')
    )
    
    colors = ['blue', 'green', 'red']
    for idx, (param_name, data) in enumerate(sensitivities.items()):
        fig.add_trace(
            go.Scatter(
                x=data['values'],
                y=data['mse'],
                mode='lines+markers',
                name=f'{param_name} 敏感性',
                line=dict(color=colors[idx], width=2)
            ),
            row=1, col=idx + 1
        )
        # 标记基准值
        base_vals = {'alpha': base_alpha, 'gamma': base_gamma, 'beta': base_beta}
        base_mse = data['mse'][np.argmin(np.abs(data['values'] - base_vals[param_name]))]
        fig.add_vline(
            x=base_vals[param_name],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"基准值={base_vals[param_name]:.3f}",
            row=1, col=idx + 1
        )
        fig.update_xaxes(title_text=param_name, row=1, col=idx + 1)
        fig.update_yaxes(title_text="MSE", row=1, col=idx + 1)
    
    fig.update_layout(
        height=400,
        title_text="参数敏感性分析",
        showlegend=False
    )
    
    return fig


def plot_stability_boundary(alpha_range: np.ndarray, gamma_range: np.ndarray, 
                           beta_fixed: float):
    """绘制稳定性边界图"""
    stability_map = np.zeros((len(gamma_range), len(alpha_range)))
    
    for i, gamma in enumerate(gamma_range):
        for j, alpha in enumerate(alpha_range):
            lambda_val = 1 + gamma * (alpha - 1) - beta_fixed
            if abs(lambda_val) < 1:
                stability_map[i, j] = 1  # 稳定
            elif abs(lambda_val) > 1:
                if lambda_val < -1:
                    stability_map[i, j] = 2  # 振荡发散
                else:
                    stability_map[i, j] = 3  # 单调发散
            else:
                stability_map[i, j] = 0.5  # 临界
    
    fig = go.Figure(data=go.Contour(
        z=stability_map,
        x=alpha_range,
        y=gamma_range,
        colorscale='RdYlGn',
        contours=dict(
            start=0,
            end=3,
            size=0.5
        ),
        colorbar=dict(
            title="稳定性",
            tickmode='array',
            tickvals=[0.5, 1, 2, 3],
            ticktext=['临界', '稳定', '振荡发散', '单调发散']
        )
    ))
    
    fig.update_layout(
        title=f"稳定性边界图 (β={beta_fixed:.3f}固定)",
        xaxis_title="α (价格在认知中的权重)",
        yaxis_title="γ (价格调整速度)",
        height=500
    )
    
    return fig


# 主应用
def main():
    st.markdown('<div class="main-header">📈 股票反身性模型回测系统</div>', 
                unsafe_allow_html=True)
    
    # 侧边栏：输入参数
    with st.sidebar:
        st.header("📊 数据输入")
        
        stock_name = st.text_input("股票名称或代码", value="平安银行", 
                                   help="输入股票名称（如：平安银行）或代码（如：000001）")
        
        lookback_weeks = st.slider("回测时间范围（周）", min_value=20, max_value=200, 
                                  value=120, step=10,
                                  help="选择要分析的历史数据周数")
        
        st.divider()
        st.header("⚙️ 参数控制")
        
        use_estimated = st.checkbox("使用反推参数", value=True,
                                    help="使用从真实数据反推出的参数",
                                    key='use_estimated')
        
        use_manual = st.checkbox("同时使用手动参数对比", value=False,
                                help="在反推参数基础上，同时使用手动参数进行对比",
                                key='use_manual')
        
        if use_estimated:
            st.info("将使用从股票数据反推出的参数")
        
        if use_manual or not use_estimated:
            st.subheader("手动调整参数")
            alpha_manual = st.slider("α (价格权重)", min_value=0.0, max_value=2.0, 
                                     value=0.8, step=0.01,
                                     help="价格在认知中的权重，>1表示极端反身性",
                                     key='alpha_manual')
            gamma_manual = st.slider("γ (价格调整速度)", min_value=0.0, max_value=5.0, 
                                    value=0.5, step=0.01,
                                    help="价格向市场预期调整的速度",
                                    key='gamma_manual')
            beta_manual = st.slider("β (基本面影响)", min_value=0.0, max_value=2.0, 
                                   value=0.1, step=0.01,
                                   help="价格对基本面的影响强度",
                                   key='beta_manual')
        
        st.divider()
        st.header("🔬 高级功能")
        show_sensitivity = st.checkbox("显示参数敏感性分析", value=False, key='show_sensitivity')
        show_stability = st.checkbox("显示稳定性边界图", value=False, key='show_stability')
    
    # 主内容区
    if st.button("🚀 开始分析", type="primary"):
        # 显示进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 步骤1：获取数据
        status_text.text("📊 正在获取股票数据...")
        progress_bar.progress(20)
        
        df, code, error = fetch_stock_data(stock_name, lookback_weeks)
        
        if error:
            st.error(f"❌ 数据获取失败: {error}")
            st.stop()
        
        if df is None or len(df) == 0:
            st.error("❌ 未获取到数据，请检查股票名称或代码")
            st.stop()
        
        progress_bar.progress(40)
        status_text.text(f"✓ 成功获取 {len(df)} 条数据")
        
        # 步骤2：参数反推
        status_text.text("🔍 正在反推模型参数...")
        progress_bar.progress(60)
        
        results_estimated, est_error = estimate_parameters(df)
        
        if est_error:
            st.error(f"❌ 参数反推失败: {est_error}")
            st.stop()
        
        progress_bar.progress(80)
        status_text.text("✓ 参数反推完成")
        
        # 步骤3：展示结果
        progress_bar.progress(100)
        status_text.text("✅ 分析完成！")
        progress_bar.empty()
        status_text.empty()
        
        # 存储到session state
        st.session_state['df'] = df
        st.session_state['code'] = code
        st.session_state['results_estimated'] = results_estimated
        st.session_state['stock_name'] = stock_name
    
    # 如果已有数据，展示结果
    if 'df' in st.session_state and 'results_estimated' in st.session_state:
        df = st.session_state['df']
        code = st.session_state['code']
        results_estimated = st.session_state['results_estimated']
        stock_name = st.session_state.get('stock_name', '股票')
        
        # 显示数据概览
        st.header("📊 数据概览")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("数据点数", len(df))
        with col2:
            st.metric("时间范围", f"{df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("价格范围", f"{df['P_t'].min():.2f} - {df['P_t'].max():.2f}")
        with col4:
            st.metric("基本面范围", f"{df['F_t'].min():.4f} - {df['F_t'].max():.4f}")
        
        # 显示反推参数
        st.header("🎯 反推参数结果")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("α (价格权重)", f"{results_estimated['parameters']['alpha']:.4f}")
        with col2:
            st.metric("γ (调整速度)", f"{results_estimated['parameters']['gamma']:.4f}")
        with col3:
            st.metric("β (基本面影响)", f"{results_estimated['parameters']['beta']:.4f}")
        with col4:
            lambda_val = results_estimated['lambda']
            stability = results_estimated['stability']
            st.metric("λ (特征值)", f"{lambda_val:.4f}", 
                     delta=stability, delta_color="normal" if abs(lambda_val) < 1 else "inverse")
        
        # 拟合效果
        st.subheader("拟合效果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{results_estimated['fitness']['r_squared']:.4f}")
        with col2:
            st.metric("RMSE", f"{results_estimated['fitness']['rmse']:.4f}")
        with col3:
            st.metric("MAE", f"{results_estimated['fitness']['mae']:.4f}")
        
        # 手动参数（如果需要对比）
        results_manual = None
        use_manual = st.session_state.get('use_manual', False)
        use_estimated = st.session_state.get('use_estimated', True)
        
        if use_manual or not use_estimated:
            # 从session_state获取手动参数
            alpha_manual = st.session_state.get('alpha_manual', 0.8)
            gamma_manual = st.session_state.get('gamma_manual', 0.5)
            beta_manual = st.session_state.get('beta_manual', 0.1)
            
            results_manual = {
                'alpha': alpha_manual,
                'gamma': gamma_manual,
                'beta': beta_manual
            }
            
            st.header("🔄 手动参数 vs 反推参数对比")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("反推参数")
                st.write(f"- α: {results_estimated['parameters']['alpha']:.4f}")
                st.write(f"- γ: {results_estimated['parameters']['gamma']:.4f}")
                st.write(f"- β: {results_estimated['parameters']['beta']:.4f}")
                lambda_est = results_estimated['lambda']
                st.write(f"- λ: {lambda_est:.4f} ({results_estimated['stability']})")
            with col2:
                st.subheader("手动参数")
                st.write(f"- α: {alpha_manual:.4f}")
                st.write(f"- γ: {gamma_manual:.4f}")
                st.write(f"- β: {beta_manual:.4f}")
                lambda_manual = 1 + gamma_manual * (alpha_manual - 1) - beta_manual
                stability_manual = "稳定收敛" if abs(lambda_manual) < 1 else "发散"
                st.write(f"- λ: {lambda_manual:.4f} ({stability_manual})")
        
        # 绘制对比图
        st.header("📈 价格对比分析")
        fig = plot_price_comparison(df, results_estimated, results_manual, stock_name)
        st.plotly_chart(fig, use_container_width=True)
        
        # 参数敏感性分析
        if st.session_state.get('show_sensitivity', False):
            st.header("🔬 参数敏感性分析")
            estimator = ParameterEstimator(df['P_t'].values, df['F_t'].values)
            sensitivity_fig = plot_sensitivity_analysis(
                estimator,
                results_estimated['parameters']['alpha'],
                results_estimated['parameters']['gamma'],
                results_estimated['parameters']['beta']
            )
            st.plotly_chart(sensitivity_fig, use_container_width=True)
        
        # 稳定性边界图
        if st.session_state.get('show_stability', False):
            st.header("🎯 稳定性边界分析")
            beta_fixed = results_estimated['parameters']['beta']
            alpha_range = np.linspace(0, 2, 50)
            gamma_range = np.linspace(0, 5, 50)
            stability_fig = plot_stability_boundary(alpha_range, gamma_range, beta_fixed)
            st.plotly_chart(stability_fig, use_container_width=True)
            
            # 在图上标记当前参数位置
            st.info(f"当前参数位置：α={results_estimated['parameters']['alpha']:.3f}, "
                   f"γ={results_estimated['parameters']['gamma']:.3f}, "
                   f"β={beta_fixed:.3f}")
        
        # 数据下载
        st.header("💾 数据下载")
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下载数据 CSV",
            data=csv,
            file_name=f"{stock_name}_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()

