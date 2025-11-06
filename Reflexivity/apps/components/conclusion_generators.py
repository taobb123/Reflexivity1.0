"""
结论生成器实现
根据拟合参数和阶段生成分析结论（中文文本描述）
"""
from typing import Dict, Any

from apps.interfaces.conclusion_generator import IConclusionGenerator


class ChineseConclusionGenerator(IConclusionGenerator):
    """中文结论生成器"""
    
    def generate(self,
                 parameters: Dict[str, float],
                 stage_result: Dict[str, Any],
                 fit_results: Dict[str, Any],
                 **kwargs) -> str:
        """生成分析结论（中文文本描述）"""
        # 生成总结性结论
        summary = self.generate_summary({
            'parameters': parameters,
            'stage_result': stage_result,
            'fit_results': fit_results
        })
        
        return summary
    
    def generate_summary(self, all_results: Dict[str, Any]) -> str:
        """生成总结性结论"""
        parameters = all_results.get('parameters', {})
        stage_result = all_results.get('stage_result', {})
        fit_results = all_results.get('fit_results', {})
        
        stage = stage_result.get('stage', '未知')
        confidence = stage_result.get('confidence', 0)
        risk_level = stage_result.get('risk_level', '未知')
        indicators = stage_result.get('indicators', {})
        
        alpha = parameters.get('alpha', 0)
        gamma = parameters.get('gamma', 0)
        beta = parameters.get('beta', 0)
        lambda_val = indicators.get('lambda', 0)
        
        # 拟合质量
        fit_metrics = fit_results.get('fit_metrics', {})
        r_squared = fit_metrics.get('r_squared', 0)
        correlation = fit_results.get('correlation', {}).get('value', 0)
        
        # 构建结论文本
        conclusion_parts = []
        
        # 1. 阶段判断
        conclusion_parts.append("=" * 60)
        conclusion_parts.append("【反身性阶段判断】")
        conclusion_parts.append(f"当前阶段: {stage}")
        conclusion_parts.append(f"置信度: {confidence:.2%}")
        conclusion_parts.append(f"风险等级: {risk_level}")
        conclusion_parts.append("")
        
        # 2. 参数分析
        conclusion_parts.append("【反身性参数分析】")
        conclusion_parts.append(f"α (价格在认知中的权重): {alpha:.4f}")
        if alpha > 1:
            conclusion_parts.append("  → 极端反身性：市场过度依赖价格信号，存在自我强化风险")
        elif alpha > 0.8:
            conclusion_parts.append("  → 强反身性：价格对市场认知影响较大")
        elif alpha > 0.5:
            conclusion_parts.append("  → 中等反身性：价格与基本面共同影响市场认知")
        else:
            conclusion_parts.append("  → 弱反身性：价格对市场认知影响较小")
        
        conclusion_parts.append(f"β (价格对基本面的影响): {beta:.4f}")
        if beta > 0.5:
            conclusion_parts.append("  → 价格对基本面影响强：价格变化会显著改变基本面")
        elif beta > 0.2:
            conclusion_parts.append("  → 价格对基本面影响中等")
        else:
            conclusion_parts.append("  → 价格对基本面影响弱：价格变化对基本面影响有限")
        
        conclusion_parts.append(f"γ (价格调整速度): {gamma:.4f}")
        conclusion_parts.append(f"λ (系统特征值): {lambda_val:.4f}")
        conclusion_parts.append("")
        
        # 3. 稳定性分析
        conclusion_parts.append("【系统稳定性分析】")
        abs_lambda = abs(lambda_val)
        if abs_lambda < 1:
            conclusion_parts.append("  ✓ 系统稳定：价格和基本面会收敛，差异将缩小")
        elif abs_lambda > 1:
            if lambda_val < -1:
                conclusion_parts.append("  ⚠️ 系统不稳定（振荡发散）：可能出现振荡且振幅发散")
            else:
                conclusion_parts.append("  ⚠️ 系统不稳定（单调发散）：差异发散，可能形成泡沫或崩溃")
        else:
            conclusion_parts.append("  ⚠️ 临界状态：系统处于临界状态，差异既不收敛也不发散")
        conclusion_parts.append("")
        
        # 4. 拟合质量分析
        conclusion_parts.append("【拟合质量分析】")
        conclusion_parts.append(f"价格与基本面相关性: {correlation:.4f}")
        if correlation > 0.7:
            conclusion_parts.append("  → 强相关：价格与基本面高度相关")
        elif correlation > 0.4:
            conclusion_parts.append("  → 中等相关：价格与基本面存在一定相关性")
        else:
            conclusion_parts.append("  → 弱相关：价格与基本面相关性较低，可能存在信息不对称")
        
        conclusion_parts.append(f"拟合决定系数 (R²): {r_squared:.4f}")
        if r_squared > 0.9:
            conclusion_parts.append("  → 拟合效果优秀")
        elif r_squared > 0.7:
            conclusion_parts.append("  → 拟合效果良好")
        elif r_squared > 0.5:
            conclusion_parts.append("  → 拟合效果一般")
        else:
            conclusion_parts.append("  → 拟合效果较差，模型可能不适用")
        conclusion_parts.append("")
        
        # 5. 背离分析
        divergence = fit_results.get('divergence', {})
        if divergence:
            mean_div = divergence.get('mean', 0)
            std_div = divergence.get('std', 0)
            conclusion_parts.append("【背离分析】")
            conclusion_parts.append(f"平均背离: {mean_div:.4f}")
            conclusion_parts.append(f"背离标准差: {std_div:.4f}")
            if abs(mean_div) > std_div:
                conclusion_parts.append("  → 存在明显背离：价格与基本面存在显著差异")
            else:
                conclusion_parts.append("  → 背离程度较小：价格与基本面基本一致")
            conclusion_parts.append("")
        
        # 6. 投资建议（基于阶段和风险）
        conclusion_parts.append("【投资建议】")
        if stage == '稳定收敛':
            conclusion_parts.append("  → 系统处于稳定状态，适合长期投资")
        elif stage == '临界状态':
            conclusion_parts.append("  → 系统处于临界状态，需要密切关注市场动态，谨慎操作")
        elif stage == '泡沫形成':
            conclusion_parts.append("  → 系统呈现泡沫特征，建议谨慎投资，注意风险控制")
        elif stage == '泡沫破灭':
            conclusion_parts.append("  → 系统处于泡沫破灭阶段，可能存在投资机会，但需谨慎")
        elif stage == '崩溃':
            conclusion_parts.append("  → 系统处于崩溃状态，建议规避风险，等待市场稳定")
        else:
            conclusion_parts.append("  → 建议根据具体情况和风险承受能力做出投资决策")
        
        conclusion_parts.append("=" * 60)
        
        return "\n".join(conclusion_parts)
    
    def generate_detailed(self, all_results: Dict[str, Any]) -> str:
        """生成详细结论"""
        summary = self.generate_summary(all_results)
        
        # 添加更详细的指标分析
        stage_result = all_results.get('stage_result', {})
        indicators = stage_result.get('indicators', {})
        stage_scores = stage_result.get('stage_scores', {})
        
        detailed_parts = [summary]
        detailed_parts.append("")
        detailed_parts.append("【详细指标分析】")
        detailed_parts.append(f"价格趋势: {indicators.get('price_trend', '未知')}")
        detailed_parts.append(f"基本面趋势: {indicators.get('fundamental_trend', '未知')}")
        detailed_parts.append(f"背离趋势: {indicators.get('divergence_trend', '未知')}")
        detailed_parts.append(f"波动率: {indicators.get('volatility', 0):.6f}")
        detailed_parts.append("")
        detailed_parts.append("【各阶段匹配分数】")
        for stage, score in sorted(stage_scores.items(), key=lambda x: x[1], reverse=True):
            detailed_parts.append(f"  {stage}: {score:.2%}")
        
        return "\n".join(detailed_parts)
