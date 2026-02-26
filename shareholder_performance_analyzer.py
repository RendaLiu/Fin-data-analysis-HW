# shareholder_performance_analyzer.py
"""
基于持股金额的股东投资能力评估
适用于只有股东持股金额数据的情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.stats as stats


class ShareholderAmountAnalyzer:
    """
    基于持股金额的股东投资能力分析器
    """
    
    def __init__(self):
        """初始化分析器"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 定义评分权重
        self.default_weights = {
            'portfolio_growth': 0.25,      # 组合增长能力
            'stability_score': 0.20,       # 投资稳定性
            'concentration_score': 0.15,   # 集中度合理性
            'turnover_score': 0.15,        # 换手率合理性
            'consistency_score': 0.15,     # 投资一致性
            'size_score': 0.10            # 规模适当性
        }
    
    def normalize_stock_codes(self, df, code_column='ts_code'):
        """
        统一股票代码格式
        
        Parameters:
        df: 包含股票代码的DataFrame
        code_column: 股票代码列名
        
        Returns:
        DataFrame: 统一格式后的数据
        """
        print("统一股票代码格式...")
        
        df = df.copy()
        
        # 定义转换函数
        def convert_code_format(code):
            if pd.isna(code):
                return code
            
            code_str = str(code).strip().upper()
            
            # 处理 xxxxxx.SH/SZ 格式 -> sh/sz.xxxxxx
            if '.' in code_str:
                parts = code_str.split('.')
                if len(parts) == 2:
                    stock_code = parts[0].zfill(6)  # 补齐6位
                    exchange = parts[1].lower()     # 转小写
                    return f"{exchange}.{stock_code}"
            
            # 处理其他格式（如果有）
            return code_str.lower()
    # 应用转换
        df[code_column] = df[code_column].apply(convert_code_format)
        
        # 统计转换结果
        unique_codes = df[code_column].nunique()
        sh_codes = df[df[code_column].str.startswith('sh.', na=False)][code_column].nunique()
        sz_codes = df[df[code_column].str.startswith('sz.', na=False)][code_column].nunique()
        
        print(f"股票代码转换完成:")
        print(f"• 总股票数: {unique_codes}")
        print(f"• 上证股票: {sh_codes}")
        print(f"• 深证股票: {sz_codes}")
    
        return df

    def filter_exchanges_and_top_holders(self, df_holdings, daily_quarterly, top_n=200):
        """
        筛选上证深证数据并选择前N大股东
        
        Parameters:
        df_holdings: 股东持股数据
        daily_quarterly: 季度股价数据
        top_n: 前N大股东
        
        Returns:
        DataFrame: 筛选后的股东数据
        """
        print(f"筛选上证深证数据并选择前{top_n}大股东...")
        
        # 1. 统一股票代码格式
        df_holdings_normalized = self.normalize_stock_codes(df_holdings, 'ts_code')
        
        # 2. 从daily_quarterly获取上证深证的股票代码
        sh_sz_codes = daily_quarterly['code'].unique()
        print(f"上证深证股票数量: {len(sh_sz_codes)}")
        
        # 3. 筛选股东数据，只保留上证深证的持股
        df_filtered = df_holdings_normalized[df_holdings_normalized['ts_code'].isin(sh_sz_codes)].copy()
        print(f"筛选后股东记录数: {len(df_filtered)} (原记录数: {len(df_holdings)})")
        
        # 4. 计算每个股东的总持股规模（使用20220630）
        recent_date = "2022-06-30"
        print(recent_date)
        recent_holdings = df_filtered[df_filtered['end_date'] == recent_date]
        
        holder_size = recent_holdings.groupby('holder_name')['hold_amount'].sum().sort_values(ascending=False)
        print(f"股东总数: {len(holder_size)}")
        print(f"最大持股股东: {holder_size.index[0]} ({holder_size.iloc[0]:,.2f}元)")
        print(holder_size[:10])
        
        # 5. 选择前N大股东
        top_holders = holder_size.head(top_n).index
        df_top_holders = df_filtered[df_filtered['holder_name'].isin(top_holders)]
        
        print(f"前{top_n}大股东记录数: {len(df_top_holders)}")
        print(f"涉及股票数量: {df_top_holders['ts_code'].nunique()}")
        
        return df_top_holders

    def calculate_holder_size_rank(self, df_holdings):
        """
        计算股东规模排名
        
        Parameters:
        df_holdings: 股东持股数据
        
        Returns:
        Series: 股东规模排名
        """
        # 使用最近季度的持股规模
        recent_date = "2022-06-30"
        recent_holdings = df_holdings[df_holdings['end_date'] == recent_date]
        
        holder_size = recent_holdings.groupby('holder_name')['hold_amount'].sum()
        holder_rank = holder_size.rank(ascending=False, method='min')
        
        return holder_rank

    def normalize_scores_to_normal_distribution(self, scored_data, score_column = 'comprehensive score', mean=50, std=10, negative = 0):
        """
        将评分投射到正态分布
        
        Parameters:
        scored_data: 原始评分数据
        mean: 目标分布的均值
        std: 目标分布的标准差
        
        Returns:
        DataFrame: 包含正态分布评分的数据
        """
        print("将评分投射到正态分布...")
        
        normalized_data = scored_data.copy()
        
        # 对综合得分进行正态分布转换
        if negative == 0:
            comprehensive_scores = normalized_data[score_column]
        else:
            comprehensive_scores = -normalized_data[score_column]
        
        # 计算原始得分的排名百分位
        ranks = comprehensive_scores.rank(method='average')
        percentiles = ranks / (len(ranks) + 1)  # 使用(len+1)避免100%分位
        
        # 将百分位映射到正态分布
        normal_scores = stats.norm.ppf(percentiles, loc=mean, scale=std)
        
        # 处理极端值（ppf可能产生inf）
        normal_scores = np.clip(normal_scores, mean - 4*std, mean + 4*std)
        
        normalized_data['normalized_score'] = normal_scores
        
        # 重新计算评级（基于正态分布得分）
        def get_normalized_rating(score):
            if score >= mean + std:
                return '优秀'
            elif score >= mean:
                return '良好'
            elif score >= mean - std:
                return '较差'
            else:
                return '很差'
        
        normalized_data['normalized_rating'] = normalized_data['normalized_score'].apply(get_normalized_rating)
        
        return normalized_data

    def plot_score_distribution_comparison(self, scored_data, normalized_data):
        """
        绘制原始评分和正态分布评分的对比图
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 原始得分分布
        ax1.hist(scored_data['comprehensive_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('原始综合得分分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('原始得分')
        ax1.set_ylabel('股东数量')
        ax1.axvline(scored_data['comprehensive_score'].mean(), color='red', linestyle='--', 
                    label=f'均值: {scored_data["comprehensive_score"].mean():.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 正态分布得分
        ax2.hist(normalized_data['normalized_score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_title('正态分布得分', fontsize=14, fontweight='bold')
        ax2.set_xlabel('正态分布得分')
        ax2.set_ylabel('股东数量')
        ax2.axvline(50, color='red', linestyle='--', label='均值: 50')
        ax2.axvline(40, color='orange', linestyle=':', label='±1标准差')
        ax2.axvline(60, color='orange', linestyle=':')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 评级分布对比
        original_ratings = scored_data['rating'].value_counts()
        normalized_ratings = normalized_data['normalized_rating'].value_counts()
        
        x = np.arange(len(original_ratings))
        width = 0.35
        
        ax3.bar(x - width/2, original_ratings.values, width, label='原始评级', alpha=0.7)
        ax3.bar(x + width/2, normalized_ratings.values, width, label='正态分布评级', alpha=0.7)
        
        ax3.set_title('评级分布对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('评级')
        ax3.set_ylabel('股东数量')
        ax3.set_xticks(x)
        ax3.set_xticklabels(original_ratings.index, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

    def calculate_portfolio_growth(self, df_holdings):
        """
        计算投资组合增长指标
        
        Parameters:
        df_holdings: 股东持股数据（需包含end_date, holder_name, hold_amount）
        
        Returns:
        DataFrame: 组合增长指标
        """
        print("计算投资组合增长指标...")
        
        # 按股东和日期汇总总持仓金额
        portfolio_value = df_holdings.groupby(['holder_name', 'end_date']).agg({
            'hold_amount': 'sum',
            'ts_code': 'nunique'  # 持股数量
        }).reset_index()
        
        portfolio_value['end_date'] = pd.to_datetime(portfolio_value['end_date'])
        portfolio_value = portfolio_value.sort_values(['holder_name', 'end_date'])
        
        growth_metrics = []
        
        for holder in portfolio_value['holder_name'].unique():
            holder_data = portfolio_value[portfolio_value['holder_name'] == holder]
            
            if len(holder_data) < 2:  # 至少需要2个时间点
                continue
            
            # 计算组合价值增长
            first_value = holder_data['hold_amount'].iloc[0]
            last_value = holder_data['hold_amount'].iloc[-1]
            total_growth = (last_value - first_value) / first_value if first_value > 0 else 0
            
            # 计算季度增长率
            holder_data = holder_data.copy()
            holder_data['quarter_growth'] = holder_data['hold_amount'].pct_change()
            avg_quarter_growth = holder_data['quarter_growth'].mean()
            
            # 增长稳定性
            growth_std = holder_data['quarter_growth'].std()
            growth_stability = 1 / (1 + growth_std) if not pd.isna(growth_std) else 0
            
            # 持续增长季度数
            positive_quarters = (holder_data['quarter_growth'] > 0).sum()
            total_quarters = len(holder_data) - 1  # 减去第一个季度（无增长率）
            growth_consistency = positive_quarters / total_quarters if total_quarters > 0 else 0
            
            growth_metrics.append({
                'holder_name': holder,
                'total_growth': total_growth,
                'avg_quarter_growth': avg_quarter_growth,
                'growth_stability': growth_stability,
                'growth_consistency': growth_consistency,
                'start_value': first_value,
                'end_value': last_value,
                'analysis_quarters': len(holder_data),
                'positive_quarters': positive_quarters
            })
        
        return pd.DataFrame(growth_metrics)
    
    def calculate_investment_stability(self, df_holdings):
        """
        计算投资稳定性指标
        
        Parameters:
        df_holdings: 股东持股数据
        
        Returns:
        DataFrame: 稳定性指标
        """
        print("计算投资稳定性指标...")
        
        stability_metrics = []
        
        for holder in df_holdings['holder_name'].unique():
            holder_data = df_holdings[df_holdings['holder_name'] == holder]
            
            # 持股数量稳定性
            stock_count_by_date = holder_data.groupby('end_date')['ts_code'].nunique()
            stock_count_stability = 1 / (1 + stock_count_by_date.std()) if len(stock_count_by_date) > 1 else 0
            
            # 持仓集中度稳定性
            concentration_by_date = []
            for date in holder_data['end_date'].unique():
                date_data = holder_data[holder_data['end_date'] == date]
                total_amount = date_data['hold_amount'].sum()
                if total_amount > 0:
                    # 赫芬达尔指数
                    hhi = ((date_data['hold_amount'] / total_amount) ** 2).sum()
                    concentration_by_date.append(hhi)
            
            concentration_stability = 1 / (1 + np.std(concentration_by_date)) if concentration_by_date else 0
            
            # 投资期限（股票平均持有期）
            holding_periods = self.calculate_avg_holding_period(holder_data)
            
            stability_metrics.append({
                'holder_name': holder,
                'stock_count_stability': stock_count_stability,
                'concentration_stability': concentration_stability,
                'avg_holding_period': holding_periods.get('avg_quarters', 0),
                'turnover_rate': holding_periods.get('turnover_rate', 0)
            })
        
        return pd.DataFrame(stability_metrics)
    
    def calculate_avg_holding_period(self, holder_data):
        """
        计算平均持有期和换手率
        
        Parameters:
        holder_data: 单个股东的数据
        
        Returns:
        dict: 持有期指标
        """
        # 按股票分析持有期
        stock_holding = []
        
        for ts_code in holder_data['ts_code'].unique():
            stock_data = holder_data[holder_data['ts_code'] == ts_code].copy()
            stock_data['end_date'] = pd.to_datetime(stock_data['end_date'])
            stock_data = stock_data.sort_values('end_date')
            
            if len(stock_data) > 1:
                holding_quarters = len(stock_data)
                stock_holding.append(holding_quarters)
        
        if stock_holding:
            avg_quarters = np.mean(stock_holding)
            # 简化换手率计算：1/平均持有期
            turnover_rate = 1 / avg_quarters if avg_quarters > 0 else 0
        else:
            avg_quarters = 0
            turnover_rate = 0
        
        return {
            'avg_quarters': avg_quarters,
            'turnover_rate': turnover_rate
        }
    
    def calculate_concentration_metrics(self, df_holdings):
        """
        计算集中度指标
        
        Parameters:
        df_holdings: 股东持股数据
        
        Returns:
        DataFrame: 集中度指标
        """
        print("计算集中度指标...")
        
        concentration_metrics = []
        
        for holder in df_holdings['holder_name'].unique():
            holder_data = df_holdings[df_holdings['holder_name'] == holder]
            
            # 使用最近季度的数据
            recent_date = '2022-06-30'
            recent_data = holder_data[holder_data['end_date'] == recent_date]
            
            if len(recent_data) == 0:
                continue
            
            total_amount = recent_data['hold_amount'].sum()
            
            if total_amount > 0:
                # 赫芬达尔指数
                weights = recent_data['hold_amount'] / total_amount
                hhi_index = (weights ** 2).sum()
                
                # 前三大持仓占比
                top3_weight = weights.nlargest(3).sum()
                
                # 持股数量
                stock_count = len(recent_data)
                
                # 集中度评分（适中的集中度更好）
                # HHI在0.1-0.25之间认为适中
                if hhi_index < 0.1:
                    concentration_score = hhi_index / 0.1  # 过于分散
                elif hhi_index > 0.25:
                    concentration_score = 1 - (hhi_index - 0.25) / 0.75  # 过于集中
                else:
                    concentration_score = 1.0  # 适中
                
                concentration_metrics.append({
                    'holder_name': holder,
                    'hhi_index': hhi_index,
                    'top3_concentration': top3_weight,
                    'stock_count': stock_count,
                    'concentration_score': max(0, min(1, concentration_score))  # 限制在0-1之间
                })
        
        return pd.DataFrame(concentration_metrics)
    
    def calculate_investment_consistency(self, df_holdings, industry_data=None):
        """
        计算投资一致性指标
        
        Parameters:
        df_holdings: 股东持股数据
        industry_data: 行业数据（可选）
        
        Returns:
        DataFrame: 一致性指标
        """
        print("计算投资一致性指标...")
        
        consistency_metrics = []
        
        for holder in df_holdings['holder_name'].unique():
            holder_data = df_holdings[df_holdings['holder_name'] == holder]
            holder_data = holder_data.sort_values('end_date')
            
            dates = holder_data['end_date'].unique()
            if len(dates) < 2:
                continue
            
            # 1. 持股连续性
            continuous_quarters = 0
            max_continuous = 0
            current_continuous = 0
            
            for i in range(len(dates)):
                if i == 0:
                    current_continuous = 1
                else:
                    prev_date = pd.to_datetime(dates[i-1])
                    curr_date = pd.to_datetime(dates[i])
                    quarter_gap = (curr_date.year - prev_date.year) * 4 + (curr_date.month - prev_date.month) / 3
                    
                    if quarter_gap <= 1.5:  # 允许一个季度的间隔
                        current_continuous += 1
                    else:
                        max_continuous = max(max_continuous, current_continuous)
                        current_continuous = 1
            
            max_continuous = max(max_continuous, current_continuous)
            continuity_score = max_continuous / len(dates) if len(dates) > 0 else 0
            
            # 2. 规模稳定性
            portfolio_sizes = holder_data.groupby('end_date')['hold_amount'].sum()
            size_stability = 1 / (1 + portfolio_sizes.std() / portfolio_sizes.mean()) if portfolio_sizes.mean() > 0 else 0
            
            consistency_metrics.append({
                'holder_name': holder,
                'continuity_score': continuity_score,
                'size_stability': size_stability,
                'total_quarters': len(dates),
                'max_continuous_quarters': max_continuous
            })
        
        return pd.DataFrame(consistency_metrics)
    
    def build_ability_score(self, growth_df, stability_df, concentration_df, consistency_df):
        """
        已废弃：请使用 run_complete_analysis_v2 的评分与输出流程。
        此方法保留为空，以避免与新版流程冲突。
        """
        print("[Deprecated] build_ability_score 已废弃，请使用 run_complete_analysis_v2。")
        return None
    
    def plot_portfolio_growth_comparison(self, df_holdings, scored_data, top_stars=5, top_problems=3):
        """
        已废弃：新版流程不再使用该绘图方法。
        保留空实现以兼容旧调用。
        """
        print("[Deprecated] plot_portfolio_growth_comparison 已废弃。")
        return None
    
    def _plot_growth_curves(self, ax, portfolio_values, shareholders, title):
        """
        已废弃的内部绘图函数。
        """
        return
    
    def run_complete_analysis_v2(self, df_holdings, daily_quarterly, industry_data=None, 
                           top_holders=200, min_quarters=8):
        """
        改进版的完整分析流程
        
        Parameters:
        df_holdings: 股东持股数据
        daily_quarterly: 季度股价数据（用于筛选上证深证）
        industry_data: 行业数据
        top_holders: 前N大股东
        min_quarters: 最小分析季度数
        """
        print("=== 开始改进版股东投资能力分析 ===")
        
        try:
            # 1. 筛选上证深证数据并选择前N大股东
            df_filtered = self.filter_exchanges_and_top_holders(
                df_holdings, daily_quarterly, top_holders
            )
            
            # 2. 进一步筛选有足够数据的股东
            holder_quarters = df_filtered.groupby('holder_name')['end_date'].nunique()
            qualified_holders = holder_quarters[holder_quarters >= min_quarters].index
            df_qualified = df_filtered[df_filtered['holder_name'].isin(qualified_holders)]
            
            print(f"最终分析 {len(qualified_holders)} 个符合条件的股东...")
            
            # 3. 计算各项指标（与之前相同）
            growth_metrics = self.calculate_portfolio_growth(df_qualified)
            growth_metrics = self.normalize_scores_to_normal_distribution(growth_metrics, score_column = 'avg_quarter_growth')
            growth_metrics.to_csv('growth_metrics.csv', encoding='gbk')
            growth_metrics = growth_metrics[['holder_name', 'avg_quarter_growth', 'normalized_score']].rename(columns={'normalized_score': 'growth_score'})

            stability_metrics = self.calculate_investment_stability(df_qualified)
            stability_metrics = self.normalize_scores_to_normal_distribution(stability_metrics, score_column = 'turnover_rate')
            stability_metrics.to_csv('stability_metrics.csv', encoding='gbk')
            stability_metrics = stability_metrics[['holder_name', 'normalized_score']].rename(columns={'normalized_score': 'stability_score'})
            
            concentration_metrics = self.calculate_concentration_metrics(df_qualified)
            concentration_metrics = self.normalize_scores_to_normal_distribution(concentration_metrics, score_column = 'concentration_score')
            concentration_metrics.to_csv('concentration_metrics.csv', encoding='gbk')
            concentration_metrics = concentration_metrics[['holder_name', 'normalized_score']].rename(columns={'normalized_score': 'concentration_score'})
            
            consistency_metrics = self.calculate_investment_consistency(df_qualified, industry_data)
            consistency_metrics = self.normalize_scores_to_normal_distribution(consistency_metrics, score_column = 'size_stability', negative = 1)
            consistency_metrics.to_csv('consistency_metrics.csv', encoding='gbk')
            consistency_metrics = consistency_metrics[['holder_name', 'normalized_score']].rename(columns={'normalized_score': 'consistency_score'})
            
            
            # 4. 构建综合评分
            scored_df = growth_metrics.merge(stability_metrics, on='holder_name', how='inner')
            scored_df = scored_df.merge(concentration_metrics, on='holder_name', how='inner')
            scored_df = scored_df.merge(consistency_metrics, on='holder_name', how='inner')
            scored_df['score'] = 0.5*scored_df['growth_score'] + 0.25*scored_df['stability_score'] + 0.1*scored_df['concentration_score'] + 0.15*scored_df['consistency_score']
            scored_df.sort_values(by='score', ascending=False, inplace=True)
            print(scored_df)
            scored_df.to_csv('scored_data.csv', encoding='gbk')
            growth_metrics.set_index('holder_name', inplace=True)


            
            # 5. 识别明星和问题股东（正数/倒数前10），以及评分
            star_shareholders = scored_df.iloc[:10]
            problem_shareholders = scored_df.iloc[-10:]
            scored_df['rating'] = pd.cut(scored_df['score'], bins=[-float('inf'), 0.2, 0.5, 0.8, float('inf')],
                                        labels=['问题', '一般', '良好', '优秀'])
            
            classification_result = {
                'star_shareholders': star_shareholders,
                'problem_shareholders': problem_shareholders,
                'rating_distribution': scored_df['rating'].value_counts(),
                'original_rating_distribution': scored_df['rating'].value_counts()
            }
            # 7. 新版输出：使用统一的新报告与保存函数
            # 注释旧版输出函数（generate_enhanced_report），改用更清晰的新版输出
            # self.generate_enhanced_report(scored_df, classification_result, growth_metrics)

            self.output_new_report(
                scored_df=scored_df,
                classification_result=classification_result,
                growth_metrics=growth_metrics,
                stability_metrics=stability_metrics,
                concentration_metrics=concentration_metrics,
                consistency_metrics=consistency_metrics,
            )

            # 可选：统一保存所有结果CSV到 outputs/ 目录
            self.save_results(
                scored_df=scored_df,
                growth_metrics=growth_metrics,
                stability_metrics=stability_metrics,
                concentration_metrics=concentration_metrics,
                consistency_metrics=consistency_metrics,
                directory='outputs'
            )
            
            print("=== 改进版分析完成 ===")
            
            return {
                'growth_metrics': growth_metrics,
                'stability_metrics': stability_metrics,
                'concentration_metrics': concentration_metrics,
                'consistency_metrics': consistency_metrics,
                'scored_data': scored_df,
                'classification_result': classification_result,
                'df_filtered': df_filtered
            }
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_enhanced_report(self, normalized_data, classification_result, growth_metrics):
        """
        已废弃：新版将使用 output_new_report 输出。
        保留空实现以避免旧调用报错。
        """
        print("[Deprecated] generate_enhanced_report 已废弃。")
        return None

    def get_expected_normal_percentage(self, rating):
        """获取理论正态分布百分比"""
        expected_percentages = {
            '明星股东': 15.9,  # 均值+1标准差以上
            '优秀股东': 34.1,  # 均值到均值+1标准差
            '一般股东': 34.1,  # 均值-1标准差到均值
            '待观察股东': 13.6, # 均值-2标准差到均值-1标准差
            '问题股东': 2.3    # 均值-2标准差以下
        }
        return expected_percentages.get(rating, 0)

    def identify_main_issue(self, holder_data):
        """识别主要问题"""
        issues = []
        
        if holder_data['total_growth'] < 0:
            issues.append("负增长")
        if holder_data['hhi_index'] > 0.3:
            issues.append("过度集中")
        elif holder_data['hhi_index'] < 0.05:
            issues.append("过度分散")
        if holder_data['turnover_rate'] > 0.4:
            issues.append("高换手")
        if holder_data['continuity_score'] < 0.5:
            issues.append("低连续性")
        
        return "、".join(issues) if issues else "多维度表现不佳"

    # ===================== 新版统一输出函数 =====================
    def output_new_report(self, scored_df, classification_result, 
                          growth_metrics, stability_metrics, 
                          concentration_metrics, consistency_metrics):
        """
        新版统一输出：按照 run_complete_analysis_v2 的思路，输出核心结果。
        - 概览统计（均值/标准差/分位）
        - TOP/Bottom 排名
        - 关键指标快照
        """
        print(f"\n{'='*80}")
        print("🎯 股东投资能力综合报告（新版输出）")
        print(f"{'='*80}")

        # 概览统计
        print("\n概览统计:")
        print(f"• 股东数量: {len(scored_df)}")
        print(f"• 得分均值: {scored_df['score'].mean():.2f}")
        print(f"• 得分标准差: {scored_df['score'].std():.2f}")
        print(f"• 得分分位(20/50/80): {scored_df['score'].quantile(0.2):.2f} / {scored_df['score'].median():.2f} / {scored_df['score'].quantile(0.8):.2f}")

        # TOP/Bottom 排名
        top_n = 10
        print(f"\nTOP{top_n} 明星股东:")
        for i, (_, holder) in enumerate(scored_df.nlargest(top_n, 'score').iterrows(), 1):
            avg_growth = growth_metrics.loc[holder['holder_name'], 'avg_quarter_growth'] if holder['holder_name'] in growth_metrics.index else np.nan
            print(f"{i:2d}. {holder['holder_name']:<25} | 得分: {holder['score']:5.1f} | 平均涨幅: {avg_growth:.2f}")

        print(f"\nBottom{top_n} 问题股东:")
        for i, (_, holder) in enumerate(scored_df.nsmallest(top_n, 'score').iterrows(), 1):
            avg_growth = growth_metrics.loc[holder['holder_name'], 'avg_quarter_growth'] if holder['holder_name'] in growth_metrics.index else np.nan
            print(f"{i:2d}. {holder['holder_name']:<25} | 得分: {holder['score']:5.1f} | 平均涨幅: {avg_growth:.2f}")

        # 评级分布
        print("\n评级分布:")
        print(scored_df['rating'].value_counts().to_string())

        # 指标快照
        def snap(df, name):
            print(f"\n{name} 指标快照:")
            print(df.head(10).to_string(index=False))
        snap(growth_metrics.reset_index(), '增长')
        snap(stability_metrics, '稳定性')
        snap(concentration_metrics, '集中度')
        snap(consistency_metrics, '一致性')

        print(f"\n{'='*80}")

    def save_results(self, scored_df, growth_metrics, stability_metrics, 
                     concentration_metrics, consistency_metrics, directory='outputs'):
        """
        将所有关键结果统一保存为 CSV，目录不存在则创建。
        """
        import os
        os.makedirs(directory, exist_ok=True)
        paths = {
            'scored_data.csv': scored_df,
            'growth_metrics.csv': growth_metrics.reset_index(),
            'stability_metrics.csv': stability_metrics,
            'concentration_metrics.csv': concentration_metrics,
            'consistency_metrics.csv': consistency_metrics,
        }
        for fname, df in paths.items():
            try:
                df.to_csv(os.path.join(directory, fname), index=False, encoding='utf-8-sig')
            except Exception:
                # 回退到 gbk，避免部分中文路径/内容问题
                df.to_csv(os.path.join(directory, fname), index=False, encoding='gbk')
        print(f"✅ 已保存结果到目录: {directory}")

# 便捷函数
def create_amount_analyzer():
    """创建基于持股金额的分析器实例"""
    return ShareholderAmountAnalyzer()