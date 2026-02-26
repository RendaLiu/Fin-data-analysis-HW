# shareholder_analyzer.py
"""
股东分析工具
封装股东行业投资分析的所有功能
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from industry_mapper import IndustryMapper, MultiIndustryAnalyzer

class ShareholderAnalyzer:
    """
    股东分析器
    封装股东行业投资分析的所有功能
    """
    
    def __init__(self):
        """初始化股东分析器"""
        self.mapper = IndustryMapper()
        self.analyzer = MultiIndustryAnalyzer(self.mapper)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def analyze_single_shareholder(self, df, hangyeshuju_with_industry, holder_name, allocation_method='equal'):
        """
        分析单个股东的行业分布
        
        Parameters:
        df: 股东持股数据
        hangyeshuju_with_industry: 已包含industry_list的行业数据
        holder_name: 股东名称
        allocation_method: 分配方法
        
        Returns:
        tuple: (industry_summary, holder_data, allocated_data)
        """
        print(f"分析股东 '{holder_name}' 的行业分布...")
        
        # 获取股东持股数据
        holder_data = df[df['holder_name'] == holder_name].copy()
        
        if holder_data.empty:
            print(f"未找到股东: {holder_name}")
            return None
        
        print(f"持股公司数量: {holder_data['ts_code'].nunique()}")
        print(f"总资金数量: {holder_data['hold_amount'].sum():,} 元")
        print(f'\n')
        
        # 检查行业数据是否包含industry_list列
        if 'industry_list' not in hangyeshuju_with_industry.columns:
            print("警告: hangyeshuju_with_industry 中未找到 industry_list 列")
            # 如果需要，可以在这里添加行业映射
            hangyeshuju_with_industry = self.mapper.add_industry_column(
                hangyeshuju_with_industry, 
                business_column='business_category', 
                new_column='industry_list', 
                format='list'
            )
        
        # 使用MultiIndustryAnalyzer的现有方法
        # 合并持股数据和行业数据
        merged_data = self.analyzer.analyze_company_industries(holder_data, hangyeshuju_with_industry)
        
        # 计算行业分配
        allocated_data = self.analyzer.calculate_industry_allocation(merged_data, allocation_method)
        
        # 汇总行业分布
        industry_summary = self.analyzer.summarize_industry_distribution(allocated_data)
        
        return industry_summary, holder_data, allocated_data
    
    def plot_shareholder_industry_chart(self, industry_summary, holder_name, allocation_method):
        """
        绘制股东行业分布图
        
        Parameters:
        industry_summary: 行业汇总数据
        holder_name: 股东名称
        allocation_method: 分配方法
        
        Returns:
        Figure: 图表对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 饼图
        industry_data = industry_summary[industry_summary['holder_name'] == holder_name]
        colors = plt.cm.Set3(np.linspace(0, 1, len(industry_data)))
        
        wedges, texts, autotexts = ax1.pie(
            industry_data['amount_percentage'],
            labels=industry_data['industry'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax1.set_title(f'{holder_name}\n行业分布（{allocation_method}分配）', fontsize=14, fontweight='bold')
        
        # 条形图
        sorted_data = industry_data.sort_values('amount_percentage', ascending=True)
        y_pos = np.arange(len(sorted_data))
        
        bars = ax2.barh(y_pos, sorted_data['amount_percentage'], color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_data['industry'])
        ax2.set_xlabel('占比 (%)')
        ax2.set_title('行业分布详情')
        
        # 在条形图上添加数值
        for i, (bar, percentage, companies) in enumerate(zip(bars, 
                                                          sorted_data['amount_percentage'], 
                                                          sorted_data['ts_code'])):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{percentage:.1f}% ({companies}家公司)', 
                    ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def calculate_concentration_metrics(self, industry_summary, holder_name):
        """
        计算集中度指标
        
        Parameters:
        industry_summary: 行业汇总数据
        holder_name: 股东名称
        
        Returns:
        dict: 集中度指标
        """
        holder_summary = industry_summary[industry_summary['holder_name'] == holder_name]
        percentages = holder_summary['amount_percentage'] / 100
        
        hhi = (percentages ** 2).sum()
        cr3 = percentages.nlargest(3).sum()
        top_industry = holder_summary.nlargest(1, 'amount_percentage')['industry'].iloc[0]
        top_percentage = holder_summary.nlargest(1, 'amount_percentage')['amount_percentage'].iloc[0]
        
        return {
            'HHI指数': hhi,
            '前三大行业占比': cr3,
            '主要行业': top_industry,
            '主要行业占比': top_percentage,
            '投资风格': '集中' if hhi > 0.25 else '分散' if hhi < 0.15 else '适中'
        }
    
    def get_top_shareholders(self, df, top_n=10):
        """
        获取持股量最大的股东
        
        Parameters:
        df: 股东持股数据
        top_n: 前N名
        
        Returns:
        Series: 持股量最大的股东
        """
        top_holders = df.groupby('holder_name')['hold_amount'].sum().nlargest(top_n)
        return top_holders
    
    def generate_concentration_report(self, df, hangyeshuju_with_industry, top_n=10, allocation_method='equal'):
        """
        生成行业集中度报告
        
        Parameters:
        df: 股东持股数据
        hangyeshuju_with_industry: 已包含industry_list的行业数据
        top_n: 前N名股东
        allocation_method: 分配方法
        
        Returns:
        DataFrame: 集中度报告
        """
        print(f"生成前{top_n}大股东的行业集中度报告...")
        
        top_holders = self.get_top_shareholders(df, top_n)
        
        report_data = []
        for holder in top_holders.index:
            result = self.analyze_single_shareholder(df, hangyeshuju_with_industry, holder, allocation_method)
            
            if result is not None:
                industry_summary, _, _ = result
                metrics = self.calculate_concentration_metrics(industry_summary, holder)
                
                report_data.append({
                    '股东名称': holder,
                    '总持股量': top_holders[holder],
                    '持股公司数': industry_summary[industry_summary['holder_name'] == holder]['ts_code'].sum(),
                    'HHI指数': metrics['HHI指数'],
                    'CR3': metrics['前三大行业占比'],
                    '主要行业': metrics['主要行业'],
                    '主要行业占比': f"{metrics['主要行业占比']:.1f}%",
                    '投资风格': metrics['投资风格']
                })
        
        report_df = pd.DataFrame(report_data)
        return report_df
    
    def run_complete_analysis(self, df, hangyeshuju_with_industry, target_holders=None, top_n=10):
        """
        运行完整分析流程
        
        Parameters:
        df: 股东持股数据
        hangyeshuju_with_industry: 已包含industry_list的行业数据
        target_holders: 特定股东列表
        top_n: 前N名股东
        
        Returns:
        dict: 分析结果
        """
        print("=== 股东行业投资分析开始 ===")
        
        # 检查数据
        print(f"股东数据形状: {df.shape}")
        print(f"行业数据形状: {hangyeshuju_with_industry.shape}")
        print(f"行业数据列: {hangyeshuju_with_industry.columns.tolist()}")
        
        if 'industry_list' in hangyeshuju_with_industry.columns:
            print("✓ 检测到 industry_list 列，跳过行业映射步骤")
        else:
            print("✗ 未检测到 industry_list 列")
            return None
        
        results = {}
        
        # 分析特定股东
        if target_holders:
            print(f"\n分析特定股东: {target_holders}")
            for holder in target_holders:
                print(f"identified target holder: {holder}")
                if holder in df['holder_name'].values:
                    result = self.analyze_single_shareholder(df, hangyeshuju_with_industry, holder)
                    
                    if result is not None:
                        
                        industry_summary, holder_data, allocated_data = result
                        
                        # 显示详细结果
                        print(f"\n{holder} 的行业分布详情:")
                        holder_summary = industry_summary[industry_summary['holder_name'] == holder]
                        print(holder_summary[['industry', 'amount_percentage', 'ts_code', 'allocated_hold_amount']].to_string(index=False))
                        
                        # 计算集中度指标
                        metrics = self.calculate_concentration_metrics(industry_summary, holder)
                        print(f"\n集中度指标:")
                        print(f"赫芬达尔指数 (HHI): {metrics['HHI指数']:.3f}")
                        print(f"前三大行业占比: {metrics['前三大行业占比']:.1%}")
                        print(f"主要行业: {metrics['主要行业']} ({metrics['主要行业占比']:.1f}%)")
                        print(f"投资风格: {metrics['投资风格']}")
                        
                        # 绘制图表
                        self.plot_shareholder_industry_chart(industry_summary, holder, 'equal')
                        
                        results[holder] = {
                            'industry_summary': industry_summary,
                            'holder_data': holder_data,
                            'allocated_data': allocated_data,
                            'metrics': metrics
                        }
                        
                        print("=" * 50)
                    else:
                        print(f"没有 result for {holder}")
                else:
                    print(f"股东 '{holder}' 未在数据中找到。")
        
        # 生成集中度报告
        print(f"\n生成前{top_n}大股东的行业集中度报告...")
        concentration_report = self.generate_concentration_report(df, hangyeshuju_with_industry, top_n)
        print("\n行业集中度报告:")
        print(concentration_report.to_string(index=False))
        
        results['concentration_report'] = concentration_report
        
        print("\n=== 分析完成 ===")
        return results


# 便捷函数
def create_shareholder_analyzer():
    """创建股东分析器实例"""
    return ShareholderAnalyzer()


def quick_analysis(df, hangyeshuju_with_industry, target_holders=None, top_n=10):
    """
    快速分析函数
    
    Parameters:
    df: 股东持股数据
    hangyeshuju_with_industry: 已包含industry_list的行业数据
    target_holders: 特定股东列表
    top_n: 前N名股东
    
    Returns:
    dict: 分析结果
    """
    analyzer = ShareholderAnalyzer()
    return analyzer.run_complete_analysis(df, hangyeshuju_with_industry, target_holders, top_n)




class ShareholderGrowthAnalyzer:
    """
    股东成长性分析器
    分析股东的换手率、持股周期和行业投资变化
    """
    
    def __init__(self):
        """初始化成长性分析器"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def calculate_turnover_rate(self, df, holder_name):
        """
        计算股东的换手率（修复：加入卖出额）
        
        Parameters:
        df: 包含时间序列的股东持股数据
        holder_name: 股东名称
        
        Returns:
        DataFrame: 换手率分析结果
        """
        print(f"计算股东 '{holder_name}' 的换手率...")
        holder_data = df[df['holder_name'] == holder_name].copy()
        if holder_data.empty:
            print(f"未找到股东: {holder_name}")
            return None

        holder_data['end_date'] = pd.to_datetime(holder_data['end_date'])
        holder_data = holder_data.sort_values('end_date')
        holder_data['quarter'] = holder_data['end_date'].dt.to_period('Q')

        quarterly_data = holder_data.groupby(['quarter', 'ts_code']).agg({
            'hold_amount': 'last',
            'hold_ratio': 'last'
        }).reset_index()

        turnover_analysis = []
        quarters = sorted(quarterly_data['quarter'].unique())
        
        for i in range(1, len(quarters)):
            current_q = quarters[i]
            previous_q = quarters[i-1]

            current_holdings = quarterly_data[quarterly_data['quarter'] == current_q].set_index('ts_code')['hold_amount']
            previous_holdings = quarterly_data[quarterly_data['quarter'] == previous_q].set_index('ts_code')['hold_amount']

            common_stocks = current_holdings.index.intersection(previous_holdings.index)
            new_stocks = current_holdings.index.difference(previous_holdings.index)
            sold_stocks = previous_holdings.index.difference(current_holdings.index)

            buys = current_holdings[new_stocks].sum() / 2 if len(new_stocks) else 0.0
            sells = previous_holdings[sold_stocks].sum() / 2 if len(sold_stocks) else 0.0
            adjustments = abs((current_holdings[common_stocks] - previous_holdings[common_stocks])).sum() / 2 if len(common_stocks) else 0.0

            total_turnover = buys + sells + adjustments
            avg_portfolio_value = (current_holdings.sum() + previous_holdings.sum()) / 2
            turnover_rate = total_turnover / avg_portfolio_value if avg_portfolio_value > 0 else 0

            turnover_analysis.append({
                'quarter': current_q,
                'turnover_rate': turnover_rate,
                'new_stocks_count': int(len(new_stocks)),
                'sold_stocks_count': int(len(sold_stocks)),
                'portfolio_size': int(len(current_holdings)),
                'total_value': float(current_holdings.sum())
            })

        return pd.DataFrame(turnover_analysis)
    
    def calculate_holding_period(self, df, holder_name):
        """
        计算股东的持股周期
        
        Parameters:
        df: 包含时间序列的股东持股数据
        holder_name: 股东名称
        
        Returns:
        dict: 持股周期统计
        """
        print(f"计算股东 '{holder_name}' 的持股周期...")
        
        holder_data = df[df['holder_name'] == holder_name].copy()
        holder_data['end_date'] = pd.to_datetime(holder_data['end_date'])
        
        # 对每只股票计算持有期
        stock_holding_periods = []
        
        for ts_code in holder_data['ts_code'].unique():
            stock_data = holder_data[holder_data['ts_code'] == ts_code].sort_values('end_date')
            
            if len(stock_data) > 1:
                # 计算持有天数
                holding_days = (stock_data['end_date'].iloc[-1] - stock_data['end_date'].iloc[0]).days
                holding_quarters = len(stock_data)
                
                stock_holding_periods.append({
                    'ts_code': ts_code,
                    'holding_days': holding_days,
                    'holding_quarters': holding_quarters,
                    'first_quarter': stock_data['end_date'].iloc[0],
                    'last_quarter': stock_data['end_date'].iloc[-1]
                })
        
        if not stock_holding_periods:
            return None
        
        holding_df = pd.DataFrame(stock_holding_periods)
        
        # 计算统计指标
        stats = {
            'avg_holding_days': holding_df['holding_days'].mean(),
            'median_holding_days': holding_df['holding_days'].median(),
            'avg_holding_quarters': holding_df['holding_quarters'].mean(),
            'median_holding_quarters': holding_df['holding_quarters'].median(),
            'total_stocks_analyzed': len(holding_df),
            'longest_holding_days': holding_df['holding_days'].max(),
            'shortest_holding_days': holding_df['holding_days'].min()
        }
        
        return stats, holding_df
    
    def analyze_industry_concentration_trend(self, df, hangyeshuju_with_industry, holder_name):
        """
        分析股东行业集中度趋势（对多行业均分持股金额，避免重复累计）
        """
        print(f"分析股东 '{holder_name}' 的行业集中度趋势...")
        holder_data = df[df['holder_name'] == holder_name].copy()
        holder_data['end_date'] = pd.to_datetime(holder_data['end_date'])

        # 按季度
        holder_data['quarter'] = holder_data['end_date'].dt.to_period('Q')
        quarters = sorted(holder_data['quarter'].unique())

        # 保障 industry_list 为列表
        merged_base = hangyeshuju_with_industry[['ts_code', 'industry_list']].copy()
        if 'industry_list' in merged_base.columns and merged_base['industry_list'].dtype == object:
            import ast
            def to_list_safe(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, str):
                    try:
                        v = ast.literal_eval(x)
                        return v if isinstance(v, list) else [x]
                    except Exception:
                        return [x]
                return []
            merged_base['industry_list'] = merged_base['industry_list'].apply(to_list_safe)

        industry_trend = []

        for quarter in quarters:
            quarter_data = holder_data[holder_data['quarter'] == quarter]

            # 合并行业数据
            merged = quarter_data.merge(merged_base, on='ts_code', how='left')
            # 均分持股金额到多行业
            expanded_rows = []
            for _, row in merged.iterrows():
                industries = row['industry_list'] if isinstance(row['industry_list'], list) else []
                if not industries or len(industries) == 0:
                    expanded_rows.append({'industry': '未分类', 'hold_amount': row['hold_amount']})
                    continue
                per_amount = row['hold_amount'] / len(industries)
                for ind in industries:
                    expanded_rows.append({'industry': ind, 'hold_amount': per_amount})

            if not expanded_rows:
                continue

            industry_df = pd.DataFrame(expanded_rows)
            industry_summary = industry_df.groupby('industry', as_index=False)['hold_amount'].sum()

            total_amount = industry_summary['hold_amount'].sum()
            industry_summary['percentage'] = (industry_summary['hold_amount'] / total_amount * 100).round(2)

            # 第一大行业
            top_industry = industry_summary.nlargest(1, 'hold_amount')
            industry_trend.append({
                'quarter': quarter,
                'top_industry': top_industry['industry'].iloc[0],
                'top_industry_percentage': top_industry['percentage'].iloc[0],
                'top_industry_hold_amount': top_industry['hold_amount'].iloc[0],
                'hhi_index': ((industry_summary['percentage'] / 100) ** 2).sum(),
                'industry_count': len(industry_summary),
                'total_hold_amount': total_amount
            })

        return pd.DataFrame(industry_trend)
    
    def plot_growth_analysis(self, turnover_df, industry_trend_df, holder_name):
        """
        绘制成长性分析图表
        
        Parameters:
        turnover_df: 换手率数据
        industry_trend_df: 行业趋势数据
        holder_name: 股东名称
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 辅助函数：安全转换季度数据为字符串
        def safe_quarter_to_str(quarter_series):
            """将季度数据安全转换为字符串"""
            return [str(q) for q in quarter_series]        
        
    # 1. 换手率趋势
        if turnover_df is not None and not turnover_df.empty:
            ax1.plot(safe_quarter_to_str(turnover_df['quarter']), turnover_df['turnover_rate'], 
                    marker='o', linewidth=2)
            ax1.set_title(f'{holder_name} - 换手率趋势', fontsize=14, fontweight='bold')
            ax1.set_xlabel('季度')
            ax1.set_ylabel('换手率')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. 第一大行业占比趋势
        if industry_trend_df is not None and not industry_trend_df.empty:
            quarters_str = safe_quarter_to_str(industry_trend_df['quarter'])
            ax2.plot(quarters_str, industry_trend_df['top_industry_percentage'], 
                    marker='s', linewidth=2, color='red')
            ax2.set_title(f'{holder_name} - 第一大行业占比趋势', fontsize=14, fontweight='bold')
            ax2.set_xlabel('季度')
            ax2.set_ylabel('第一大行业占比 (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 在图上标注行业名称
            for i, row in industry_trend_df.iterrows():
                ax2.annotate(row['top_industry'], 
                        (quarters_str[i], row['top_industry_percentage']),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        # 3. HHI指数趋势
        if industry_trend_df is not None and not industry_trend_df.empty:
            ax3.plot(quarters_str, industry_trend_df['hhi_index'], 
                    marker='^', linewidth=2, color='green')
            ax3.set_title(f'{holder_name} - 行业集中度趋势 (HHI)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('季度')
            ax3.set_ylabel('HHI指数')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. 投资组合规模
        if turnover_df is not None and not turnover_df.empty:
            ax4.bar(safe_quarter_to_str(turnover_df['quarter']), turnover_df['portfolio_size'], alpha=0.7)
            ax4.set_title(f'{holder_name} - 投资组合规模', fontsize=14, fontweight='bold')
            ax4.set_xlabel('季度')
            ax4.set_ylabel('持股公司数量')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def run_complete_growth_analysis(self, df, hangyeshuju_with_industry, holder_name):
        """
        运行完整的成长性分析
        
        Parameters:
        df: 股东持股数据
        hangyeshuju_with_industry: 行业数据
        holder_name: 股东名称
        
        Returns:
        dict: 成长性分析结果
        """
        print(f"=== 开始分析股东 '{holder_name}' 的成长性 ===")
        
        results = {}
        
        # 1. 计算换手率
        turnover_df = self.calculate_turnover_rate(df, holder_name)
        results['turnover_analysis'] = turnover_df
        
        # 2. 计算持股周期
        holding_stats, holding_df = self.calculate_holding_period(df, holder_name)
        results['holding_period_stats'] = holding_stats
        results['holding_period_details'] = holding_df
        
        # 3. 分析行业集中度趋势
        industry_trend_df = self.analyze_industry_concentration_trend(df, hangyeshuju_with_industry, holder_name)
        results['industry_trend'] = industry_trend_df
        
        # 4. 绘制图表
        if turnover_df is not None or industry_trend_df is not None:
            self.plot_growth_analysis(turnover_df, industry_trend_df, holder_name)
        
        # 5. 输出总结报告
        self.print_growth_summary(results, holder_name)
        
        print(f"=== 股东 '{holder_name}' 成长性分析完成 ===")
        return results
    
    def print_growth_summary(self, results, holder_name):
        """打印成长性分析总结"""
        print(f"\n{'='*60}")
        print(f"股东 '{holder_name}' 成长性分析总结")
        print(f"{'='*60}")
        
        if results['turnover_analysis'] is not None:
            turnover_df = results['turnover_analysis']
            avg_turnover = turnover_df['turnover_rate'].mean()
            print(f"平均换手率: {avg_turnover:.2%}")
            print(f"最高换手率: {turnover_df['turnover_rate'].max():.2%} (季度: {turnover_df.loc[turnover_df['turnover_rate'].idxmax(), 'quarter']})")
            print(f"最低换手率: {turnover_df['turnover_rate'].min():.2%} (季度: {turnover_df.loc[turnover_df['turnover_rate'].idxmin(), 'quarter']})")
        
        if results['holding_period_stats'] is not None:
            stats = results['holding_period_stats']
            print(f"\n持股周期分析:")
            print(f"平均持股天数: {stats['avg_holding_days']:.0f} 天")
            print(f"平均持股季度: {stats['avg_holding_quarters']:.1f} 个季度")
            print(f"最长持股: {stats['longest_holding_days']} 天")
            print(f"分析股票数量: {stats['total_stocks_analyzed']} 只")
        
        if results['industry_trend'] is not None:
            industry_df = results['industry_trend']
            if not industry_df.empty:
                first_q = industry_df.iloc[0]
                last_q = industry_df.iloc[-1]
                print(f"\n行业投资趋势:")
                print(f"初期第一大行业: {first_q['top_industry']} ({first_q['top_industry_percentage']:.1f}%)")
                print(f"近期第一大行业: {last_q['top_industry']} ({last_q['top_industry_percentage']:.1f}%)")
                print(f"行业集中度变化: {first_q['hhi_index']:.3f} → {last_q['hhi_index']:.3f}")
                
                # 计算行业稳定性
                industry_changes = len(industry_df['top_industry'].unique())
                print(f"行业切换次数: {industry_changes - 1}")

        print(f"{'='*60}")


# 便捷函数
def create_growth_analyzer():
    """创建成长性分析器实例"""
    return ShareholderGrowthAnalyzer()