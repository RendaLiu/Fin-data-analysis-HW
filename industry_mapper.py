# industry_mapper.py
"""
行业映射工具
用于将非标准化的主营业务类型映射到统一的行业分类
支持多标签处理和一对多关系
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict
import jieba

class IndustryMapper:
    """
    行业映射器类
    将非标准化的主营业务类型映射到统一的行业分类
    """
    
    def __init__(self, custom_mapping=None):
        """
        初始化行业映射器
        
        Parameters:
        custom_mapping: 自定义行业映射规则，字典格式 {关键词: 行业}
        """
        # 内置行业分类体系
        self.industry_hierarchy = self._build_default_industry_hierarchy()
        
        # 合并自定义映射
        if custom_mapping:
            self.industry_hierarchy.update(custom_mapping)
    
    def _build_default_industry_hierarchy(self):
        """构建默认行业分类体系"""
        industry_mapping = {
            # 食品饮料相关
            '果汁': '食品饮料', '乳制品': '食品饮料', '食用油': '食品饮料', 
            '调味品': '食品饮料', '饮料': '食品饮料', '白酒': '食品饮料',
            '啤酒': '食品饮料', '葡萄酒': '食品饮料', '零食': '食品饮料', 
            '肉制品': '食品饮料', '食品': '食品饮料', '餐饮': '食品饮料',
            '糖果': '食品饮料', '巧克力': '食品饮料', '方便食品': '食品饮料',
            '茶叶': '食品饮料', '咖啡': '食品饮料',
            
            # 物流运输相关
            '物流': '交通运输', '快递': '交通运输', '货运': '交通运输',
            '仓储': '交通运输', '航运': '交通运输', '航空': '交通运输',
            '铁路': '交通运输', '公路': '交通运输', '港口': '交通运输',
            '运输': '交通运输', '供应链': '交通运输',
            
            # 咨询服务相关
            '咨询': '商业服务', '专业服务': '商业服务', '技术服务': '商业服务',
            '管理咨询': '商业服务', '商务服务': '商业服务', '外包': '商业服务',
            
            # 科技相关
            '软件': '信息技术', '硬件': '信息技术', '半导体': '信息技术',
            '芯片': '信息技术', '电子': '信息技术', '通信': '信息技术',
            '互联网': '信息技术', '计算机': '信息技术', '人工智能': '信息技术',
            '大数据': '信息技术', '云计算': '信息技术', '物联网': '信息技术',
            '网络安全': '信息技术', '游戏': '信息技术',
            '航空航天': '航空航天与国防', '卫星': '航空航天与国防', '航空': '航空航天与国防',
            '军队': '航空航天与国防', '国防': '航空航天与国防', '航天': '航空航天与国防', 


            
            # 金融相关
            '银行': '金融', '保险': '金融', '证券': '金融', '信托': '金融',
            '基金': '金融', '投资': '金融', '信贷': '金融', '租赁': '金融',
            '支付': '金融', '理财': '金融',
            
            # 医药相关
            '医药': '医药生物', '生物': '医药生物', '医疗': '医药生物',
            '制药': '医药生物', '医疗器械': '医药生物', '医院': '医药生物',
            '保健': '医药生物', '健康': '医药生物', '疫苗': '医药生物',
            '中药': '医药生物', '西药': '医药生物', '药店': '医药生物',
            
            # 房地产相关
            '房地产': '房地产', '物业': '房地产', 
            '装修': '房地产', '园区': '房地产',
            '地产': '房地产', '住宅': '房地产', '商业地产': '房地产',
            
            # 能源相关
            '石油': '能源', '煤炭': '能源', '电力': '能源', '新能源': '能源',
            '天然气': '能源', '太阳能': '能源', '风能': '能源', '核电': '能源',
            '水电': '能源', '火电': '能源', '风泵': '能源', '煤': '能源',
            
            # 制造业
            '机械': '机械设备', '设备': '机械设备', '制造': '机械设备',
            '汽车': '汽车', '零部件': '汽车', '整车': '汽车', '新能源汽车': '汽车', '轿车': '汽车', '新能源汽车': '汽车', '货车': '汽车',
            '家电': '家用电器', '空调': '家用电器', '洗衣机': '家用电器', '冰箱': '家用电器',
            '电器': '电器', '电子产品': '消费电子',
            '纺织': '纺织服装', '服装': '纺织服装', '鞋类': '纺织服装',
            '衣服': '纺织服装', '服饰': '纺织服装',
            
            # 消费零售
            '零售': '商业贸易', '批发': '商业贸易', '超市': '商业贸易',
            '百货': '商业贸易', '电商': '商业贸易', '购物': '商业贸易',
            
            # 原材料
            '化工': '原材料', '材料': '原材料', '金属': '原材料', '钢铁': '原材料',
            '有色金属': '原材料', '新材料': '原材料', '塑料': '原材料', '橡胶': '原材料',
            '合金': '原材料', '玻璃': '原材料', '水泥': '原材料', '白银': '原材料',
            '煤制品': '原材料',
            
            # 文化传媒
            '媒体': '传媒', '出版': '传媒', '广告': '传媒', '影视': '传媒',
            '文化': '传媒', '娱乐': '传媒', '体育': '传媒', '旅游': '休闲服务',
            
            # 公共事业
            '环保': '公用事业', '水务': '公用事业', '燃气': '公用事业',
            '公共': '公用事业', '基础设施': '公用事业', '变电设备': '公用事业',
            '火电': '公用事业', '水电': '公用事业', '水务': '公用事业',
            
            # 农业
            '农业': '农林牧渔', '林业': '农林牧渔', '牧业': '农林牧渔',
            '渔业': '农林牧渔', '种植': '农林牧渔', '养殖': '农林牧渔',
            '农副产品': '农林牧渔', '生物肥': '农林牧渔',
        }
        
        return industry_mapping
    
    def add_custom_mapping(self, keyword, industry):
        """
        添加自定义映射规则
        
        Parameters:
        keyword: 关键词
        industry: 对应的行业
        """
        self.industry_hierarchy[keyword] = industry
    
    def categorize_business(self, business_str):
        """
        将业务描述分类到标准行业
        
        Parameters:
        business_str: 业务描述字符串
        
        Returns:
        list: 行业列表
        """
        if pd.isna(business_str) or business_str == '' or business_str == 'nan':
            return ['未知']
        
        categories = set()
        business_str = str(business_str).strip()
        
        # 如果已经是标准行业，直接返回
        if business_str in set(self.industry_hierarchy.values()):
            return [business_str]
        
        # 分割业务描述
        business_items = re.split(r'[,，、;；\s]', business_str)
        
        # 标记是否有匹配到任何已知行业
        has_known_industry = False
        
        for item in business_items:
            item = item.strip()
            if not item:
                continue
                
            # 精确匹配
            matched = False
            for keyword, industry in self.industry_hierarchy.items():
                if keyword in item:
                    categories.add(industry)
                    matched = True
                    has_known_industry = True
            
            # 如果没有匹配到，使用模糊匹配
            if not matched and len(item) > 1:
                # 尝试分词匹配
                words = jieba.lcut(item)
                for word in words:
                    if len(word) > 1 and word in self.industry_hierarchy:
                        categories.add(self.industry_hierarchy[word])
                        matched = True
                        has_known_industry = True
        
        # 只有当没有任何已知行业匹配时，才添加"其他"
        if not has_known_industry:
            categories.add('其他')
        
        return list(categories) if categories else ['未知']
    
    def map_industry_data(self, df, business_column='business_category', ts_code_column='ts_code'):
        """
        映射行业数据
        
        Parameters:
        df: 包含业务分类数据的DataFrame
        business_column: 业务分类列名
        ts_code_column: 股票代码列名
        
        Returns:
        DataFrame: 展开后的行业数据，包含ts_code, industry, original_business列
        """
        industry_expanded = []
        
        for _, row in df.iterrows():
            ts_code = row[ts_code_column]
            business_categories = self.categorize_business(row[business_column])
            
            for industry in business_categories:
                industry_expanded.append({
                    'ts_code': ts_code,
                    'industry': industry,
                    'original_business': row[business_column]
                })
        
        return pd.DataFrame(industry_expanded)
    
    def add_industry_column(self, df, business_column='business_category', 
                           new_column='mapped_industry', format='list'):
        """
        将映射后的行业添加到原始DataFrame中
        
        Parameters:
        df: 原始DataFrame
        business_column: 业务分类列名
        new_column: 新列名
        format: 输出格式 ('list', 'string', 'primary')
        
        Returns:
        DataFrame: 添加了行业列的新DataFrame
        """
        df_copy = df.copy()
        
        # 映射行业数据
        industry_expanded = self.map_industry_data(df_copy, business_column)
        
        # 根据格式要求处理行业数据
        if format == 'list':
            # 将行业列表作为新列
            industry_lists = industry_expanded.groupby('ts_code')['industry'].apply(list).reset_index()
            industry_lists.columns = ['ts_code', new_column]
            df_copy = df_copy.merge(industry_lists, on='ts_code', how='left')
            
        elif format == 'string':
            # 将行业列表转换为逗号分隔的字符串
            industry_strings = industry_expanded.groupby('ts_code')['industry'].apply(
                lambda x: ', '.join(x)
            ).reset_index()
            industry_strings.columns = ['ts_code', new_column]
            df_copy = df_copy.merge(industry_strings, on='ts_code', how='left')
            
        elif format == 'primary':
            # 只取第一个行业作为主要行业
            primary_industries = industry_expanded.groupby('ts_code')['industry'].first().reset_index()
            primary_industries.columns = ['ts_code', new_column]
            df_copy = df_copy.merge(primary_industries, on='ts_code', how='left')
            
        else:
            raise ValueError("format参数必须是 'list', 'string' 或 'primary'")
        
        # 处理缺失值
        df_copy[new_column] = df_copy[new_column].fillna('未知')
        
        return df_copy
    
    def get_industry_statistics(self, df, business_column='business_category', with_industry = 0):
        """
        获取行业统计信息
        
        Parameters:
        df: 包含业务分类数据的DataFrame
        business_column: 业务分类列名
        
        Returns:
        dict: 行业统计信息
        """

        
        # 映射行业数据
        if with_industry == 0:
            df_with_industry = self.add_industry_column(df, business_column, 'industry_list', 'list')
            print(df_with_industry)
        else: 
            df_with_industry = df.copy()
        
        # 计算每个公司的行业数量
        industry_expanded = []
        for _, row in df_with_industry.iterrows():
            if isinstance(row['industry_list'], list):
                for industry in row['industry_list']:
                    industry_expanded.append({
                        'ts_code': row['ts_code'],
                        'industry': industry
                    })
        
        industry_df = pd.DataFrame(industry_expanded)
        
        # 计算每个公司的行业数量
        company_industry_count = industry_df.groupby('ts_code')['industry'].nunique().reset_index()
        company_industry_count.columns = ['ts_code', 'industry_count']
        
        # 合并行业数据和行业数量
        industry_weighted = industry_df.merge(company_industry_count, on='ts_code')
        
        # 计算每个行业的权重（1/n）
        industry_weighted['weight'] = 1 / industry_weighted['industry_count']
        
        # 统计各行业加权公司数量
        weighted_industry_stats = industry_weighted.groupby('industry')['weight'].sum().to_dict()
        
        # 统计原始行业分布（不加权）
        raw_industry_stats = industry_df['industry'].value_counts().to_dict()
        
        
        # 统计多行业公司数量
        multi_industry_companies = (company_industry_count['industry_count'] > 1).sum()
        
        # 统计"其他"行业的比例
        other_ratio = (industry_df['industry'] == '其他').sum() / len(industry_df) if len(industry_df) > 0 else 0
        
        # 计算加权"其他"行业比例
        weighted_other_ratio = weighted_industry_stats.get('其他', 0) / sum(weighted_industry_stats.values()) if weighted_industry_stats else 0
        
        return {
            'raw_industry_distribution': raw_industry_stats,  # 原始行业分布
            'weighted_industry_distribution': weighted_industry_stats,  # 加权行业分布
            'total_companies': len(df),
            'multi_industry_companies': multi_industry_companies,
            'multi_industry_ratio': multi_industry_companies / len(df) if len(df) > 0 else 0,
            'other_industry_ratio': other_ratio,
            'weighted_other_ratio': weighted_other_ratio
        }
    
    def export_mapping_rules(self, filepath):
        """
        导出映射规则到文件
        
        Parameters:
        filepath: 文件路径
        """
        mapping_df = pd.DataFrame([
            {'keyword': k, 'industry': v} 
            for k, v in self.industry_hierarchy.items()
        ])
        mapping_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    
    def import_mapping_rules(self, filepath):
        """
        从文件导入映射规则
        
        Parameters:
        filepath: 文件路径
        """
        mapping_df = pd.read_csv(filepath)
        for _, row in mapping_df.iterrows():
            self.industry_hierarchy[row['keyword']] = row['industry']


class MultiIndustryAnalyzer:
    """
    多行业分析器
    处理一对多行业关系和分析
    """
    
    def __init__(self, industry_mapper):
        """
        初始化多行业分析器
        
        Parameters:
        industry_mapper: IndustryMapper实例
        """
        self.mapper = industry_mapper
    
    def analyze_company_industries(self, df_holdings, df_industry, 
                                 business_column='business_category', 
                                 ts_code_column='ts_code'):
        """
        分析公司行业分布
        
        Parameters:
        df_holdings: 持股数据
        df_industry: 行业数据
        business_column: 业务分类列名
        ts_code_column: 股票代码列名
        
        Returns:
        DataFrame: 合并后的行业持股数据
        """
        # 映射行业数据
        industry_expanded = self.mapper.map_industry_data(
            df_industry, business_column, ts_code_column
        )
        
        # 合并持股数据和行业数据
        merged_data = df_holdings.merge(
            industry_expanded, on=ts_code_column, how='left'
        )
        
        # 处理缺失行业数据
        merged_data['industry'] = merged_data['industry'].fillna('未知')
        
        return merged_data
    
    def calculate_industry_allocation(self, merged_data, allocation_method='equal'):
        """
        计算行业分配
        
        Parameters:
        merged_data: 合并后的行业持股数据
        allocation_method: 分配方法 ('equal', 'weighted')
        
        Returns:
        DataFrame: 行业分配结果
        """
        if allocation_method == 'equal':
            # 平均分配：如果一个公司有N个行业，每个行业分配1/N的持股数量
            company_industry_count = merged_data.groupby('ts_code')['industry'].nunique().reset_index()
            company_industry_count.columns = ['ts_code', 'industry_count']
            
            merged_data = merged_data.merge(company_industry_count, on='ts_code')
            merged_data['allocated_hold_amount'] = merged_data['hold_amount'] / merged_data['industry_count']
        
        elif allocation_method == 'weighted':
            # 加权分配：这里简化处理，实际中可以根据业务收入占比等数据进行加权
            company_industry_count = merged_data.groupby('ts_code')['industry'].nunique().reset_index()
            company_industry_count.columns = ['ts_code', 'industry_count']
            
            merged_data = merged_data.merge(company_industry_count, on='ts_code')
            merged_data['allocated_hold_amount'] = merged_data['hold_amount'] / merged_data['industry_count']
        
        else:
            raise ValueError(f"不支持的分配方法: {allocation_method}")
        
        return merged_data
    
    def summarize_industry_distribution(self, allocated_data, group_by='holder_name'):
        """
        汇总行业分布
        
        Parameters:
        allocated_data: 分配后的数据
        group_by: 分组字段
        
        Returns:
        DataFrame: 行业分布汇总
        """
        industry_summary = allocated_data.groupby([group_by, 'industry']).agg({
            'allocated_hold_amount': 'sum',
            'ts_code': 'nunique',
            'hold_ratio': 'mean'
        }).reset_index()
        
        # 计算每个分组的总量
        group_totals = industry_summary.groupby(group_by)['allocated_hold_amount'].sum().reset_index()
        group_totals.columns = [group_by, 'total_allocated_amount']
        
        industry_summary = industry_summary.merge(group_totals, on=group_by)
        industry_summary['amount_percentage'] = (
            industry_summary['allocated_hold_amount'] / industry_summary['total_allocated_amount'] * 100
        ).round(2)
        
        return industry_summary


# 便捷函数
def create_industry_mapper(custom_mapping=None):
    """
    创建行业映射器实例
    
    Parameters:
    custom_mapping: 自定义映射规则
    
    Returns:
    IndustryMapper实例
    """
    return IndustryMapper(custom_mapping)


def quick_industry_mapping(df, business_column='business_category', custom_mapping=None):
    """
    快速行业映射
    
    Parameters:
    df: 包含业务分类数据的DataFrame
    business_column: 业务分类列名
    custom_mapping: 自定义映射规则
    
    Returns:
    DataFrame: 映射后的行业数据
    """
    mapper = IndustryMapper(custom_mapping)
    return mapper.map_industry_data(df, business_column)


def add_industry_column_to_df(df, business_column='business_category', 
                             new_column='mapped_industry', format='list', custom_mapping=None):
    """
    便捷函数：将映射后的行业添加到原始DataFrame中
    
    Parameters:
    df: 原始DataFrame
    business_column: 业务分类列名
    new_column: 新列名
    format: 输出格式 ('list', 'string', 'primary')
    custom_mapping: 自定义映射规则
    
    Returns:
    DataFrame: 添加了行业列的新DataFrame
    """
    mapper = IndustryMapper(custom_mapping)
    return mapper.add_industry_column(df, business_column, new_column, format)

# 示例使用
if __name__ == "__main__":
    # 示例数据
    sample_data = pd.DataFrame({
        'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ'],
        'business_category': ['存贷款业务、国际业务、机构业务、结算业务、人民币理财、外汇理财、银行卡、证券业务', 
                              '住宅楼盘', 
                              '半导体材料、半导体分立器件、电子设备及加工、电子元器件',
                              '未知的类型'
                              ]
    })
    
    # 创建映射器
    mapper = IndustryMapper()
    
    # 映射行业
    result = mapper.map_industry_data(sample_data)
    # print("行业映射结果:")
    # print(result)

    sample_with_industry = mapper.add_industry_column(sample_data, 'business_category', 'mapped_industry', 'list')
    # print("\n添加行业列后的DataFrame:")
    # print(sample_with_industry)

    
    # 获取统计信息
    stats = mapper.get_industry_statistics(sample_data)
    print("\n行业统计信息:")
    print(stats)

