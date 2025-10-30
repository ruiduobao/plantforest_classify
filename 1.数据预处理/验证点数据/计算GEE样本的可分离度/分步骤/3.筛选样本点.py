#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本点筛选脚本
功能：基于偏离度分析结果筛选样本点，删除异常样本以提高分类效果
筛选策略：
- 删除所有 Extreme 等级样本点
- 删除部分 High 等级样本点 (随机删除70%)
- 保留所有 Normal 和 Moderate 等级样本点
作者：锐多宝 (ruiduobao)
日期：2025年10月21日
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import random
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
INPUT_SHAPEFILE = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\计算样本点的偏离度分析\样本点偏离度分析_20251023_215650.shp"
output_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\筛选样本点(通过偏离度)"

HIGH_REMOVAL_RATIO = 0.7  # 删除70%的High等级样本点
RANDOM_SEED = 42  # 随机种子，确保结果可重现

def read_deviation_data(shapefile_path):
    """
    读取偏离度分析结果数据
    参数：
        shapefile_path: 偏离度分析结果shapefile路径
    返回：
        gdf: GeoDataFrame格式的数据
    """
    print(f"正在读取偏离度分析结果: {shapefile_path}")
    
    try:
        gdf = gpd.read_file(shapefile_path)
        print(f"成功读取 {len(gdf)} 个样本点")
        return gdf
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        raise

def analyze_deviation_distribution(gdf):
    """
    分析偏离度等级分布
    参数：
        gdf: 包含偏离度信息的GeoDataFrame
    """
    print("\n" + "="*50)
    print("偏离度等级分布分析")
    print("="*50)
    
    # 统计各等级样本数量
    level_counts = gdf['dev_level'].value_counts()
    total_samples = len(gdf)
    
    print(f"\n总样本数: {total_samples}")
    print("\n各等级分布:")
    for level, count in level_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  - {level}: {count} 个样本 ({percentage:.1f}%)")
    
    # 统计各类别的偏离度分布
    print(f"\n各类别偏离度分布:")
    class_deviation = pd.crosstab(gdf['landcover'], gdf['dev_level'])
    print(class_deviation)
    
    return level_counts, class_deviation

def filter_samples(gdf, high_removal_ratio=0.7, random_seed=42):
    """
    根据偏离度等级筛选样本点
    参数：
        gdf: 包含偏离度信息的GeoDataFrame
        high_removal_ratio: High等级样本点的删除比例
        random_seed: 随机种子
    返回：
        filtered_gdf: 筛选后的GeoDataFrame
        removal_stats: 删除统计信息
    """
    print(f"\n正在执行样本点筛选...")
    print(f"筛选策略:")
    print(f"  - 删除所有 Extreme 等级样本点")
    print(f"  - 删除 {high_removal_ratio*100:.0f}% 的 High 等级样本点")
    print(f"  - 保留所有 Normal 和 Moderate 等级样本点")
    
    # 设置随机种子
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 初始化删除统计
    removal_stats = {
        'Extreme': {'total': 0, 'removed': 0},
        'High': {'total': 0, 'removed': 0},
        'Moderate': {'total': 0, 'removed': 0},
        'Normal': {'total': 0, 'removed': 0}
    }
    
    # 统计各等级样本数量
    for level in ['Extreme', 'High', 'Moderate', 'Normal']:
        if level in gdf['dev_level'].values:
            removal_stats[level]['total'] = len(gdf[gdf['dev_level'] == level])
    
    # 创建筛选后的数据副本
    filtered_gdf = gdf.copy()
    
    # 1. 删除所有 Extreme 等级样本点
    extreme_mask = filtered_gdf['dev_level'] == 'Extreme'
    extreme_count = extreme_mask.sum()
    if extreme_count > 0:
        filtered_gdf = filtered_gdf[~extreme_mask]
        removal_stats['Extreme']['removed'] = extreme_count
        print(f"  删除了 {extreme_count} 个 Extreme 等级样本点")
    
    # 2. 随机删除部分 High 等级样本点
    high_samples = filtered_gdf[filtered_gdf['dev_level'] == 'High']
    high_count = len(high_samples)
    if high_count > 0:
        # 计算要删除的数量
        remove_count = int(high_count * high_removal_ratio)
        
        # 随机选择要删除的样本索引
        remove_indices = np.random.choice(high_samples.index, size=remove_count, replace=False)
        
        # 删除选中的样本
        filtered_gdf = filtered_gdf.drop(remove_indices)
        removal_stats['High']['removed'] = remove_count
        print(f"  随机删除了 {remove_count} 个 High 等级样本点 (总共{high_count}个)")
    
    # 3. 保留所有 Normal 和 Moderate 等级样本点
    normal_count = len(filtered_gdf[filtered_gdf['dev_level'] == 'Normal'])
    moderate_count = len(filtered_gdf[filtered_gdf['dev_level'] == 'Moderate'])
    print(f"  保留了 {normal_count} 个 Normal 等级样本点")
    print(f"  保留了 {moderate_count} 个 Moderate 等级样本点")
    
    print(f"\n筛选结果:")
    print(f"  原始样本数: {len(gdf)}")
    print(f"  筛选后样本数: {len(filtered_gdf)}")
    print(f"  删除样本数: {len(gdf) - len(filtered_gdf)}")
    print(f"  保留比例: {len(filtered_gdf)/len(gdf)*100:.1f}%")
    
    return filtered_gdf, removal_stats

def save_filtered_samples(filtered_gdf, output_path):
    """
    保存筛选后的样本点到shapefile
    参数：
        filtered_gdf: 筛选后的GeoDataFrame
        output_path: 输出文件路径
    """
    print(f"\n正在保存筛选后的样本点到: {output_path}")
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存到shapefile
        filtered_gdf.to_file(output_path, driver='ESRI Shapefile')
        print(f"成功保存 {len(filtered_gdf)} 个筛选后的样本点")
        
    except Exception as e:
        print(f"保存文件失败: {str(e)}")
        raise

def create_comparison_visualizations(original_gdf, filtered_gdf, removal_stats, output_dir, timestamp):
    """
    创建筛选前后的对比可视化图表
    参数：
        original_gdf: 原始数据
        filtered_gdf: 筛选后数据
        removal_stats: 删除统计信息
        output_dir: 输出目录
        timestamp: 时间戳
    """
    print("\n正在生成筛选对比可视化图表...")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sample Filtering Comparison Report', fontsize=16, fontweight='bold')
    
    # 子图1：筛选前后偏离度等级分布对比
    ax1 = axes[0, 0]
    
    # 统计筛选前后各等级数量
    original_counts = original_gdf['dev_level'].value_counts()
    filtered_counts = filtered_gdf['dev_level'].value_counts()
    
    # 创建对比数据
    levels = ['Normal', 'Moderate', 'High', 'Extreme']
    original_values = [original_counts.get(level, 0) for level in levels]
    filtered_values = [filtered_counts.get(level, 0) for level in levels]
    
    x = np.arange(len(levels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_values, width, label='筛选前', alpha=0.8)
    bars2 = ax1.bar(x + width/2, filtered_values, width, label='筛选后', alpha=0.8)
    
    ax1.set_title('Deviation Level Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Deviation Level')
    ax1.set_ylabel('Sample Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.legend()
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 子图2：各类别样本数量对比
    ax2 = axes[0, 1]
    
    original_class_counts = original_gdf['landcover'].value_counts().sort_index()
    filtered_class_counts = filtered_gdf['landcover'].value_counts().sort_index()
    
    classes = sorted(original_gdf['landcover'].unique())
    original_class_values = [original_class_counts.get(cls, 0) for cls in classes]
    filtered_class_values = [filtered_class_counts.get(cls, 0) for cls in classes]
    
    x_class = np.arange(len(classes))
    bars3 = ax2.bar(x_class - width/2, original_class_values, width, label='筛选前', alpha=0.8)
    bars4 = ax2.bar(x_class + width/2, filtered_class_values, width, label='筛选后', alpha=0.8)
    
    ax2.set_title('Sample Count by Class Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Land Cover Class')
    ax2.set_ylabel('Sample Count')
    ax2.set_xticks(x_class)
    ax2.set_xticklabels([f'Class {cls}' for cls in classes])
    ax2.legend()
    
    # 子图3：删除统计饼图
    ax3 = axes[1, 0]
    
    removal_data = []
    removal_labels = []
    colors = []
    
    for level in ['Normal', 'Moderate', 'High', 'Extreme']:
        if level in removal_stats and removal_stats[level]['total'] > 0:
            removed = removal_stats[level]['removed']
            retained = removal_stats[level]['total'] - removed
            
            if retained > 0:
                removal_data.append(retained)
                removal_labels.append(f'{level} (保留)')
                colors.append('green' if level in ['Normal', 'Moderate'] else 'orange')
            
            if removed > 0:
                removal_data.append(removed)
                removal_labels.append(f'{level} (删除)')
                colors.append('red')
    
    if removal_data:
        wedges, texts, autotexts = ax3.pie(removal_data, labels=removal_labels, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax3.set_title('Sample Retention/Removal Distribution', fontsize=12, fontweight='bold')
    
    # 子图4：马氏距离分布对比
    ax4 = axes[1, 1]
    
    ax4.hist(original_gdf['mahal_dist'], bins=50, alpha=0.5, label='筛选前', density=True)
    ax4.hist(filtered_gdf['mahal_dist'], bins=50, alpha=0.5, label='筛选后', density=True)
    ax4.set_title('Mahalanobis Distance Distribution Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mahalanobis Distance')
    ax4.set_ylabel('Density')
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, f"样本点筛选对比分析_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"对比分析图表已保存到: {plot_file}")
    plt.show()

def print_filtering_summary(original_gdf, filtered_gdf, removal_stats):
    """
    打印筛选汇总统计
    参数：
        original_gdf: 原始数据
        filtered_gdf: 筛选后数据
        removal_stats: 删除统计信息
    """
    print("\n" + "="*60)
    print("样本点筛选汇总报告")
    print("="*60)
    
    print(f"\n1. 筛选前后对比:")
    print(f"   - 原始样本数: {len(original_gdf)}")
    print(f"   - 筛选后样本数: {len(filtered_gdf)}")
    print(f"   - 删除样本数: {len(original_gdf) - len(filtered_gdf)}")
    print(f"   - 保留比例: {len(filtered_gdf)/len(original_gdf)*100:.1f}%")
    
    print(f"\n2. 各等级删除统计:")
    for level, stats in removal_stats.items():
        if stats['total'] > 0:
            retention_rate = (stats['total'] - stats['removed']) / stats['total'] * 100
            print(f"   - {level}: 删除 {stats['removed']}/{stats['total']} "
                  f"({stats['removed']/stats['total']*100:.1f}%), "
                  f"保留率 {retention_rate:.1f}%")
    
    print(f"\n3. 筛选后各类别分布:")
    class_counts = filtered_gdf['landcover'].value_counts().sort_index()
    for cls, count in class_counts.items():
        original_count = len(original_gdf[original_gdf['landcover'] == cls])
        retention_rate = count / original_count * 100
        print(f"   - 类别 {cls}: {count} 个样本 (保留率 {retention_rate:.1f}%)")
    
    print(f"\n4. 筛选后偏离度等级分布:")
    level_counts = filtered_gdf['dev_level'].value_counts()
    total_filtered = len(filtered_gdf)
    for level, count in level_counts.items():
        percentage = (count / total_filtered) * 100
        print(f"   - {level}: {count} 个样本 ({percentage:.1f}%)")
    
    print(f"\n5. 马氏距离统计对比:")
    print(f"   筛选前:")
    print(f"     - 平均值: {original_gdf['mahal_dist'].mean():.3f}")
    print(f"     - 标准差: {original_gdf['mahal_dist'].std():.3f}")
    print(f"     - 最大值: {original_gdf['mahal_dist'].max():.3f}")
    
    print(f"   筛选后:")
    print(f"     - 平均值: {filtered_gdf['mahal_dist'].mean():.3f}")
    print(f"     - 标准差: {filtered_gdf['mahal_dist'].std():.3f}")
    print(f"     - 最大值: {filtered_gdf['mahal_dist'].max():.3f}")
    
    print("="*60)

def main():
    """
    主函数：执行样本点筛选
    """
    print("开始样本点筛选...")
    print(f"输入文件: {INPUT_SHAPEFILE}")
    print(f"High等级删除比例: {HIGH_REMOVAL_RATIO*100:.0f}%")
    print(f"随机种子: {RANDOM_SEED}")
    
    try:
        # 读取偏离度分析结果
        print("\n正在读取偏离度分析结果...")
        original_gdf = read_deviation_data(INPUT_SHAPEFILE)
        
        # 分析偏离度分布
        level_counts, class_deviation = analyze_deviation_distribution(original_gdf)
        
        # 执行样本点筛选
        filtered_gdf, removal_stats = filter_samples(
            original_gdf, 
            high_removal_ratio=HIGH_REMOVAL_RATIO, 
            random_seed=RANDOM_SEED
        )
        
        # 创建输出目录

        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存筛选后的样本点
        output_shapefile = os.path.join(output_dir, f"筛选后样本点_{timestamp}.shp")
        save_filtered_samples(filtered_gdf, output_shapefile)
        
        # 保存筛选统计CSV
        print("正在保存筛选统计结果...")
        
        # 创建筛选前后对比统计
        comparison_stats = []
        for level in ['Normal', 'Moderate', 'High', 'Extreme']:
            original_count = len(original_gdf[original_gdf['dev_level'] == level])
            filtered_count = len(filtered_gdf[filtered_gdf['dev_level'] == level])
            removed_count = original_count - filtered_count
            retention_rate = filtered_count / original_count * 100 if original_count > 0 else 0
            
            comparison_stats.append({
                'deviation_level': level,
                'original_count': original_count,
                'filtered_count': filtered_count,
                'removed_count': removed_count,
                'retention_rate': retention_rate
            })
        
        comparison_df = pd.DataFrame(comparison_stats)
        csv_file = os.path.join(output_dir, f"筛选统计对比_{timestamp}.csv")
        comparison_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"筛选统计已保存到: {csv_file}")
        
        # 生成对比可视化图表
        create_comparison_visualizations(original_gdf, filtered_gdf, removal_stats, output_dir, timestamp)
        
        # 打印汇总统计
        print_filtering_summary(original_gdf, filtered_gdf, removal_stats)
        
        print(f"\n筛选完成！所有结果已保存到: {output_dir}")
        print(f"输出文件包括:")
        print(f"  - 筛选后样本点: {output_shapefile}")
        print(f"  - 筛选统计对比: {csv_file}")
        print(f"  - 对比分析图表: 样本点筛选对比分析_{timestamp}.png")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()