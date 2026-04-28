#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于结果进行统计分析
目的：对等面积投影统计结果进行面积变化分析，生成折线图
作者：锐多宝 (ruiduobao)
创建时间：2025年11月3日
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置参数
EXCEL_PATH = r'D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\统计面积\等面积投影面积统计结果_20251103_192624.xlsx'
OUTPUT_DIR = r'D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\统计面积\面积分析'

# 颜色配置
COLORS = {
    'plantation': '#2E8B57',  # 海绿色 - 人工林
    'natural': '#228B22',     # 森林绿 - 自然林  
    'other': '#CD853F',       # 秘鲁色 - 其他地类
    'total': '#4169E1'        # 皇家蓝 - 总面积
}

def setup_output_directory():
    """
    创建输出目录
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    else:
        print(f"输出目录已存在: {OUTPUT_DIR}")

def load_and_prepare_data():
    """
    加载并预处理数据
    返回：处理后的DataFrame
    """
    print("正在加载Excel数据...")
    
    # 检查文件是否存在
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel文件不存在: {EXCEL_PATH}")
    
    # 读取数据
    df = pd.read_excel(EXCEL_PATH)
    print(f"数据加载完成，共 {len(df)} 行数据")
    
    # 数据基本信息
    print(f"Zone数量: {len(df['Zone'].unique())}")
    print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Zone列表: {sorted(df['Zone'].unique())}")
    
    # 检查数据完整性
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("发现缺失数据:")
        print(missing_data[missing_data > 0])
    
    # 检查负数面积
    negative_areas = df[df['Total_Area_Ha'] < 0]
    if len(negative_areas) > 0:
        print(f"警告：发现 {len(negative_areas)} 行负数面积数据")
        print(negative_areas[['Zone', 'Year', 'Total_Area_Ha']])
    
    return df

def plot_individual_zone_trends(df):
    """
    绘制每个zone在2017-2024年的变化折线图
    参数：df - 数据DataFrame
    """
    print("正在生成每个zone的变化折线图...")
    
    zones = sorted(df['Zone'].unique())
    years = sorted(df['Year'].unique())
    
    # 创建子图 - 2行5列布局
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    fig.suptitle('各Zone森林面积变化趋势 (2017-2024)', fontsize=16, fontweight='bold')
    
    # 扁平化axes数组以便遍历
    axes_flat = axes.flatten()
    
    for i, zone in enumerate(zones):
        ax = axes_flat[i]
        
        # 筛选当前zone的数据
        zone_data = df[df['Zone'] == zone].sort_values('Year')
        
        # 绘制折线图
        ax.plot(zone_data['Year'], zone_data['Plantation_Forest_Ha'] / 10000, 
                color=COLORS['plantation'], marker='o', linewidth=2, 
                label='人工林', markersize=4)
        
        ax.plot(zone_data['Year'], zone_data['Natural_Forest_Ha'] / 10000, 
                color=COLORS['natural'], marker='s', linewidth=2, 
                label='自然林', markersize=4)
        
        ax.plot(zone_data['Year'], zone_data['Other_Ha'] / 10000, 
                color=COLORS['other'], marker='^', linewidth=2, 
                label='其他地类', markersize=4)
        
        # 设置标题和标签
        ax.set_title(f'{zone.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('年份', fontsize=10)
        ax.set_ylabel('面积 (万公顷)', fontsize=10)
        
        # 设置x轴刻度
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        
        # 网格
        ax.grid(True, alpha=0.3)
        
        # 只在第一个子图显示图例
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, f'各Zone森林面积变化趋势_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"各Zone变化趋势图已保存: {output_path}")
    
    plt.show()

def plot_total_area_trends(df):
    """
    绘制所有区域汇总的2017-2024年变化折线图
    参数：df - 数据DataFrame
    """
    print("正在生成所有区域汇总的变化折线图...")
    
    # 按年份汇总所有zone的面积
    yearly_summary = df.groupby('Year').agg({
        'Plantation_Forest_Ha': 'sum',
        'Natural_Forest_Ha': 'sum', 
        'Other_Ha': 'sum',
        'Total_Area_Ha': 'sum'
    }).reset_index()
    
    # 转换为万公顷
    yearly_summary['Plantation_Forest_万Ha'] = yearly_summary['Plantation_Forest_Ha'] / 10000
    yearly_summary['Natural_Forest_万Ha'] = yearly_summary['Natural_Forest_Ha'] / 10000
    yearly_summary['Other_万Ha'] = yearly_summary['Other_Ha'] / 10000
    yearly_summary['Total_万Ha'] = yearly_summary['Total_Area_Ha'] / 10000
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：分类面积变化
    ax1.plot(yearly_summary['Year'], yearly_summary['Plantation_Forest_万Ha'], 
             color=COLORS['plantation'], marker='o', linewidth=3, 
             label='人工林', markersize=6)
    
    ax1.plot(yearly_summary['Year'], yearly_summary['Natural_Forest_万Ha'], 
             color=COLORS['natural'], marker='s', linewidth=3, 
             label='自然林', markersize=6)
    
    ax1.plot(yearly_summary['Year'], yearly_summary['Other_万Ha'], 
             color=COLORS['other'], marker='^', linewidth=3, 
             label='其他地类', markersize=6)
    
    ax1.set_title('东南亚各地类面积变化趋势 (2017-2024)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('年份', fontsize=12)
    ax1.set_ylabel('面积 (万公顷)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(yearly_summary['Year'])
    
    # 右图：总面积变化
    ax2.plot(yearly_summary['Year'], yearly_summary['Total_万Ha'], 
             color=COLORS['total'], marker='o', linewidth=3, 
             label='总面积', markersize=6)
    
    ax2.set_title('东南亚总面积变化趋势 (2017-2024)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('年份', fontsize=12)
    ax2.set_ylabel('面积 (万公顷)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(yearly_summary['Year'])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, f'东南亚森林面积汇总变化趋势_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"汇总变化趋势图已保存: {output_path}")
    
    plt.show()
    
    # 打印汇总统计信息
    print("\n=== 年度汇总统计 ===")
    print(yearly_summary[['Year', 'Plantation_Forest_万Ha', 'Natural_Forest_万Ha', 'Other_万Ha', 'Total_万Ha']].round(2))
    
    return yearly_summary

def calculate_change_statistics(df):
    """
    计算变化统计信息
    参数：df - 数据DataFrame
    """
    print("正在计算变化统计信息...")
    
    # 按zone计算2017-2024年的变化
    zones = sorted(df['Zone'].unique())
    change_stats = []
    
    for zone in zones:
        zone_data = df[df['Zone'] == zone].sort_values('Year')
        
        if len(zone_data) >= 2:
            # 获取首末年数据
            first_year = zone_data.iloc[0]
            last_year = zone_data.iloc[-1]
            
            # 计算变化量和变化率
            plantation_change = last_year['Plantation_Forest_Ha'] - first_year['Plantation_Forest_Ha']
            natural_change = last_year['Natural_Forest_Ha'] - first_year['Natural_Forest_Ha']
            other_change = last_year['Other_Ha'] - first_year['Other_Ha']
            
            plantation_rate = (plantation_change / first_year['Plantation_Forest_Ha']) * 100 if first_year['Plantation_Forest_Ha'] > 0 else 0
            natural_rate = (natural_change / first_year['Natural_Forest_Ha']) * 100 if first_year['Natural_Forest_Ha'] > 0 else 0
            other_rate = (other_change / first_year['Other_Ha']) * 100 if first_year['Other_Ha'] > 0 else 0
            
            change_stats.append({
                'Zone': zone,
                '人工林变化量(万Ha)': plantation_change / 10000,
                '人工林变化率(%)': plantation_rate,
                '自然林变化量(万Ha)': natural_change / 10000,
                '自然林变化率(%)': natural_rate,
                '其他地类变化量(万Ha)': other_change / 10000,
                '其他地类变化率(%)': other_rate
            })
    
    # 转换为DataFrame并保存
    change_df = pd.DataFrame(change_stats)
    
    # 保存变化统计表
    change_output_path = os.path.join(OUTPUT_DIR, f'面积变化统计表_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
    change_df.to_excel(change_output_path, index=False)
    print(f"变化统计表已保存: {change_output_path}")
    
    # 打印变化统计
    print("\n=== 2017-2024年各Zone面积变化统计 ===")
    print(change_df.round(2))
    
    return change_df

def main():
    """
    主函数：执行完整的面积分析流程
    """
    print("=" * 60)
    print("东南亚森林面积变化分析")
    print("=" * 60)
    
    try:
        # 1. 创建输出目录
        setup_output_directory()
        
        # 2. 加载数据
        df = load_and_prepare_data()
        
        # 3. 生成每个zone的变化折线图
        plot_individual_zone_trends(df)
        
        # 4. 生成所有区域汇总的变化折线图
        yearly_summary = plot_total_area_trends(df)
        
        # 5. 计算变化统计信息
        change_stats = calculate_change_statistics(df)
        
        print("\n" + "=" * 60)
        print("面积分析完成！所有图表和统计结果已保存到输出目录。")
        print("=" * 60)
        
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()