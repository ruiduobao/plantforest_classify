#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桑基图制图脚本
功能：基于土地利用转换矩阵数据创建桑基图，展示人工林、自然林、其他类别的连续演变
作者：锐多宝 (ruiduobao)
创建时间：2025年1月
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

def load_transition_data(excel_path, sheet_name='转换矩阵汇总'):
    """
    从Excel文件加载转换矩阵数据
    
    参数:
    - excel_path: Excel文件路径
    - sheet_name: 工作表名称
    
    返回:
    - DataFrame: 转换数据
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"成功加载数据，共 {len(df)} 条记录")
        print(f"数据列: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def create_continuous_sankey_data(df, min_area_threshold=50):
    """
    创建连续桑基图数据结构
    
    参数:
    - df: 转换数据DataFrame
    - min_area_threshold: 最小面积阈值（km²），小于此值的转换将被过滤
    
    返回:
    - dict: 包含节点和链接信息的字典
    """
    # 过滤小面积转换以简化图表
    df_filtered = df[df['面积(km²)'] >= min_area_threshold].copy()
    print(f"过滤后保留 {len(df_filtered)} 条记录（面积 >= {min_area_threshold} km²）")
    
    # 获取所有时期和类别
    periods = sorted(df_filtered['时期'].unique())
    categories = ['人工林', '自然林', '其他']
    
    print(f"时期: {periods}")
    print(f"类别: {categories}")
    
    # 计算每个年份每个类别的总面积（用于确定节点大小）
    node_areas = {}
    
    # 计算起始年份的面积（基于流出面积）
    start_year = periods[0].split('-')[0]
    for category in categories:
        # 计算该类别在起始年份的总面积（所有流出的面积之和）
        outflow_area = df_filtered[
            (df_filtered['时期'] == periods[0]) & 
            (df_filtered['源类别'] == category)
        ]['面积(km²)'].sum()
        node_areas[f"{category}_{start_year}"] = outflow_area
    
    # 计算后续年份的面积（基于流入面积）
    for period in periods:
        end_year = period.split('-')[1]
        for category in categories:
            # 计算该类别在结束年份的总面积（所有流入的面积之和）
            inflow_area = df_filtered[
                (df_filtered['时期'] == period) & 
                (df_filtered['目标类别'] == category)
            ]['面积(km²)'].sum()
            node_areas[f"{category}_{end_year}"] = inflow_area
    
    # 创建节点列表
    nodes = []
    node_colors = []
    
    # 定义类别颜色
    category_colors = {
        '人工林': 'rgba(34, 139, 34, 0.8)',    # 森林绿
        '自然林': 'rgba(0, 100, 0, 0.8)',      # 深绿
        '其他': 'rgba(139, 69, 19, 0.8)'       # 棕色
    }
    
    # 为每个时期的每个类别创建节点
    node_dict = {}
    node_index = 0
    
    # 创建起始年份节点
    for category in categories:
        node_name = f"{category}_{start_year}"
        nodes.append(node_name)
        node_colors.append(category_colors[category])
        node_dict[node_name] = node_index
        node_index += 1
    
    # 创建后续年份节点
    for period in periods:
        end_year = period.split('-')[1]
        for category in categories:
            node_name = f"{category}_{end_year}"
            if node_name not in node_dict:
                nodes.append(node_name)
                node_colors.append(category_colors[category])
                node_dict[node_name] = node_index
                node_index += 1
    
    # 创建链接
    source_indices = []
    target_indices = []
    values = []
    link_colors = []
    link_labels = []
    
    for _, row in df_filtered.iterrows():
        period = row['时期']
        source_cat = row['源类别']
        target_cat = row['目标类别']
        area = row['面积(km²)']
        
        start_year, end_year = period.split('-')
        source_node = f"{source_cat}_{start_year}"
        target_node = f"{target_cat}_{end_year}"
        
        if source_node in node_dict and target_node in node_dict:
            source_indices.append(node_dict[source_node])
            target_indices.append(node_dict[target_node])
            values.append(area)
            
            # 设置链接颜色和标签
            if source_cat == target_cat:
                # 保持不变的流向用蓝色
                link_colors.append('rgba(0, 100, 200, 0.4)')
                link_labels.append(f"{source_cat}保持: {area:.1f}km²")
            else:
                # 转换的流向用红色
                link_colors.append('rgba(255, 100, 100, 0.4)')
                link_labels.append(f"{source_cat}→{target_cat}: {area:.1f}km²")
    
    return {
        'nodes': nodes,
        'node_colors': node_colors,
        'node_areas': node_areas,  # 添加节点面积信息
        'source_indices': source_indices,
        'target_indices': target_indices,
        'values': values,
        'link_colors': link_colors,
        'link_labels': link_labels,
        'periods': periods
    }

def create_sankey_diagram(sankey_data, output_dir, title="土地利用连续演变桑基图"):
    """
    创建桑基图
    
    参数:
    - sankey_data: 桑基图数据字典
    - output_dir: 输出目录
    - title: 图表标题
    
    返回:
    - str: 输出文件路径
    """
    # 计算节点位置
    periods = sankey_data['periods']
    years = []
    for period in periods:
        start_year, end_year = period.split('-')
        if start_year not in years:
            years.append(start_year)
        if end_year not in years:
            years.append(end_year)
    
    unique_years = sorted(set(years))
    print(f"年份序列: {unique_years}")
    
    # 为每个年份分配x坐标
    year_x_positions = {}
    if len(unique_years) > 1:
        for i, year in enumerate(unique_years):
            year_x_positions[year] = i / (len(unique_years) - 1)
    else:
        year_x_positions[unique_years[0]] = 0.5
    
    # 计算每个年份各类别的面积，用于确定y坐标
    year_category_areas = {}
    for node_name, area in sankey_data['node_areas'].items():
        category, year = node_name.rsplit('_', 1)
        if year not in year_category_areas:
            year_category_areas[year] = {}
        year_category_areas[year][category] = area
    
    # 为每个年份的类别分配y坐标（基于面积比例）
    node_x_positions = []
    node_y_positions = []
    node_customdata = []  # 存储节点的面积信息
    
    for node_name in sankey_data['nodes']:
        category, year = node_name.rsplit('_', 1)
        
        # x坐标基于年份
        x_pos = year_x_positions[year]
        node_x_positions.append(x_pos)
        
        # y坐标基于该年份内各类别的面积比例
        year_areas = year_category_areas[year]
        total_area = sum(year_areas.values())
        
        # 计算累积面积比例来确定y坐标
        categories = ['人工林', '自然林', '其他']
        cumulative_area = 0
        for cat in categories:
            if cat == category:
                # 该类别的中心位置
                category_area = year_areas.get(cat, 0)
                y_pos = (cumulative_area + category_area / 2) / total_area
                node_y_positions.append(y_pos)
                node_customdata.append(category_area)  # 存储面积信息
                break
            cumulative_area += year_areas.get(cat, 0)
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,  # 减少节点间距
            thickness=20,  # 节点厚度
            line=dict(color="black", width=1),
            label=[f"{node.replace('_', ' ')}<br>{sankey_data['node_areas'][node]:,.0f} km²" 
                   for node in sankey_data['nodes']],  # 在标签中显示面积
            color=sankey_data['node_colors'],
            x=node_x_positions,
            y=node_y_positions,
            customdata=node_customdata,
            hovertemplate='<b>%{label}</b><br>面积: %{customdata:,.0f} km²<extra></extra>'
        ),
        link=dict(
            source=sankey_data['source_indices'],
            target=sankey_data['target_indices'],
            value=sankey_data['values'],  # 这个值决定了流的宽度，与面积成正比
            color=sankey_data['link_colors'],
            hovertemplate='%{label}<br>转换面积: %{value:,.1f} km²<extra></extra>',
            label=sankey_data['link_labels']
        )
    )])
    
    # 设置布局
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Microsoft YaHei'}
        },
        font=dict(size=14, family="Microsoft YaHei"),
        width=1400,
        height=800,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # 添加年份标注
    for year, x_pos in year_x_positions.items():
        fig.add_annotation(
            x=x_pos,
            y=1.05,
            text=f"<b>{year}年</b>",
            showarrow=False,
            font=dict(size=16, family="Microsoft YaHei"),
            xref="paper",
            yref="paper"
        )
    
    # 添加面积统计信息
    stats_text = "各年份类别面积统计:<br>"
    for year in unique_years:
        if year in year_category_areas:
            stats_text += f"<b>{year}年:</b><br>"
            for category in ['人工林', '自然林', '其他']:
                area = year_category_areas[year].get(category, 0)
                stats_text += f"  {category}: {area:,.0f} km²<br>"
            stats_text += "<br>"
    
    fig.add_annotation(
        x=0.02,
        y=0.98,
        text=stats_text,
        showarrow=False,
        font=dict(size=10, family="Microsoft YaHei"),
        xref="paper",
        yref="paper",
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # 保存文件
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存HTML版本
    html_path = os.path.join(output_dir, f"continuous_sankey_diagram_{timestamp}.html")
    fig.write_html(html_path)
    print(f"桑基图HTML版本已保存: {html_path}")
    
    # 尝试保存PNG版本
    try:
        png_path = os.path.join(output_dir, f"continuous_sankey_diagram_{timestamp}.png")
        fig.write_image(png_path, width=1400, height=800, scale=2)
        print(f"桑基图PNG版本已保存: {png_path}")
        return png_path
    except Exception as e:
        print(f"保存PNG版本失败: {e}")
        print("请安装 kaleido 库以支持PNG导出: pip install kaleido")
        return html_path

def create_period_sankey(df, period, output_dir):
    """
    为单个时期创建桑基图
    
    参数:
    - df: 转换数据DataFrame
    - period: 时期（如 '2020-2021'）
    - output_dir: 输出目录
    
    返回:
    - str: 输出文件路径
    """
    period_data = df[df['时期'] == period].copy()
    
    if len(period_data) == 0:
        print(f"未找到时期 {period} 的数据")
        return None
    
    # 创建节点
    categories = ['人工林', '自然林', '其他']
    start_year, end_year = period.split('-')
    
    nodes = []
    node_colors = []
    category_colors = {
        '人工林': 'rgba(34, 139, 34, 0.8)',
        '自然林': 'rgba(0, 100, 0, 0.8)',
        '其他': 'rgba(139, 69, 19, 0.8)'
    }
    
    # 创建源节点和目标节点
    for category in categories:
        nodes.append(f"{category}_{start_year}")
        nodes.append(f"{category}_{end_year}")
        node_colors.extend([category_colors[category], category_colors[category]])
    
    # 创建节点索引映射
    node_dict = {node: i for i, node in enumerate(nodes)}
    
    # 创建链接
    source_indices = []
    target_indices = []
    values = []
    link_colors = []
    
    for _, row in period_data.iterrows():
        source_cat = row['源类别']
        target_cat = row['目标类别']
        area = row['面积(km²)']
        
        source_node = f"{source_cat}_{start_year}"
        target_node = f"{target_cat}_{end_year}"
        
        source_indices.append(node_dict[source_node])
        target_indices.append(node_dict[target_node])
        values.append(area)
        
        # 设置链接颜色
        if source_cat == target_cat:
            link_colors.append('rgba(0, 100, 200, 0.6)')  # 蓝色表示保持
        else:
            link_colors.append('rgba(255, 100, 100, 0.6)')  # 红色表示转换
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title=f"{period}年土地利用转换桑基图",
        font_size=12,
        font_family="Microsoft YaHei",
        width=1000,
        height=600
    )
    
    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"sankey_{period}_{timestamp}.html")
    fig.write_html(html_path)
    
    try:
        png_path = os.path.join(output_dir, f"sankey_{period}_{timestamp}.png")
        fig.write_image(png_path, width=1000, height=600, scale=2)
        print(f"{period}年桑基图已保存: {png_path}")
        return png_path
    except Exception as e:
        print(f"{period}年桑基图HTML版本已保存: {html_path}")
        return html_path

def create_summary_statistics(df, output_dir):
    """
    创建转换统计摘要
    
    参数:
    - df: 转换数据DataFrame
    - output_dir: 输出目录
    """
    print("\n=== 土地利用转换统计摘要 ===")
    
    # 按时期统计
    for period in sorted(df['时期'].unique()):
        period_data = df[df['时期'] == period]
        print(f"\n{period}年转换统计:")
        
        # 计算各类别的净变化
        categories = ['人工林', '自然林', '其他']
        for category in categories:
            # 流入该类别的面积
            inflow = period_data[period_data['目标类别'] == category]['面积(km²)'].sum()
            # 流出该类别的面积
            outflow = period_data[period_data['源类别'] == category]['面积(km²)'].sum()
            # 保持不变的面积
            stable = period_data[(period_data['源类别'] == category) & 
                               (period_data['目标类别'] == category)]['面积(km²)'].sum()
            # 净变化
            net_change = inflow - outflow
            
            print(f"  {category}:")
            print(f"    总流入: {inflow:,.1f} km²")
            print(f"    总流出: {outflow:,.1f} km²")
            print(f"    保持不变: {stable:,.1f} km²")
            print(f"    净变化: {net_change:+,.1f} km²")
    
    # 创建统计表格
    summary_data = []
    for period in sorted(df['时期'].unique()):
        period_data = df[df['时期'] == period]
        
        for category in ['人工林', '自然林', '其他']:
            inflow = period_data[period_data['目标类别'] == category]['面积(km²)'].sum()
            outflow = period_data[period_data['源类别'] == category]['面积(km²)'].sum()
            stable = period_data[(period_data['源类别'] == category) & 
                               (period_data['目标类别'] == category)]['面积(km²)'].sum()
            net_change = inflow - outflow
            
            summary_data.append({
                '时期': period,
                '类别': category,
                '总流入(km²)': inflow,
                '总流出(km²)': outflow,
                '保持不变(km²)': stable,
                '净变化(km²)': net_change
            })
    
    # 保存统计表格
    summary_df = pd.DataFrame(summary_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"transition_summary_{timestamp}.csv")
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n统计摘要已保存: {csv_path}")

def main():
    """
    主函数：创建桑基图
    """
    # 配置参数
    EXCEL_PATH = r"K:\地理所\论文\东南亚10m人工林提取\数据\GEE分类\转换分析\ZONE5\land_use_transition_matrices_20251021_155609.xlsx"
    OUTPUT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\GEE分类\桑基图\ZONE5"
    MIN_AREA_THRESHOLD = 50  # 最小面积阈值（km²）
    
    print("开始创建土地利用转换桑基图")
    print(f"数据文件: {EXCEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"最小面积阈值: {MIN_AREA_THRESHOLD} km²")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 加载数据
        print("\n1. 加载转换矩阵数据...")
        df = load_transition_data(EXCEL_PATH)
        
        if df is None:
            print("数据加载失败，程序退出")
            return
        
        # 显示数据概览
        print(f"\n数据概览:")
        print(f"  总记录数: {len(df)}")
        print(f"  时期数量: {len(df['时期'].unique())}")
        print(f"  时期列表: {sorted(df['时期'].unique())}")
        print(f"  类别列表: {sorted(df['源类别'].unique())}")
        
        # 创建连续桑基图数据
        print(f"\n2. 创建连续桑基图数据...")
        sankey_data = create_continuous_sankey_data(df, MIN_AREA_THRESHOLD)
        
        # 创建连续桑基图
        print(f"\n3. 创建连续演变桑基图...")
        continuous_path = create_sankey_diagram(
            sankey_data, 
            OUTPUT_DIR, 
            "东南亚ZONE5土地利用连续演变桑基图 (2020-2022)"
        )
        
        # 为每个时期创建单独的桑基图
        print(f"\n4. 创建各时期单独桑基图...")
        periods = sorted(df['时期'].unique())
        for period in periods:
            period_path = create_period_sankey(df, period, OUTPUT_DIR)
        
        # 创建统计摘要
        print(f"\n5. 创建统计摘要...")
        create_summary_statistics(df, OUTPUT_DIR)
        
        print(f"\n=== 桑基图制作完成 ===")
        print(f"所有文件已保存至: {OUTPUT_DIR}")
        print(f"主要输出文件:")
        print(f"  - 连续演变桑基图: {os.path.basename(continuous_path)}")
        print(f"  - 各时期单独桑基图: sankey_YYYY-YYYY_*.html/png")
        print(f"  - 统计摘要: transition_summary_*.csv")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()