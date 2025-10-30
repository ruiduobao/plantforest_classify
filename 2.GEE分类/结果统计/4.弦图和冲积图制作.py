#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
土地利用转换可视化脚本 - 弦图和冲积图
功能：创建弦图展示类别间双向转换关系，冲积图展示时间序列转换
作者：锐多宝 (ruiduobao)
创建时间：2025年1月
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

def create_chord_diagram_data(df, period=None):
    """
    创建弦图数据
    
    参数:
    - df: 转换数据DataFrame
    - period: 指定时期，如果为None则汇总所有时期
    
    返回:
    - dict: 弦图数据
    """
    if period:
        df_filtered = df[df['时期'] == period].copy()
        print(f"创建 {period} 时期的弦图数据")
    else:
        df_filtered = df.copy()
        print("创建所有时期汇总的弦图数据")
    
    # 定义类别和颜色
    categories = ['人工林', '自然林', '其他']
    colors = {
        '人工林': '#228B22',  # 森林绿
        '自然林': '#006400',  # 深绿
        '其他': '#8B4513'     # 棕色
    }
    
    # 创建转换矩阵
    matrix = np.zeros((len(categories), len(categories)))
    
    for _, row in df_filtered.iterrows():
        source_idx = categories.index(row['源类别'])
        target_idx = categories.index(row['目标类别'])
        matrix[source_idx][target_idx] += row['面积(km²)']
    
    return {
        'matrix': matrix,
        'categories': categories,
        'colors': colors,
        'total_area': matrix.sum()
    }

def create_chord_diagram(chord_data, output_dir, title="土地利用转换弦图", filename_prefix="chord_diagram"):
    """
    创建弦图（改进版，支持双向箭头）
    
    参数:
    - chord_data: 弦图数据字典
    - output_dir: 输出目录
    - title: 图表标题
    - filename_prefix: 文件名前缀，用于区分不同类型的弦图
    
    返回:
    - str: 输出文件路径
    """
    matrix = chord_data['matrix']
    categories = chord_data['categories']
    colors = chord_data['colors']
    
    # 计算角度位置
    n_categories = len(categories)
    angles = np.linspace(0, 2*np.pi, n_categories, endpoint=False)
    
    # 创建弦图
    fig = go.Figure()
    
    # 添加类别扇形
    for i, (category, angle) in enumerate(zip(categories, angles)):
        # 计算该类别的总流出面积
        total_outflow = matrix[i].sum()
        
        # 扇形的角度范围（基于流出面积比例）
        angle_span = (total_outflow / matrix.sum()) * 2 * np.pi * 0.8  # 0.8是为了留出间隙
        
        # 创建扇形
        theta = np.linspace(angle - angle_span/2, angle + angle_span/2, 50)
        r_inner = 0.8
        r_outer = 1.0
        
        # 扇形的x, y坐标
        x_inner = r_inner * np.cos(theta)
        y_inner = r_inner * np.sin(theta)
        x_outer = r_outer * np.cos(theta)
        y_outer = r_outer * np.sin(theta)
        
        # 添加扇形
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_inner, x_outer[::-1], [x_inner[0]]]),
            y=np.concatenate([y_inner, y_outer[::-1], [y_inner[0]]]),
            fill='toself',
            fillcolor=colors[category],
            line=dict(color='white', width=2),
            name=f"{category}<br>{total_outflow:,.0f} km²",
            hovertemplate=f"<b>{category}</b><br>总流出: {total_outflow:,.0f} km²<extra></extra>",
            showlegend=True
        ))
        
        # 添加类别标签
        label_angle = angle
        label_r = 1.15
        fig.add_annotation(
            x=label_r * np.cos(label_angle),
            y=label_r * np.sin(label_angle),
            text=f"<b>{category}</b><br>{total_outflow:,.0f} km²",
            showarrow=False,
            font=dict(size=12, family="Microsoft YaHei"),
            xanchor='center',
            yanchor='middle'
        )
    
    # 添加双向转换弦（改进版）
    processed_pairs = set()  # 记录已处理的类别对，避免重复
    
    for i in range(n_categories):
        for j in range(n_categories):
            if i != j and (i, j) not in processed_pairs and (j, i) not in processed_pairs:
                # 获取双向转换面积
                area_i_to_j = matrix[i][j]  # i -> j
                area_j_to_i = matrix[j][i]  # j -> i
                
                if area_i_to_j > 0 or area_j_to_i > 0:
                    # 计算弦的起始和结束角度
                    source_angle = angles[i]
                    target_angle = angles[j]
                    
                    # 计算弦的中点角度
                    mid_angle = (source_angle + target_angle) / 2
                    if abs(source_angle - target_angle) > np.pi:
                        mid_angle += np.pi
                    
                    # 弦的控制点（弯曲程度）
                    control_r = 0.2
                    
                    # 绘制双向弦
                    if area_i_to_j > 0:
                        # i -> j 方向的弦
                        t = np.linspace(0, 1, 100)
                        r = 0.9
                        x1, y1 = r * np.cos(source_angle), r * np.sin(source_angle)
                        x2, y2 = r * np.cos(target_angle), r * np.sin(target_angle)
                        cx, cy = control_r * np.cos(mid_angle), control_r * np.sin(mid_angle)
                        
                        # 贝塞尔曲线
                        x_curve = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
                        y_curve = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
                        
                        # 弦的宽度基于转换面积（更加夸张的计算）
                        # 使用对数缩放和更大的基础宽度，确保差异明显
                        max_area = matrix.max()
                        min_width = 3  # 最小宽度
                        max_width = 25  # 最大宽度
                        if max_area > 0:
                            # 使用平方根缩放使差异更明显
                            width_ratio = np.sqrt(area_i_to_j / max_area)
                            chord_width = min_width + (max_width - min_width) * width_ratio
                        else:
                            chord_width = min_width
                        
                        # 添加弦
                        fig.add_trace(go.Scatter(
                            x=x_curve,
                            y=y_curve,
                            mode='lines',
                            line=dict(
                                color=colors[categories[i]],
                                width=chord_width
                            ),
                            opacity=0.8,  # 提高透明度使标签更清晰
                            name=f"{categories[i]}→{categories[j]}",
                            hovertemplate=f"<b>{categories[i]} → {categories[j]}</b><br>转换面积: {area_i_to_j:,.0f} km²<extra></extra>",
                            showlegend=False
                        ))
                        
                        # 在弦的中点添加面积标签
                        mid_idx = len(x_curve) // 2
                        label_x, label_y = x_curve[mid_idx], y_curve[mid_idx]
                        
                        # 计算标签位置偏移（垂直于弦的方向）
                        if mid_idx < len(x_curve) - 1:
                            dx = x_curve[mid_idx + 1] - x_curve[mid_idx]
                            dy = y_curve[mid_idx + 1] - y_curve[mid_idx]
                            # 垂直方向
                            perp_dx = -dy
                            perp_dy = dx
                            # 标准化
                            length = np.sqrt(perp_dx**2 + perp_dy**2)
                            if length > 0:
                                perp_dx /= length
                                perp_dy /= length
                            # 偏移距离
                            offset = 0.08
                            label_x += perp_dx * offset
                            label_y += perp_dy * offset
                        
                        # 添加面积标签
                        if area_i_to_j >= 1000:  # 只为较大的转换添加标签，避免拥挤
                            area_text = f"{area_i_to_j:,.0f}"
                            if area_i_to_j >= 10000:
                                area_text = f"{area_i_to_j/1000:.1f}k"
                            
                            fig.add_annotation(
                                x=label_x,
                                y=label_y,
                                text=f"<b>{area_text}</b>",
                                showarrow=False,
                                font=dict(
                                    size=10,
                                    family="Microsoft YaHei",
                                    color="white"
                                ),
                                bgcolor=colors[categories[i]],
                                bordercolor="white",
                                borderwidth=1,
                                borderpad=2,
                                xanchor='center',
                                yanchor='middle'
                            )
                        
                        # 添加箭头（在弦的中点）
                        arrow_x, arrow_y = x_curve[mid_idx], y_curve[mid_idx]
                        
                        # 计算箭头方向
                        if mid_idx < len(x_curve) - 1:
                            dx = x_curve[mid_idx + 1] - x_curve[mid_idx]
                            dy = y_curve[mid_idx + 1] - y_curve[mid_idx]
                            arrow_angle = np.arctan2(dy, dx)
                        else:
                            arrow_angle = target_angle
                        
                        # 添加箭头标记
                        fig.add_annotation(
                            x=arrow_x,
                            y=arrow_y,
                            ax=arrow_x - 0.05 * np.cos(arrow_angle),
                            ay=arrow_y - 0.05 * np.sin(arrow_angle),
                            arrowhead=2,
                            arrowsize=2,  # 增大箭头
                            arrowwidth=3,
                            arrowcolor=colors[categories[i]],
                            showarrow=True,
                            text=""
                        )
                    
                    if area_j_to_i > 0:
                        # j -> i 方向的弦（稍微偏移以避免重叠）
                        t = np.linspace(0, 1, 100)
                        r = 0.85  # 稍微内缩
                        x1, y1 = r * np.cos(target_angle), r * np.sin(target_angle)
                        x2, y2 = r * np.cos(source_angle), r * np.sin(source_angle)
                        cx, cy = (control_r + 0.1) * np.cos(mid_angle), (control_r + 0.1) * np.sin(mid_angle)
                        
                        # 贝塞尔曲线
                        x_curve = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
                        y_curve = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
                        
                        # 弦的宽度基于转换面积（更加夸张的计算）
                        max_area = matrix.max()
                        min_width = 3  # 最小宽度
                        max_width = 25  # 最大宽度
                        if max_area > 0:
                            # 使用平方根缩放使差异更明显
                            width_ratio = np.sqrt(area_j_to_i / max_area)
                            chord_width = min_width + (max_width - min_width) * width_ratio
                        else:
                            chord_width = min_width
                        
                        # 添加弦
                        fig.add_trace(go.Scatter(
                            x=x_curve,
                            y=y_curve,
                            mode='lines',
                            line=dict(
                                color=colors[categories[j]],
                                width=chord_width
                            ),
                            opacity=0.8,  # 提高透明度使标签更清晰
                            name=f"{categories[j]}→{categories[i]}",
                            hovertemplate=f"<b>{categories[j]} → {categories[i]}</b><br>转换面积: {area_j_to_i:,.0f} km²<extra></extra>",
                            showlegend=False
                        ))
                        
                        # 在弦的中点添加面积标签
                        mid_idx = len(x_curve) // 2
                        label_x, label_y = x_curve[mid_idx], y_curve[mid_idx]
                        
                        # 计算标签位置偏移（垂直于弦的方向）
                        if mid_idx < len(x_curve) - 1:
                            dx = x_curve[mid_idx + 1] - x_curve[mid_idx]
                            dy = y_curve[mid_idx + 1] - y_curve[mid_idx]
                            # 垂直方向
                            perp_dx = -dy
                            perp_dy = dx
                            # 标准化
                            length = np.sqrt(perp_dx**2 + perp_dy**2)
                            if length > 0:
                                perp_dx /= length
                                perp_dy /= length
                            # 偏移距离
                            offset = 0.08
                            label_x += perp_dx * offset
                            label_y += perp_dy * offset
                        
                        # 添加面积标签
                        if area_j_to_i >= 1000:  # 只为较大的转换添加标签，避免拥挤
                            area_text = f"{area_j_to_i:,.0f}"
                            if area_j_to_i >= 10000:
                                area_text = f"{area_j_to_i/1000:.1f}k"
                            
                            fig.add_annotation(
                                x=label_x,
                                y=label_y,
                                text=f"<b>{area_text}</b>",
                                showarrow=False,
                                font=dict(
                                    size=10,
                                    family="Microsoft YaHei",
                                    color="white"
                                ),
                                bgcolor=colors[categories[j]],
                                bordercolor="white",
                                borderwidth=1,
                                borderpad=2,
                                xanchor='center',
                                yanchor='middle'
                            )
                        
                        # 添加箭头（在弦的中点）
                        arrow_x, arrow_y = x_curve[mid_idx], y_curve[mid_idx]
                        
                        # 计算箭头方向
                        if mid_idx < len(x_curve) - 1:
                            dx = x_curve[mid_idx + 1] - x_curve[mid_idx]
                            dy = y_curve[mid_idx + 1] - y_curve[mid_idx]
                            arrow_angle = np.arctan2(dy, dx)
                        else:
                            arrow_angle = source_angle
                        
                        # 添加箭头标记
                        fig.add_annotation(
                            x=arrow_x,
                            y=arrow_y,
                            ax=arrow_x - 0.05 * np.cos(arrow_angle),
                            ay=arrow_y - 0.05 * np.sin(arrow_angle),
                            arrowhead=2,
                            arrowsize=2,  # 增大箭头
                            arrowwidth=3,
                            arrowcolor=colors[categories[j]],
                            showarrow=True,
                            text=""
                        )
                
                # 标记该类别对已处理
                processed_pairs.add((i, j))
    
    # 设置布局
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Microsoft YaHei'}
        },
        xaxis=dict(
            range=[-1.5, 1.5],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-1.5, 1.5],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        font=dict(size=12, family="Microsoft YaHei"),
        width=800,
        height=800,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        )
    )
    
    # 添加说明文本
    explanation = (
        "弦图说明:<br>"
        "• 扇形大小 = 该类别总流出面积<br>"
        "• 弦的宽度 = 转换面积大小<br>"
        "• 箭头显示转换方向<br>"
        "• 双向箭头表示相互转换<br>"
        "• 颜色表示源类别"
    )
    
    fig.add_annotation(
        x=-1.4,
        y=-1.2,
        text=explanation,
        showarrow=False,
        font=dict(size=10, family="Microsoft YaHei"),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    # 保存文件
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存HTML版本
    html_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.html")
    fig.write_html(html_path)
    print(f"弦图HTML版本已保存: {html_path}")
    
    # 尝试保存PNG版本
    try:
        png_path = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.png")
        fig.write_image(png_path, width=800, height=800, scale=2)
        print(f"弦图PNG版本已保存: {png_path}")
        return png_path
    except Exception as e:
        print(f"保存PNG版本失败: {e}")
        return html_path

def create_transition_heatmap(df, output_dir):
    """
    创建转换矩阵热力图
    
    参数:
    - df: 转换数据DataFrame
    - output_dir: 输出目录
    
    返回:
    - str: 输出文件路径
    """
    periods = sorted(df['时期'].unique())
    categories = ['人工林', '自然林', '其他']
    
    # 创建子图
    n_periods = len(periods)
    cols = min(3, n_periods)
    rows = (n_periods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_periods == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, period in enumerate(periods):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # 创建该时期转换矩阵
        period_data = df[df['时期'] == period]
        matrix = np.zeros((len(categories), len(categories)))
        
        for _, row_data in period_data.iterrows():
            source_idx = categories.index(row_data['源类别'])
            target_idx = categories.index(row_data['目标类别'])
            matrix[source_idx][target_idx] = row_data['面积(km²)']
        
        # 创建热力图
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置标签
        ax.set_xticks(range(len(categories)))
        ax.set_yticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45)
        ax.set_yticklabels(categories)
        ax.set_xlabel('目标类别')
        ax.set_ylabel('源类别')
        ax.set_title(f'{period}年转换矩阵')
        
        # 添加数值标注
        for i in range(len(categories)):
            for j in range(len(categories)):
                value = matrix[i, j]
                if value > 0:
                    color = 'white' if value > matrix.max() * 0.5 else 'black'
                    ax.text(j, i, f'{value:,.0f}', ha='center', va='center', 
                           color=color, fontsize=8, weight='bold')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8, label='面积 (km²)')
    
    # 隐藏多余的子图
    for i in range(n_periods, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(output_dir, f"transition_heatmap_{timestamp}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"转换矩阵热力图已保存: {png_path}")
    return png_path

def create_alluvial_diagram(df, output_dir):
    """
    创建冲积图展示时间序列转换
    
    参数:
    - df: 转换数据DataFrame
    - output_dir: 输出目录
    
    返回:
    - str: 输出文件路径
    """
    # 计算每年各类别的面积
    periods = sorted(df['时期'].unique())
    categories = ['人工林', '自然林', '其他']
    
    # 获取所有年份
    years = []
    for period in periods:
        start_year, end_year = period.split('-')
        if start_year not in years:
            years.append(start_year)
        if end_year not in years:
            years.append(end_year)
    
    unique_years = sorted(set(years))
    
    # 计算每年各类别面积
    year_areas = {}
    
    # 计算起始年份面积（基于流出）
    start_year = unique_years[0]
    for category in categories:
        outflow = df[
            (df['时期'] == periods[0]) & 
            (df['源类别'] == category)
        ]['面积(km²)'].sum()
        if start_year not in year_areas:
            year_areas[start_year] = {}
        year_areas[start_year][category] = outflow
    
    # 计算后续年份面积（基于流入）
    for period in periods:
        end_year = period.split('-')[1]
        for category in categories:
            inflow = df[
                (df['时期'] == period) & 
                (df['目标类别'] == category)
            ]['面积(km²)'].sum()
            if end_year not in year_areas:
                year_areas[end_year] = {}
            year_areas[end_year][category] = inflow
    
    # 创建冲积图数据
    fig = go.Figure()
    
    # 定义颜色
    colors = {
        '人工林': 'rgba(34, 139, 34, 0.8)',
        '自然林': 'rgba(0, 100, 0, 0.8)',
        '其他': 'rgba(139, 69, 19, 0.8)'
    }
    
    # 为每个类别创建面积条
    for category in categories:
        x_vals = []
        y_vals = []
        
        for year in unique_years:
            area = year_areas.get(year, {}).get(category, 0)
            x_vals.append(year)
            y_vals.append(area)
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines+markers',
            name=category,
            line=dict(color=colors[category], width=4),
            marker=dict(size=8),
            fill='tonexty' if category != '人工林' else 'tozeroy',
            fillcolor=colors[category],
            hovertemplate=f"<b>{category}</b><br>年份: %{{x}}<br>面积: %{{y:,.0f}} km²<extra></extra>"
        ))
    
    # 设置布局
    fig.update_layout(
        title={
            'text': '土地利用类别面积时间序列变化',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Microsoft YaHei'}
        },
        xaxis=dict(
            title='年份',
            tickmode='array',
            tickvals=unique_years,
            ticktext=unique_years
        ),
        yaxis=dict(
            title='面积 (km²)',
            tickformat=',.0f'
        ),
        font=dict(size=14, family="Microsoft YaHei"),
        width=1000,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"alluvial_diagram_{timestamp}.html")
    fig.write_html(html_path)
    print(f"冲积图HTML版本已保存: {html_path}")
    
    try:
        png_path = os.path.join(output_dir, f"alluvial_diagram_{timestamp}.png")
        fig.write_image(png_path, width=1000, height=600, scale=2)
        print(f"冲积图PNG版本已保存: {png_path}")
        return png_path
    except Exception as e:
        print(f"保存PNG版本失败: {e}")
        return html_path

def analyze_key_transitions(df, output_dir):
    """
    分析关键转换并生成统计报告
    
    参数:
    - df: 转换数据DataFrame
    - output_dir: 输出目录
    """
    print("\n=== 关键转换分析 ===")
    
    # 分析各时期的关键转换
    key_transitions = []
    
    for period in sorted(df['时期'].unique()):
        period_data = df[df['时期'] == period]
        print(f"\n{period}年关键转换:")
        
        # 人工林 ↔ 自然林 转换
        af_to_nf = period_data[
            (period_data['源类别'] == '人工林') & 
            (period_data['目标类别'] == '自然林')
        ]['面积(km²)'].sum()
        
        nf_to_af = period_data[
            (period_data['源类别'] == '自然林') & 
            (period_data['目标类别'] == '人工林')
        ]['面积(km²)'].sum()
        
        # 转为其他类别
        af_to_other = period_data[
            (period_data['源类别'] == '人工林') & 
            (period_data['目标类别'] == '其他')
        ]['面积(km²)'].sum()
        
        nf_to_other = period_data[
            (period_data['源类别'] == '自然林') & 
            (period_data['目标类别'] == '其他')
        ]['面积(km²)'].sum()
        
        # 从其他类别转入
        other_to_af = period_data[
            (period_data['源类别'] == '其他') & 
            (period_data['目标类别'] == '人工林')
        ]['面积(km²)'].sum()
        
        other_to_nf = period_data[
            (period_data['源类别'] == '其他') & 
            (period_data['目标类别'] == '自然林')
        ]['面积(km²)'].sum()
        
        print(f"  人工林 → 自然林: {af_to_nf:,.1f} km²")
        print(f"  自然林 → 人工林: {nf_to_af:,.1f} km²")
        print(f"  人工林 → 其他: {af_to_other:,.1f} km²")
        print(f"  自然林 → 其他: {nf_to_other:,.1f} km²")
        print(f"  其他 → 人工林: {other_to_af:,.1f} km²")
        print(f"  其他 → 自然林: {other_to_nf:,.1f} km²")
        
        # 净转换
        net_af_nf = nf_to_af - af_to_nf  # 正值表示自然林净转为人工林
        print(f"  净转换 (自然林→人工林): {net_af_nf:+,.1f} km²")
        
        key_transitions.append({
            '时期': period,
            '人工林→自然林': af_to_nf,
            '自然林→人工林': nf_to_af,
            '人工林→其他': af_to_other,
            '自然林→其他': nf_to_other,
            '其他→人工林': other_to_af,
            '其他→自然林': other_to_nf,
            '净转换(自然林→人工林)': net_af_nf
        })
    
    # 保存关键转换统计
    transitions_df = pd.DataFrame(key_transitions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"key_transitions_analysis_{timestamp}.csv")
    transitions_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n关键转换分析已保存: {csv_path}")
    
    return transitions_df

def create_multi_period_chord_data(df, periods_list):
    """
    创建多时期汇总的弦图数据
    
    参数:
    - df: 转换数据DataFrame
    - periods_list: 时期列表，如['2020-2021', '2021-2022']
    
    返回:
    - dict: 弦图数据
    """
    df_filtered = df[df['时期'].isin(periods_list)].copy()
    print(f"创建多时期汇总弦图数据: {', '.join(periods_list)}")
    
    # 定义类别和颜色
    categories = ['人工林', '自然林', '其他']
    colors = {
        '人工林': '#228B22',  # 森林绿
        '自然林': '#006400',  # 深绿
        '其他': '#8B4513'     # 棕色
    }
    
    # 创建转换矩阵（汇总多个时期）
    matrix = np.zeros((len(categories), len(categories)))
    
    for _, row in df_filtered.iterrows():
        source_idx = categories.index(row['源类别'])
        target_idx = categories.index(row['目标类别'])
        matrix[source_idx][target_idx] += row['面积(km²)']
    
    return {
        'matrix': matrix,
        'categories': categories,
        'colors': colors,
        'total_area': matrix.sum()
    }

def main():
    """
    主函数：创建多种土地利用转换可视化（改进版，支持多时间跨度）
    """
    # 配置参数
    EXCEL_PATH = r"D:\地理所\论文\东南亚10m人工林提取\数据\第三次分类_样本筛选_每年逐年分类\GEE下载_逐年镶嵌_马尔科夫模型\2.土地利用转换矩阵和桑基图分析_变化检测分类\land_use_transition_matrices_20251028_020313.xlsx"
    OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\第三次分类_样本筛选_每年逐年分类\GEE下载_逐年镶嵌_马尔科夫模型\3.弦图可视化_变化检测分类"
    
    print("开始创建土地利用转换可视化（改进版）")
    print(f"数据文件: {EXCEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 加载数据
        print("\n1. 加载转换矩阵数据...")
        df = load_transition_data(EXCEL_PATH)
        
        if df is None:
            print("数据加载失败，程序退出")
            return
        
        # 获取所有可用时期
        available_periods = sorted(df['时期'].unique())
        print(f"可用时期: {available_periods}")
        
        # 1. 创建每隔一年的弦图（单独时期）
        print("\n2. 创建每隔一年的弦图...")
        for period in available_periods:
            period_chord_data = create_chord_diagram_data(df, period)
            period_chord_path = create_chord_diagram(
                period_chord_data,
                OUTPUT_DIR,
                f"东南亚ZONE5土地利用转换关系弦图 ({period})",
                f"chord_diagram_annual_{period.replace('-', '_')}"
            )
        
        # 2. 创建每隔3年的弦图（多时期汇总）
        print("\n3. 创建每隔3年的弦图...")
        
        # 2020-2023年（前3年）
        if all(p in available_periods for p in ['2020-2021', '2021-2022', '2022-2023']):
            periods_2020_2023 = ['2020-2021', '2021-2022', '2022-2023']
            chord_data_2020_2023 = create_multi_period_chord_data(df, periods_2020_2023)
            chord_path_2020_2023 = create_chord_diagram(
                chord_data_2020_2023,
                OUTPUT_DIR,
                "东南亚ZONE5土地利用转换关系弦图 (2020-2023年汇总)",
                "chord_diagram_3year_2020_2023"
            )
        
        # 2021-2024年（后3年）
        if all(p in available_periods for p in ['2021-2022', '2022-2023', '2023-2024']):
            periods_2021_2024 = ['2021-2022', '2022-2023', '2023-2024']
            chord_data_2021_2024 = create_multi_period_chord_data(df, periods_2021_2024)
            chord_path_2021_2024 = create_chord_diagram(
                chord_data_2021_2024,
                OUTPUT_DIR,
                "东南亚ZONE5土地利用转换关系弦图 (2021-2024年汇总)",
                "chord_diagram_3year_2021_2024"
            )
        
        # 3. 创建2017-2024年长期弦图（如果有2017年数据）
        print("\n4. 创建长期弦图...")
        
        # 检查是否有2017年相关数据
        periods_with_2017 = [p for p in available_periods if '2017' in p]
        if periods_with_2017:
            print(f"发现2017年相关数据: {periods_with_2017}")
            # 创建包含2017年的所有时期弦图
            all_periods_from_2017 = [p for p in available_periods if int(p.split('-')[0]) >= 2017]
            chord_data_long_term = create_multi_period_chord_data(df, all_periods_from_2017)
            chord_path_long_term = create_chord_diagram(
                chord_data_long_term,
                OUTPUT_DIR,
                f"东南亚ZONE5土地利用转换关系弦图 (2017-2024年长期汇总)",
                "chord_diagram_longterm_2017_2024"
            )
        else:
            print("未发现2017年数据，创建2020-2024年汇总弦图")
            # 创建2020-2024年汇总弦图
            chord_data_all = create_chord_diagram_data(df)
            chord_path_all = create_chord_diagram(
                chord_data_all, 
                OUTPUT_DIR, 
                "东南亚ZONE5土地利用转换关系弦图 (2020-2024年汇总)",
                "chord_diagram_longterm_2020_2024"
            )
        
        # 4. 创建转换矩阵热力图
        print("\n5. 创建转换矩阵热力图...")
        heatmap_path = create_transition_heatmap(df, OUTPUT_DIR)
        
        # 5. 创建冲积图
        print("\n6. 创建冲积图...")
        alluvial_path = create_alluvial_diagram(df, OUTPUT_DIR)
        
        # 6. 分析关键转换
        print("\n7. 分析关键转换...")
        transitions_df = analyze_key_transitions(df, OUTPUT_DIR)
        
        print(f"\n=== 可视化制作完成 ===")
        print(f"所有文件已保存至: {OUTPUT_DIR}")
        print(f"主要输出文件:")
        print(f"  - 单年弦图: chord_diagram_*年.html/png")
        print(f"  - 3年汇总弦图: chord_diagram_*年汇总.html/png")
        print(f"  - 长期汇总弦图: chord_diagram_*长期汇总.html/png")
        print(f"  - 转换矩阵热力图: transition_heatmap_*.png")
        print(f"  - 冲积图: alluvial_diagram_*.html/png")
        print(f"  - 关键转换分析: key_transitions_analysis_*.csv")
        
        # 输出弦图创建总结
        print(f"\n=== 弦图创建总结 ===")
        print(f"✓ 每隔一年弦图: {len(available_periods)} 个")
        print(f"✓ 每隔3年弦图: 2 个 (2020-2023, 2021-2024)")
        if periods_with_2017:
            print(f"✓ 长期弦图: 1 个 (2017-2024)")
        else:
            print(f"✓ 汇总弦图: 1 个 (2020-2024)")
        print(f"✓ 所有弦图均支持双向箭头显示相互转换关系")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()