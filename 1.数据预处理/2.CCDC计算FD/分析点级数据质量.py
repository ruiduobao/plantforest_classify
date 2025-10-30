#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析每个点的数据质量分布
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

def convert_qa_pixel_to_cfmask(qa_pixel):
    """
    将Landsat Collection 2的QA_PIXEL转换为cfmask格式
    """
    # 处理NaN值和非数值
    if pd.isna(qa_pixel) or not isinstance(qa_pixel, (int, float)):
        return 255
    
    qa_pixel = int(qa_pixel)
    
    if qa_pixel == 65535:  # 填充值
        return 255
    
    # 提取关键位
    fill = (qa_pixel >> 0) & 1
    cloud = (qa_pixel >> 3) & 1
    cloud_shadow = (qa_pixel >> 4) & 1
    snow = (qa_pixel >> 5) & 1
    clear = (qa_pixel >> 6) & 1
    water = (qa_pixel >> 7) & 1
    
    # 提取置信度位
    cloud_conf = (qa_pixel >> 8) & 3
    shadow_conf = (qa_pixel >> 10) & 3
    
    # 填充值
    if fill == 1:
        return 255
    
    # 水体
    if water == 1:
        return 1
    
    # 雪/冰（热带地区很少，但保留检测）
    if snow == 1:
        return 3
    
    # 极度宽松的云检测：只有明确的高置信度云才标记
    if cloud == 1 and cloud_conf == 3:
        return 4
    
    # 极度宽松的云阴影检测：只有明确的高置信度阴影才标记
    if cloud_shadow == 1 and shadow_conf == 3:
        return 2
    
    # 卷云完全忽略（热带地区卷云常见但不影响地表反射率太多）
    # 所有其他情况都视为清晰观测
    return 0

def analyze_point_data_quality():
    """
    分析每个点的数据质量
    """
    print("开始分析点级数据质量...")
    
    # 读取所有年份的数据
    data_dir = Path(r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.CCDC计算FD\数据\Singapore_PyCCD_RAW")
    all_data = []
    
    for year in range(1985, 2025):
        csv_file = data_dir / f'raw_data_{year}.csv'
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    df['year'] = year
                    all_data.append(df)
                    print(f"读取{year}年数据: {len(df)}条记录")
            except Exception as e:
                print(f"读取{year}年数据失败: {e}")
    
    if not all_data:
        print("没有找到有效数据文件")
        return
    
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\n总数据量: {len(combined_data)}条记录")
    
    # 转换QA为cfmask
    combined_data['cfmask'] = combined_data['QA'].apply(convert_qa_pixel_to_cfmask)
    
    # 过滤有效数据（排除填充值和异常值）
    valid_mask = (
        (combined_data['cfmask'] != 255) &  # 非填充值
        (combined_data['B'].notna()) &
        (combined_data['G'].notna()) &
        (combined_data['R'].notna()) &
        (combined_data['NIR'].notna()) &
        (combined_data['SWIR1'].notna()) &
        (combined_data['SWIR2'].notna())
        # THERMAL可能为空，不作为必需条件
    )
    
    valid_data = combined_data[valid_mask].copy()
    print(f"有效数据量: {len(valid_data)}条记录 ({len(valid_data)/len(combined_data)*100:.1f}%)")
    
    # 按点分组分析
    point_stats = []
    
    for point_id in valid_data['id'].unique():
        point_data = valid_data[valid_data['id'] == point_id]
        
        # 计算清晰观测
        clear_obs = point_data[point_data['cfmask'] == 0]
        
        # 计算各种质量标记的数量
        water_count = len(point_data[point_data['cfmask'] == 1])
        shadow_count = len(point_data[point_data['cfmask'] == 2])
        snow_count = len(point_data[point_data['cfmask'] == 3])
        cloud_count = len(point_data[point_data['cfmask'] == 4])
        clear_count = len(clear_obs)
        
        # 计算时间跨度
        years = point_data['year'].unique()
        time_span = max(years) - min(years) + 1
        
        point_stats.append({
            'point_id': point_id,
            'total_obs': len(point_data),
            'clear_obs': clear_count,
            'clear_ratio': clear_count / len(point_data) * 100,
            'water_obs': water_count,
            'shadow_obs': shadow_count,
            'snow_obs': snow_count,
            'cloud_obs': cloud_count,
            'time_span': time_span,
            'years_covered': len(years),
            'avg_obs_per_year': len(point_data) / len(years)
        })
    
    # 转换为DataFrame
    stats_df = pd.DataFrame(point_stats)
    
    print(f"\n=== 点级数据质量统计 ===")
    print(f"总点数: {len(stats_df)}")
    print(f"平均总观测数: {stats_df['total_obs'].mean():.1f}")
    print(f"平均清晰观测数: {stats_df['clear_obs'].mean():.1f}")
    print(f"平均清晰观测比例: {stats_df['clear_ratio'].mean():.1f}%")
    
    print(f"\n=== 清晰观测数量分布 ===")
    clear_bins = [0, 6, 12, 24, 50, 100, float('inf')]
    clear_labels = ['<6', '6-11', '12-23', '24-49', '50-99', '≥100']
    stats_df['clear_bin'] = pd.cut(stats_df['clear_obs'], bins=clear_bins, labels=clear_labels, right=False)
    clear_dist = stats_df['clear_bin'].value_counts().sort_index()
    
    for bin_name, count in clear_dist.items():
        percentage = count / len(stats_df) * 100
        print(f"{bin_name}个清晰观测: {count}个点 ({percentage:.1f}%)")
    
    print(f"\n=== 清晰观测比例分布 ===")
    ratio_bins = [0, 10, 20, 30, 40, 50, float('inf')]
    ratio_labels = ['<10%', '10-19%', '20-29%', '30-39%', '40-49%', '≥50%']
    stats_df['ratio_bin'] = pd.cut(stats_df['clear_ratio'], bins=ratio_bins, labels=ratio_labels, right=False)
    ratio_dist = stats_df['ratio_bin'].value_counts().sort_index()
    
    for bin_name, count in ratio_dist.items():
        percentage = count / len(stats_df) * 100
        print(f"{bin_name}清晰观测比例: {count}个点 ({percentage:.1f}%)")
    
    # 找出清晰观测最多和最少的点
    print(f"\n=== 极值点分析 ===")
    best_points = stats_df.nlargest(5, 'clear_obs')
    worst_points = stats_df.nsmallest(5, 'clear_obs')
    
    print("清晰观测最多的5个点:")
    for _, row in best_points.iterrows():
        print(f"  点{int(row['point_id'])}: {int(row['clear_obs'])}个清晰观测 ({row['clear_ratio']:.1f}%), 总观测{int(row['total_obs'])}个")
    
    print("清晰观测最少的5个点:")
    for _, row in worst_points.iterrows():
        print(f"  点{int(row['point_id'])}: {int(row['clear_obs'])}个清晰观测 ({row['clear_ratio']:.1f}%), 总观测{int(row['total_obs'])}个")
    
    # 保存详细统计
    stats_df.to_csv('点级数据质量统计.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细统计已保存到: 点级数据质量统计.csv")
    
    print("\n点级数据质量分析完成")

if __name__ == "__main__":
    analyze_point_data_quality()