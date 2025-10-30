#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析新加坡Landsat数据的质量分布
了解清晰观测的数量和分布
"""

import pandas as pd
import numpy as np
from pathlib import Path

def convert_qa_pixel_to_cfmask(qa_pixel):
    """
    将QA_PIXEL转换为cfmask格式
    """
    if pd.isna(qa_pixel):
        return 255
    
    qa_pixel = int(qa_pixel)
    
    # 检查填充值 (bit 0)
    if qa_pixel & (1 << 0):
        return 255  # fill
    
    # 检查云 (bit 3)
    if qa_pixel & (1 << 3):
        return 4  # cloud
    
    # 检查云阴影 (bit 4)
    if qa_pixel & (1 << 4):
        return 2  # cloud shadow
    
    # 检查雪 (bit 5)
    if qa_pixel & (1 << 5):
        return 3  # snow
    
    # 检查水体 (bit 7)
    if qa_pixel & (1 << 7):
        return 1  # water
    
    # 检查云置信度 (bits 8-9)
    cloud_conf = (qa_pixel >> 8) & 3
    if cloud_conf == 3:  # 最高置信度
        return 4  # cloud
    
    # 检查云阴影置信度 (bits 10-11)
    shadow_conf = (qa_pixel >> 10) & 3
    if shadow_conf == 3:  # 最高置信度
        return 2  # cloud shadow
    
    # 检查卷云 (bit 2)
    if qa_pixel & (1 << 2):
        if cloud_conf <= 1 and shadow_conf <= 1:
            return 0  # 轻微卷云视为清晰
        else:
            return 4  # 有其他问题时标记为云
    
    return 0  # clear

def analyze_data_quality():
    """
    分析数据质量
    """
    data_dir = Path(r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.CCDC计算FD\数据\Singapore_PyCCD_RAW")
    
    print("开始分析数据质量...")
    
    all_data = []
    
    # 读取所有年份的数据
    for year in range(1985, 2025):
        file_path = data_dir / f"raw_data_{year}.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['year'] = year
                all_data.append(df)
                print(f"读取{year}年数据: {len(df)}条记录")
            except Exception as e:
                print(f"读取{year}年数据失败: {e}")
    
    if not all_data:
        print("没有找到数据文件")
        return
    
    # 合并数据
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\n总数据量: {len(combined_data)}条记录")
    print(f"唯一点数: {combined_data['id'].nunique()}")
    
    # 转换QA为cfmask
    combined_data['cfmask'] = combined_data['QA'].apply(convert_qa_pixel_to_cfmask)
    
    # 分析QA质量分布
    print("\n=== QA质量分布 ===")
    cfmask_labels = {0: 'Clear', 1: 'Water', 2: 'Cloud Shadow', 3: 'Snow', 4: 'Cloud', 255: 'Fill'}
    cfmask_counts = combined_data['cfmask'].value_counts().sort_index()
    for cfmask, count in cfmask_counts.items():
        label = cfmask_labels.get(cfmask, f'Unknown({cfmask})')
        percentage = count / len(combined_data) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")
    
    # 分析每个点的清晰观测数量
    print("\n=== 每个点的清晰观测统计 ===")
    clear_obs_per_point = combined_data[combined_data['cfmask'] == 0].groupby('id').size()
    
    print(f"有清晰观测的点数: {len(clear_obs_per_point)}")
    print(f"清晰观测数量统计:")
    print(f"  平均: {clear_obs_per_point.mean():.1f}")
    print(f"  中位数: {clear_obs_per_point.median():.1f}")
    print(f"  最小值: {clear_obs_per_point.min()}")
    print(f"  最大值: {clear_obs_per_point.max()}")
    
    # 分析不同观测数量阈值下的点数
    print("\n=== 不同清晰观测阈值下的可用点数 ===")
    for threshold in [5, 8, 10, 12, 15, 20, 30]:
        valid_points = (clear_obs_per_point >= threshold).sum()
        percentage = valid_points / combined_data['id'].nunique() * 100
        print(f"≥{threshold}个清晰观测: {valid_points}个点 ({percentage:.1f}%)")
    
    # 分析光谱数据质量
    print("\n=== 光谱数据质量 ===")
    spectral_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
    
    for band in spectral_bands:
        valid_count = combined_data[band].notna().sum()
        fill_count = (combined_data[band] == 65535).sum()
        percentage_valid = valid_count / len(combined_data) * 100
        percentage_fill = fill_count / len(combined_data) * 100
        print(f"{band}: 有效值 {valid_count} ({percentage_valid:.1f}%), 填充值 {fill_count} ({percentage_fill:.1f}%)")
    
    # 分析年度数据分布
    print("\n=== 年度数据分布 ===")
    yearly_stats = combined_data.groupby('year').agg({
        'id': 'count',
        'cfmask': lambda x: (x == 0).sum()
    }).rename(columns={'id': 'total_obs', 'cfmask': 'clear_obs'})
    
    yearly_stats['clear_percentage'] = yearly_stats['clear_obs'] / yearly_stats['total_obs'] * 100
    
    for year, row in yearly_stats.iterrows():
        print(f"{year}: 总观测 {row['total_obs']}, 清晰观测 {row['clear_obs']} ({row['clear_percentage']:.1f}%)")
    
    print("\n数据质量分析完成")

if __name__ == "__main__":
    analyze_data_quality()