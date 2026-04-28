#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断QA_PIXEL处理效果
分析每个点的QA值分布和清晰观测数量
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def convert_qa_pixel_to_cfmask(qa_pixel):
    """
    将QA_PIXEL转换为cfmask（与主脚本保持一致）
    """
    if pd.isna(qa_pixel) or not isinstance(qa_pixel, (int, float, np.integer, np.floating)):
        return 255
    
    qa_pixel = int(qa_pixel)
    
    # 检查填充值 (bit 0)
    if qa_pixel & (1 << 0):
        return 255
    
    # 检查水体 (bit 7)
    if qa_pixel & (1 << 7):
        return 1
    
    # 检查雪 (bit 5)
    if qa_pixel & (1 << 5):
        return 3
    
    # 云检测
    cloud_flag = qa_pixel & (1 << 3)
    cloud_conf = (qa_pixel >> 8) & 3
    if cloud_flag and cloud_conf >= 2:
        return 4
    
    # 阴影检测
    shadow_flag = qa_pixel & (1 << 4)
    shadow_conf = (qa_pixel >> 10) & 3
    if shadow_flag and shadow_conf >= 2:
        return 2
    
    # 卷云检测
    cirrus_flag = qa_pixel & (1 << 2)
    cirrus_conf = (qa_pixel >> 14) & 3
    if cirrus_flag and cirrus_conf == 3:
        return 4
    
    # 检查Clear位
    clear_flag = qa_pixel & (1 << 6)
    if clear_flag:
        return 0
    
    return 0

def analyze_qa_distribution():
    """
    分析QA值分布和处理效果
    """
    data_dir = Path(r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.CCDC计算FD\数据\Singapore_PyCCD_RAW")
    
    print("开始分析QA值分布...")
    
    # 读取2022年数据作为样本
    sample_file = data_dir / "raw_data_2022.csv"
    df = pd.read_csv(sample_file)
    
    print(f"样本数据: {len(df)} 条记录")
    print(f"唯一点数: {df['id'].nunique()}")
    
    # 分析QA值分布
    qa_values = df['QA'].value_counts().head(20)
    print("\n=== 最常见的QA值 ===")
    for qa_val, count in qa_values.items():
        print(f"QA={qa_val}: {count}次 ({count/len(df)*100:.1f}%)")
    
    # 转换为cfmask
    df['cfmask'] = df['QA'].apply(convert_qa_pixel_to_cfmask)
    
    # 分析cfmask分布
    cfmask_labels = {0: 'clear', 1: 'water', 2: 'shadow', 3: 'snow', 4: 'cloud', 255: 'fill'}
    cfmask_counts = df['cfmask'].value_counts()
    
    print("\n=== cfmask分布 ===")
    for cfmask_val, count in cfmask_counts.items():
        label = cfmask_labels.get(cfmask_val, f'unknown({cfmask_val})')
        print(f"{label}: {count}次 ({count/len(df)*100:.1f}%)")
    
    # 分析每个点的清晰观测数量
    print("\n=== 按点分析清晰观测数量 ===")
    
    point_stats = []
    for point_id in df['id'].unique()[:20]:  # 分析前20个点
        point_df = df[df['id'] == point_id]
        
        # 过滤有效光谱数据
        spectral_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
        valid_spectral = point_df[spectral_bands].notna().all(axis=1)
        valid_df = point_df[valid_spectral]
        
        if len(valid_df) == 0:
            continue
            
        # 统计cfmask分布
        cfmask_dist = valid_df['cfmask'].value_counts()
        clear_count = cfmask_dist.get(0, 0)
        total_count = len(valid_df)
        
        point_stats.append({
            'point_id': point_id,
            'total_obs': len(point_df),
            'valid_spectral': len(valid_df),
            'clear_obs': clear_count,
            'clear_ratio': clear_count / len(valid_df) if len(valid_df) > 0 else 0,
            'cfmask_dist': dict(cfmask_dist)
        })
        
        print(f"点 {point_id}: 总观测={len(point_df)}, 有效光谱={len(valid_df)}, 清晰观测={clear_count} ({clear_count/len(valid_df)*100:.1f}%)")
    
    # 统计清晰观测数量分布
    clear_counts = [stat['clear_obs'] for stat in point_stats]
    clear_dist = Counter(clear_counts)
    
    print("\n=== 清晰观测数量分布 ===")
    for clear_num in sorted(clear_dist.keys()):
        count = clear_dist[clear_num]
        print(f"{clear_num}个清晰观测: {count}个点")
    
    # 分析具体QA值的位信息
    print("\n=== 常见QA值的位分析 ===")
    common_qa_values = [5896, 23888, 62820]  # 从之前的分析中选择
    
    for qa_val in common_qa_values:
        if qa_val in df['QA'].values:
            print(f"\nQA={qa_val} (二进制: {bin(qa_val)})")
            print(f"  Bit 0 (Fill): {bool(qa_val & (1 << 0))}")
            print(f"  Bit 1 (Dilated Cloud): {bool(qa_val & (1 << 1))}")
            print(f"  Bit 2 (Cirrus): {bool(qa_val & (1 << 2))}")
            print(f"  Bit 3 (Cloud): {bool(qa_val & (1 << 3))}")
            print(f"  Bit 4 (Shadow): {bool(qa_val & (1 << 4))}")
            print(f"  Bit 5 (Snow): {bool(qa_val & (1 << 5))}")
            print(f"  Bit 6 (Clear): {bool(qa_val & (1 << 6))}")
            print(f"  Bit 7 (Water): {bool(qa_val & (1 << 7))}")
            print(f"  Cloud Conf (8-9): {(qa_val >> 8) & 3}")
            print(f"  Shadow Conf (10-11): {(qa_val >> 10) & 3}")
            print(f"  Snow Conf (12-13): {(qa_val >> 12) & 3}")
            print(f"  Cirrus Conf (14-15): {(qa_val >> 14) & 3}")
            print(f"  转换为cfmask: {convert_qa_pixel_to_cfmask(qa_val)}")

if __name__ == "__main__":
    analyze_qa_distribution()