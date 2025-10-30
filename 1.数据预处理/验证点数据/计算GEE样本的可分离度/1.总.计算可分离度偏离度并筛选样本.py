#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的样本点分析流程脚本
功能：
1. 计算样本点的光谱可分离度（JM距离、Bhattacharyya距离）
2. 分析每个样本点的偏离度（马氏距离）
3. 基于偏离度结果筛选样本点

流程：
可分离度计算 → 偏离度分析 → 样本点筛选

作者：锐多宝 (ruiduobao)
日期：2025年1月21日
"""

from osgeo import ogr, osr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import geopandas as gpd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import math
import sys
import random
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置参数 ====================
# 输入文件路径
INPUT_SHAPEFILE = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果\Zone5_2024_Merged_AllBand_Sample.shp"

# 统一输出目录
OUTPUT_BASE_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_本地分析\2024"

# 字段配置
LAYER_NAME = None
CLASS_FIELD = "landcover"
BANDS = [f"A{i:02d}" for i in range(0, 64)]  # 波段A00到A63

# 筛选参数
HIGH_REMOVAL_RATIO = 0.7  # 删除70%的High等级样本点
RANDOM_SEED = 42  # 随机种子，确保结果可重现

# 数值稳定性参数
EPS = 1e-6  # 协方差正则化项

# ==================== 第一部分：可分离度计算 ====================

def read_features(path, layer_name=None):
    """
    读取矢量文件中的要素属性
    参数：
        path: 矢量文件路径
        layer_name: 图层名称，None则取第一个图层
    返回：
        features: 要素属性列表
    """
    ds = ogr.Open(path, 0)
    if ds is None:
        raise RuntimeError("无法打开矢量文件: " + path)
    if layer_name is None:
        layer = ds.GetLayer(0)
    else:
        layer = ds.GetLayerByName(layer_name)
    features = []
    layer.ResetReading()
    for feat in layer:
        props = feat.items()
        features.append(props)
    return features

def group_by_class(features, class_field, bands):
    """
    按类别分组样本数据
    参数：
        features: 要素属性列表
        class_field: 类别字段名
        bands: 波段名称列表
    返回：
        groups: 按类别分组的光谱数据字典
    """
    groups = {}
    for f in features:
        cls = f[class_field]
        vec = np.array([float(f[b]) for b in bands], dtype=np.float64)
        groups.setdefault(cls, []).append(vec)
    for k in list(groups.keys()):
        groups[k] = np.vstack(groups[k])
    return groups

def mean_and_cov(X):
    """
    计算均值和协方差矩阵
    参数：
        X: 样本数据矩阵 (n x d)
    返回：
        mu: 均值向量
        cov: 协方差矩阵
    """
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = np.dot(Xc.T, Xc) / (X.shape[0] - 1) if X.shape[0] > 1 else np.eye(X.shape[1]) * EPS
    return mu, cov

def bhattacharyya_distance(mu1, cov1, mu2, cov2, reg=1e-6):
    """
    计算Bhattacharyya距离
    参数：
        mu1, mu2: 两个类别的均值向量
        cov1, cov2: 两个类别的协方差矩阵
        reg: 正则化参数
    返回：
        B: Bhattacharyya距离
    """
    d = mu1.shape[0]
    cov1_r = cov1 + np.eye(d) * reg
    cov2_r = cov2 + np.eye(d) * reg
    covm = 0.5 * (cov1_r + cov2_r)
    
    sign_m, logdet_m = np.linalg.slogdet(covm)
    sign1, logdet1 = np.linalg.slogdet(cov1_r)
    sign2, logdet2 = np.linalg.slogdet(cov2_r)
    
    if sign_m <= 0 or sign1 <= 0 or sign2 <= 0:
        logdet_term = 0.0
    else:
        logdet_term = 0.5 * (logdet_m - 0.5*(logdet1 + logdet2))
    
    try:
        inv_covm = np.linalg.inv(covm)
    except np.linalg.LinAlgError:
        inv_covm = np.linalg.pinv(covm)
    
    diff = (mu1 - mu2).reshape((d,1))
    term1 = 0.125 * float(diff.T.dot(inv_covm).dot(diff))
    term2 = logdet_term
    B = term1 + term2
    return float(B)

def compute_pairwise_separability(groups):
    """
    计算所有类别对之间的可分离度
    参数：
        groups: 按类别分组的光谱数据
    返回：
        results: 可分离度结果列表
        stats: 各类别统计信息
    """
    classes = sorted(groups.keys())
    stats = {}
    for c in classes:
        X = groups[c]
        mu, cov = mean_and_cov(X)
        stats[c] = {"n": X.shape[0], "mu": mu, "cov": cov}
    
    results = []
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            c1 = classes[i]; c2 = classes[j]
            s1 = stats[c1]; s2 = stats[c2]
            B = bhattacharyya_distance(s1["mu"], s1["cov"], s2["mu"], s2["cov"], reg=EPS)
            JM = 2.0 * (1.0 - math.exp(-B))
            results.append((c1, c2, s1["n"], s2["n"], B, JM))
    
    return results, stats

# ==================== 第二部分：偏离度分析 ====================

def read_features_with_geometry(path, layer_name=None):
    """
    读取矢量文件中的要素属性和几何信息
    参数：
        path: 矢量文件路径
        layer_name: 图层名称，None则取第一个图层
    返回：
        features: 要素属性列表
        geometries: 几何信息列表
        spatial_ref: 空间参考系统
    """
    ds = ogr.Open(path, 0)
    if ds is None:
        raise RuntimeError("无法打开矢量文件: " + path)
    
    if layer_name is None:
        layer = ds.GetLayer(0)
    else:
        layer = ds.GetLayerByName(layer_name)
    
    features = []
    geometries = []
    spatial_ref = layer.GetSpatialRef()
    
    layer.ResetReading()
    for feat in layer:
        props = feat.items()
        geom = feat.GetGeometryRef()
        if geom:
            x = geom.GetX()
            y = geom.GetY()
            props['geometry_x'] = x
            props['geometry_y'] = y
        features.append(props)
        geometries.append(geom.Clone() if geom else None)
    
    ds = None
    return features, geometries, spatial_ref

def group_by_class_with_indices(features, class_field, bands):
    """
    按类别分组样本数据，同时保留原始索引
    参数：
        features: 要素属性列表
        class_field: 类别字段名
        bands: 波段名称列表
    返回：
        groups: 按类别分组的光谱数据字典
        indices: 每个类别对应的原始索引
    """
    groups = {}
    indices = {}
    
    for i, f in enumerate(features):
        cls = f[class_field]
        vec = np.array([float(f[b]) for b in bands], dtype=np.float64)
        
        groups.setdefault(cls, []).append(vec)
        indices.setdefault(cls, []).append(i)
    
    for k in list(groups.keys()):
        groups[k] = np.vstack(groups[k])
        indices[k] = np.array(indices[k])
    
    return groups, indices

def calculate_mahalanobis_distances(groups, indices, features, bands):
    """
    计算每个样本点相对于其类别群体的马氏距离
    参数：
        groups: 按类别分组的光谱数据
        indices: 每个类别对应的原始索引
        features: 原始要素列表
        bands: 波段名称列表
    返回：
        deviation_results: 包含偏离度信息的列表
    """
    deviation_results = []
    
    print("正在计算每个样本点的偏离度...")
    
    for cls, data in groups.items():
        print(f"处理类别 {cls}，样本数量: {len(data)}")
        
        mean_vec = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False)
        
        reg_term = 1e-6
        cov_matrix += reg_term * np.eye(cov_matrix.shape[0])
        
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print(f"警告：类别 {cls} 的协方差矩阵奇异，使用伪逆")
            cov_inv = np.linalg.pinv(cov_matrix)
        
        class_indices = indices[cls]
        for i, idx in enumerate(class_indices):
            sample_vec = data[i]
            
            diff = sample_vec - mean_vec
            mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)
            
            chi2_stat = mahal_dist ** 2
            p_value = 1 - chi2.cdf(chi2_stat, df=len(bands))
            
            if p_value > 0.05:
                deviation_level = "Normal"
            elif p_value > 0.01:
                deviation_level = "Moderate"
            elif p_value > 0.001:
                deviation_level = "High"
            else:
                deviation_level = "Extreme"
            
            original_feature = features[idx]
            
            result = {
                'original_index': idx,
                'landcover': cls,
                'mahalanobis_distance': mahal_dist,
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'deviation_level': deviation_level,
                'geometry_x': original_feature.get('geometry_x', 0),
                'geometry_y': original_feature.get('geometry_y', 0)
            }
            
            for key, value in original_feature.items():
                if key not in result:
                    result[key] = value
            
            deviation_results.append(result)
    
    return deviation_results

def create_deviation_shapefile(deviation_results, output_path, spatial_ref):
    """
    创建包含偏离度信息的输出shapefile
    参数：
        deviation_results: 偏离度分析结果
        output_path: 输出文件路径
        spatial_ref: 空间参考系统
    """
    print(f"正在创建偏离度分析shapefile: {output_path}")
    
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    
    out_ds = driver.CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer("deviation_analysis", spatial_ref, ogr.wkbPoint)
    
    fields = [
        ("landcover", ogr.OFTReal),
        ("mahal_dist", ogr.OFTReal),
        ("chi2_stat", ogr.OFTReal),
        ("p_value", ogr.OFTReal),
        ("dev_level", ogr.OFTString)
    ]
    
    for band in BANDS:
        fields.append((band, ogr.OFTReal))
    
    for field_name, field_type in fields:
        field_def = ogr.FieldDefn(field_name, field_type)
        if field_type == ogr.OFTString:
            field_def.SetWidth(20)
        out_layer.CreateField(field_def)
    
    for result in deviation_results:
        feature = ogr.Feature(out_layer.GetLayerDefn())
        
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(result['geometry_x'], result['geometry_y'])
        feature.SetGeometry(point)
        
        feature.SetField("landcover", float(result['landcover']))
        feature.SetField("mahal_dist", result['mahalanobis_distance'])
        feature.SetField("chi2_stat", result['chi2_statistic'])
        feature.SetField("p_value", result['p_value'])
        feature.SetField("dev_level", result['deviation_level'])
        
        for band in BANDS:
            if band in result:
                feature.SetField(band, float(result[band]))
        
        out_layer.CreateFeature(feature)
        feature = None
    
    out_ds = None
    print(f"成功创建偏离度shapefile，包含 {len(deviation_results)} 个要素")

# ==================== 第三部分：样本点筛选 ====================

def filter_samples_by_deviation(deviation_results, high_removal_ratio=0.7, random_seed=42):
    """
    根据偏离度等级筛选样本点
    参数：
        deviation_results: 偏离度分析结果
        high_removal_ratio: High等级样本点的删除比例
        random_seed: 随机种子
    返回：
        filtered_results: 筛选后的结果
        removal_stats: 删除统计信息
    """
    print(f"\n正在执行样本点筛选...")
    print(f"筛选策略:")
    print(f"  - 删除所有 Extreme 等级样本点")
    print(f"  - 删除 {high_removal_ratio*100:.0f}% 的 High 等级样本点")
    print(f"  - 保留所有 Normal 和 Moderate 等级样本点")
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    df = pd.DataFrame(deviation_results)
    
    removal_stats = {
        'Extreme': {'total': 0, 'removed': 0},
        'High': {'total': 0, 'removed': 0},
        'Moderate': {'total': 0, 'removed': 0},
        'Normal': {'total': 0, 'removed': 0}
    }
    
    for level in ['Extreme', 'High', 'Moderate', 'Normal']:
        if level in df['deviation_level'].values:
            removal_stats[level]['total'] = len(df[df['deviation_level'] == level])
    
    filtered_df = df.copy()
    
    # 删除所有 Extreme 等级样本点
    extreme_mask = filtered_df['deviation_level'] == 'Extreme'
    extreme_count = extreme_mask.sum()
    if extreme_count > 0:
        filtered_df = filtered_df[~extreme_mask]
        removal_stats['Extreme']['removed'] = extreme_count
        print(f"  删除了 {extreme_count} 个 Extreme 等级样本点")
    
    # 随机删除部分 High 等级样本点
    high_samples = filtered_df[filtered_df['deviation_level'] == 'High']
    high_count = len(high_samples)
    if high_count > 0:
        remove_count = int(high_count * high_removal_ratio)
        remove_indices = np.random.choice(high_samples.index, size=remove_count, replace=False)
        filtered_df = filtered_df.drop(remove_indices)
        removal_stats['High']['removed'] = remove_count
        print(f"  随机删除了 {remove_count} 个 High 等级样本点 (总共{high_count}个)")
    
    normal_count = len(filtered_df[filtered_df['deviation_level'] == 'Normal'])
    moderate_count = len(filtered_df[filtered_df['deviation_level'] == 'Moderate'])
    print(f"  保留了 {normal_count} 个 Normal 等级样本点")
    print(f"  保留了 {moderate_count} 个 Moderate 等级样本点")
    
    print(f"\n筛选结果:")
    print(f"  原始样本数: {len(df)}")
    print(f"  筛选后样本数: {len(filtered_df)}")
    print(f"  删除样本数: {len(df) - len(filtered_df)}")
    print(f"  保留比例: {len(filtered_df)/len(df)*100:.1f}%")
    
    return filtered_df.to_dict('records'), removal_stats

def save_filtered_samples_shapefile(filtered_results, output_path, spatial_ref):
    """
    保存筛选后的样本点到shapefile
    参数：
        filtered_results: 筛选后的结果
        output_path: 输出文件路径
        spatial_ref: 空间参考系统
    """
    print(f"正在保存筛选后的样本点到: {output_path}")
    
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    
    out_ds = driver.CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer("filtered_samples", spatial_ref, ogr.wkbPoint)
    
    fields = [
        ("landcover", ogr.OFTReal),
        ("mahal_dist", ogr.OFTReal),
        ("chi2_stat", ogr.OFTReal),
        ("p_value", ogr.OFTReal),
        ("dev_level", ogr.OFTString)
    ]
    
    for band in BANDS:
        fields.append((band, ogr.OFTReal))
    
    for field_name, field_type in fields:
        field_def = ogr.FieldDefn(field_name, field_type)
        if field_type == ogr.OFTString:
            field_def.SetWidth(20)
        out_layer.CreateField(field_def)
    
    for result in filtered_results:
        feature = ogr.Feature(out_layer.GetLayerDefn())
        
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(result['geometry_x'], result['geometry_y'])
        feature.SetGeometry(point)
        
        feature.SetField("landcover", float(result['landcover']))
        feature.SetField("mahal_dist", result['mahalanobis_distance'])
        feature.SetField("chi2_stat", result['chi2_statistic'])
        feature.SetField("p_value", result['p_value'])
        feature.SetField("dev_level", result['deviation_level'])
        
        for band in BANDS:
            if band in result:
                feature.SetField(band, float(result[band]))
        
        out_layer.CreateFeature(feature)
        feature = None
    
    out_ds = None
    print(f"成功保存筛选后的样本点，包含 {len(filtered_results)} 个要素")

# ==================== 可视化和统计函数 ====================

def create_separability_visualizations(df_results, df_stats, output_dir, timestamp):
    """
    创建可分离度分析的可视化图表
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sample Separability Analysis Report', fontsize=16, fontweight='bold')
    
    # 子图1：各类别样本数量
    ax1 = axes[0, 0]
    bars = ax1.bar(df_stats['类别'], df_stats['样本数量'], color=sns.color_palette("husl", len(df_stats)))
    ax1.set_title('Sample Count Distribution by Class', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Land Cover Class')
    ax1.set_ylabel('Sample Count')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # 子图2：JM距离分布直方图
    ax2 = axes[0, 1]
    ax2.hist(df_results['JM距离'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('JM Distance Distribution Histogram', fontsize=12, fontweight='bold')
    ax2.set_xlabel('JM Distance')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df_results['JM距离'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_results["JM距离"].mean():.3f}')
    ax2.legend()
    
    # 子图3：可分离度热力图
    ax3 = axes[1, 0]
    classes = sorted(set(df_results['类别1'].tolist() + df_results['类别2'].tolist()))
    separability_matrix = np.zeros((len(classes), len(classes)))
    
    for _, row in df_results.iterrows():
        i = classes.index(row['类别1'])
        j = classes.index(row['类别2'])
        separability_matrix[i, j] = row['JM距离']
        separability_matrix[j, i] = row['JM距离']
    
    im = ax3.imshow(separability_matrix, cmap='RdYlBu_r', aspect='auto')
    ax3.set_title('Inter-class JM Distance Heatmap', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(classes)))
    ax3.set_yticks(range(len(classes)))
    ax3.set_xticklabels(classes, rotation=45)
    ax3.set_yticklabels(classes)
    
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('JM Distance')
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            if separability_matrix[i, j] > 0:
                text = ax3.text(j, i, f'{separability_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    # 子图4：可分离度等级统计
    ax4 = axes[1, 1]
    def classify_separability(jm_distance):
        if jm_distance < 1.0:
            return 'Poor (<1.0)'
        elif jm_distance < 1.5:
            return 'Fair (1.0-1.5)'
        elif jm_distance < 1.8:
            return 'Good (1.5-1.8)'
        else:
            return 'Excellent (≥1.8)'
    
    df_results['可分离度等级'] = df_results['JM距离'].apply(classify_separability)
    separability_counts = df_results['可分离度等级'].value_counts()
    
    colors = ['red', 'orange', 'lightgreen', 'green']
    wedges, texts, autotexts = ax4.pie(separability_counts.values, labels=separability_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title('Separability Level Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"1.样本可分离度分析图_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"可分离度可视化图表已保存到: {plot_file}")
    plt.show()

def create_deviation_visualizations(deviation_results, output_dir, timestamp):
    """
    创建偏离度分析的可视化图表
    """
    df = pd.DataFrame(deviation_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sample Point Deviation Analysis Report', fontsize=16, fontweight='bold')
    
    # 子图1：马氏距离分布直方图
    ax1 = axes[0, 0]
    ax1.hist(df['mahalanobis_distance'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Mahalanobis Distance Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Mahalanobis Distance')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['mahalanobis_distance'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["mahalanobis_distance"].mean():.3f}')
    ax1.legend()
    
    # 子图2：各类别偏离度等级分布
    ax2 = axes[0, 1]
    deviation_counts = df.groupby(['landcover', 'deviation_level']).size().unstack(fill_value=0)
    deviation_counts.plot(kind='bar', ax=ax2, stacked=True)
    ax2.set_title('Deviation Level Distribution by Class', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Land Cover Class')
    ax2.set_ylabel('Sample Count')
    ax2.legend(title='Deviation Level')
    ax2.tick_params(axis='x', rotation=45)
    
    # 子图3：偏离度等级饼图
    ax3 = axes[1, 0]
    level_counts = df['deviation_level'].value_counts()
    colors = ['green', 'yellow', 'orange', 'red']
    wedges, texts, autotexts = ax3.pie(level_counts.values, labels=level_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Overall Deviation Level Distribution', fontsize=12, fontweight='bold')
    
    # 子图4：各类别马氏距离箱线图
    ax4 = axes[1, 1]
    df.boxplot(column='mahalanobis_distance', by='landcover', ax=ax4)
    ax4.set_title('Mahalanobis Distance by Class', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Land Cover Class')
    ax4.set_ylabel('Mahalanobis Distance')
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"2.样本点偏离度分析图_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"偏离度可视化图表已保存到: {plot_file}")
    plt.show()

def create_filtering_visualizations(original_results, filtered_results, removal_stats, output_dir, timestamp):
    """
    创建筛选对比的可视化图表
    """
    original_df = pd.DataFrame(original_results)
    filtered_df = pd.DataFrame(filtered_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sample Filtering Comparison Report', fontsize=16, fontweight='bold')
    
    # 子图1：筛选前后偏离度等级分布对比
    ax1 = axes[0, 0]
    
    original_counts = original_df['deviation_level'].value_counts()
    filtered_counts = filtered_df['deviation_level'].value_counts()
    
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
    
    original_class_counts = original_df['landcover'].value_counts().sort_index()
    filtered_class_counts = filtered_df['landcover'].value_counts().sort_index()
    
    classes = sorted(original_df['landcover'].unique())
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
    
    ax4.hist(original_df['mahalanobis_distance'], bins=50, alpha=0.5, label='筛选前', density=True)
    ax4.hist(filtered_df['mahalanobis_distance'], bins=50, alpha=0.5, label='筛选后', density=True)
    ax4.set_title('Mahalanobis Distance Distribution Comparison', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Mahalanobis Distance')
    ax4.set_ylabel('Density')
    ax4.legend()
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"3.样本点筛选对比分析_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"筛选对比可视化图表已保存到: {plot_file}")
    plt.show()

def print_comprehensive_summary(separability_results, separability_stats, deviation_results, filtered_results, removal_stats):
    """
    打印完整的分析汇总报告
    """
    print("\n" + "="*80)
    print("完整样本点分析汇总报告")
    print("="*80)
    
    # 基本信息
    print(f"\n【基本信息】")
    print(f"  输入文件: {INPUT_SHAPEFILE}")
    print(f"  类别字段: {CLASS_FIELD}")
    print(f"  波段数量: {len(BANDS)}")
    print(f"  总样本数: {len(deviation_results)}")
    print(f"  类别数量: {len(separability_stats)}")
    
    # 可分离度分析结果
    print(f"\n【可分离度分析结果】")
    df_sep = pd.DataFrame(separability_results, columns=['类别1', '类别2', '类别1样本数', '类别2样本数', 'Bhattacharyya距离', 'JM距离'])
    print(f"  类别对数量: {len(df_sep)}")
    print(f"  JM距离范围: {df_sep['JM距离'].min():.3f} - {df_sep['JM距离'].max():.3f}")
    print(f"  JM距离平均值: {df_sep['JM距离'].mean():.3f}")
    
    # 偏离度分析结果
    print(f"\n【偏离度分析结果】")
    df_dev = pd.DataFrame(deviation_results)
    level_counts = df_dev['deviation_level'].value_counts()
    total_samples = len(df_dev)
    for level, count in level_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {level}: {count} 个样本 ({percentage:.1f}%)")
    
    # 筛选结果
    print(f"\n【样本筛选结果】")
    print(f"  筛选前样本数: {len(deviation_results)}")
    print(f"  筛选后样本数: {len(filtered_results)}")
    print(f"  删除样本数: {len(deviation_results) - len(filtered_results)}")
    print(f"  保留比例: {len(filtered_results)/len(deviation_results)*100:.1f}%")
    
    for level, stats in removal_stats.items():
        if stats['total'] > 0:
            retention_rate = (stats['total'] - stats['removed']) / stats['total'] * 100
            print(f"  {level}: 删除 {stats['removed']}/{stats['total']} ({stats['removed']/stats['total']*100:.1f}%), 保留率 {retention_rate:.1f}%")
    
    print("="*80)

# ==================== 主函数 ====================

def main():
    """
    主函数：执行完整的样本点分析流程
    """
    print("="*80)
    print("开始完整的样本点分析流程")
    print("="*80)
    print(f"输入文件: {INPUT_SHAPEFILE}")
    print(f"输出目录: {OUTPUT_BASE_DIR}")
    print(f"类别字段: {CLASS_FIELD}")
    print(f"波段数量: {len(BANDS)}")
    print(f"High等级删除比例: {HIGH_REMOVAL_RATIO*100:.0f}%")
    print(f"随机种子: {RANDOM_SEED}")
    
    try:
        # 创建输出目录
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ==================== 第一步：可分离度计算 ====================
        print(f"\n{'='*20} 第一步：可分离度计算 {'='*20}")
        
        print("正在读取矢量数据...")
        features = read_features(INPUT_SHAPEFILE, LAYER_NAME)
        print(f"成功读取 {len(features)} 个要素")
        
        print("正在按类别分组数据...")
        groups = group_by_class(features, CLASS_FIELD, BANDS)
        print(f"发现 {len(groups)} 个类别")
        
        print("正在计算可分离度...")
        separability_results, separability_stats = compute_pairwise_separability(groups)
        print(f"计算了 {len(separability_results)} 个类别对的可分离度")
        
        # 保存可分离度结果
        df_results = pd.DataFrame(separability_results, columns=['类别1', '类别2', '类别1样本数', '类别2样本数', 'Bhattacharyya距离', 'JM距离'])
        results_file = os.path.join(OUTPUT_BASE_DIR, f"1.可分离度结果_{timestamp}.csv")
        df_results.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"可分离度结果已保存到: {results_file}")
        
        # 保存类别统计信息
        stats_data = []
        for cls, stat in separability_stats.items():
            stats_data.append({
                '类别': cls,
                '样本数量': stat['n'],
                '均值_最小值': np.min(stat['mu']),
                '均值_最大值': np.max(stat['mu']),
                '均值_平均值': np.mean(stat['mu']),
                '协方差矩阵_迹': np.trace(stat['cov']),
                '协方差矩阵_行列式': np.linalg.det(stat['cov'])
            })
        
        df_stats = pd.DataFrame(stats_data)
        stats_file = os.path.join(OUTPUT_BASE_DIR, f"1.类别统计信息_{timestamp}.csv")
        df_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"类别统计信息已保存到: {stats_file}")
        
        # 生成可分离度可视化图表
        create_separability_visualizations(df_results, df_stats, OUTPUT_BASE_DIR, timestamp)
        
        # ==================== 第二步：偏离度分析 ====================
        print(f"\n{'='*20} 第二步：偏离度分析 {'='*20}")
        
        print("正在读取矢量数据（包含几何信息）...")
        features_with_geom, geometries, spatial_ref = read_features_with_geometry(INPUT_SHAPEFILE, LAYER_NAME)
        print(f"成功读取 {len(features_with_geom)} 个要素")
        
        print("正在按类别分组数据（保留索引）...")
        groups_with_indices, indices = group_by_class_with_indices(features_with_geom, CLASS_FIELD, BANDS)
        print(f"发现 {len(groups_with_indices)} 个类别")
        
        print("正在计算样本点偏离度...")
        deviation_results = calculate_mahalanobis_distances(groups_with_indices, indices, features_with_geom, BANDS)
        print(f"完成 {len(deviation_results)} 个样本点的偏离度计算")
        
        # 保存偏离度分析结果
        deviation_shapefile = os.path.join(OUTPUT_BASE_DIR, f"2.样本点偏离度分析_{timestamp}.shp")
        create_deviation_shapefile(deviation_results, deviation_shapefile, spatial_ref)
        
        deviation_csv = os.path.join(OUTPUT_BASE_DIR, f"2.样本点偏离度分析_{timestamp}.csv")
        df_deviation = pd.DataFrame(deviation_results)
        df_deviation.to_csv(deviation_csv, index=False, encoding='utf-8-sig')
        print(f"偏离度分析CSV已保存到: {deviation_csv}")
        
        # 生成偏离度可视化图表
        create_deviation_visualizations(deviation_results, OUTPUT_BASE_DIR, timestamp)
        
        # ==================== 第三步：样本点筛选 ====================
        print(f"\n{'='*20} 第三步：样本点筛选 {'='*20}")
        
        filtered_results, removal_stats = filter_samples_by_deviation(
            deviation_results, 
            high_removal_ratio=HIGH_REMOVAL_RATIO, 
            random_seed=RANDOM_SEED
        )
        
        # 保存筛选后的样本点
        filtered_shapefile = os.path.join(OUTPUT_BASE_DIR, f"3.筛选后样本点_{timestamp}.shp")
        save_filtered_samples_shapefile(filtered_results, filtered_shapefile, spatial_ref)
        
        filtered_csv = os.path.join(OUTPUT_BASE_DIR, f"3.筛选后样本点_{timestamp}.csv")
        df_filtered = pd.DataFrame(filtered_results)
        df_filtered.to_csv(filtered_csv, index=False, encoding='utf-8-sig')
        print(f"筛选后样本点CSV已保存到: {filtered_csv}")
        
        # 保存筛选统计对比
        comparison_stats = []
        for level in ['Normal', 'Moderate', 'High', 'Extreme']:
            original_count = len([r for r in deviation_results if r['deviation_level'] == level])
            filtered_count = len([r for r in filtered_results if r['deviation_level'] == level])
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
        comparison_csv = os.path.join(OUTPUT_BASE_DIR, f"3.筛选统计对比_{timestamp}.csv")
        comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8-sig')
        print(f"筛选统计对比已保存到: {comparison_csv}")
        
        # 生成筛选对比可视化图表
        create_filtering_visualizations(deviation_results, filtered_results, removal_stats, OUTPUT_BASE_DIR, timestamp)
        
        # ==================== 生成完整汇总报告 ====================
        print_comprehensive_summary(separability_results, separability_stats, deviation_results, filtered_results, removal_stats)
        
        print(f"\n{'='*20} 分析完成 {'='*20}")
        print(f"所有结果已保存到: {OUTPUT_BASE_DIR}")
        print(f"\n输出文件列表:")
        print(f"  1. 可分离度结果: 1.可分离度结果_{timestamp}.csv")
        print(f"  2. 类别统计信息: 1.类别统计信息_{timestamp}.csv")
        print(f"  3. 可分离度可视化: 1.样本可分离度分析图_{timestamp}.png")
        print(f"  4. 偏离度分析Shapefile: 2.样本点偏离度分析_{timestamp}.shp")
        print(f"  5. 偏离度分析CSV: 2.样本点偏离度分析_{timestamp}.csv")
        print(f"  6. 偏离度可视化: 2.样本点偏离度分析图_{timestamp}.png")
        print(f"  7. 筛选后样本Shapefile: 3.筛选后样本点_{timestamp}.shp")
        print(f"  8. 筛选后样本CSV: 3.筛选后样本点_{timestamp}.csv")
        print(f"  9. 筛选统计对比: 3.筛选统计对比_{timestamp}.csv")
        print(f"  10. 筛选对比可视化: 3.样本点筛选对比分析_{timestamp}.png")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()