#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本点偏离度分析脚本
功能：计算每个样本点相对于其类别群体的偏离程度（马氏距离），并输出到矢量文件中
作者：锐多宝 (ruiduobao)
日期：2025年10月21日
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
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
VECTOR_PATH = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果\Zone5_2017_Merged_AllBand_Sample (1).shp"   # 替换为你的文件
output_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\计算样本点的偏离度分析"
LAYER_NAME = None
class_field = "landcover"
bands = [f"A{i:02d}" for i in range(0, 64)]  # 波段A00到A63

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
            # 获取点坐标
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
    
    # 转换为numpy数组
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
        
        # 计算类别的均值和协方差矩阵
        mean_vec = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False)
        
        # 添加正则化项防止奇异矩阵
        reg_term = 1e-6
        cov_matrix += reg_term * np.eye(cov_matrix.shape[0])
        
        try:
            # 计算协方差矩阵的逆
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            print(f"警告：类别 {cls} 的协方差矩阵奇异，使用伪逆")
            cov_inv = np.linalg.pinv(cov_matrix)
        
        # 计算每个点的马氏距离
        class_indices = indices[cls]
        for i, idx in enumerate(class_indices):
            sample_vec = data[i]
            
            # 计算马氏距离
            diff = sample_vec - mean_vec
            mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)
            
            # 计算p值（基于卡方分布）
            chi2_stat = mahal_dist ** 2
            p_value = 1 - chi2.cdf(chi2_stat, df=len(bands))
            
            # 分类偏离程度
            if p_value > 0.05:
                deviation_level = "Normal"
            elif p_value > 0.01:
                deviation_level = "Moderate"
            elif p_value > 0.001:
                deviation_level = "High"
            else:
                deviation_level = "Extreme"
            
            # 获取原始要素信息
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
            
            # 添加原始属性
            for key, value in original_feature.items():
                if key not in result:
                    result[key] = value
            
            deviation_results.append(result)
    
    return deviation_results

def create_output_shapefile(deviation_results, output_path, spatial_ref):
    """
    创建包含偏离度信息的输出shapefile
    参数：
        deviation_results: 偏离度分析结果
        output_path: 输出文件路径
        spatial_ref: 空间参考系统
    """
    print(f"正在创建输出shapefile: {output_path}")
    
    # 创建输出数据源
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    
    out_ds = driver.CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer("deviation_analysis", spatial_ref, ogr.wkbPoint)
    
    # 创建字段定义
    fields = [
        ("landcover", ogr.OFTReal),
        ("mahal_dist", ogr.OFTReal),
        ("chi2_stat", ogr.OFTReal),
        ("p_value", ogr.OFTReal),
        ("dev_level", ogr.OFTString)
    ]
    
    # 添加波段字段
    for band in bands:
        fields.append((band, ogr.OFTReal))
    
    # 创建字段
    for field_name, field_type in fields:
        field_def = ogr.FieldDefn(field_name, field_type)
        if field_type == ogr.OFTString:
            field_def.SetWidth(20)
        out_layer.CreateField(field_def)
    
    # 添加要素
    for result in deviation_results:
        # 创建要素
        feature = ogr.Feature(out_layer.GetLayerDefn())
        
        # 设置几何
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(result['geometry_x'], result['geometry_y'])
        feature.SetGeometry(point)
        
        # 设置属性
        feature.SetField("landcover", float(result['landcover']))
        feature.SetField("mahal_dist", result['mahalanobis_distance'])
        feature.SetField("chi2_stat", result['chi2_statistic'])
        feature.SetField("p_value", result['p_value'])
        feature.SetField("dev_level", result['deviation_level'])
        
        # 设置波段值
        for band in bands:
            if band in result:
                feature.SetField(band, float(result[band]))
        
        # 添加要素到图层
        out_layer.CreateFeature(feature)
        feature = None
    
    out_ds = None
    print(f"成功创建shapefile，包含 {len(deviation_results)} 个要素")

def create_deviation_visualizations(deviation_results, output_dir, timestamp):
    """
    创建偏离度分析的可视化图表
    参数：
        deviation_results: 偏离度分析结果
        output_dir: 输出目录
        timestamp: 时间戳
    """
    print("正在生成偏离度可视化图表...")
    
    # 转换为DataFrame
    df = pd.DataFrame(deviation_results)
    
    # 创建图表
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
    
    # 保存图表
    plot_file = os.path.join(output_dir, f"样本点偏离度分析图_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {plot_file}")
    plt.show()
    
    # 创建空间分布图
    fig2, ax = plt.subplots(figsize=(12, 10))
    
    # 根据偏离度等级设置颜色
    color_map = {'Normal': 'green', 'Moderate': 'yellow', 'High': 'orange', 'Extreme': 'red'}
    colors = [color_map[level] for level in df['deviation_level']]
    
    scatter = ax.scatter(df['geometry_x'], df['geometry_y'], 
                        c=colors, s=df['mahalanobis_distance']*10, 
                        alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_title('Spatial Distribution of Sample Point Deviations', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=level) 
                      for level, color in color_map.items()]
    ax.legend(handles=legend_elements, title='Deviation Level', loc='upper right')
    
    plt.tight_layout()
    
    # 保存空间分布图
    spatial_plot_file = os.path.join(output_dir, f"样本点空间偏离度分布图_{timestamp}.png")
    plt.savefig(spatial_plot_file, dpi=300, bbox_inches='tight')
    print(f"空间分布图已保存到: {spatial_plot_file}")
    plt.show()

def print_deviation_summary(deviation_results):
    """
    打印偏离度分析汇总统计
    参数：
        deviation_results: 偏离度分析结果
    """
    df = pd.DataFrame(deviation_results)
    
    print("\n" + "="*60)
    print("样本点偏离度分析汇总报告")
    print("="*60)
    
    print(f"\n1. 基本信息:")
    print(f"   - 总样本数: {len(df)}")
    print(f"   - 类别数: {df['landcover'].nunique()}")
    print(f"   - 波段数: {len(bands)}")
    
    print(f"\n2. 各类别样本分布:")
    class_counts = df['landcover'].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"   - 类别 {cls}: {count} 个样本")
    
    print(f"\n3. 马氏距离统计:")
    print(f"   - 最小值: {df['mahalanobis_distance'].min():.3f}")
    print(f"   - 最大值: {df['mahalanobis_distance'].max():.3f}")
    print(f"   - 平均值: {df['mahalanobis_distance'].mean():.3f}")
    print(f"   - 标准差: {df['mahalanobis_distance'].std():.3f}")
    
    print(f"\n4. 偏离度等级分布:")
    level_counts = df['deviation_level'].value_counts()
    total_samples = len(df)
    for level, count in level_counts.items():
        percentage = (count / total_samples) * 100
        print(f"   - {level}: {count} 个样本 ({percentage:.1f}%)")
    
    print(f"\n5. 各类别偏离度统计:")
    for cls in sorted(df['landcover'].unique()):
        class_data = df[df['landcover'] == cls]
        print(f"   - 类别 {cls}:")
        print(f"     * 平均马氏距离: {class_data['mahalanobis_distance'].mean():.3f}")
        print(f"     * 异常样本数 (p<0.05): {len(class_data[class_data['p_value'] < 0.05])}")
        
        # 找出最偏离的样本
        max_deviation_idx = class_data['mahalanobis_distance'].idxmax()
        max_deviation_sample = class_data.loc[max_deviation_idx]
        print(f"     * 最大偏离样本: 马氏距离={max_deviation_sample['mahalanobis_distance']:.3f}, "
              f"等级={max_deviation_sample['deviation_level']}")
    
    print("="*60)

def main():
    """
    主函数：执行样本点偏离度分析
    """
    print("开始样本点偏离度分析...")
    print(f"输入文件: {VECTOR_PATH}")
    print(f"类别字段: {class_field}")
    print(f"波段数量: {len(bands)}")
    
    try:
        # 读取矢量数据
        print("\n正在读取矢量数据...")
        features, geometries, spatial_ref = read_features_with_geometry(VECTOR_PATH, LAYER_NAME)
        print(f"成功读取 {len(features)} 个要素")
        
        # 按类别分组数据
        print("正在按类别分组数据...")
        groups, indices = group_by_class_with_indices(features, class_field, bands)
        print(f"发现 {len(groups)} 个类别")
        
        # 计算偏离度
        print("正在计算样本点偏离度...")
        deviation_results = calculate_mahalanobis_distances(groups, indices, features, bands)
        print(f"完成 {len(deviation_results)} 个样本点的偏离度计算")
        
        # 创建输出目录

        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 输出shapefile
        print("\n正在输出结果shapefile...")
        output_shapefile = os.path.join(output_dir, f"样本点偏离度分析_{timestamp}.shp")
        create_output_shapefile(deviation_results, output_shapefile, spatial_ref)
        
        # 保存CSV结果
        print("正在保存CSV结果...")
        df_results = pd.DataFrame(deviation_results)
        csv_file = os.path.join(output_dir, f"样本点偏离度分析_{timestamp}.csv")
        df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"CSV结果已保存到: {csv_file}")
        
        # 生成可视化图表
        print("正在生成可视化图表...")
        create_deviation_visualizations(deviation_results, output_dir, timestamp)
        
        # 打印汇总统计
        print_deviation_summary(deviation_results)
        
        print(f"\n分析完成！所有结果已保存到: {output_dir}")
        print(f"输出文件包括:")
        print(f"  - Shapefile: {output_shapefile}")
        print(f"  - CSV文件: {csv_file}")
        print(f"  - 可视化图表: 样本点偏离度分析图_{timestamp}.png")
        print(f"  - 空间分布图: 样本点空间偏离度分布图_{timestamp}.png")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()