#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理矢量文件筛选脚本 - 多进程版本
功能：
1. 使用多进程并行处理指定目录中的所有shapefile
2. 对每个文件进行样本点偏离度分析和筛选
3. 只输出筛选后的矢量文件，不生成分析图表
4. 显著提升处理速度

基于原始脚本：1.总.计算可分离度偏离度并筛选样本_只输出矢量.py
作者：锐多宝 (ruiduobao)
日期：2025年1月21日
版本：多进程优化版
"""

from osgeo import ogr, osr
import numpy as np
import pandas as pd
from datetime import datetime
import os
import glob
from scipy.stats import chi2
import random
import warnings
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time
from tqdm import tqdm
import sys

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 输入目录
INPUT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并"

# 输出目录
OUTPUT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并_剔除异常样本"

# 字段配置
LAYER_NAME = None
CLASS_FIELD = "landcover"
BANDS = [f"A{i:02d}" for i in range(0, 64)]  # 波段A00到A63

# 筛选参数
HIGH_REMOVAL_RATIO = 0.7  # 删除70%的High等级样本点
RANDOM_SEED = 42  # 随机种子，确保结果可重现

# 数值稳定性参数
EPS = 1e-6  # 协方差正则化项

# 多进程参数
MAX_WORKERS = mp.cpu_count() -  20 # 使用CPU核心数-1个进程
CHUNK_SIZE = 1  # 每个进程处理的文件数量

# ==================== 核心函数 ====================

def read_features_with_geometry(path, layer_name=None):
    """
    读取矢量文件中的要素属性和几何信息
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
    """
    deviation_results = []
    
    for cls, data in groups.items():
        mean_vec = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False)
        
        reg_term = 1e-6
        cov_matrix += reg_term * np.eye(cov_matrix.shape[0])
        
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
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

def filter_samples_by_deviation(deviation_results, high_removal_ratio=0.7, random_seed=42):
    """
    根据偏离度等级筛选样本点
    """
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
    
    # 随机删除部分 High 等级样本点
    high_samples = filtered_df[filtered_df['deviation_level'] == 'High']
    high_count = len(high_samples)
    if high_count > 0:
        remove_count = int(high_count * high_removal_ratio)
        remove_indices = np.random.choice(high_samples.index, size=remove_count, replace=False)
        filtered_df = filtered_df.drop(remove_indices)
        removal_stats['High']['removed'] = remove_count
    
    return filtered_df.to_dict('records'), removal_stats

def save_filtered_samples_shapefile(filtered_results, output_path, spatial_ref):
    """
    保存筛选后的样本点到shapefile
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    
    out_ds = driver.CreateDataSource(output_path)
    out_layer = out_ds.CreateLayer("filtered_samples", spatial_ref, ogr.wkbPoint)
    
    # 创建字段
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
    
    # 写入要素
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

def process_single_shapefile_worker(args):
    """
    处理单个shapefile文件的工作函数（用于多进程）
    """
    input_path, output_path, progress_queue = args
    
    try:
        # 读取数据
        features, geometries, spatial_ref = read_features_with_geometry(input_path, LAYER_NAME)
        
        # 检查是否包含必要的字段
        if not features:
            progress_queue.put(1)
            return {
                'input_file': os.path.basename(input_path),
                'original_count': 0,
                'filtered_count': 0,
                'retention_rate': 0,
                'original_class_stats': {},
                'filtered_class_stats': {},
                'deviation_level_stats': {},
                'removal_stats': {},
                'status': 'Failed: 文件为空'
            }
        
        # 检查是否包含landcover字段
        if CLASS_FIELD not in features[0]:
            progress_queue.put(1)
            return {
                'input_file': os.path.basename(input_path),
                'original_count': 0,
                'filtered_count': 0,
                'retention_rate': 0,
                'original_class_stats': {},
                'filtered_class_stats': {},
                'deviation_level_stats': {},
                'removal_stats': {},
                'status': f'Failed: 缺少 {CLASS_FIELD} 字段'
            }
        
        # 检查是否包含波段字段
        missing_bands = [band for band in BANDS if band not in features[0]]
        if missing_bands:
            progress_queue.put(1)
            return {
                'input_file': os.path.basename(input_path),
                'original_count': 0,
                'filtered_count': 0,
                'retention_rate': 0,
                'original_class_stats': {},
                'filtered_class_stats': {},
                'deviation_level_stats': {},
                'removal_stats': {},
                'status': f'Failed: 缺少波段字段'
            }
        
        # 按类别分组
        groups, indices = group_by_class_with_indices(features, CLASS_FIELD, BANDS)
        
        # 统计原始各类别样本数量
        original_class_stats = {}
        for cls, data in groups.items():
            original_class_stats[cls] = len(data)
        
        # 计算马氏距离
        deviation_results = calculate_mahalanobis_distances(groups, indices, features, BANDS)
        
        # 筛选样本点
        filtered_results, removal_stats = filter_samples_by_deviation(
            deviation_results, HIGH_REMOVAL_RATIO, RANDOM_SEED
        )
        
        # 统计筛选后各类别样本数量
        filtered_class_stats = {}
        for result in filtered_results:
            cls = result['landcover']
            filtered_class_stats[cls] = filtered_class_stats.get(cls, 0) + 1
        
        # 统计各偏离度等级的详细信息
        deviation_level_stats = {}
        for result in deviation_results:
            level = result['deviation_level']
            cls = result['landcover']
            
            if level not in deviation_level_stats:
                deviation_level_stats[level] = {'total': 0, 'by_class': {}}
            
            deviation_level_stats[level]['total'] += 1
            deviation_level_stats[level]['by_class'][cls] = deviation_level_stats[level]['by_class'].get(cls, 0) + 1
        
        # 保存结果
        save_filtered_samples_shapefile(filtered_results, output_path, spatial_ref)
        
        # 更新进度
        progress_queue.put(1)
        
        return {
            'input_file': os.path.basename(input_path),
            'original_count': len(features),
            'filtered_count': len(filtered_results),
            'retention_rate': len(filtered_results) / len(features) * 100,
            'original_class_stats': original_class_stats,
            'filtered_class_stats': filtered_class_stats,
            'deviation_level_stats': deviation_level_stats,
            'removal_stats': removal_stats,
            'status': 'Success'
        }
        
    except Exception as e:
        progress_queue.put(1)
        return {
            'input_file': os.path.basename(input_path),
            'original_count': 0,
            'filtered_count': 0,
            'retention_rate': 0,
            'original_class_stats': {},
            'filtered_class_stats': {},
            'deviation_level_stats': {},
            'removal_stats': {},
            'status': f'Failed: {str(e)}'
        }

def get_shapefile_list(input_dir):
    """
    获取输入目录中的所有shapefile
    """
    pattern = os.path.join(input_dir, "*.shp")
    shp_files = glob.glob(pattern)
    return shp_files

def create_output_directory(output_dir):
    """
    创建输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")

def update_progress_bar(pbar, progress_queue, total_files):
    """
    更新进度条的函数
    """
    completed = 0
    while completed < total_files:
        try:
            progress_queue.get(timeout=1)
            completed += 1
            pbar.update(1)
        except:
            continue

def main():
    """
    主函数 - 多进程版本
    """
    print("=" * 80)
    print("批量处理矢量文件筛选脚本 - 多进程版本")
    print("=" * 80)
    
    # 创建输出目录
    create_output_directory(OUTPUT_DIR)
    
    # 获取所有shapefile列表
    print(f"\n正在扫描输入目录: {INPUT_DIR}")
    shp_files = get_shapefile_list(INPUT_DIR)
    
    if not shp_files:
        print("错误：未找到shapefile文件")
        return
    
    print(f"找到 {len(shp_files)} 个shapefile文件")
    print(f"使用 {MAX_WORKERS} 个进程并行处理")
    
    # 准备任务参数
    tasks = []
    manager = Manager()
    progress_queue = manager.Queue()
    
    for input_path in shp_files:
        input_filename = os.path.basename(input_path)
        output_filename = os.path.splitext(input_filename)[0] + "_filtered.shp"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        tasks.append((input_path, output_path, progress_queue))
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建进度条
    print(f"\n开始多进程处理...")
    with tqdm(total=len(shp_files), desc="处理进度", unit="文件") as pbar:
        # 启动多进程池
        with Pool(processes=MAX_WORKERS) as pool:
            # 异步提交所有任务
            result = pool.map_async(process_single_shapefile_worker, tasks, chunksize=CHUNK_SIZE)
            
            # 监控进度
            completed = 0
            while not result.ready():
                try:
                    progress_queue.get(timeout=0.1)
                    completed += 1
                    pbar.update(1)
                except:
                    continue
            
            # 获取所有结果
            process_stats = result.get()
            
            # 更新剩余进度
            while completed < len(shp_files):
                try:
                    progress_queue.get(timeout=0.1)
                    completed += 1
                    pbar.update(1)
                except:
                    break
    
    # 计算处理时间
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 生成处理统计报告
    print("\n" + "=" * 80)
    print("批量处理统计报告")
    print("=" * 80)
    
    if process_stats:
        df_stats = pd.DataFrame(process_stats)
        
        # 显示统计信息
        print(f"\n处理文件总数: {len(process_stats)}")
        success_count = len(df_stats[df_stats['status'] == 'Success'])
        print(f"成功处理: {success_count}")
        print(f"处理失败: {len(process_stats) - success_count}")
        print(f"总处理时间: {processing_time:.2f} 秒")
        print(f"平均每文件处理时间: {processing_time/len(process_stats):.2f} 秒")
        print(f"处理速度: {len(process_stats)/processing_time:.2f} 文件/秒")
        
        if success_count > 0:
            successful_df = df_stats[df_stats['status'] == 'Success']
            total_original = successful_df['original_count'].sum()
            total_filtered = successful_df['filtered_count'].sum()
            print(f"\n样本点统计:")
            print(f"  原始样本总数: {total_original:,}")
            print(f"  筛选后总数: {total_filtered:,}")
            print(f"  总体保留率: {total_filtered/total_original*100:.1f}%")
        
        # 保存简要统计报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(OUTPUT_DIR, f"批量处理统计报告_多进程_{timestamp}.csv")
        df_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"\n简要统计报告已保存到: {stats_file}")
        
        # 生成详细统计表
        detailed_stats = []
        for stat in process_stats:
            if stat['status'] == 'Success':
                base_info = {
                    '文件名': stat['input_file'],
                    '原始样本总数': stat['original_count'],
                    '筛选后总数': stat['filtered_count'],
                    '保留率(%)': f"{stat['retention_rate']:.1f}"
                }
                
                # 添加原始各类别统计
                for cls, count in stat['original_class_stats'].items():
                    base_info[f'原始类别{cls}数量'] = count
                
                # 添加筛选后各类别统计
                for cls, count in stat['filtered_class_stats'].items():
                    base_info[f'筛选后类别{cls}数量'] = count
                    # 计算各类别保留率
                    original_count = stat['original_class_stats'].get(cls, 0)
                    if original_count > 0:
                        retention_rate = count / original_count * 100
                        base_info[f'类别{cls}保留率(%)'] = f"{retention_rate:.1f}"
                
                # 添加偏离度等级统计
                for level, level_stats in stat['deviation_level_stats'].items():
                    base_info[f'{level}等级总数'] = level_stats['total']
                    for cls, count in level_stats['by_class'].items():
                        base_info[f'{level}等级类别{cls}数量'] = count
                
                # 添加删除统计
                for level, removal_info in stat['removal_stats'].items():
                    if removal_info['removed'] > 0:
                        base_info[f'删除{level}等级数量'] = removal_info['removed']
                        base_info[f'{level}等级删除率(%)'] = f"{removal_info['removed']/removal_info['total']*100:.1f}" if removal_info['total'] > 0 else "0.0"
                
                detailed_stats.append(base_info)
        
        # 保存详细统计表
        if detailed_stats:
            detailed_df = pd.DataFrame(detailed_stats)
            detailed_file = os.path.join(OUTPUT_DIR, f"详细处理统计表_多进程_{timestamp}.csv")
            detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
            print(f"详细统计表已保存到: {detailed_file}")
    
    print(f"\n多进程批量处理完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"性能提升: 使用 {MAX_WORKERS} 个进程并行处理")

if __name__ == "__main__":
    # 在Windows上需要这个保护
    mp.freeze_support()
    main()