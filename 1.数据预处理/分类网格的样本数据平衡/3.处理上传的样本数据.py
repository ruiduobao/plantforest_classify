#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本数据平衡处理脚本
目的：对各个zone的样本数据进行平衡处理，确保传统机器学习算法的分类效果
作者：锐多宝 (ruiduobao)
创建时间：2025年10月14日

处理策略：
- Zone 1,2,3,4,5,10: 将非林地和自然林样本随机筛选到与人工林相同数量
- Zone 6: 全部样本随机筛选到5万个（每类约1.67万个）
- Zone 7,8,9: 人工林从SDPT数据补充到1万个，其他类别随机筛选到1万个
"""

import os
import sys
import time
import random
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

def setup_logging(output_dir):
    """
    设置日志记录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"sample_balancing_log_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def load_zone_boundary(zone_folder, zone_name):
    """
    加载zone边界数据
    """
    try:
        # 检查不同的命名格式
        if zone_name in ['6', '7', '8', '9', '10']:
            boundary_file = os.path.join(zone_folder, f"BOUNDARY_ZONE_{zone_name}.shp")
        else:
            boundary_file = os.path.join(zone_folder, f"zone_{zone_name}_boundary.shp")
        
        if os.path.exists(boundary_file):
            boundary_gdf = gpd.read_file(boundary_file)
            logging.info(f"成功加载Zone {zone_name}边界数据: {len(boundary_gdf)} 个要素")
            return boundary_gdf
        else:
            logging.error(f"Zone {zone_name}边界文件不存在: {boundary_file}")
            return None
    except Exception as e:
        logging.error(f"加载Zone {zone_name}边界数据失败: {str(e)}")
        return None

def load_sample_points(zone_folder, zone_name, sample_type):
    """
    加载样本点数据
    """
    try:
        # 根据zone编号确定文件命名格式
        if zone_name in ['6', '7', '8', '9', '10']:
            if sample_type == 'plantation':
                file_pattern = f"POINTS_PLANTATION_ZONE_{zone_name}.shp"
            elif sample_type == 'nonforest':
                file_pattern = f"POINTS_NONFOREST_ZONE_{zone_name}.shp"
            elif sample_type == 'natural_forest':
                file_pattern = f"POINTS_NATURAL_FOREST_ZONE_{zone_name}.shp"
        else:
            if sample_type == 'plantation':
                file_pattern = f"plantation_points_zone_{zone_name}.shp"
            elif sample_type == 'nonforest':
                file_pattern = f"nonforest_points_zone_{zone_name}.shp"
            elif sample_type == 'natural_forest':
                file_pattern = f"natural_forest_points_zone_{zone_name}.shp"
        
        file_path = os.path.join(zone_folder, file_pattern)
        
        if os.path.exists(file_path):
            gdf = gpd.read_file(file_path)
            logging.info(f"Zone {zone_name} {sample_type}样本点: {len(gdf)} 个")
            return gdf
        else:
            logging.warning(f"文件不存在: {file_path}")
            return gpd.GeoDataFrame()
    except Exception as e:
        logging.error(f"加载Zone {zone_name} {sample_type}样本点失败: {str(e)}")
        return gpd.GeoDataFrame()

def load_sdpt_data(sdpt_file_path):
    """
    加载SDPT补充样本数据
    """
    try:
        sdpt_gdf = gpd.read_file(sdpt_file_path)
        logging.info(f"成功加载SDPT数据: {len(sdpt_gdf)} 个样本点")
        return sdpt_gdf
    except Exception as e:
        logging.error(f"加载SDPT数据失败: {str(e)}")
        return None

def random_sample_points(gdf, target_count, random_seed=42):
    """
    随机采样点数据
    """
    if len(gdf) == 0:
        return gdf
    
    if len(gdf) <= target_count:
        return gdf
    
    # 设置随机种子确保结果可重现
    np.random.seed(random_seed)
    sampled_indices = np.random.choice(len(gdf), target_count, replace=False)
    return gdf.iloc[sampled_indices].copy()

def supplement_plantation_from_sdpt(zone_boundary, sdpt_gdf, target_count, existing_count):
    """
    从SDPT数据中补充人工林样本点
    """
    try:
        # 需要补充的数量
        supplement_count = target_count - existing_count
        
        if supplement_count <= 0:
            return gpd.GeoDataFrame()
        
        logging.info(f"需要从SDPT补充 {supplement_count} 个人工林样本点")
        
        # 将SDPT数据裁剪到zone边界内
        sdpt_in_zone = gpd.clip(sdpt_gdf, zone_boundary)
        logging.info(f"Zone边界内的SDPT样本点: {len(sdpt_in_zone)} 个")
        
        if len(sdpt_in_zone) == 0:
            logging.warning("Zone边界内没有SDPT样本点可用于补充")
            return gpd.GeoDataFrame()
        
        # 随机选择需要的数量
        if len(sdpt_in_zone) >= supplement_count:
            sampled_sdpt = random_sample_points(sdpt_in_zone, supplement_count)
        else:
            sampled_sdpt = sdpt_in_zone.copy()
            logging.warning(f"SDPT样本点不足，只能补充 {len(sampled_sdpt)} 个")
        
        # 创建新的GeoDataFrame，只保留几何信息和landcover属性
        supplement_gdf = gpd.GeoDataFrame({
            'geometry': sampled_sdpt.geometry,
            'landcover': [1] * len(sampled_sdpt)  # 人工林标记为1
        }, crs=sampled_sdpt.crs)
        
        logging.info(f"成功补充 {len(supplement_gdf)} 个人工林样本点")
        return supplement_gdf
        
    except Exception as e:
        logging.error(f"从SDPT补充样本点失败: {str(e)}")
        return gpd.GeoDataFrame()

def process_zone_samples(zone_info):
    """
    处理单个zone的样本平衡
    """
    zone_name, input_dir, output_dir, sdpt_gdf = zone_info
    
    try:
        logging.info(f"开始处理Zone {zone_name}")
        
        # 创建输出文件夹
        zone_output_dir = os.path.join(output_dir, f"zone_{zone_name}")
        os.makedirs(zone_output_dir, exist_ok=True)
        
        # 加载zone文件夹路径
        zone_input_dir = os.path.join(input_dir, f"zone_{zone_name}")
        
        if not os.path.exists(zone_input_dir):
            logging.error(f"Zone {zone_name}输入文件夹不存在: {zone_input_dir}")
            return False
        
        # 加载边界数据
        zone_boundary = load_zone_boundary(zone_input_dir, zone_name)
        if zone_boundary is None:
            return False
        
        # 加载现有样本点数据
        plantation_gdf = load_sample_points(zone_input_dir, zone_name, 'plantation')
        nonforest_gdf = load_sample_points(zone_input_dir, zone_name, 'nonforest')
        natural_forest_gdf = load_sample_points(zone_input_dir, zone_name, 'natural_forest')
        
        # 统计原始数量
        plantation_count = len(plantation_gdf)
        nonforest_count = len(nonforest_gdf)
        natural_forest_count = len(natural_forest_gdf)
        
        logging.info(f"Zone {zone_name}原始样本数量 - 人工林: {plantation_count}, 非林地: {nonforest_count}, 自然林: {natural_forest_count}")
        
        # 根据zone编号确定处理策略
        if zone_name in ['7', '8', '9']:
            # Zone 7,8,9: 人工林补充到1万，其他类别筛选到1万
            target_plantation = 10000
            target_others = 10000
            
            # 补充人工林样本
            if plantation_count < target_plantation and sdpt_gdf is not None:
                supplement_gdf = supplement_plantation_from_sdpt(
                    zone_boundary, sdpt_gdf, target_plantation, plantation_count
                )
                if len(supplement_gdf) > 0:
                    # 合并现有和补充的样本
                    plantation_gdf_balanced = pd.concat([plantation_gdf, supplement_gdf], ignore_index=True)
                else:
                    plantation_gdf_balanced = plantation_gdf.copy()
            else:
                plantation_gdf_balanced = plantation_gdf.copy()
            
            # 随机筛选其他类别
            nonforest_gdf_balanced = random_sample_points(nonforest_gdf, target_others)
            natural_forest_gdf_balanced = random_sample_points(natural_forest_gdf, target_others)
            
        elif zone_name == '6' or zone_name == '2'or zone_name == '3'or zone_name == '4'or zone_name == '10':
            # Zone 6: 全部缩小到5万个样本点（每类约1.67万）
            total_target = 30000
            per_class_target = total_target // 3
            
            plantation_gdf_balanced = random_sample_points(plantation_gdf, per_class_target)
            nonforest_gdf_balanced = random_sample_points(nonforest_gdf, per_class_target)
            natural_forest_gdf_balanced = random_sample_points(natural_forest_gdf, per_class_target)
            
        else:
            # Zone 1,5: 非林地和自然林与人工林数量一致
            target_count = plantation_count
            
            plantation_gdf_balanced = plantation_gdf.copy()
            nonforest_gdf_balanced = random_sample_points(nonforest_gdf, target_count)
            natural_forest_gdf_balanced = random_sample_points(natural_forest_gdf, target_count)
        
        # 确保所有数据都有landcover属性
        if 'landcover' not in plantation_gdf_balanced.columns:
            plantation_gdf_balanced['landcover'] = 1
        if 'landcover' not in nonforest_gdf_balanced.columns:
            nonforest_gdf_balanced['landcover'] = 2
        if 'landcover' not in natural_forest_gdf_balanced.columns:
            natural_forest_gdf_balanced['landcover'] = 3
        
        # 保存平衡后的样本数据
        if len(plantation_gdf_balanced) > 0:
            plantation_output = os.path.join(zone_output_dir, f"plantation_points_balanced_zone_{zone_name}.shp")
            plantation_gdf_balanced.to_file(plantation_output, encoding='utf-8')
        
        if len(nonforest_gdf_balanced) > 0:
            nonforest_output = os.path.join(zone_output_dir, f"nonforest_points_balanced_zone_{zone_name}.shp")
            nonforest_gdf_balanced.to_file(nonforest_output, encoding='utf-8')
        
        if len(natural_forest_gdf_balanced) > 0:
            natural_forest_output = os.path.join(zone_output_dir, f"natural_forest_points_balanced_zone_{zone_name}.shp")
            natural_forest_gdf_balanced.to_file(natural_forest_output, encoding='utf-8')
        
        # 保存zone边界
        boundary_output = os.path.join(zone_output_dir, f"zone_{zone_name}_boundary.shp")
        zone_boundary.to_file(boundary_output, encoding='utf-8')
        
        # 统计平衡后的数量
        balanced_plantation = len(plantation_gdf_balanced)
        balanced_nonforest = len(nonforest_gdf_balanced)
        balanced_natural_forest = len(natural_forest_gdf_balanced)
        
        logging.info(f"Zone {zone_name}平衡后样本数量 - 人工林: {balanced_plantation}, 非林地: {balanced_nonforest}, 自然林: {balanced_natural_forest}")
        
        return {
            'zone': zone_name,
            'original': {
                'plantation': plantation_count,
                'nonforest': nonforest_count,
                'natural_forest': natural_forest_count
            },
            'balanced': {
                'plantation': balanced_plantation,
                'nonforest': balanced_nonforest,
                'natural_forest': balanced_natural_forest
            }
        }
        
    except Exception as e:
        logging.error(f"处理Zone {zone_name}失败: {str(e)}")
        return False

def main():
    """
    主函数
    """
    # 设置路径
    input_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter"
    output_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本"
    sdpt_file = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\1.人工林样本点\2.根据产品数据和土地覆盖数据筛选\points_with_ccdc_values_final_20251013_172745.gpkg"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_file = setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info("开始样本数据平衡处理")
    logging.info("="*80)
    
    start_time = time.time()
    
    # 加载SDPT数据（用于zone 7,8,9的样本补充）
    sdpt_gdf = load_sdpt_data(sdpt_file)
    
    # 获取所有zone
    zones = []
    for i in range(1, 11):
        zone_folder = os.path.join(input_dir, f"zone_{i}")
        if os.path.exists(zone_folder):
            zones.append(str(i))
    
    logging.info(f"发现 {len(zones)} 个zone需要处理: {zones}")
    
    # 准备多进程参数
    zone_infos = [(zone, input_dir, output_dir, sdpt_gdf) for zone in zones]
    
    # 使用多进程处理
    num_processes = min(cpu_count(), len(zones))
    logging.info(f"使用 {num_processes} 个进程进行并行处理")
    
    results = []
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_zone_samples, zone_infos)
    
    # 统计处理结果
    successful_zones = [r for r in results if r is not False]
    failed_zones = len(results) - len(successful_zones)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 生成统计报告
    logging.info("="*80)
    logging.info("样本平衡处理完成统计")
    logging.info("="*80)
    
    # 创建详细统计表格
    print("\n各Zone样本平衡处理结果:")
    print("="*100)
    print(f"{'Zone':<6} {'原始-人工林':<12} {'原始-非林地':<12} {'原始-自然林':<12} {'平衡-人工林':<12} {'平衡-非林地':<12} {'平衡-自然林':<12}")
    print("-"*100)
    
    total_original = {'plantation': 0, 'nonforest': 0, 'natural_forest': 0}
    total_balanced = {'plantation': 0, 'nonforest': 0, 'natural_forest': 0}
    
    for result in successful_zones:
        if isinstance(result, dict):
            zone = result['zone']
            orig = result['original']
            bal = result['balanced']
            
            print(f"{zone:<6} {orig['plantation']:<12,} {orig['nonforest']:<12,} {orig['natural_forest']:<12,} "
                  f"{bal['plantation']:<12,} {bal['nonforest']:<12,} {bal['natural_forest']:<12,}")
            
            # 累计统计
            for key in total_original:
                total_original[key] += orig[key]
                total_balanced[key] += bal[key]
    
    print("-"*100)
    print(f"{'总计':<6} {total_original['plantation']:<12,} {total_original['nonforest']:<12,} {total_original['natural_forest']:<12,} "
          f"{total_balanced['plantation']:<12,} {total_balanced['nonforest']:<12,} {total_balanced['natural_forest']:<12,}")
    
    logging.info(f"成功处理的zone数量: {len(successful_zones)}")
    logging.info(f"处理失败的zone数量: {failed_zones}")
    logging.info(f"总处理时间: {processing_time:.2f} 秒")
    
    # 保存统计结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = os.path.join(output_dir, f"balancing_statistics_{timestamp}.txt")
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("样本平衡处理统计结果\n")
        f.write("="*80 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"SDPT数据文件: {sdpt_file}\n\n")
        
        f.write("各Zone样本平衡处理结果:\n")
        f.write("="*100 + "\n")
        f.write(f"{'Zone':<6} {'原始-人工林':<12} {'原始-非林地':<12} {'原始-自然林':<12} {'平衡-人工林':<12} {'平衡-非林地':<12} {'平衡-自然林':<12}\n")
        f.write("-"*100 + "\n")
        
        for result in successful_zones:
            if isinstance(result, dict):
                zone = result['zone']
                orig = result['original']
                bal = result['balanced']
                
                f.write(f"{zone:<6} {orig['plantation']:<12,} {orig['nonforest']:<12,} {orig['natural_forest']:<12,} "
                       f"{bal['plantation']:<12,} {bal['nonforest']:<12,} {bal['natural_forest']:<12,}\n")
        
        f.write("-"*100 + "\n")
        f.write(f"{'总计':<6} {total_original['plantation']:<12,} {total_original['nonforest']:<12,} {total_original['natural_forest']:<12,} "
               f"{total_balanced['plantation']:<12,} {total_balanced['nonforest']:<12,} {total_balanced['natural_forest']:<12,}\n")
        
        f.write(f"\n成功处理的zone数量: {len(successful_zones)}\n")
        f.write(f"处理失败的zone数量: {failed_zones}\n")
        f.write(f"总处理时间: {processing_time:.2f} 秒\n")
    
    logging.info(f"统计结果已保存到: {stats_file}")
    logging.info("样本平衡处理完成！")

if __name__ == "__main__":
    main()