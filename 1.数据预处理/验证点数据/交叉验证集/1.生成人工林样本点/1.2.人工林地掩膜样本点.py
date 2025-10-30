#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本点掩膜筛选脚本
功能：根据土地覆盖数据和人工林产品数据对SDPT样本点进行筛选
作者：锐多宝 (ruiduobao)
创建时间：2025年1月9日
"""

import os
import sys
import time
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from shapely.geometry import Point, box
import warnings
warnings.filterwarnings('ignore')

# 配置参数
INPUT_POINTS_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\1.人工林样本点\1.生成SDPT的样本点\sdpt_sample_points_fixed_no_attr_20251011_215214.gpkg"
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\2.根据产品数据和土地覆盖数据筛选"
LANDCOVER_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\土地覆盖数据\southeast_asia_landcover_2024_mosaic.tif"
FOREST_VALUE = 2  # 林地像素值
PLANTTREE_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\全球人工林产品数据\southeast_asia_PlantTree_2021_mosaic.tif"

def setup_logging(output_dir):
    """
    设置日志记录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"mask_filtering_log_{timestamp}.txt")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def check_input_files():
    """
    检查输入文件是否存在
    """
    files_to_check = {
        "样本点文件": INPUT_POINTS_FILE,
        "土地覆盖数据": LANDCOVER_FILE,
        "人工林产品数据": PLANTTREE_FILE
    }
    
    missing_files = []
    for name, file_path in files_to_check.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{name}: {file_path}")
            logging.error(f"文件不存在: {name} - {file_path}")
        else:
            logging.info(f"文件检查通过: {name} - {file_path}")
    
    if missing_files:
        raise FileNotFoundError(f"以下文件不存在:\n" + "\n".join(missing_files))
    
    return True

def get_raster_info(raster_path):
    """
    获取栅格数据信息
    """
    with rasterio.open(raster_path) as src:
        info = {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'dtype': src.dtypes[0],
            'nodata': src.nodata
        }
    return info

def sample_raster_values_batch(points_gdf, raster_path, chunk_size=10000):
    """
    批量采样栅格数据值
    使用分块处理提高内存效率
    """
    logging.info(f"开始从栅格文件采样: {os.path.basename(raster_path)}")
    
    # 获取栅格信息
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        nodata_value = src.nodata
        
        # 确保点数据与栅格数据坐标系一致
        if points_gdf.crs != raster_crs:
            logging.info(f"转换坐标系从 {points_gdf.crs} 到 {raster_crs}")
            points_gdf = points_gdf.to_crs(raster_crs)
        
        # 过滤在栅格范围内的点
        points_in_bounds = points_gdf.cx[
            raster_bounds[0]:raster_bounds[2],
            raster_bounds[1]:raster_bounds[3]
        ]
        
        logging.info(f"栅格范围内的点数: {len(points_in_bounds)} / {len(points_gdf)}")
        
        # 初始化结果数组
        values = np.full(len(points_gdf), nodata_value if nodata_value is not None else -9999, dtype=src.dtypes[0])
        
        # 分块处理点数据
        total_chunks = (len(points_in_bounds) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(points_in_bounds), chunk_size):
            chunk_end = min(i + chunk_size, len(points_in_bounds))
            chunk_points = points_in_bounds.iloc[i:chunk_end]
            
            if len(chunk_points) == 0:
                continue
            
            # 获取点的坐标
            coords = [(point.x, point.y) for point in chunk_points.geometry]
            
            # 采样栅格值
            try:
                sampled_values = list(src.sample(coords))
                chunk_values = [val[0] if len(val) > 0 else (nodata_value if nodata_value is not None else -9999) 
                              for val in sampled_values]
                
                # 将结果存储到对应位置
                chunk_indices = points_in_bounds.iloc[i:chunk_end].index
                values[chunk_indices] = chunk_values
                
            except Exception as e:
                logging.warning(f"采样块 {i//chunk_size + 1}/{total_chunks} 时出错: {str(e)}")
                continue
            
            if (i // chunk_size + 1) % 10 == 0:
                logging.info(f"已处理 {i//chunk_size + 1}/{total_chunks} 个块")
    
    logging.info(f"栅格采样完成，有效值数量: {np.sum(values != (nodata_value if nodata_value is not None else -9999))}")
    return values

def filter_sample_points():
    """
    主要的样本点筛选函数
    """
    start_time = time.time()
    
    # 设置日志
    log_file = setup_logging(OUTPUT_DIR)
    logging.info("="*80)
    logging.info("开始样本点掩膜筛选处理")
    logging.info("="*80)
    
    # 记录配置参数
    logging.info("配置参数:")
    logging.info(f"  输入样本点文件: {INPUT_POINTS_FILE}")
    logging.info(f"  输出目录: {OUTPUT_DIR}")
    logging.info(f"  土地覆盖数据: {LANDCOVER_FILE}")
    logging.info(f"  林地像素值: {FOREST_VALUE}")
    logging.info(f"  人工林产品数据: {PLANTTREE_FILE}")
    
    # 检查输入文件
    check_input_files()
    
    # 读取样本点数据
    logging.info("读取样本点数据...")
    points_gdf = gpd.read_file(INPUT_POINTS_FILE)
    logging.info(f"成功读取 {len(points_gdf)} 个样本点")
    logging.info(f"样本点坐标系: {points_gdf.crs}")
    logging.info(f"样本点属性字段: {list(points_gdf.columns)}")
    
    # 获取栅格数据信息
    logging.info("获取栅格数据信息...")
    landcover_info = get_raster_info(LANDCOVER_FILE)
    planttree_info = get_raster_info(PLANTTREE_FILE)
    
    logging.info(f"土地覆盖数据信息: CRS={landcover_info['crs']}, 尺寸={landcover_info['width']}x{landcover_info['height']}")
    logging.info(f"人工林产品数据信息: CRS={planttree_info['crs']}, 尺寸={planttree_info['width']}x{planttree_info['height']}")
    
    # 采样土地覆盖数据
    logging.info("采样土地覆盖数据...")
    landcover_values = sample_raster_values_batch(points_gdf, LANDCOVER_FILE)
    points_gdf['landcover'] = landcover_values
    
    # 采样人工林产品数据
    logging.info("采样人工林产品数据...")
    planttree_values = sample_raster_values_batch(points_gdf, PLANTTREE_FILE)
    points_gdf['planttree'] = planttree_values
    
    # 应用筛选条件
    logging.info("应用筛选条件...")
    
    # 统计原始数据
    total_points = len(points_gdf)
    logging.info(f"原始样本点总数: {total_points}")
    
    # 筛选条件1: 土地覆盖为林地
    forest_mask = points_gdf['landcover'] == FOREST_VALUE
    forest_points = np.sum(forest_mask)
    logging.info(f"土地覆盖为林地的点数: {forest_points} ({forest_points/total_points*100:.2f}%)")
    
    # 筛选条件2: 人工林产品数据值为2（人工林）
    planttree_mask = points_gdf['planttree'] == 2  # 只保留值为2的点（人工林）
    planttree_points = np.sum(planttree_mask)
    logging.info(f"人工林产品数据为人工林(值=2)的点数: {planttree_points} ({planttree_points/total_points*100:.2f}%)")
    
    # 组合筛选条件
    combined_mask = forest_mask & planttree_mask
    filtered_points = np.sum(combined_mask)
    logging.info(f"同时满足两个条件的点数: {filtered_points} ({filtered_points/total_points*100:.2f}%)")
    
    # 应用筛选
    filtered_gdf = points_gdf[combined_mask].copy()
    
    # 统计筛选结果
    logging.info("筛选结果统计:")
    logging.info(f"  原始点数: {total_points}")
    logging.info(f"  筛选后点数: {len(filtered_gdf)}")
    logging.info(f"  保留比例: {len(filtered_gdf)/total_points*100:.2f}%")
    
    # 按国家统计筛选结果
    if 'country' in filtered_gdf.columns:
        country_stats = filtered_gdf['country'].value_counts()
        logging.info("按国家统计筛选后的点数:")
        for country, count in country_stats.items():
            percentage = count / len(filtered_gdf) * 100
            logging.info(f"  {country}: {count} 点 ({percentage:.2f}%)")
    
    # 保存筛选结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"sdpt_filtered_points_{timestamp}.gpkg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    logging.info(f"保存筛选结果到: {output_path}")
    filtered_gdf.to_file(output_path, driver='GPKG')
    
    # 计算文件大小
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"输出文件大小: {file_size_mb:.2f} MB")
    
    # 保存统计报告
    stats_filename = f"filtering_statistics_{timestamp}.txt"
    stats_path = os.path.join(OUTPUT_DIR, stats_filename)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("SDPT样本点筛选统计报告\n")
        f.write("="*50 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入文件: {INPUT_POINTS_FILE}\n")
        f.write(f"输出文件: {output_path}\n")
        f.write(f"土地覆盖数据: {LANDCOVER_FILE}\n")
        f.write(f"人工林产品数据: {PLANTTREE_FILE}\n\n")
        
        f.write("筛选条件:\n")
        f.write(f"1. 土地覆盖 = {FOREST_VALUE} (林地)\n")
        f.write(f"2. 人工林产品数据 = 2 (人工林)\n\n")
        
        f.write("筛选结果:\n")
        f.write(f"原始点数: {total_points}\n")
        f.write(f"土地覆盖为林地: {forest_points} ({forest_points/total_points*100:.2f}%)\n")
        f.write(f"人工林产品数据为人工林(值=2): {planttree_points} ({planttree_points/total_points*100:.2f}%)\n")
        f.write(f"筛选后点数: {len(filtered_gdf)} ({len(filtered_gdf)/total_points*100:.2f}%)\n\n")
        
        if 'country' in filtered_gdf.columns:
            f.write("按国家统计:\n")
            for country, count in country_stats.items():
                percentage = count / len(filtered_gdf) * 100
                f.write(f"{country}: {count} 点 ({percentage:.2f}%)\n")
    
    # 计算总处理时间
    total_time = time.time() - start_time
    logging.info(f"样本点筛选完成！总耗时: {total_time:.2f} 秒")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"统计报告: {stats_path}")
    logging.info("="*80)
    
    return output_path, stats_path

if __name__ == "__main__":
    try:
        # 设置多进程启动方法（Windows系统需要）
        if sys.platform.startswith('win'):
            mp.set_start_method('spawn', force=True)
        
        # 执行样本点筛选
        output_file, stats_file = filter_sample_points()
        
        print(f"\n筛选完成！")
        print(f"输出文件: {output_file}")
        print(f"统计报告: {stats_file}")
        
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        import traceback
        logging.error(f"错误详情:\n{traceback.format_exc()}")
        sys.exit(1)