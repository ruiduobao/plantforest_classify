#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
目的：使用rasterio实现对ESRI2024数据进行筛选，根据东南亚区域矢量范围的外接矩形框筛选tif文件
作者：锐多宝 (ruiduobao)
日期：2024
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime

import rasterio
import geopandas as gpd
from rasterio.warp import transform_bounds
from shapely.geometry import box
from tqdm import tqdm

# 配置日志
def setup_logging(output_dir):
    """
    设置日志记录
    """
    log_file = os.path.join(output_dir, f'esri_filter_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def read_vector_bounds(vector_path):
    """
    读取矢量数据并获取外接矩形范围
    
    Args:
        vector_path (str): 矢量文件路径
    
    Returns:
        tuple: (minx, miny, maxx, maxy, crs)
    """
    try:
        # 读取矢量数据
        gdf = gpd.read_file(vector_path)
        logging.info(f"成功读取矢量数据: {vector_path}")
        logging.info(f"矢量数据坐标系: {gdf.crs}")
        logging.info(f"矢量数据要素数量: {len(gdf)}")
        
        # 获取总的外接矩形
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        logging.info(f"矢量数据外接矩形范围: {bounds}")
        
        return bounds[0], bounds[1], bounds[2], bounds[3], gdf.crs
    
    except Exception as e:
        logging.error(f"读取矢量数据失败: {e}")
        raise

def get_raster_info(raster_path):
    """
    获取栅格数据的基本信息
    
    Args:
        raster_path (str): 栅格文件路径
    
    Returns:
        dict: 包含bounds, crs等信息的字典
    """
    try:
        with rasterio.open(raster_path) as src:
            return {
                'bounds': src.bounds,
                'crs': src.crs,
                'width': src.width,
                'height': src.height,
                'transform': src.transform
            }
    except Exception as e:
        logging.warning(f"无法读取栅格文件 {raster_path}: {e}")
        return None

def check_spatial_overlap(raster_info, target_bounds, target_crs):
    """
    检查栅格数据是否与目标范围有空间重叠
    
    Args:
        raster_info (dict): 栅格信息
        target_bounds (tuple): 目标范围 (minx, miny, maxx, maxy)
        target_crs: 目标坐标系
    
    Returns:
        bool: 是否有重叠
    """
    if raster_info is None:
        return False
    
    try:
        raster_bounds = raster_info['bounds']
        raster_crs = raster_info['crs']
        
        # 如果坐标系不同，需要转换栅格边界到目标坐标系
        if raster_crs != target_crs:
            # 将栅格边界转换到目标坐标系
            transformed_bounds = transform_bounds(
                raster_crs, target_crs,
                raster_bounds.left, raster_bounds.bottom,
                raster_bounds.right, raster_bounds.top
            )
            raster_box = box(*transformed_bounds)
        else:
            raster_box = box(raster_bounds.left, raster_bounds.bottom,
                           raster_bounds.right, raster_bounds.top)
        
        # 创建目标范围的几何对象
        target_box = box(*target_bounds)
        
        # 检查是否有重叠
        return raster_box.intersects(target_box)
    
    except Exception as e:
        logging.warning(f"坐标系转换或重叠检查失败: {e}")
        return False

def process_single_file(args):
    """
    处理单个文件的函数（用于多进程）
    
    Args:
        args (tuple): (文件路径, 目标范围, 目标坐标系, 输出目录)
    
    Returns:
        tuple: (文件名, 是否成功, 消息)
    """
    file_path, target_bounds, target_crs, output_dir = args
    file_name = os.path.basename(file_path)
    
    try:
        # 获取栅格信息
        raster_info = get_raster_info(file_path)
        
        if raster_info is None:
            return file_name, False, "无法读取栅格信息"
        
        # 检查空间重叠
        if check_spatial_overlap(raster_info, target_bounds, target_crs):
            # 复制文件到输出目录
            output_path = os.path.join(output_dir, file_name)
            shutil.copy2(file_path, output_path)
            return file_name, True, "成功复制"
        else:
            return file_name, False, "不在目标范围内"
    
    except Exception as e:
        return file_name, False, f"处理失败: {str(e)}"

def filter_esri_data(source_dir, vector_path, output_dir, num_processes=None):
    """
    主函数：筛选ESRI数据
    
    Args:
        source_dir (str): 源数据目录
        vector_path (str): 矢量文件路径
        output_dir (str): 输出目录
        num_processes (int): 进程数，默认为CPU核心数
    """
    # 设置日志
    log_file = setup_logging(output_dir)
    logging.info("开始ESRI数据筛选任务")
    logging.info(f"源数据目录: {source_dir}")
    logging.info(f"矢量文件: {vector_path}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"日志文件: {log_file}")
    
    # 读取矢量数据边界
    logging.info("正在读取矢量数据...")
    minx, miny, maxx, maxy, vector_crs = read_vector_bounds(vector_path)
    target_bounds = (minx, miny, maxx, maxy)
    
    # 获取所有tif文件
    logging.info("正在扫描源数据目录...")
    tif_files = []
    for file_name in os.listdir(source_dir):
        if file_name.lower().endswith('.tif'):
            tif_files.append(os.path.join(source_dir, file_name))
    
    logging.info(f"找到 {len(tif_files)} 个tif文件")
    
    if not tif_files:
        logging.warning("未找到任何tif文件")
        return
    
    # 设置进程数
    if num_processes is None:
        num_processes = min(cpu_count(), len(tif_files))
    
    logging.info(f"使用 {num_processes} 个进程进行并行处理")
    
    # 准备多进程参数
    process_args = [(file_path, target_bounds, vector_crs, output_dir) 
                   for file_path in tif_files]
    
    # 多进程处理
    successful_files = []
    failed_files = []
    
    with Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度条
        results = list(tqdm(
            pool.imap(process_single_file, process_args),
            total=len(tif_files),
            desc="处理文件"
        ))
    
    # 统计结果
    for file_name, success, message in results:
        if success:
            successful_files.append(file_name)
            logging.info(f"✓ {file_name}: {message}")
        else:
            failed_files.append((file_name, message))
            logging.info(f"✗ {file_name}: {message}")
    
    # 输出统计信息
    logging.info("\n=== 处理结果统计 ===")
    logging.info(f"总文件数: {len(tif_files)}")
    logging.info(f"成功复制: {len(successful_files)}")
    logging.info(f"跳过文件: {len(failed_files)}")
    
    if successful_files:
        logging.info("\n成功复制的文件:")
        for file_name in successful_files:
            logging.info(f"  - {file_name}")
    
    if failed_files:
        logging.info("\n跳过的文件:")
        for file_name, reason in failed_files:
            logging.info(f"  - {file_name}: {reason}")
    
    logging.info("ESRI数据筛选任务完成")

def main():
    """
    主程序入口
    """
    # 配置路径
    source_dir = r"K:\数据\ESRI2024\2024"  # 源数据目录
    vector_path = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\区划数据\EAST_ASIA_worldregions\worldregions.shp"  # 矢量文件路径
    output_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\ESRI_2024\东南亚区域的分块"  # 输出目录
    
    # 检查路径是否存在
    if not os.path.exists(source_dir):
        print(f"错误：源数据目录不存在 - {source_dir}")
        return
    
    if not os.path.exists(vector_path):
        print(f"错误：矢量文件不存在 - {vector_path}")
        return
    
    if not os.path.exists(output_dir):
        print(f"错误：输出目录不存在 - {output_dir}")
        return
    
    try:
        # 执行筛选任务
        filter_esri_data(source_dir, vector_path, output_dir)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()