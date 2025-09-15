#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码目的：筛选和裁剪人工林产品数据
功能：
1. 读取指定文件夹中的所有tif文件
2. 筛选出与东南亚国家边界有交集的栅格文件
3. 使用东南亚国家边界矢量文件进行裁剪
4. 将RGB三波段数据转换为单波段uint8数据
5. 按照指定规则重分类：RGB(127,127,127)→0(nodata), RGB(0,127,0)→1, RGB(127,127,0)→2
6. 输出为LZW压缩的BIGTIFF格式
7. 使用多进程加速处理
8. 输出详细的处理日志

作者：锐多宝 (ruiduobao)
日期：2025年1月
"""

import os
import sys
import logging
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import mapping, box
import numpy as np
from tqdm import tqdm

# 设置日志
def setup_logging(output_dir):
    """
    设置日志记录
    """
    log_filename = f"plantation_clip_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(output_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_path

def check_intersection(raster_file, clip_gdf):
    """
    检查栅格文件是否与裁剪区域有交集
    
    参数:
        raster_file: 栅格文件路径
        clip_gdf: 裁剪矢量GeoDataFrame
    
    返回:
        bool: 是否有交集
    """
    try:
        with rasterio.open(raster_file) as src:
            # 获取栅格边界
            raster_bounds = src.bounds
            raster_crs = src.crs
            
            # 创建栅格边界几何
            raster_geometry = box(raster_bounds.left, raster_bounds.bottom, 
                                raster_bounds.right, raster_bounds.top)
            raster_gdf = gpd.GeoDataFrame([1], geometry=[raster_geometry], crs=raster_crs)
            
            # 将裁剪矢量重投影到栅格坐标系
            if clip_gdf.crs != raster_crs:
                clip_gdf_reproj = clip_gdf.to_crs(raster_crs)
            else:
                clip_gdf_reproj = clip_gdf
            
            # 检查是否有交集
            intersection = gpd.overlay(raster_gdf, clip_gdf_reproj, how='intersection')
            return not intersection.empty
            
    except Exception as e:
        print(f"检查交集时出错 {os.path.basename(raster_file)}: {str(e)}")
        return False

def reclassify_rgb_to_single_band(rgb_array):
    """
    将RGB三波段数据重分类为单波段数据
    
    参数:
        rgb_array: RGB数组，形状为 (3, height, width)
    
    返回:
        numpy.ndarray: 单波段分类结果，形状为 (height, width)
    """
    # 获取数组维度
    if len(rgb_array.shape) != 3 or rgb_array.shape[0] != 3:
        raise ValueError(f"输入数组形状应为 (3, height, width)，实际为 {rgb_array.shape}")
    
    height, width = rgb_array.shape[1], rgb_array.shape[2]
    
    # 创建输出数组，初始化为0（nodata）
    output_array = np.zeros((height, width), dtype=np.uint8)
    
    # 提取RGB波段
    r_band = rgb_array[0]
    g_band = rgb_array[1] 
    b_band = rgb_array[2]
    
    # 定义RGB值映射规则
    # RGB(127,127,127) → 0 (nodata)
    mask_gray = (r_band == 127) & (g_band == 127) & (b_band == 127)
    output_array[mask_gray] = 0
    
    # RGB(0,127,0) → 1
    mask_green = (r_band == 0) & (g_band == 127) & (b_band == 0)
    output_array[mask_green] = 1
    
    # RGB(127,127,10) → 2  
    mask_yellow = (r_band == 127) & (g_band == 127) & (b_band == 0)
    output_array[mask_yellow] = 2
    
    return output_array

def process_single_raster(args):
    """
    处理单个栅格文件的函数（用于多进程）
    
    参数:
        args: 包含(input_file, output_file, clip_gdf)的元组
    
    返回:
        tuple: (文件名, 是否成功, 错误信息, 统计信息)
    """
    input_file, output_file, clip_gdf = args
    filename = os.path.basename(input_file)
    
    try:
        print(f"开始处理: {filename}")
        
        # 读取原始栅格
        with rasterio.open(input_file) as src:
            print(f"{filename}: 读取栅格完成，坐标系: {src.crs}，波段数: {src.count}")
            
            # 检查波段数
            if src.count != 3:
                return (filename, False, f"波段数不正确，期望3个波段，实际{src.count}个波段", None)
            
            # 获取栅格的坐标系
            raster_crs = src.crs
            
            # 将矢量重投影到栅格坐标系
            if clip_gdf.crs != raster_crs:
                print(f"{filename}: 重投影矢量从 {clip_gdf.crs} 到 {raster_crs}")
                clip_gdf_reproj = clip_gdf.to_crs(raster_crs)
            else:
                print(f"{filename}: 矢量坐标系匹配，无需重投影")
                clip_gdf_reproj = clip_gdf
            
            # 生成栅格的矢量范围
            print(f"{filename}: 生成栅格矢量范围")
            raster_bounds = src.bounds
            raster_geometry = box(raster_bounds.left, raster_bounds.bottom, 
                                raster_bounds.right, raster_bounds.top)
            
            # 创建栅格范围的GeoDataFrame
            raster_gdf = gpd.GeoDataFrame([1], geometry=[raster_geometry], crs=raster_crs)
            
            # 计算栅格范围与东南亚边界的交集
            print(f"{filename}: 计算交集")
            intersection_gdf = gpd.overlay(raster_gdf, clip_gdf_reproj, how='intersection')
            
            if intersection_gdf.empty:
                print(f"{filename}: 警告 - 栅格与裁剪区域无交集，跳过处理")
                return (filename, False, "栅格与裁剪区域无交集", None)
            
            # 合并交集几何体并转换为字典格式
            print(f"{filename}: 合并交集几何体")
            intersection_geometry = intersection_gdf.unary_union
            clip_geometry_dict = mapping(intersection_geometry)
            
            # 执行裁剪操作
            print(f"{filename}: 开始裁剪操作")
            out_image, out_transform = mask(
                src, 
                [clip_geometry_dict], 
                crop=True,  # 裁剪到几何体的边界框
                nodata=src.nodata,  # 保持原始的nodata值
                filled=False  # 不填充masked区域
            )
            print(f"{filename}: 裁剪完成，输出形状: {out_image.shape}")
            
            # 将RGB数据重分类为单波段
            print(f"{filename}: 开始RGB重分类")
            classified_array = reclassify_rgb_to_single_band(out_image)
            print(f"{filename}: 重分类完成")
            
            # 统计分类结果
            unique_values, counts = np.unique(classified_array, return_counts=True)
            stats_info = {}
            for val, count in zip(unique_values, counts):
                if val == 0:
                    stats_info['nodata_pixels'] = int(count)
                elif val == 1:
                    stats_info['class1_pixels'] = int(count)
                elif val == 2:
                    stats_info['class2_pixels'] = int(count)
            
            # 更新元数据
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": classified_array.shape[0],
                "width": classified_array.shape[1],
                "count": 1,  # 单波段
                "transform": out_transform,
                "dtype": "uint8",
                "nodata": 0,  # 设置nodata为0
                "compress": "lzw",  # 使用LZW压缩
                "tiled": True,  # 使用瓦片存储
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "YES"  # 使用BIGTIFF格式
            })
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 写入分类后的栅格
            print(f"{filename}: 开始写入输出文件")
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(classified_array, 1)  # 写入第一个波段
                
                # 添加描述信息
                dest.update_tags(
                    DESCRIPTION="人工林产品数据重分类结果",
                    CLASS_0="nodata (原RGB: 127,127,127)",
                    CLASS_1="人工林类型1 (原RGB: 0,127,0)", 
                    CLASS_2="人工林类型2 (原RGB: 127,127,10)",
                    PROCESSING_DATE=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
            
            print(f"{filename}: 处理完成")
        
        return (filename, True, None, stats_info)
        
    except Exception as e:
        error_msg = f"处理文件 {filename} 时出错: {str(e)}"
        return (filename, False, error_msg, None)

def main():
    """
    主函数
    """
    # 定义路径
    input_dir = r"D:\浏览器下载\人工林分布数据\合并"
    output_dir = r"K:\数据\全球人工林产品数据\筛选出东南亚区域_裁剪"
    clip_shapefile = r"K:\数据\GDAM全球\东南亚国家\southeast_asia_combine.shp"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(output_dir)
    logging.info(f"开始人工林产品数据筛选和裁剪处理")
    logging.info(f"输入目录: {input_dir}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"裁剪矢量: {clip_shapefile}")
    logging.info(f"日志文件: {log_path}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logging.error(f"输入目录不存在: {input_dir}")
        return
    
    # 检查裁剪矢量文件是否存在
    if not os.path.exists(clip_shapefile):
        logging.error(f"裁剪矢量文件不存在: {clip_shapefile}")
        return
    
    # 读取裁剪矢量
    logging.info("读取裁剪矢量文件...")
    try:
        clip_gdf = gpd.read_file(clip_shapefile)
        logging.info(f"裁剪矢量坐标系: {clip_gdf.crs}")
        logging.info(f"裁剪矢量要素数量: {len(clip_gdf)}")
        
    except Exception as e:
        logging.error(f"读取裁剪矢量文件失败: {str(e)}")
        return
    
    # 获取所有tif文件
    tif_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
            tif_files.append(file)
    
    logging.info(f"找到 {len(tif_files)} 个tif文件")
    
    if not tif_files:
        logging.warning("未找到任何tif文件")
        return
    
    # 第一步：筛选有交集的文件
    logging.info("第一步：筛选与东南亚区域有交集的文件...")
    intersecting_files = []
    
    for tif_file in tqdm(tif_files, desc="筛选文件"):
        input_path = os.path.join(input_dir, tif_file)
        if check_intersection(input_path, clip_gdf):
            intersecting_files.append(tif_file)
            logging.info(f"✓ 有交集: {tif_file}")
        else:
            logging.info(f"✗ 无交集: {tif_file}")
    
    logging.info(f"筛选结果：{len(intersecting_files)}/{len(tif_files)} 个文件与东南亚区域有交集")
    
    if not intersecting_files:
        logging.warning("没有文件与东南亚区域有交集")
        return
    
    # 第二步：裁剪和重分类处理
    logging.info("第二步：开始裁剪和重分类处理...")
    
    # 准备多进程参数
    process_args = []
    for tif_file in intersecting_files:
        input_path = os.path.join(input_dir, tif_file)
        output_path = os.path.join(output_dir, tif_file)
        process_args.append((input_path, output_path, clip_gdf))
    
    # 确定进程数（使用CPU核心数的一半，避免内存压力过大）
    # num_processes = max(1, cpu_count() // 2)
    num_processes = 1
    logging.info(f"使用 {num_processes} 个进程进行并行处理")
    
    # 开始处理
    start_time = time.time()
    successful_files = []
    failed_files = []
    total_stats = {
        'total_nodata_pixels': 0,
        'total_class1_pixels': 0, 
        'total_class2_pixels': 0
    }
    
    try:
        # 使用多进程处理
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_raster, process_args),
                total=len(process_args),
                desc="处理文件"
            ))
        
        # 处理结果
        for filename, success, error_msg, stats_info in results:
            if success:
                successful_files.append(filename)
                logging.info(f"✓ 成功处理: {filename}")
                
                # 累计统计信息
                if stats_info:
                    total_stats['total_nodata_pixels'] += stats_info.get('nodata_pixels', 0)
                    total_stats['total_class1_pixels'] += stats_info.get('class1_pixels', 0)
                    total_stats['total_class2_pixels'] += stats_info.get('class2_pixels', 0)
                    
                    logging.info(f"  - nodata像素: {stats_info.get('nodata_pixels', 0)}")
                    logging.info(f"  - 类别1像素: {stats_info.get('class1_pixels', 0)}")
                    logging.info(f"  - 类别2像素: {stats_info.get('class2_pixels', 0)}")
            else:
                failed_files.append(filename)
                logging.error(f"✗ 处理失败: {error_msg}")
    
    except Exception as e:
        logging.error(f"多进程处理过程中出错: {str(e)}")
        return
    
    # 处理完成统计
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info("\n" + "="*60)
    logging.info("处理完成统计:")
    logging.info(f"原始文件数: {len(tif_files)}")
    logging.info(f"有交集文件数: {len(intersecting_files)}")
    logging.info(f"成功处理: {len(successful_files)}")
    logging.info(f"处理失败: {len(failed_files)}")
    logging.info(f"总耗时: {total_time:.2f} 秒")
    if intersecting_files:
        logging.info(f"平均每文件耗时: {total_time/len(intersecting_files):.2f} 秒")
    
    # 输出分类统计
    logging.info("\n分类统计汇总:")
    logging.info(f"总nodata像素数: {total_stats['total_nodata_pixels']:,}")
    logging.info(f"总类别1像素数: {total_stats['total_class1_pixels']:,}")
    logging.info(f"总类别2像素数: {total_stats['total_class2_pixels']:,}")
    
    if failed_files:
        logging.info(f"\n失败文件列表: {', '.join(failed_files)}")
    
    logging.info("\n人工林产品数据筛选和裁剪处理完成!")
    logging.info("="*60)

if __name__ == "__main__":
    main()