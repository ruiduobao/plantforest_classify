#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码目的：使用矢量文件批量裁剪ESRI土地覆盖数据的tif文件
功能：
1. 读取指定文件夹中的所有tif文件
2. 使用东南亚国家边界矢量文件进行裁剪
3. 保持原栅格的空值、颜色映射等属性
4. 使用多进程加速处理
5. 输出裁剪后的栅格到指定文件夹

作者：锐多宝 (ruiduobao)
日期：2025年1月
"""

import os
import sys
import logging
import time
from datetime import datetime
# from multiprocessing import Pool, cpu_count  # 改为单进程处理，不再需要
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np

# 设置日志
def setup_logging(output_dir):
    """
    设置日志记录
    """
    log_filename = f"esri_clip_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

def clip_single_raster(args):
    """
    裁剪单个栅格文件的函数（用于多进程）
    
    参数:
        args: 包含(input_file, output_file, clip_gdf)的元组
    
    返回:
        tuple: (文件名, 是否成功, 错误信息)
    """
    input_file, output_file, clip_gdf = args
    filename = os.path.basename(input_file)
    
    try:
        print(f"开始处理: {filename}")
        
        # 读取原始栅格
        with rasterio.open(input_file) as src:
            print(f"{filename}: 读取栅格完成，坐标系: {src.crs}")
            
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
            from shapely.geometry import box
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
                return (os.path.basename(input_file), False, "栅格与裁剪区域无交集")
            
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
            print(f"{filename}: 裁剪完成")
            
            # 更新元数据
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": "uint8",  # ESRI土地覆盖数据使用uint8类型（0-255）
                "compress": "lzw",  # 使用LZW压缩
                "tiled": True,  # 使用瓦片存储
                "blockxsize": 512,
                "blockysize": 512
            })
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 写入裁剪后的栅格
            with rasterio.open(output_file, "w", **out_meta) as dest:
                # 确保数据类型为uint8
                out_image_uint8 = out_image.astype(np.uint8)
                dest.write(out_image_uint8)
                
                # 复制颜色映射表（如果存在）
                if src.colormap(1) is not None:
                    dest.write_colormap(1, src.colormap(1))
                
                # 复制其他属性
                for i in range(1, src.count + 1):
                    if src.tags(i):
                        dest.update_tags(i, **src.tags(i))
        
        return (os.path.basename(input_file), True, None)
        
    except Exception as e:
        error_msg = f"处理文件 {os.path.basename(input_file)} 时出错: {str(e)}"
        return (os.path.basename(input_file), False, error_msg)

def main():
    """
    主函数
    """
    # 定义路径
    input_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\ESRI_2017_2023\2018年"
    output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\ESRI_2017_20233\2018年_分块裁剪"
    clip_shapefile = r"D:\地理所\论文\东南亚10m人工林提取\数据\东南亚国家\southeast_asia_combine.shp"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(output_dir)
    logging.info(f"开始ESRI数据批量裁剪处理")
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
        if file.lower().endswith('.tif'):
            tif_files.append(file)
    
    logging.info(f"找到 {len(tif_files)} 个tif文件")
    
    if not tif_files:
        logging.warning("未找到任何tif文件")
        return
    
    # 准备多进程参数
    process_args = []
    for tif_file in tif_files:
        input_path = os.path.join(input_dir, tif_file)
        output_path = os.path.join(output_dir, tif_file)
        process_args.append((input_path, output_path, clip_gdf))
    
    # 使用单进程处理避免内存问题
    logging.info(f"使用单进程进行处理（避免内存竞争问题）")
    
    # 开始处理
    start_time = time.time()
    successful_files = []
    failed_files = []
    
    try:
        # 单进程逐个处理文件
        for i, args in enumerate(process_args, 1):
            logging.info(f"处理进度: {i}/{len(process_args)}")
            filename, success, error_msg = clip_single_raster(args)
            
            if success:
                successful_files.append(filename)
                logging.info(f"✓ 成功处理: {filename}")
            else:
                failed_files.append(filename)
                logging.error(f"✗ 处理失败: {error_msg}")
    
    except Exception as e:
        logging.error(f"处理过程中出错: {str(e)}")
        return
    
    # 处理完成统计
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info("\n" + "="*50)
    logging.info("处理完成统计:")
    logging.info(f"总文件数: {len(tif_files)}")
    logging.info(f"成功处理: {len(successful_files)}")
    logging.info(f"处理失败: {len(failed_files)}")
    logging.info(f"总耗时: {total_time:.2f} 秒")
    logging.info(f"平均每文件耗时: {total_time/len(tif_files):.2f} 秒")
    
    if failed_files:
        logging.info(f"失败文件列表: {', '.join(failed_files)}")
    
    logging.info("ESRI数据批量裁剪处理完成!")
    logging.info("="*50)

if __name__ == "__main__":
    main()