#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据SDPT面数据生成验证点
目的：根据每个SDPT面的面积，按1km²生成1个验证点的密度生成样本点
作者：锐多宝 (ruiduobao)
创建时间：2025年1月6日
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from shapely.geometry import Point
from shapely.ops import transform
import pyproj
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def setup_logging(output_dir):
    """
    设置日志记录系统
    Args:
        output_dir: 输出目录路径
    Returns:
        logger: 配置好的日志记录器
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"sdpt_sample_points_log_{timestamp}.txt")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件输出
            logging.StreamHandler()  # 控制台输出
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存至: {log_file}")
    return logger

def calculate_area_km2(geometry, crs):
    """
    计算几何体的面积（平方公里）
    Args:
        geometry: 几何体
        crs: 坐标参考系统
    Returns:
        area_km2: 面积（平方公里）
    """
    try:
        # 如果是地理坐标系，需要投影到等面积投影
        if crs.is_geographic:
            # 使用等面积投影（Mollweide）
            proj_crs = pyproj.CRS.from_string('+proj=moll +datum=WGS84')
            project = pyproj.Transformer.from_crs(crs, proj_crs, always_xy=True).transform
            projected_geom = transform(project, geometry)
            area_m2 = projected_geom.area
        else:
            area_m2 = geometry.area
        
        # 转换为平方公里
        area_km2 = area_m2 / 1000000
        return area_km2
    except Exception as e:
        return 0.0

def generate_random_points_in_polygon(polygon, num_points, max_attempts=1000):
    """
    在多边形内生成随机点
    Args:
        polygon: 多边形几何体
        num_points: 需要生成的点数
        max_attempts: 最大尝试次数
    Returns:
        points: 生成的点列表
    """
    points = []
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    
    attempts = 0
    while len(points) < num_points and attempts < max_attempts:
        # 在边界框内生成随机点
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        point = Point(x, y)
        
        # 检查点是否在多边形内
        if polygon.contains(point):
            points.append(point)
        
        attempts += 1
    
    # 如果无法生成足够的点，使用质心作为备选
    while len(points) < num_points:
        centroid = polygon.centroid
        if polygon.contains(centroid):
            points.append(centroid)
        else:
            # 如果质心不在多边形内，使用代表点
            points.append(polygon.representative_point())
        break
    
    return points[:num_points]

def process_polygon_chunk(args):
    """
    处理多边形块的函数（用于多进程）
    Args:
        args: 包含(chunk_data, target_attributes)的元组
    Returns:
        result_points: 生成的点数据列表
    """
    chunk_data, target_attributes = args
    result_points = []
    
    for idx, row in chunk_data.iterrows():
        try:
            # 计算面积（平方公里）
            area_km2 = calculate_area_km2(row.geometry, chunk_data.crs)
            
            # 计算需要生成的点数（每平方公里1个点）
            num_points = max(1, int(round(area_km2)))  # 至少生成1个点
            
            # 在多边形内生成随机点
            points = generate_random_points_in_polygon(row.geometry, num_points)
            
            # 为每个点创建属性记录
            for i, point in enumerate(points):
                point_data = {'geometry': point}
                
                # 继承指定的属性
                for attr in target_attributes:
                    if attr in row.index:
                        point_data[attr] = row[attr]
                    else:
                        point_data[attr] = None
                
                # 添加额外信息
                point_data['source_area_km2'] = area_km2
                point_data['point_id'] = f"{row.get('gfw_fid', idx)}_{i+1}"
                point_data['source_fid'] = row.get('gfw_fid', idx)
                
                result_points.append(point_data)
                
        except Exception as e:
            # 记录错误但继续处理
            print(f"处理多边形 {idx} 时发生错误: {str(e)}")
            continue
    
    return result_points

def generate_sample_points(sdpt_path, target_attributes, output_dir, logger, chunk_size=1000):
    """
    根据SDPT面数据生成样本点
    Args:
        sdpt_path: SDPT数据文件路径
        target_attributes: 需要继承的属性列表
        output_dir: 输出目录
        logger: 日志记录器
        chunk_size: 分块处理大小
    Returns:
        output_file: 输出文件路径
    """
    logger.info("开始读取SDPT数据...")
    logger.info(f"数据路径: {sdpt_path}")
    
    try:
        # 读取SDPT数据
        gdf = gpd.read_file(sdpt_path)
        logger.info(f"成功读取SDPT数据，共 {len(gdf)} 个面")
        
        # 检查目标属性是否存在
        missing_attrs = [attr for attr in target_attributes if attr not in gdf.columns]
        if missing_attrs:
            logger.warning(f"以下属性在数据中不存在: {missing_attrs}")
            target_attributes = [attr for attr in target_attributes if attr in gdf.columns]
        
        logger.info(f"将继承的属性: {target_attributes}")
        
        # 计算总面积和预估点数
        logger.info("计算总面积和预估点数...")
        total_area = 0
        estimated_points = 0
        
        for idx, row in gdf.iterrows():
            area_km2 = calculate_area_km2(row.geometry, gdf.crs)
            total_area += area_km2
            estimated_points += max(1, int(round(area_km2)))
            
            if (idx + 1) % 10000 == 0:
                logger.info(f"已处理 {idx + 1} 个面...")
        
        logger.info(f"总面积: {total_area:.2f} km²")
        logger.info(f"预估生成点数: {estimated_points:,}")
        
        # 分块处理数据
        logger.info(f"开始分块处理数据，块大小: {chunk_size}")
        chunks = [gdf.iloc[i:i+chunk_size] for i in range(0, len(gdf), chunk_size)]
        logger.info(f"共分为 {len(chunks)} 个块")
        
        # 准备多进程参数
        chunk_args = [(chunk, target_attributes) for chunk in chunks]
        
        # 使用多进程处理
        num_processes = min(mp.cpu_count(), len(chunks))
        logger.info(f"使用 {num_processes} 个进程进行并行处理")
        
        all_points = []
        with mp.Pool(processes=num_processes) as pool:
            # 使用tqdm显示进度
            results = list(tqdm(
                pool.imap(process_polygon_chunk, chunk_args),
                total=len(chunk_args),
                desc="处理多边形块"
            ))
            
            # 合并结果
            for result in results:
                all_points.extend(result)
        
        logger.info(f"成功生成 {len(all_points)} 个样本点")
        
        # 创建点数据的GeoDataFrame
        logger.info("创建点数据GeoDataFrame...")
        points_gdf = gpd.GeoDataFrame(all_points, crs=gdf.crs)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"sdpt_sample_points_{timestamp}.gpkg")
        
        logger.info(f"保存样本点数据到: {output_file}")
        points_gdf.to_file(output_file, driver='GPKG')
        
        # 获取文件大小
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"文件保存成功，大小: {file_size_mb:.2f} MB")
        
        # 统计信息
        logger.info("\n样本点生成统计:")
        logger.info(f"  - 总点数: {len(points_gdf):,}")
        logger.info(f"  - 平均每km²点数: {len(points_gdf)/total_area:.2f}")
        logger.info(f"  - 数据覆盖面积: {total_area:.2f} km²")
        
        # 按国家统计
        if 'country' in points_gdf.columns:
            country_stats = points_gdf['country'].value_counts()
            logger.info("\n各国家样本点统计:")
            for country, count in country_stats.items():
                percentage = (count / len(points_gdf)) * 100
                logger.info(f"  - {country}: {count:,} 个点 ({percentage:.2f}%)")
        
        return output_file
        
    except Exception as e:
        logger.error(f"生成样本点时发生错误: {str(e)}")
        return None

def main():
    """
    主函数：执行完整的样本点生成流程
    """
    # 设置路径
    sdpt_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\筛选结果\sdpt_filtered_20251006_174752.gpkg"
    output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\1.生成SDPT的样本点"
    
    # 需要继承的属性字段
    target_attributes = [
        'gfw_fid',      # 对应用户要求的fid
        'final_id',     # final_id
        'iso',          # iso
        'common_nam',   # 对应用户要求的commom_name（注意原数据中是common_nam）
        'species_si',   # 对应用户要求的species_simp（注意原数据中是species_si）
        'size',         # size
        'creation_y',   # 对应用户要求的creation_year（注意原数据中是creation_y）
        'method',       # method
        'country'       # 额外添加国家信息
    ]
    
    # 设置日志
    logger = setup_logging(output_dir)
    logger.info("="*60)
    logger.info("开始SDPT样本点生成处理")
    logger.info("="*60)
    logger.info(f"输入数据: {sdpt_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"生成密度: 1点/km²")
    logger.info(f"继承属性: {target_attributes}")
    
    # 生成样本点
    output_file = generate_sample_points(sdpt_path, target_attributes, output_dir, logger)
    
    if output_file:
        logger.info("="*60)
        logger.info("SDPT样本点生成处理完成！")
        logger.info(f"输出文件: {output_file}")
        logger.info("="*60)
    else:
        logger.error("样本点生成失败！")

if __name__ == "__main__":
    main()