#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据SDPT面数据生成验证点（优化版）
目的：根据每个SDPT面的面积，按1km²生成1个验证点的密度生成样本点
优化策略：
1. 批量投影转换，避免重复计算
2. 优化面积计算算法
3. 改进多进程处理策略
4. 简化随机点生成算法
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
    log_file = os.path.join(output_dir, f"sdpt_sample_points_optimized_log_{timestamp}.txt")
    
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

def batch_calculate_areas(geometries, crs):
    """
    批量计算几何体面积（平方公里）- 优化版
    Args:
        geometries: 几何体列表
        crs: 坐标参考系统
    Returns:
        areas_km2: 面积列表（平方公里）
    """
    try:
        if crs.is_geographic:
            # 使用等面积投影（Mollweide）- 一次性转换所有几何体
            proj_crs = pyproj.CRS.from_string('+proj=moll +datum=WGS84')
            transformer = pyproj.Transformer.from_crs(crs, proj_crs, always_xy=True)
            
            # 批量转换几何体
            projected_geoms = [transform(transformer.transform, geom) for geom in geometries]
            areas_m2 = [geom.area for geom in projected_geoms]
        else:
            areas_m2 = [geom.area for geom in geometries]
        
        # 转换为平方公里
        areas_km2 = [area / 1000000 for area in areas_m2]
        return areas_km2
    except Exception as e:
        # 如果批量计算失败，返回零面积列表
        return [0.0] * len(geometries)

def fast_random_points_in_polygon(polygon, num_points, max_attempts=500):
    """
    快速在多边形内生成随机点 - 优化版
    Args:
        polygon: 多边形几何体
        num_points: 需要生成的点数
        max_attempts: 最大尝试次数（减少以提高速度）
    Returns:
        points: 生成的点列表
    """
    if num_points <= 0:
        return []
    
    points = []
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    
    # 预计算多边形面积，用于估算成功率
    bbox_area = (maxx - minx) * (maxy - miny)
    poly_area = polygon.area
    success_rate = poly_area / bbox_area if bbox_area > 0 else 0.5
    
    # 根据成功率调整生成策略
    if success_rate > 0.7:  # 高成功率，使用随机生成
        attempts = 0
        batch_size = min(num_points * 3, 100)  # 批量生成
        
        while len(points) < num_points and attempts < max_attempts:
            # 批量生成随机点
            x_coords = np.random.uniform(minx, maxx, batch_size)
            y_coords = np.random.uniform(miny, maxy, batch_size)
            
            for x, y in zip(x_coords, y_coords):
                if len(points) >= num_points:
                    break
                point = Point(x, y)
                if polygon.contains(point):
                    points.append(point)
            
            attempts += batch_size
    
    # 如果随机生成不够，使用网格采样
    if len(points) < num_points:
        remaining = num_points - len(points)
        grid_points = generate_grid_points_in_polygon(polygon, remaining)
        points.extend(grid_points)
    
    return points[:num_points]

def generate_grid_points_in_polygon(polygon, num_points):
    """
    在多边形内生成网格点
    Args:
        polygon: 多边形几何体
        num_points: 需要生成的点数
    Returns:
        points: 生成的点列表
    """
    points = []
    bounds = polygon.bounds
    minx, miny, maxx, maxy = bounds
    
    # 计算网格密度
    grid_size = int(np.ceil(np.sqrt(num_points * 2)))  # 稍微密一点确保有足够的点
    
    x_step = (maxx - minx) / grid_size
    y_step = (maxy - miny) / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= num_points:
                break
            
            x = minx + (i + 0.5) * x_step
            y = miny + (j + 0.5) * y_step
            point = Point(x, y)
            
            if polygon.contains(point):
                points.append(point)
    
    # 如果网格点不够，添加质心
    while len(points) < num_points:
        if polygon.centroid.within(polygon):
            points.append(polygon.centroid)
        else:
            points.append(polygon.representative_point())
        break
    
    return points[:num_points]

def clean_text_data(value):
    """
    清理文本数据，处理编码问题
    Args:
        value: 输入值
    Returns:
        cleaned_value: 清理后的值
    """
    if value is None:
        return None
    
    if isinstance(value, str):
        try:
            # 尝试编码为UTF-8并解码，去除无效字符
            return value.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            # 如果仍然失败，返回字符串表示
            return str(value)
    
    return value

def process_chunk_optimized(args):
    """
    优化的多进程处理函数（修复编码问题）
    Args:
        args: 包含(chunk_indices, target_attributes, chunk_id, crs_info)的元组
    Returns:
        result_points: 生成的点数据列表
    """
    chunk_indices, target_attributes, chunk_id, crs_info = args
    result_points = []
    
    try:
        # 在子进程中重新读取数据，避免序列化大量几何数据
        import geopandas as gpd
        sdpt_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\按国家筛选\sdpt_southeast_asia_20251006_181147.gpkg"
        
        # 只读取需要的行
        chunk_data = gpd.read_file(sdpt_path, rows=slice(chunk_indices[0], chunk_indices[1]))
        
        # 批量计算面积
        geometries = chunk_data.geometry.tolist()
        areas_km2 = batch_calculate_areas(geometries, chunk_data.crs)
        
        # 处理每个多边形
        for idx, (_, row) in enumerate(chunk_data.iterrows()):
            try:
                area_km2 = areas_km2[idx]
                
                # 计算需要生成的点数（每平方公里1个点）
                num_points = max(1, int(round(area_km2))*5)
                
                # 快速生成随机点
                points = fast_random_points_in_polygon(row.geometry, num_points)
                
                # 为每个点创建属性记录
                for i, point in enumerate(points):
                    point_data = {'geometry': point}
                    
                    # # 继承指定的属性，并清理文本数据
                    # for attr in target_attributes:
                    #     if attr in row.index:
                    #         raw_value = row[attr]
                    #         point_data[attr] = clean_text_data(raw_value)
                    #     else:
                    #         point_data[attr] = None
                    
                    # 添加额外信息
                    # point_data['source_area_km2'] = area_km2
                    # point_data['point_id'] = f"{clean_text_data(str(row.get('final_id', row.name)))}_{i+1}"
                    # point_data['source_fid'] = clean_text_data(str(row.get('final_id', row.name)))
                    
                    result_points.append(point_data)
                    
            except Exception as e:
                # 记录错误但继续处理
                continue
                
    except Exception as e:
        print(f"处理块 {chunk_id} 时发生错误: {str(e)}")
    
    return result_points

def generate_sample_points_optimized(sdpt_path, target_attributes, output_dir, logger, chunk_size=1000):
    """
    优化版样本点生成函数（修复编码问题）
    Args:
        sdpt_path: SDPT数据文件路径
        target_attributes: 需要继承的属性列表
        output_dir: 输出目录
        logger: 日志记录器
        chunk_size: 分块处理大小（减小以避免内存问题）
    Returns:
        output_file: 输出文件路径
    """
    logger.info("开始读取SDPT数据...")
    logger.info(f"数据路径: {sdpt_path}")
    
    try:
        # 先读取数据基本信息
        gdf_info = gpd.read_file(sdpt_path, rows=1)  # 只读取第一行获取基本信息
        total_rows = len(gpd.read_file(sdpt_path, rows=slice(None, None)))  # 获取总行数
        logger.info(f"SDPT数据总共 {total_rows} 个面")
        
        # 检查目标属性是否存在
        missing_attrs = [attr for attr in target_attributes if attr not in gdf_info.columns]
        if missing_attrs:
            logger.warning(f"以下属性在数据中不存在: {missing_attrs}")
            target_attributes = [attr for attr in target_attributes if attr in gdf_info.columns]
        
        logger.info(f"将继承的属性: {target_attributes}")
        
        # 快速估算总面积和点数（使用边界框面积估算）
        logger.info("快速估算总面积和点数...")
        bounds = gdf_info.total_bounds
        bbox_area_deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        # 粗略转换为km²（1度约等于111km）
        estimated_area_km2 = bbox_area_deg2 * (111 * 111) * total_rows / 1000  # 调整估算
        estimated_points = int(estimated_area_km2 * 0.3)  # 假设实际面积是边界框的30%
        
        logger.info(f"估算覆盖面积: ~{estimated_area_km2:.0f} km²")
        logger.info(f"估算生成点数: ~{estimated_points:,}")
        
        # 创建分块索引，避免传递大量几何数据
        logger.info(f"开始分块处理数据，块大小: {chunk_size}")
        chunk_indices = []
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_indices.append(((i, end_idx), target_attributes, i//chunk_size, str(gdf_info.crs)))
        
        logger.info(f"共分为 {len(chunk_indices)} 个块")
        
        # 使用较少的进程数避免内存问题
        num_processes = min(8, mp.cpu_count()//2, len(chunk_indices))  # 减少进程数
        logger.info(f"使用 {num_processes} 个进程进行并行处理")
        
        all_points = []
        with mp.Pool(processes=num_processes) as pool:
            # 使用tqdm显示进度
            results = list(tqdm(
                pool.imap(process_chunk_optimized, chunk_indices),
                total=len(chunk_indices),
                desc="处理多边形块"
            ))
            
            # 合并结果
            for result in results:
                all_points.extend(result)
        
        logger.info(f"成功生成 {len(all_points)} 个样本点")
        
        # 创建点数据的GeoDataFrame
        logger.info("创建点数据GeoDataFrame...")
        points_gdf = gpd.GeoDataFrame(all_points, crs=gdf_info.crs)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"sdpt_sample_points_fixed_no_attr_{timestamp}.gpkg")
        
        logger.info(f"保存样本点数据到: {output_file}")
        points_gdf.to_file(output_file, driver='GPKG')
        
        # 获取文件大小
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"文件保存成功，大小: {file_size_mb:.2f} MB")
        
        # 统计信息
        actual_area = points_gdf['source_area_km2'].sum()
        logger.info("\n样本点生成统计:")
        logger.info(f"  - 总点数: {len(points_gdf):,}")
        logger.info(f"  - 实际覆盖面积: {actual_area:.2f} km²")
        logger.info(f"  - 平均每km²点数: {len(points_gdf)/actual_area:.2f}")
        
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
    主函数：执行完整的样本点生成流程（优化版）
    """
    # 设置路径
    sdpt_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\按国家筛选\sdpt_southeast_asia_20251006_181147.gpkg"
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
    logger.info("开始SDPT样本点生成处理（优化版）")
    logger.info("="*60)
    logger.info(f"输入数据: {sdpt_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"生成密度: 1点/km²")
    logger.info(f"继承属性: {target_attributes}")
    logger.info("优化策略: 批量面积计算 + 快速点生成 + 大块处理")
    
    # 生成样本点
    output_file = generate_sample_points_optimized(sdpt_path, target_attributes, output_dir, logger)
    
    if output_file:
        logger.info("="*60)
        logger.info("SDPT样本点生成处理完成！")
        logger.info(f"输出文件: {output_file}")
        logger.info("="*60)
    else:
        logger.error("样本点生成失败！")

if __name__ == "__main__":
    main()