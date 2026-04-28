#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码目的：诊断多UTM区域镶嵌效果
功能：
1. 分析镶嵌结果的空间覆盖范围
2. 检查UTM区域间的空间连续性
3. 计算空间偏移和重叠情况
4. 提供优化建议

作者：锐多宝 (ruiduobao)
日期：2025年1月
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from osgeo import gdal, osr

def setup_logging(output_dir):
    """设置日志记录"""
    log_file = os.path.join(output_dir, f"多UTM区域镶嵌诊断_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"诊断日志保存到: {log_file}")
    return log_file

def get_image_info(file_path):
    """获取影像的详细信息"""
    try:
        ds = gdal.Open(file_path)
        if ds is None:
            return None
        
        # 基本信息
        width = ds.RasterXSize
        height = ds.RasterYSize
        bands = ds.RasterCount
        
        # 地理变换信息
        gt = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # 坐标系信息
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)
        
        # 计算边界
        min_x = gt[0]
        max_y = gt[3]
        max_x = min_x + gt[1] * width
        min_y = max_y + gt[5] * height
        
        # 像元大小
        pixel_width = abs(gt[1])
        pixel_height = abs(gt[5])
        
        info = {
            'path': file_path,
            'width': width,
            'height': height,
            'bands': bands,
            'geotransform': gt,
            'projection': projection,
            'srs': srs,
            'bounds': [min_x, min_y, max_x, max_y],
            'pixel_size': [pixel_width, pixel_height],
            'epsg': srs.GetAttrValue('AUTHORITY', 1) if srs.GetAttrValue('AUTHORITY', 1) else 'Unknown',
            'is_geographic': srs.IsGeographic(),
            'units': srs.GetLinearUnitsName() if not srs.IsGeographic() else 'degrees'
        }
        
        ds = None
        return info
        
    except Exception as e:
        logging.error(f"获取影像信息失败 {file_path}: {str(e)}")
        return None

def analyze_utm_zones_coverage(input_files):
    """分析UTM区域的覆盖情况"""
    logging.info("=== UTM区域覆盖分析 ===")
    
    zone_info = {}
    all_bounds = []
    
    for i, file_path in enumerate(input_files):
        info = get_image_info(file_path)
        if info is None:
            continue
        
        filename = os.path.basename(file_path)
        zone_info[filename] = info
        all_bounds.append(info['bounds'])
        
        logging.info(f"Zone {i+1} ({filename}):")
        logging.info(f"  尺寸: {info['width']} x {info['height']}")
        logging.info(f"  边界: [{info['bounds'][0]:.6f}, {info['bounds'][1]:.6f}, {info['bounds'][2]:.6f}, {info['bounds'][3]:.6f}]")
        logging.info(f"  坐标系: {info['epsg']} ({info['units']})")
        logging.info(f"  像元大小: {info['pixel_size'][0]:.8f} x {info['pixel_size'][1]:.8f}")
    
    # 计算总体覆盖范围
    if all_bounds:
        min_x_global = min(bounds[0] for bounds in all_bounds)
        min_y_global = min(bounds[1] for bounds in all_bounds)
        max_x_global = max(bounds[2] for bounds in all_bounds)
        max_y_global = max(bounds[3] for bounds in all_bounds)
        
        logging.info(f"\n总体覆盖范围:")
        logging.info(f"  经度: {min_x_global:.6f} 到 {max_x_global:.6f} (跨度 {max_x_global-min_x_global:.6f}°)")
        logging.info(f"  纬度: {min_y_global:.6f} 到 {max_y_global:.6f} (跨度 {max_y_global-min_y_global:.6f}°)")
        
        return zone_info, [min_x_global, min_y_global, max_x_global, max_y_global]
    
    return zone_info, None

def analyze_zone_overlaps(zone_info):
    """分析UTM区域间的重叠情况"""
    logging.info("\n=== UTM区域重叠分析 ===")
    
    zones = list(zone_info.keys())
    overlaps = []
    
    for i in range(len(zones)):
        for j in range(i+1, len(zones)):
            zone1 = zones[i]
            zone2 = zones[j]
            
            bounds1 = zone_info[zone1]['bounds']
            bounds2 = zone_info[zone2]['bounds']
            
            # 计算重叠区域
            overlap_min_x = max(bounds1[0], bounds2[0])
            overlap_min_y = max(bounds1[1], bounds2[1])
            overlap_max_x = min(bounds1[2], bounds2[2])
            overlap_max_y = min(bounds1[3], bounds2[3])
            
            # 检查是否有重叠
            if overlap_min_x < overlap_max_x and overlap_min_y < overlap_max_y:
                overlap_width = overlap_max_x - overlap_min_x
                overlap_height = overlap_max_y - overlap_min_y
                overlap_area = overlap_width * overlap_height
                
                overlaps.append({
                    'zone1': zone1,
                    'zone2': zone2,
                    'overlap_bounds': [overlap_min_x, overlap_min_y, overlap_max_x, overlap_max_y],
                    'overlap_size': [overlap_width, overlap_height],
                    'overlap_area': overlap_area
                })
                
                logging.info(f"{zone1} 与 {zone2} 重叠:")
                logging.info(f"  重叠区域: [{overlap_min_x:.6f}, {overlap_min_y:.6f}, {overlap_max_x:.6f}, {overlap_max_y:.6f}]")
                logging.info(f"  重叠大小: {overlap_width:.6f}° x {overlap_height:.6f}°")
                logging.info(f"  重叠面积: {overlap_area:.8f} 平方度")
            else:
                # 计算间隙
                gap_x = max(0, overlap_min_x - overlap_max_x)
                gap_y = max(0, overlap_min_y - overlap_max_y)
                
                if gap_x > 0 or gap_y > 0:
                    logging.info(f"{zone1} 与 {zone2} 存在间隙:")
                    logging.info(f"  经度间隙: {gap_x:.6f}°")
                    logging.info(f"  纬度间隙: {gap_y:.6f}°")
    
    return overlaps

def analyze_mosaic_result(mosaic_path, expected_bounds):
    """分析镶嵌结果"""
    logging.info("\n=== 镶嵌结果分析 ===")
    
    if not os.path.exists(mosaic_path):
        logging.error(f"镶嵌结果文件不存在: {mosaic_path}")
        return None
    
    info = get_image_info(mosaic_path)
    if info is None:
        logging.error("无法读取镶嵌结果信息")
        return None
    
    logging.info(f"镶嵌结果文件: {os.path.basename(mosaic_path)}")
    logging.info(f"  尺寸: {info['width']} x {info['height']}")
    logging.info(f"  波段数: {info['bands']}")
    logging.info(f"  坐标系: {info['epsg']}")
    logging.info(f"  像元大小: {info['pixel_size'][0]:.8f} x {info['pixel_size'][1]:.8f}")
    logging.info(f"  实际边界: [{info['bounds'][0]:.6f}, {info['bounds'][1]:.6f}, {info['bounds'][2]:.6f}, {info['bounds'][3]:.6f}]")
    
    # 与期望边界比较
    if expected_bounds:
        logging.info(f"  期望边界: [{expected_bounds[0]:.6f}, {expected_bounds[1]:.6f}, {expected_bounds[2]:.6f}, {expected_bounds[3]:.6f}]")
        
        # 计算覆盖率
        actual_width = info['bounds'][2] - info['bounds'][0]
        actual_height = info['bounds'][3] - info['bounds'][1]
        expected_width = expected_bounds[2] - expected_bounds[0]
        expected_height = expected_bounds[3] - expected_bounds[1]
        
        width_coverage = (actual_width / expected_width) * 100
        height_coverage = (actual_height / expected_height) * 100
        
        logging.info(f"  覆盖率: 经度 {width_coverage:.1f}%, 纬度 {height_coverage:.1f}%")
        
        # 计算偏移
        x_offset = info['bounds'][0] - expected_bounds[0]
        y_offset = info['bounds'][1] - expected_bounds[1]
        
        logging.info(f"  边界偏移: X {x_offset:.6f}°, Y {y_offset:.6f}°")
        
        if abs(x_offset) > info['pixel_size'][0] or abs(y_offset) > info['pixel_size'][1]:
            logging.warning("检测到显著的边界偏移，可能影响空间精度")
    
    return info

def provide_optimization_suggestions(zone_info, overlaps, mosaic_info, expected_bounds):
    """提供优化建议"""
    logging.info("\n=== 优化建议 ===")
    
    suggestions = []
    
    # 检查像元大小一致性
    pixel_sizes = [info['pixel_size'][0] for info in zone_info.values()]
    if len(set([round(ps, 8) for ps in pixel_sizes])) > 1:
        suggestions.append("检测到不同的像元大小，建议统一重采样到相同分辨率")
        logging.warning(f"像元大小变化范围: {min(pixel_sizes):.8f} - {max(pixel_sizes):.8f}")
    
    # 检查重叠情况
    if overlaps:
        total_overlap_area = sum(overlap['overlap_area'] for overlap in overlaps)
        suggestions.append(f"检测到 {len(overlaps)} 个重叠区域，总面积 {total_overlap_area:.8f} 平方度")
        suggestions.append("建议使用更精确的边界裁剪或融合算法处理重叠区域")
    
    # 检查覆盖完整性
    if mosaic_info and expected_bounds:
        actual_area = (mosaic_info['bounds'][2] - mosaic_info['bounds'][0]) * (mosaic_info['bounds'][3] - mosaic_info['bounds'][1])
        expected_area = (expected_bounds[2] - expected_bounds[0]) * (expected_bounds[3] - expected_bounds[1])
        coverage_ratio = actual_area / expected_area
        
        if coverage_ratio < 0.95:
            suggestions.append(f"镶嵌结果覆盖率仅 {coverage_ratio*100:.1f}%，可能存在数据缺失")
        elif coverage_ratio > 1.05:
            suggestions.append(f"镶嵌结果覆盖率达 {coverage_ratio*100:.1f}%，可能存在额外数据")
    
    # 输出建议
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            logging.info(f"{i}. {suggestion}")
    else:
        logging.info("镶嵌质量良好，无明显问题")

def main():
    """主函数"""
    # 设置路径
    input_root = r"f:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.GEE分类\数据统计\裁剪后的结果"
    output_dir = r"f:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.GEE分类\数据统计\合并结果"
    
    # 设置日志
    log_file = setup_logging(output_dir)
    
    # 示例：分析2020年的镶嵌效果
    year = 2020
    
    logging.info(f"开始诊断 {year} 年多UTM区域镶嵌效果")
    
    # 查找输入文件
    input_files = []
    for zone in range(1, 11):  # zone1 到 zone10
        zone_dir = os.path.join(input_root, f"zone{zone}")
        if os.path.exists(zone_dir):
            for file in os.listdir(zone_dir):
                if file.endswith('.tif') and str(year) in file:
                    input_files.append(os.path.join(zone_dir, file))
    
    if not input_files:
        logging.error(f"未找到 {year} 年的输入文件")
        return
    
    logging.info(f"找到 {len(input_files)} 个输入文件")
    
    # 分析UTM区域覆盖
    zone_info, expected_bounds = analyze_utm_zones_coverage(input_files)
    
    # 分析区域重叠
    overlaps = analyze_zone_overlaps(zone_info)
    
    # 分析镶嵌结果
    mosaic_path = os.path.join(output_dir, f"merged_{year}.tif")
    mosaic_info = analyze_mosaic_result(mosaic_path, expected_bounds)
    
    # 提供优化建议
    provide_optimization_suggestions(zone_info, overlaps, mosaic_info, expected_bounds)
    
    logging.info(f"\n诊断完成，详细日志保存到: {log_file}")

if __name__ == "__main__":
    main()