#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间偏移诊断脚本
用于检查镶嵌前后影像的空间对齐情况，诊断偏移原因

作者: 锐多宝 (ruiduobao)
功能: 
1. 比较原始影像和镶嵌结果的坐标系统、分辨率、原点等信息
2. 计算空间偏移量并输出详细诊断报告
3. 生成日志文件记录所有诊断信息
"""

import os
import sys
import logging
from osgeo import gdal, osr
import numpy as np
from datetime import datetime

def setup_logging(output_dir):
    """
    设置日志记录
    """
    log_file = os.path.join(output_dir, f"空间偏移诊断_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"诊断日志保存至: {log_file}")
    return log_file

def get_image_info(image_path):
    """
    获取影像的详细信息
    
    参数:
        image_path: 影像文件路径
    
    返回:
        dict: 包含影像信息的字典
    """
    try:
        ds = gdal.Open(image_path)
        if ds is None:
            return None
        
        # 获取基本信息
        info = {
            'path': image_path,
            'width': ds.RasterXSize,
            'height': ds.RasterYSize,
            'bands': ds.RasterCount
        }
        
        # 获取地理变换信息
        gt = ds.GetGeoTransform()
        if gt:
            info['geotransform'] = gt
            info['origin_x'] = gt[0]  # 左上角X坐标
            info['origin_y'] = gt[3]  # 左上角Y坐标
            info['pixel_width'] = gt[1]  # 像元宽度
            info['pixel_height'] = abs(gt[5])  # 像元高度（取绝对值）
            info['rotation_x'] = gt[2]  # X方向旋转
            info['rotation_y'] = gt[4]  # Y方向旋转
            
            # 计算影像范围
            info['min_x'] = gt[0]
            info['max_x'] = gt[0] + gt[1] * ds.RasterXSize
            info['max_y'] = gt[3]
            info['min_y'] = gt[3] + gt[5] * ds.RasterYSize
        
        # 获取投影信息
        proj_wkt = ds.GetProjection()
        if proj_wkt:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj_wkt)
            info['projection'] = proj_wkt
            info['epsg_code'] = srs.GetAuthorityCode(None)
            info['is_geographic'] = srs.IsGeographic()
            info['is_projected'] = srs.IsProjected()
            info['units'] = srs.GetLinearUnitsName() if srs.IsProjected() else 'degrees'
        
        ds = None
        return info
        
    except Exception as e:
        logging.error(f"读取影像信息失败 {image_path}: {str(e)}")
        return None

def calculate_offset(info1, info2):
    """
    计算两个影像之间的空间偏移
    
    参数:
        info1: 第一个影像信息
        info2: 第二个影像信息
    
    返回:
        dict: 偏移信息
    """
    offset_info = {}
    
    # 计算原点偏移
    if 'origin_x' in info1 and 'origin_x' in info2:
        offset_x = abs(info1['origin_x'] - info2['origin_x'])
        offset_y = abs(info1['origin_y'] - info2['origin_y'])
        
        offset_info['origin_offset_x'] = offset_x
        offset_info['origin_offset_y'] = offset_y
        
        # 如果是地理坐标系，转换为米
        if info1.get('is_geographic', False):
            # 在赤道附近，1度约等于111320米
            offset_x_m = offset_x * 111320
            offset_y_m = offset_y * 111320
            offset_info['origin_offset_x_meters'] = offset_x_m
            offset_info['origin_offset_y_meters'] = offset_y_m
            offset_info['total_offset_meters'] = np.sqrt(offset_x_m**2 + offset_y_m**2)
        else:
            offset_info['origin_offset_x_meters'] = offset_x
            offset_info['origin_offset_y_meters'] = offset_y
            offset_info['total_offset_meters'] = np.sqrt(offset_x**2 + offset_y**2)
    
    # 计算分辨率差异
    if 'pixel_width' in info1 and 'pixel_width' in info2:
        res_diff_x = abs(info1['pixel_width'] - info2['pixel_width'])
        res_diff_y = abs(info1['pixel_height'] - info2['pixel_height'])
        
        offset_info['resolution_diff_x'] = res_diff_x
        offset_info['resolution_diff_y'] = res_diff_y
        
        # 计算分辨率差异百分比
        if info1['pixel_width'] > 0:
            offset_info['resolution_diff_x_percent'] = (res_diff_x / info1['pixel_width']) * 100
        if info1['pixel_height'] > 0:
            offset_info['resolution_diff_y_percent'] = (res_diff_y / info1['pixel_height']) * 100
    
    return offset_info

def diagnose_spatial_alignment(original_files, mosaic_file, output_dir):
    """
    诊断空间对齐问题
    
    参数:
        original_files: 原始影像文件列表
        mosaic_file: 镶嵌结果文件
        output_dir: 输出目录
    """
    logging.info("="*60)
    logging.info("开始空间偏移诊断")
    logging.info("="*60)
    
    # 获取镶嵌结果信息
    logging.info(f"分析镶嵌结果: {mosaic_file}")
    mosaic_info = get_image_info(mosaic_file)
    
    if mosaic_info is None:
        logging.error("无法读取镶嵌结果文件")
        return
    
    # 输出镶嵌结果信息
    logging.info("\n镶嵌结果信息:")
    logging.info(f"  尺寸: {mosaic_info['width']} x {mosaic_info['height']}")
    logging.info(f"  波段数: {mosaic_info['bands']}")
    logging.info(f"  EPSG代码: {mosaic_info.get('epsg_code', 'Unknown')}")
    logging.info(f"  坐标系类型: {'地理坐标系' if mosaic_info.get('is_geographic') else '投影坐标系'}")
    logging.info(f"  单位: {mosaic_info.get('units', 'Unknown')}")
    logging.info(f"  像元大小: {mosaic_info.get('pixel_width', 0):.8f} x {mosaic_info.get('pixel_height', 0):.8f}")
    logging.info(f"  原点坐标: ({mosaic_info.get('origin_x', 0):.8f}, {mosaic_info.get('origin_y', 0):.8f})")
    logging.info(f"  影像范围: X[{mosaic_info.get('min_x', 0):.6f}, {mosaic_info.get('max_x', 0):.6f}], Y[{mosaic_info.get('min_y', 0):.6f}, {mosaic_info.get('max_y', 0):.6f}]")
    
    # 分析原始文件
    logging.info("\n原始影像分析:")
    logging.info("-" * 40)
    
    max_offset = 0
    max_offset_file = ""
    
    for i, original_file in enumerate(original_files[:5]):  # 只分析前5个文件
        if not os.path.exists(original_file):
            logging.warning(f"文件不存在: {original_file}")
            continue
            
        logging.info(f"\n分析原始文件 {i+1}: {os.path.basename(original_file)}")
        
        original_info = get_image_info(original_file)
        if original_info is None:
            continue
        
        # 输出原始文件信息
        logging.info(f"  尺寸: {original_info['width']} x {original_info['height']}")
        logging.info(f"  EPSG代码: {original_info.get('epsg_code', 'Unknown')}")
        logging.info(f"  坐标系类型: {'地理坐标系' if original_info.get('is_geographic') else '投影坐标系'}")
        logging.info(f"  像元大小: {original_info.get('pixel_width', 0):.8f} x {original_info.get('pixel_height', 0):.8f}")
        logging.info(f"  原点坐标: ({original_info.get('origin_x', 0):.8f}, {original_info.get('origin_y', 0):.8f})")
        
        # 计算偏移
        offset_info = calculate_offset(original_info, mosaic_info)
        
        if 'total_offset_meters' in offset_info:
            total_offset = offset_info['total_offset_meters']
            logging.info(f"  空间偏移: {total_offset:.2f} 米")
            logging.info(f"    X方向: {offset_info.get('origin_offset_x_meters', 0):.2f} 米")
            logging.info(f"    Y方向: {offset_info.get('origin_offset_y_meters', 0):.2f} 米")
            
            if total_offset > max_offset:
                max_offset = total_offset
                max_offset_file = original_file
        
        # 检查分辨率差异
        if 'resolution_diff_x_percent' in offset_info:
            logging.info(f"  分辨率差异: X方向 {offset_info['resolution_diff_x_percent']:.2f}%, Y方向 {offset_info['resolution_diff_y_percent']:.2f}%")
    
    # 输出诊断总结
    logging.info("\n" + "="*60)
    logging.info("诊断总结")
    logging.info("="*60)
    logging.info(f"最大空间偏移: {max_offset:.2f} 米")
    if max_offset_file:
        logging.info(f"最大偏移文件: {os.path.basename(max_offset_file)}")
    
    # 提供建议
    logging.info("\n改进建议:")
    if max_offset > 10:
        logging.info("- 偏移较大(>10米)，建议检查:")
        logging.info("  1. 源数据坐标系是否正确")
        logging.info("  2. 重投影参数设置")
        logging.info("  3. 输出边界对齐设置")
    elif max_offset > 5:
        logging.info("- 偏移中等(5-10米)，建议:")
        logging.info("  1. 启用输出边界对齐(outputBounds)")
        logging.info("  2. 检查像元对齐设置")
    elif max_offset > 1:
        logging.info("- 偏移较小(1-5米)，可能原因:")
        logging.info("  1. 子像元级别的网格对齐差异")
        logging.info("  2. 重采样算法影响")
    else:
        logging.info("- 偏移很小(<1米)，在可接受范围内")

def main():
    """
    主函数
    """
    # 设置输出目录
    output_dir = r"f:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.GEE分类\数据统计\诊断结果"
    
    # 设置日志
    log_file = setup_logging(output_dir)
    
    # 示例：诊断2020年的结果
    year = 2020
    
    # 原始文件目录
    input_dir = r"f:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.GEE分类\数据统计\分年度结果"
    year_dir = os.path.join(input_dir, str(year))
    
    # 镶嵌结果文件
    mosaic_file = os.path.join(input_dir, f"东南亚人工林分类结果_{year}_合并.tif")
    
    if not os.path.exists(mosaic_file):
        logging.error(f"镶嵌结果文件不存在: {mosaic_file}")
        return
    
    # 获取原始文件列表
    original_files = []
    if os.path.exists(year_dir):
        for file in os.listdir(year_dir):
            if file.endswith('.tif'):
                original_files.append(os.path.join(year_dir, file))
    
    if not original_files:
        logging.error(f"未找到原始文件: {year_dir}")
        return
    
    logging.info(f"找到 {len(original_files)} 个原始文件")
    
    # 执行诊断
    diagnose_spatial_alignment(original_files, mosaic_file, output_dir)
    
    logging.info(f"\n诊断完成！详细日志已保存至: {log_file}")

if __name__ == "__main__":
    main()