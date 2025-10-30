#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本功能：为全球人工林产品数据添加颜色映射表
作者：锐多宝 (ruiduobao)
创建时间：2025年
用途：为30米分辨率人工林产品数据tif文件添加颜色映射表，便于可视化显示

颜色映射规则：
- 0: 空值/无数据 (透明)
- 1: 自然林 (绿色)
- 2: 人工林 (深绿色)
"""

import os
import sys
import logging
from datetime import datetime
from osgeo import gdal, gdalconst
import numpy as np

# 配置GDAL环境
gdal.UseExceptions()  # 启用GDAL异常处理

def setup_logging(log_dir="日志"):
    """
    设置日志记录系统
    
    Args:
        log_dir (str): 日志文件存储目录
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"颜色映射表_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger

def create_color_table():
    """
    创建颜色映射表
    
    Returns:
        gdal.ColorTable: GDAL颜色表对象
    """
    # 创建颜色表对象
    color_table = gdal.ColorTable()
    
    # 定义颜色映射 (R, G, B, Alpha)
    # 0: 空值/无数据 - 透明
    color_table.SetColorEntry(0, (0, 0, 0, 0))
    
    # 1: 自然林 - 绿色 (RGB: 34, 139, 34)
    color_table.SetColorEntry(1, (30, 144, 255, 255))
    
    # 2: 人工林 - 深绿色 (RGB: 0, 100, 0) 
    color_table.SetColorEntry(2, (34, 139, 34, 255))
    
    return color_table

def apply_color_table_to_raster(input_file, output_file, logger):
    """
    为栅格文件应用颜色映射表
    
    Args:
        input_file (str): 输入tif文件路径
        output_file (str): 输出tif文件路径
        logger (logging.Logger): 日志记录器
    
    Returns:
        bool: 处理是否成功
    """
    try:
        logger.info(f"开始处理文件: {input_file}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            logger.error(f"输入文件不存在: {input_file}")
            return False
        
        # 打开输入数据集
        logger.info("正在打开输入数据集...")
        src_ds = gdal.Open(input_file, gdalconst.GA_ReadOnly)
        if src_ds is None:
            logger.error(f"无法打开输入文件: {input_file}")
            return False
        
        # 获取数据集信息
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        bands = src_ds.RasterCount
        data_type = src_ds.GetRasterBand(1).DataType
        projection = src_ds.GetProjection()
        geotransform = src_ds.GetGeoTransform()
        
        logger.info(f"数据集信息: 宽度={width}, 高度={height}, 波段数={bands}")
        logger.info(f"数据类型: {gdal.GetDataTypeName(data_type)}")
        logger.info(f"投影信息: {projection[:100]}...")
        
        # 创建输出数据集 - 使用特殊的创建选项支持颜色表
        logger.info("正在创建输出数据集...")
        driver = gdal.GetDriverByName('GTiff')
        
        # 设置创建选项，专门为调色板模式优化
        creation_options = [
            'COMPRESS=LZW',           # LZW压缩
            'TILED=YES',              # 分块存储
            'BIGTIFF=IF_SAFER',       # 大文件支持
            'PHOTOMETRIC=PALETTE'     # 调色板模式 - 在创建时就设置
        ]
        
        dst_ds = driver.Create(
            output_file, 
            width, 
            height, 
            bands, 
            data_type,
            creation_options
        )
        
        if dst_ds is None:
            logger.error(f"无法创建输出文件: {output_file}")
            src_ds = None
            return False
        
        # 设置地理信息
        dst_ds.SetProjection(projection)
        dst_ds.SetGeoTransform(geotransform)
        
        # 创建颜色表
        logger.info("正在创建颜色映射表...")
        color_table = create_color_table()
        
        # 处理每个波段
        for band_idx in range(1, bands + 1):
            logger.info(f"正在处理第 {band_idx} 波段...")
            
            # 获取源波段
            src_band = src_ds.GetRasterBand(band_idx)
            dst_band = dst_ds.GetRasterBand(band_idx)
            
            # 先设置颜色表和颜色解释（在写入数据之前）
            logger.info("正在设置颜色映射表...")
            dst_band.SetColorInterpretation(gdalconst.GCI_PaletteIndex)
            result = dst_band.SetColorTable(color_table)
            if result != 0:
                logger.warning(f"设置颜色表返回代码: {result}")
            else:
                logger.info("颜色表设置成功")
            
            # 设置NoData值
            nodata_value = src_band.GetNoDataValue()
            if nodata_value is not None:
                dst_band.SetNoDataValue(nodata_value)
                logger.info(f"设置NoData值: {nodata_value}")
            else:
                dst_band.SetNoDataValue(0)
                logger.info("设置NoData值: 0")
            
            # 分块复制数据，避免内存溢出
            logger.info("正在分块复制栅格数据...")
            
            # 设置分块大小（根据可用内存调整）
            block_size = 4096 # 1024x1024像素块
            
            # 计算分块数量
            x_blocks = (width + block_size - 1) // block_size
            y_blocks = (height + block_size - 1) // block_size
            total_blocks = x_blocks * y_blocks
            
            logger.info(f"数据将分为 {x_blocks} x {y_blocks} = {total_blocks} 个块进行处理")
            
            # 逐块处理数据
            processed_blocks = 0
            for y_block in range(y_blocks):
                for x_block in range(x_blocks):
                    # 计算当前块的范围
                    x_start = x_block * block_size
                    y_start = y_block * block_size
                    x_size = min(block_size, width - x_start)
                    y_size = min(block_size, height - y_start)
                    
                    # 读取当前块数据
                    block_data = src_band.ReadAsArray(x_start, y_start, x_size, y_size)
                    
                    # 写入当前块数据
                    dst_band.WriteArray(block_data, x_start, y_start)
                    
                    processed_blocks += 1
                    
                    # 每处理100个块显示一次进度
                    if processed_blocks % 100 == 0 or processed_blocks == total_blocks:
                        progress = (processed_blocks / total_blocks) * 100
                        logger.info(f"处理进度: {processed_blocks}/{total_blocks} ({progress:.1f}%)")
            
            logger.info("栅格数据复制完成")
            
            # 计算统计信息
            logger.info("正在计算统计信息...")
            dst_band.ComputeStatistics(False)
        
        # 强制写入磁盘
        dst_ds.FlushCache()
        
        # 关闭数据集
        src_ds = None
        dst_ds = None
        
        # 验证输出文件
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            logger.info(f"颜色映射表应用成功!")
            logger.info(f"输出文件: {output_file}")
            logger.info(f"文件大小: {file_size:.2f} MB")
            return True
        else:
            logger.error("输出文件创建失败")
            return False
            
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        return False
    
    finally:
        # 确保数据集被正确关闭
        try:
            if 'src_ds' in locals():
                src_ds = None
            if 'dst_ds' in locals():
                dst_ds = None
        except:
            pass

def main():
    """
    主函数：执行颜色映射表应用流程
    """
    # 设置日志系统
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("全球人工林产品数据颜色映射表应用程序启动")
    logger.info("作者: 锐多宝 (ruiduobao)")
    logger.info("=" * 60)
    
    # 定义输入输出路径
    input_file = r"D:\地理所\论文\东南亚10m人工林提取\数据\全球人工林产品数据\southeast_asia_PlantTree_2021_mosaic.tif"
    output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\全球人工林产品数据"
    output_file = os.path.join(output_dir, "southeast_asia_PlantTree_2021_mosaic_带颜色映射表.tif")
    
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    
    # 检查输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"处理开始时间: {start_time}")
    
    # 应用颜色映射表
    success = apply_color_table_to_raster(input_file, output_file, logger)
    
    # 记录结束时间和统计信息
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 60)
    if success:
        logger.info("颜色映射表应用完成!")
        logger.info("颜色映射规则:")
        logger.info("  0: 空值/无数据 (透明)")
        logger.info("  1: 自然林 (绿色)")
        logger.info("  2: 人工林 (深绿色)")
    else:
        logger.error("颜色映射表应用失败!")
    
    logger.info(f"总耗时: {duration:.2f} 秒")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    # 执行主程序
    success = main()
    
    # 设置退出代码
    sys.exit(0 if success else 1)