#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本功能：为GEE分类结果添加颜色映射表并构建金字塔
作者：锐多宝 (ruiduobao)
创建时间：2025年
用途：为人工林分类结果tif文件添加颜色映射表，便于可视化显示，并自动构建外置金字塔提升显示性能

颜色映射规则：
- 0: 空值/无数据 (透明)
- 1: 人工林 (绿色)
- 2: 其他植被 (蓝色) 
- 3: 非植被 (浅灰色)

金字塔功能：
- 自动构建外置.ovr金字塔文件
- 使用LZW压缩减小金字塔文件大小
- 使用最近邻重采样保持分类数据完整性
- 默认层级：2, 4, 8, 16, 32, 64倍缩放
- 显著提升大文件的显示和缩放性能
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
    
    # 1: 人工林 - 绿色 (RGB: 34, 139, 34)
    color_table.SetColorEntry(1, (30, 144, 255, 255))
    
    # 2: 其他植被 - 蓝色 (RGB: 30, 144, 255) 
    color_table.SetColorEntry(2, (34, 139, 34, 255))
    
    # 3: 非植被 - 浅灰色 (RGB: 192, 192, 192)
    color_table.SetColorEntry(3, (192, 192, 192, 255))
    
    return color_table

def build_overviews(file_path, logger, overview_levels=None):
    """
    为栅格文件构建外置金字塔（.ovr文件）
    
    Args:
        file_path (str): 栅格文件路径
        logger (logging.Logger): 日志记录器
        overview_levels (list): 金字塔层级列表，默认为[2, 4, 8, 16, 32, 64]
    
    Returns:
        bool: 构建是否成功
    """
    try:
        logger.info(f"开始为文件构建金字塔: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
        
        # 设置默认金字塔层级
        if overview_levels is None:
            overview_levels = [2, 4, 8, 16, 32, 64]
        
        logger.info(f"金字塔层级: {overview_levels}")
        
        # 打开数据集
        dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
        if dataset is None:
            logger.error(f"无法打开文件: {file_path}")
            return False
        
        # 获取数据集信息
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount
        
        logger.info(f"数据集信息: {width}x{height} 像素, {bands} 个波段")
        
        # 设置金字塔构建选项
        # NEAREST: 最近邻重采样，适合分类数据
        # AVERAGE: 平均值重采样，适合连续数据
        # MODE: 众数重采样，适合分类数据，保持类别完整性
        resampling_method = "NEAREST"  # 对于分类数据使用最近邻
        
        logger.info(f"使用重采样方法: {resampling_method}")
        
        # 设置金字塔压缩选项
        # 为金字塔文件启用LZW压缩以减小文件大小
        logger.info("设置金字塔压缩选项: LZW")
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
        gdal.SetConfigOption('TILED_OVERVIEW', 'YES')  # 启用分块存储
        gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'IF_SAFER')  # 大文件支持
        
        # 构建金字塔
        logger.info("正在构建压缩金字塔...")
        
        # 使用GDAL的BuildOverviews方法
        # 参数说明:
        # - resampling_method: 重采样方法
        # - overview_levels: 金字塔层级列表
        # - callback: 进度回调函数（可选）
        result = dataset.BuildOverviews(resampling_method, overview_levels)
        
        if result == 0:
            logger.info("金字塔构建成功!")
            
            # 检查生成的.ovr文件
            ovr_file = file_path + ".ovr"
            if os.path.exists(ovr_file):
                ovr_size = os.path.getsize(ovr_file) / (1024 * 1024)  # MB
                logger.info(f"金字塔文件: {ovr_file}")
                logger.info(f"金字塔文件大小: {ovr_size:.2f} MB")
            else:
                logger.warning("未找到.ovr文件，金字塔可能内嵌在原文件中")
            
            # 获取金字塔信息
            band = dataset.GetRasterBand(1)
            overview_count = band.GetOverviewCount()
            logger.info(f"金字塔层数: {overview_count}")
            
            # 显示每层金字塔的尺寸
            for i in range(overview_count):
                overview = band.GetOverview(i)
                ov_width = overview.XSize
                ov_height = overview.YSize
                scale_factor = width / ov_width
                logger.info(f"  层级 {i+1}: {ov_width}x{ov_height} (缩放比例: 1:{scale_factor:.0f})")
            
            dataset = None  # 关闭数据集
            return True
            
        else:
            logger.error(f"金字塔构建失败，错误代码: {result}")
            dataset = None
            return False
            
    except Exception as e:
        logger.error(f"构建金字塔时发生错误: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        return False
    
    finally:
        # 清理GDAL配置选项
        gdal.SetConfigOption('COMPRESS_OVERVIEW', None)
        gdal.SetConfigOption('TILED_OVERVIEW', None)
        gdal.SetConfigOption('BIGTIFF_OVERVIEW', None)
        
        # 确保数据集被正确关闭
        try:
            if 'dataset' in locals() and dataset is not None:
                dataset = None
        except:
            pass

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
            
            # 构建金字塔
            logger.info("正在为输出文件构建金字塔...")
            pyramid_success = build_overviews(output_file, logger)
            
            if pyramid_success:
                logger.info("金字塔构建完成!")
            else:
                logger.warning("金字塔构建失败，但主文件处理成功")
            
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

def process_folder(input_folder, logger):
    """
    批量处理文件夹中的所有tif文件
    
    Args:
        input_folder (str): 输入文件夹路径
        logger (logging.Logger): 日志记录器
    
    Returns:
        tuple: (成功处理的文件数, 总文件数)
    """
    logger.info(f"开始扫描文件夹: {input_folder}")
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        logger.error(f"输入文件夹不存在: {input_folder}")
        return 0, 0
    
    # 查找所有tif文件
    tif_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.tif', '.tiff')):
            # 跳过已经处理过的文件（包含"_添加颜色映射表"的文件）
            if "_添加颜色映射表" not in file:
                tif_files.append(file)
    
    if not tif_files:
        logger.warning("未找到需要处理的tif文件")
        return 0, 0
    
    logger.info(f"找到 {len(tif_files)} 个tif文件需要处理")
    
    # 统计处理结果
    success_count = 0
    total_count = len(tif_files)
    
    # 逐个处理文件
    for i, filename in enumerate(tif_files, 1):
        logger.info("=" * 50)
        logger.info(f"处理文件 {i}/{total_count}: {filename}")
        
        # 构建输入和输出文件路径
        input_file = os.path.join(input_folder, filename)
        
        # 生成输出文件名：在原文件名后添加"_添加颜色映射表"
        name_without_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        output_filename = f"{name_without_ext}_添加颜色映射表{ext}"
        output_file = os.path.join(input_folder, output_filename)
        
        logger.info(f"输入文件: {input_file}")
        logger.info(f"输出文件: {output_file}")
        
        # 检查输出文件是否已存在
        if os.path.exists(output_file):
            logger.warning(f"输出文件已存在，跳过处理: {output_file}")
            continue
        
        # 处理单个文件
        file_start_time = datetime.now()
        success = apply_color_table_to_raster(input_file, output_file, logger)
        file_end_time = datetime.now()
        file_duration = (file_end_time - file_start_time).total_seconds()
        
        if success:
            success_count += 1
            logger.info(f"✓ 文件处理成功，耗时: {file_duration:.2f} 秒")
        else:
            logger.error(f"✗ 文件处理失败，耗时: {file_duration:.2f} 秒")
    
    return success_count, total_count

def main():
    """
    主函数：执行颜色映射表应用流程
    """
    # 设置日志系统
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("GEE分类结果颜色映射表批量应用程序启动")
    logger.info("作者: 锐多宝 (ruiduobao)")
    logger.info("功能: 批量为文件夹中的所有tif文件添加颜色映射表")
    logger.info("=" * 60)
    
    # 定义输入文件夹路径
    # 用户可以修改这个路径来指定要处理的文件夹
    input_folder = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\zone7首尾填充"
    
    logger.info(f"输入文件夹: {input_folder}")
    logger.info("输出文件将保存在同一文件夹中，文件名添加'_添加颜色映射表'后缀")
    
    # 记录开始时间
    start_time = datetime.now()
    logger.info(f"批量处理开始时间: {start_time}")
    
    # 批量处理文件夹
    success_count, total_count = process_folder(input_folder, logger)
    
    # 记录结束时间和统计信息
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 60)
    logger.info("批量处理完成!")
    logger.info(f"处理结果: {success_count}/{total_count} 个文件成功")
    
    if success_count > 0:
        logger.info("颜色映射规则:")
        logger.info("  0: 空值/无数据 (透明)")
        logger.info("  1: 人工林 (蓝色)")
        logger.info("  2: 其他植被 (绿色)")
        logger.info("  3: 非植被 (浅灰色)")
        logger.info("已自动为所有文件构建LZW压缩的外置金字塔(.ovr)文件以提升显示性能")
    
    if success_count < total_count:
        failed_count = total_count - success_count
        logger.warning(f"有 {failed_count} 个文件处理失败，请检查日志获取详细信息")
    
    logger.info(f"总耗时: {duration:.2f} 秒")
    logger.info("=" * 60)
    
    return success_count == total_count

if __name__ == "__main__":
    # 执行主程序
    success = main()
    
    # 设置退出代码
    sys.exit(0 if success else 1)