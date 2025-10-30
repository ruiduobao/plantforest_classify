#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码目的：对tif文件进行高效镶嵌合并（按区域和年份分组）
功能：
1. 读取裁剪后的所有tif文件
2. 按区域和年份分组进行镶嵌
3. 使用分层镶嵌策略提高处理效率
4. 支持大文件处理，避免内存溢出
5. 输出按区域年份命名的镶嵌结果
6. 可选择使用VRT虚拟栅格技术加速
7. 自动为输出文件添加颜色映射表和构建金字塔

镶嵌策略分析：
- VRT技术 - 虚拟镶嵌，速度最快，内存消耗最小
- 特别适合含大量nodata值的栅格数据
- 充分利用多核CPU，避免64GB内存限制

作者：锐多宝 (ruiduobao)
日期：2025年1月
"""

import os
import sys
import logging
import time
import re
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# 设置日志
def setup_logging(output_dir):
    """
    设置日志记录
    """
    log_filename = f"esri_mosaic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

def create_color_table():
    """
    创建颜色映射表
    
    Returns:
        gdal.ColorTable: GDAL颜色表对象
    """
    try:
        from osgeo import gdal
        
        # 创建颜色表对象
        color_table = gdal.ColorTable()
        
        # 定义颜色映射 (R, G, B, Alpha)
        # 0: 空值/无数据 - 透明
        color_table.SetColorEntry(0, (0, 0, 0, 0))
        
        # 1: 人工林 - 蓝色 (RGB: 30, 144, 255)
        color_table.SetColorEntry(1, (30, 144, 255, 255))
        
        # 2: 其他植被 - 绿色 (RGB: 34, 139, 34) 
        color_table.SetColorEntry(2, (34, 139, 34, 255))
        
        # 3: 非植被 - 浅灰色 (RGB: 192, 192, 192)
        color_table.SetColorEntry(3, (192, 192, 192, 255))
        
        return color_table
        
    except ImportError:
        logging.error("GDAL未安装，无法创建颜色映射表")
        return None

def build_overviews(file_path, overview_levels=None):
    """
    为栅格文件构建外置金字塔（.ovr文件）
    
    Args:
        file_path (str): 栅格文件路径
        overview_levels (list): 金字塔层级列表，默认为[2, 4, 8, 16, 32, 64]
    
    Returns:
        bool: 构建是否成功
    """
    try:
        from osgeo import gdal
        
        logging.info(f"开始为文件构建金字塔: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            return False
        
        # 设置默认金字塔层级
        if overview_levels is None:
            overview_levels = [2, 4, 8, 16, 32, 64]
        
        logging.info(f"金字塔层级: {overview_levels}")
        
        # 打开数据集
        dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
        if dataset is None:
            logging.error(f"无法打开文件: {file_path}")
            return False
        
        # 获取数据集信息
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount
        
        logging.info(f"数据集信息: {width}x{height} 像素, {bands} 个波段")
        
        # 设置金字塔构建选项
        resampling_method = "NEAREST"  # 对于分类数据使用最近邻
        
        logging.info(f"使用重采样方法: {resampling_method}")
        
        # 设置金字塔压缩选项
        logging.info("设置金字塔压缩选项: LZW")
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
        gdal.SetConfigOption('TILED_OVERVIEW', 'YES')  # 启用分块存储
        gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'IF_SAFER')  # 大文件支持
        
        # 构建金字塔
        logging.info("正在构建压缩金字塔...")
        
        result = dataset.BuildOverviews(resampling_method, overview_levels)
        
        if result == 0:
            logging.info("金字塔构建成功!")
            
            # 检查生成的.ovr文件
            ovr_file = file_path + ".ovr"
            if os.path.exists(ovr_file):
                ovr_size = os.path.getsize(ovr_file) / (1024 * 1024)  # MB
                logging.info(f"金字塔文件: {ovr_file}")
                logging.info(f"金字塔文件大小: {ovr_size:.2f} MB")
            else:
                logging.warning("未找到.ovr文件，金字塔可能内嵌在原文件中")
            
            # 获取金字塔信息
            band = dataset.GetRasterBand(1)
            overview_count = band.GetOverviewCount()
            logging.info(f"金字塔层数: {overview_count}")
            
            # 显示每层金字塔的尺寸
            for i in range(overview_count):
                overview = band.GetOverview(i)
                ov_width = overview.XSize
                ov_height = overview.YSize
                scale_factor = width / ov_width
                logging.info(f"  层级 {i+1}: {ov_width}x{ov_height} (缩放比例: 1:{scale_factor:.0f})")
            
            dataset = None  # 关闭数据集
            return True
            
        else:
            logging.error(f"金字塔构建失败，错误代码: {result}")
            dataset = None
            return False
            
    except ImportError:
        logging.error("GDAL未安装，无法构建金字塔")
        return False
    except Exception as e:
        logging.error(f"构建金字塔时发生错误: {str(e)}")
        logging.error(f"错误类型: {type(e).__name__}")
        return False
    
    finally:
        # 清理GDAL配置选项
        try:
            from osgeo import gdal
            gdal.SetConfigOption('COMPRESS_OVERVIEW', None)
            gdal.SetConfigOption('TILED_OVERVIEW', None)
            gdal.SetConfigOption('BIGTIFF_OVERVIEW', None)
        except:
            pass
        
        # 确保数据集被正确关闭
        try:
            if 'dataset' in locals() and dataset is not None:
                dataset = None
        except:
            pass

def apply_color_table_and_build_pyramids(file_path):
    """
    为栅格文件应用颜色映射表并构建金字塔
    
    Args:
        file_path (str): 栅格文件路径
    
    Returns:
        bool: 处理是否成功
    """
    try:
        from osgeo import gdal, gdalconst
        
        logging.info(f"开始为文件应用颜色映射表和构建金字塔: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            return False
        
        # 打开数据集（可写模式）
        dataset = gdal.Open(file_path, gdal.GA_Update)
        if dataset is None:
            logging.error(f"无法打开文件: {file_path}")
            return False
        
        # 创建颜色映射表
        color_table = create_color_table()
        if color_table is None:
            logging.error("无法创建颜色映射表")
            dataset = None
            return False
        
        # 应用颜色映射表到第一个波段
        band = dataset.GetRasterBand(1)
        band.SetColorInterpretation(gdalconst.GCI_PaletteIndex)
        result = band.SetColorTable(color_table)
        
        if result == 0:
            logging.info("颜色映射表应用成功")
        else:
            logging.warning(f"颜色映射表应用返回代码: {result}")
        
        # 强制写入磁盘
        dataset.FlushCache()
        dataset = None  # 关闭数据集
        
        # 构建金字塔
        pyramid_success = build_overviews(file_path)
        
        if pyramid_success:
            logging.info("颜色映射表和金字塔处理完成")
            return True
        else:
            logging.warning("颜色映射表应用成功，但金字塔构建失败")
            return True  # 颜色映射表成功就算成功
            
    except ImportError:
        logging.error("GDAL未安装，无法处理颜色映射表和金字塔")
        return False
    except Exception as e:
        logging.error(f"应用颜色映射表和构建金字塔时出错: {str(e)}")
        return False
    
    finally:
        # 确保数据集被正确关闭
        try:
            if 'dataset' in locals() and dataset is not None:
                dataset = None
        except:
            pass

def force_remove_directory(dir_path, max_retries=3):
    """
    强制删除目录及其所有内容（包括非空目录）
    
    参数:
        dir_path: 要删除的目录路径
        max_retries: 最大重试次数
    
    返回:
        bool: 删除是否成功
    """
    if not os.path.exists(dir_path):
        logging.info(f"目录不存在，无需删除: {dir_path}")
        return True
    
    for attempt in range(max_retries):
        try:
            # 使用shutil.rmtree强制删除整个目录树
            shutil.rmtree(dir_path, ignore_errors=False)
            logging.info(f"临时目录删除成功: {dir_path}")
            return True
            
        except PermissionError as e:
            logging.warning(f"删除目录权限错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)  # 等待1秒后重试
                continue
            else:
                logging.error(f"删除目录失败，权限不足: {dir_path}")
                return False
                
        except OSError as e:
            logging.warning(f"删除目录系统错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)  # 等待1秒后重试
                continue
            else:
                logging.error(f"删除目录失败，系统错误: {dir_path}")
                return False
                
        except Exception as e:
            logging.error(f"删除目录时发生未知错误: {str(e)}")
            return False
    
    return False

def parse_filename(filename):
    """
    解析文件名，提取区域和年份信息
    
    参数:
        filename: 文件名，如 "zone10_classification_2017_rf100-0000000000-0000000000.tif"
    
    返回:
        tuple: (zone, year) 或 (None, None) 如果解析失败
    """
    try:
        # 匹配文件名模式：zone{数字}_classification_{年份}_...
        pattern = r'zone(\d+)_classification_(\d{4})_'
        match = re.search(pattern, filename)
        
        if match:
            zone = f"zone{match.group(1)}"
            year = match.group(2)
            return zone, year
        else:
            logging.warning(f"无法解析文件名: {filename}")
            return None, None
            
    except Exception as e:
        logging.error(f"解析文件名时出错 {filename}: {str(e)}")
        return None, None

def group_files_by_zone_year(input_dir):
    """
    按区域和年份对文件进行分组
    
    参数:
        input_dir: 输入目录
    
    返回:
        dict: {(zone, year): [file_paths]}
    """
    grouped_files = defaultdict(list)
    
    try:
        # 获取所有tif文件
        for file in os.listdir(input_dir):
            if file.lower().endswith('.tif'):
                file_path = os.path.join(input_dir, file)
                zone, year = parse_filename(file)
                
                if zone and year:
                    grouped_files[(zone, year)].append(file_path)
                    logging.debug(f"文件 {file} 归类到 {zone}_{year}")
                else:
                    logging.warning(f"跳过无法解析的文件: {file}")
        
        # 输出分组统计
        logging.info(f"文件分组完成，共找到 {len(grouped_files)} 个区域-年份组合:")
        for (zone, year), files in grouped_files.items():
            logging.info(f"  {zone}_{year}: {len(files)} 个文件")
        
        return dict(grouped_files)
        
    except Exception as e:
        logging.error(f"文件分组时出错: {str(e)}")
        return {}

# VRT相关函数保留，其他函数已移除以优化性能

def create_unified_mosaic(input_files, output_path):
    """
    使用gdalwarp创建统一投影的镶嵌文件（解决投影不一致问题）
    
    参数:
        input_files: 输入文件列表
        output_path: 输出路径
    
    返回:
        bool: 是否成功
    """
    try:
        from osgeo import gdal
        
        logging.info(f"使用gdalwarp创建统一投影镶嵌，包含 {len(input_files)} 个文件")
        logging.info("统一投影到WGS84地理坐标系，解决UTM投影不一致问题")
        logging.info("输出数据类型: uint8 (Byte)，像素值范围: 0-255")
        logging.info("NoData值设置为: 0 (表示空值/无数据区域)")
        
        # 设置GDAL配置以优化性能和内存使用（高性能配置）
        gdal.SetConfigOption('GDAL_CACHEMAX', '4096')  # 4GB缓存（提升）
        gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')   # 使用所有CPU核心
        gdal.SetConfigOption('VSI_CACHE', 'TRUE')  # 启用VSI缓存
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')  # 优化文件打开
        gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '200')  # 增加数据集池大小
        gdal.SetConfigOption('GDAL_SWATH_SIZE', '0')  # 禁用swath限制，提升大文件处理
        gdal.SetConfigOption('GDAL_MAX_RAW_BLOCK_CACHE_SIZE', '200000000')  # 200MB原始块缓存
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 使用gdalwarp进行重投影和镶嵌
        warp_options = gdal.WarpOptions(
            format='GTiff',
            dstSRS='EPSG:4326',  # 目标投影：WGS84地理坐标系
            xRes=0.0001,  # 约10米分辨率（地理坐标）
            yRes=0.0001,
            resampleAlg='nearest',  # 最快的重采样方法
            outputType=gdal.GDT_Byte,  # 输出数据类型为uint8（Byte）
            srcNodata=0,  # 源数据nodata值
            dstNodata=0,  # 目标nodata值
            creationOptions=[
                'COMPRESS=LZW',  # LZW压缩
                'TILED=YES',  # 瓦片存储
                'BLOCKXSIZE=1024',
                'BLOCKYSIZE=1024',
                'NUM_THREADS=ALL_CPUS',  # 使用所有CPU核心
                'BIGTIFF=IF_SAFER',
                'SPARSE_OK=TRUE'
            ],
            multithread=True,  # 启用多线程
            warpMemoryLimit=2048,  # 2GB内存限制（提升）
            callback=gdal.TermProgress_nocb  # 显示进度
        )
        
        logging.info("开始重投影和镶嵌处理...")
        
        # 添加详细的错误处理和进度监控
        try:
            # 启用GDAL异常处理
            gdal.UseExceptions()
            
            ds = gdal.Warp(output_path, input_files, options=warp_options)
            if ds is None:
                logging.error("gdalwarp镶嵌失败：返回空数据集")
                return False
            
            # 检查输出文件是否真正创建
            if not os.path.exists(output_path):
                logging.error(f"输出文件未创建: {output_path}")
                return False
                
        except Exception as e:
            logging.error(f"gdalwarp处理过程中发生异常: {str(e)}")
            logging.error(f"异常类型: {type(e).__name__}")
            return False
        finally:
            # 恢复GDAL默认错误处理
            gdal.DontUseExceptions()
        

        
        # 获取输出信息
        logging.info(f"镶嵌结果尺寸: {ds.RasterXSize} x {ds.RasterYSize}")
        logging.info(f"波段数: {ds.RasterCount}")
        
        # 获取地理范围信息
        geotransform = ds.GetGeoTransform()
        if geotransform:
            min_x = geotransform[0]
            max_y = geotransform[3]
            max_x = min_x + geotransform[1] * ds.RasterXSize
            min_y = max_y + geotransform[5] * ds.RasterYSize
            logging.info(f"地理范围: X({min_x:.6f}, {max_x:.6f}), Y({min_y:.6f}, {max_y:.6f})")
        
        # 获取输出文件大小
        ds = None  # 关闭文件
        file_size_gb = os.path.getsize(output_path) / (1024 * 1024 * 1024)
        logging.info(f"输出文件大小: {file_size_gb:.2f} GB")
        logging.info(f"镶嵌创建成功: {output_path}")
        
        # 应用颜色映射表和构建金字塔
        logging.info("开始为输出文件应用颜色映射表和构建金字塔...")
        color_pyramid_success = apply_color_table_and_build_pyramids(output_path)
        
        if color_pyramid_success:
            logging.info("颜色映射表和金字塔处理完成")
        else:
            logging.warning("颜色映射表和金字塔处理失败，但主文件镶嵌成功")
        
        return True
        
    except ImportError:
        logging.error("GDAL未安装，无法使用镶嵌功能")
        return False
    except Exception as e:
        logging.error(f"创建镶嵌时出错: {str(e)}")
        return False

def create_vrt_mosaic(input_files, vrt_path):
    """
    创建VRT虚拟镶嵌文件（最快的镶嵌方法）
    
    参数:
        input_files: 输入文件列表
        vrt_path: VRT输出路径
    
    返回:
        bool: 是否成功
    """
    try:
        from osgeo import gdal
        
        logging.info(f"创建VRT虚拟镶嵌，包含 {len(input_files)} 个文件")
        logging.info("VRT技术：虚拟镶嵌，速度最快，内存消耗最小")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(vrt_path), exist_ok=True)
        
        # 创建VRT选项
        vrt_options = gdal.BuildVRTOptions(
            resolution='highest',  # 使用最高分辨率
            outputSRS='EPSG:4326',  # 输出投影
            srcNodata=0,  # 源数据nodata值
            VRTNodata=0,  # VRT nodata值
            separate=False,  # 不分离波段
            allowProjectionDifference=True  # 允许投影差异
        )
        
        # 创建VRT
        logging.info("开始创建VRT文件...")
        
        try:
            # 启用GDAL异常处理
            gdal.UseExceptions()
            
            vrt_ds = gdal.BuildVRT(vrt_path, input_files, options=vrt_options)
            if vrt_ds is None:
                logging.error("VRT创建失败：返回空数据集")
                return False
            
            # 检查VRT文件是否创建成功
            if not os.path.exists(vrt_path):
                logging.error(f"VRT文件未创建: {vrt_path}")
                return False
                
        except Exception as e:
            logging.error(f"VRT创建过程中发生异常: {str(e)}")
            logging.error(f"异常类型: {type(e).__name__}")
            return False
        finally:
            # 恢复GDAL默认错误处理
            gdal.DontUseExceptions()
        
        # 获取VRT信息
        logging.info(f"VRT尺寸: {vrt_ds.RasterXSize} x {vrt_ds.RasterYSize}")
        logging.info(f"波段数: {vrt_ds.RasterCount}")
        
        vrt_ds = None  # 关闭VRT
        logging.info(f"VRT创建成功: {vrt_path}")
        return True
        
    except ImportError:
        logging.error("GDAL未安装，无法使用VRT功能")
        return False
    except Exception as e:
        logging.error(f"创建VRT时出错: {str(e)}")
        return False

def convert_vrt_to_tiff(vrt_path, output_path):
    """
    将VRT转换为实际的TIFF文件（多线程优化，内存友好）
    
    参数:
        vrt_path: VRT文件路径
        output_path: 输出TIFF路径
    
    返回:
        bool: 是否成功
    """
    try:
        from osgeo import gdal
        
        logging.info(f"将VRT转换为TIFF: {output_path}")
        logging.info("使用多线程转换，充分利用所有CPU核心（高性能配置）")
        logging.info("输出数据类型: uint8 (Byte)，像素值范围: 0-255")
        logging.info("NoData值设置为: 0 (表示空值/无数据区域)")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 转换选项，专门优化大量nodata的处理
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=[
                'COMPRESS=LZW',  # LZW压缩，对nodata区域压缩效果好
                'TILED=YES',  # 瓦片存储，提高访问速度
                'BLOCKXSIZE=1024',  # 增大块大小，减少I/O次数
                'BLOCKYSIZE=1024',
                'NUM_THREADS=ALL_CPUS',  # 使用所有CPU核心
                'BIGTIFF=IF_SAFER',  # 大文件自动使用BigTIFF
                'SPARSE_OK=TRUE'  # 稀疏文件优化，对大量nodata有效
            ],
            outputType=gdal.GDT_Byte,  # 输出数据类型为uint8（Byte）  
            noData=0,  # nodata值
            stats=False,  # 跳过统计计算以提高速度
            callback=gdal.TermProgress_nocb  # 显示进度但不回调
        )
        
        # 执行转换
        logging.info("开始VRT到TIFF转换...")
        
        # 添加详细的错误处理
        try:
            # 启用GDAL异常处理
            gdal.UseExceptions()
            
            ds = gdal.Translate(output_path, vrt_path, options=translate_options)
            if ds is None:
                logging.error("VRT转换失败：返回空数据集")
                return False
                
            # 检查输出文件是否真正创建
            if not os.path.exists(output_path):
                logging.error(f"VRT转换输出文件未创建: {output_path}")
                return False
                
        except Exception as e:
            logging.error(f"VRT转换过程中发生异常: {str(e)}")
            logging.error(f"异常类型: {type(e).__name__}")
            return False
        finally:
            # 恢复GDAL默认错误处理
            gdal.DontUseExceptions()
        
        # 保留颜色映射表
        try:
            vrt_ds = gdal.Open(vrt_path)
            if vrt_ds is not None:
                vrt_band = vrt_ds.GetRasterBand(1)
                color_table = vrt_band.GetColorTable()
                if color_table is not None:
                    output_band = ds.GetRasterBand(1)
                    output_band.SetColorTable(color_table)
                    logging.info("已保留颜色映射表到输出文件")
                vrt_ds = None
        except Exception as e:
            logging.warning(f"保留颜色映射表到输出文件时出错: {str(e)}")
        
        # 跳过统计计算以提高处理速度
        logging.info("已跳过栅格统计计算以优化性能")
        
        # 获取输出文件信息
        file_size_gb = os.path.getsize(output_path) / (1024 * 1024 * 1024)
        logging.info(f"输出文件大小: {file_size_gb:.2f} GB")
        
        ds = None  # 关闭文件
        logging.info(f"VRT转换完成: {output_path}")
        
        # 应用颜色映射表和构建金字塔
        logging.info("开始为输出文件应用颜色映射表和构建金字塔...")
        color_pyramid_success = apply_color_table_and_build_pyramids(output_path)
        
        if color_pyramid_success:
            logging.info("颜色映射表和金字塔处理完成")
        else:
            logging.warning("颜色映射表和金字塔处理失败，但主文件转换成功")
        
        return True
        
    except Exception as e:
        logging.error(f"VRT转换时出错: {str(e)}")
        return False

def fast_vrt_mosaic(input_dir, output_path):
    """
    最快的VRT镶嵌方法（专为25核CPU和64GB内存优化）
    
    参数:
        input_dir: 输入目录
        output_path: 输出文件路径
    
    返回:
        bool: 是否成功
    """
    try:
        # 获取所有tif文件
        tif_files = []
        for file in os.listdir(input_dir):
            if file.lower().endswith('.tif'):
                file_path = os.path.join(input_dir, file)
                tif_files.append(file_path)
        
        if not tif_files:
            logging.error("未找到任何tif文件")
            return False
        
        logging.info(f"找到 {len(tif_files)} 个tif文件")
        
        # 计算总文件大小
        total_size_gb = sum(os.path.getsize(f) for f in tif_files) / (1024**3)
        logging.info(f"总数据大小: {total_size_gb:.2f} GB")
        
        # 使用VRT技术（最快方法）
        temp_vrt = output_path.replace('.tif', '_temp.vrt')
        
        logging.info("=== 开始VRT镶嵌处理 ===")
        if create_vrt_mosaic(tif_files, temp_vrt):
            # 将VRT转换为TIFF
            success = convert_vrt_to_tiff(temp_vrt, output_path)
            
            # 清理临时文件
            if os.path.exists(temp_vrt):
                os.remove(temp_vrt)
                logging.info("清理临时VRT文件完成")
            
            return success
        else:
            logging.error("VRT镶嵌失败")
            return False
        
    except Exception as e:
        logging.error(f"VRT镶嵌时出错: {str(e)}")
        return False

def create_batch_mosaic(input_files, output_path, batch_size=20):
    """
    分批处理镶嵌，避免内存溢出（专为大量文件优化）
    
    参数:
        input_files: 输入文件列表
        output_path: 输出路径
        batch_size: 每批处理的文件数量
    
    返回:
        bool: 是否成功
    """
    try:
        from osgeo import gdal
        
        logging.info(f"使用分批镶嵌策略，每批处理 {batch_size} 个文件")
        logging.info(f"总文件数: {len(input_files)}，预计分 {(len(input_files) + batch_size - 1) // batch_size} 批处理")
        
        # 创建临时目录存放中间结果
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_batch")
        os.makedirs(temp_dir, exist_ok=True)
        
        batch_files = []
        
        # 分批处理
        for i in range(0, len(input_files), batch_size):
            batch = input_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            batch_output = os.path.join(temp_dir, f"batch_{batch_num:03d}.tif")
            
            logging.info(f"处理第 {batch_num} 批，包含 {len(batch)} 个文件")
            
            # 处理当前批次
            success = create_unified_mosaic(batch, batch_output)
            if not success:
                logging.error(f"第 {batch_num} 批处理失败")
                return False
            
            batch_files.append(batch_output)
            logging.info(f"第 {batch_num} 批处理完成")
        
        # 合并所有批次结果
        logging.info("开始合并所有批次结果...")
        final_success = create_unified_mosaic(batch_files, output_path)
        
        # 清理临时文件
        logging.info("清理临时文件...")
        cleaned_count = 0
        for batch_file in batch_files:
            try:
                if os.path.exists(batch_file):
                    os.remove(batch_file)
                    cleaned_count += 1
                    logging.debug(f"删除临时文件: {batch_file}")
            except Exception as e:
                logging.warning(f"删除临时文件失败 {batch_file}: {str(e)}")
        
        logging.info(f"已清理 {cleaned_count} 个临时文件")
        
        # 强制删除临时目录（使用新的强制删除函数）
        temp_dir_removed = force_remove_directory(temp_dir)
        if not temp_dir_removed:
            logging.warning(f"临时目录删除失败，但不影响主要功能: {temp_dir}")
        
        return final_success
        
    except Exception as e:
        logging.error(f"分批镶嵌时出错: {str(e)}")
        
        # 即使出错也要清理临时文件
        try:
            logging.info("出错时清理临时文件...")
            if 'batch_files' in locals():
                for batch_file in batch_files:
                    try:
                        if os.path.exists(batch_file):
                            os.remove(batch_file)
                            logging.debug(f"清理临时文件: {batch_file}")
                    except Exception as cleanup_e:
                        logging.warning(f"清理临时文件失败 {batch_file}: {str(cleanup_e)}")
            
            # 清理临时目录
            if 'temp_dir' in locals():
                force_remove_directory(temp_dir)
                
        except Exception as cleanup_e:
            logging.error(f"清理临时文件时出错: {str(cleanup_e)}")
        
        return False

def main():
    """
    主函数 - 按区域和年份分组进行镶嵌
    """
    # 定义路径
    input_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\1.GEE导出结果\4"
    output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\2.GEE导出结果_结果合并"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(output_dir)
    logging.info(f"开始按区域和年份分组的镶嵌处理")
    logging.info(f"输入目录: {input_dir}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"日志文件: {log_path}")
    
    # 记录系统信息和性能配置
    import psutil
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    logging.info(f"系统信息: {cpu_count} 个CPU核心, {memory_gb:.1f}GB 内存")
    logging.info("性能配置: 高性能模式 - 4GB GDAL缓存, 2GB Warp内存, 使用所有CPU核心")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logging.error(f"输入目录不存在: {input_dir}")
        return
    
    # 开始处理
    start_time = time.time()
    
    try:
        # 按区域和年份分组文件
        logging.info("="*50)
        logging.info("第一步：按区域和年份分组文件")
        grouped_files = group_files_by_zone_year(input_dir)
        
        if not grouped_files:
            logging.error("未找到任何可处理的文件")
            return
        
        # 统计信息
        total_groups = len(grouped_files)
        total_files = sum(len(files) for files in grouped_files.values())
        logging.info(f"总共找到 {total_files} 个文件，分为 {total_groups} 个区域-年份组合")
        
        # 逐个处理每个区域-年份组合
        logging.info("="*50)
        logging.info("第二步：开始逐个镶嵌处理")
        
        success_count = 0
        failed_groups = []
        
        for i, ((zone, year), files) in enumerate(grouped_files.items(), 1):
            group_start_time = time.time()
            
            # 生成输出文件名
            output_filename = f"{zone}_{year}.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            logging.info(f"处理第 {i}/{total_groups} 组: {zone}_{year}")
            logging.info(f"  包含文件数: {len(files)}")
            logging.info(f"  输出文件: {output_filename}")
            
            # 计算当前组文件总大小
            try:
                group_size_gb = sum(os.path.getsize(f) for f in files) / (1024**3)
                logging.info(f"  数据大小: {group_size_gb:.2f} GB")
            except Exception as e:
                logging.warning(f"  无法计算文件大小: {str(e)}")
                group_size_gb = 0
            
            # 选择镶嵌策略
            if len(files) > 10:
                logging.info(f"  使用分批镶嵌策略（文件数量: {len(files)}）")
                success = create_batch_mosaic(files, output_path, batch_size=8)
            else:
                logging.info(f"  使用直接镶嵌策略（文件数量: {len(files)}）")
                success = create_unified_mosaic(files, output_path)
            
            group_end_time = time.time()
            group_time = group_end_time - group_start_time
            
            if success:
                success_count += 1
                # 获取输出文件信息
                if os.path.exists(output_path):
                    file_size_gb = os.path.getsize(output_path) / (1024**3)
                    processing_speed = file_size_gb / group_time * 60 if group_time > 0 else 0
                    logging.info(f"  ✓ 镶嵌成功！")
                    logging.info(f"    输出大小: {file_size_gb:.2f} GB")
                    logging.info(f"    处理时间: {group_time:.2f} 秒")
                    logging.info(f"    处理速度: {processing_speed:.2f} GB/分钟")
                else:
                    logging.error(f"  ✗ 镶嵌失败：输出文件不存在")
                    failed_groups.append(f"{zone}_{year}")
            else:
                logging.error(f"  ✗ 镶嵌失败：{zone}_{year}")
                failed_groups.append(f"{zone}_{year}")
            
            logging.info("-" * 30)
        
        # 处理完成统计
        end_time = time.time()
        total_time = end_time - start_time
        
        logging.info("="*50)
        logging.info("镶嵌处理完成统计:")
        logging.info(f"总处理组数: {total_groups}")
        logging.info(f"成功组数: {success_count}")
        logging.info(f"失败组数: {len(failed_groups)}")
        logging.info(f"成功率: {success_count/total_groups*100:.1f}%")
        logging.info(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
        
        if failed_groups:
            logging.error("失败的组合:")
            for group in failed_groups:
                logging.error(f"  - {group}")
        
        # 输出文件列表
        logging.info("\n生成的镶嵌文件:")
        for file in os.listdir(output_dir):
            if file.endswith('.tif'):
                file_path = os.path.join(output_dir, file)
                file_size_gb = os.path.getsize(file_path) / (1024**3)
                logging.info(f"  {file} ({file_size_gb:.2f} GB)")
        
        if success_count == total_groups:
            logging.info("🎉 所有区域-年份组合镶嵌处理成功完成!")
        else:
            logging.warning(f"⚠️  部分组合处理失败，请检查失败的 {len(failed_groups)} 个组合")
            
    except Exception as e:
        logging.error(f"镶嵌过程中出错: {str(e)}")
        import traceback
        logging.error(f"详细错误信息: {traceback.format_exc()}")
    
    logging.info("="*50)

if __name__ == "__main__":
    main()