#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码目的：对tif文件进行高效镶嵌合并
功能：
1. 读取裁剪后的所有tif文件
2. 使用分层镶嵌策略提高处理效率
3. 支持大文件处理，避免内存溢出
4. 输出最终的镶嵌结果
5. 可选择使用VRT虚拟栅格技术加速

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
import numpy as np  # 用于数学计算和数组操作
from datetime import datetime
from pathlib import Path

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

# VRT相关函数保留，其他函数已移除以优化性能

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
            logging.error("金字塔构建失败")
            return False
            
    except ImportError:
        logging.error("GDAL未安装，无法处理颜色映射表和金字塔")
        return False
    except Exception as e:
        logging.error(f"处理颜色映射表和金字塔时发生错误: {str(e)}")
        logging.error(f"错误类型: {type(e).__name__}")
        return False

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
        from osgeo import osr
        
        logging.info(f"使用gdalwarp创建统一投影镶嵌，包含 {len(input_files)} 个文件")
        logging.info("统一投影到WGS84地理坐标系，解决UTM投影不一致问题")
        logging.info("输出数据类型: uint8 (Byte)，像素值范围: 0-255")
        logging.info("NoData值设置为: 0 (表示空值/无数据区域)")
        
        # 设置GDAL配置以优化性能和内存使用（保守配置）
        gdal.SetConfigOption('GDAL_CACHEMAX', '1024')  # 1GB缓存（降低）
        gdal.SetConfigOption('GDAL_NUM_THREADS', '8')   # 8个线程（降低）
        gdal.SetConfigOption('VSI_CACHE', 'TRUE')  # 启用VSI缓存
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')  # 优化文件打开
        gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '100')  # 限制数据集池大小
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 智能分辨率计算和多UTM区域边界计算
        x_res = 0.0001  # 默认经度方向分辨率（度），约10米
        y_res = 0.0001  # 默认纬度方向分辨率（度），约10米
        output_bounds = None  # 输出边界，用于网格对齐
        
        # 统一的目标坐标系
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)  # WGS84
        
        # 存储所有文件的边界信息
        all_bounds = []
        pixel_sizes = []
        
        logging.info(f"开始分析 {len(input_files)} 个UTM区域文件的边界和分辨率...")
        
        try:
            # 遍历所有输入文件，计算统一的边界和分辨率
            for i, file_path in enumerate(input_files):
                try:
                    ds = gdal.Open(file_path)
                    if ds is None:
                        logging.warning(f"无法打开文件 {i+1}: {os.path.basename(file_path)}")
                        continue
                    
                    gt = ds.GetGeoTransform()
                    srs_wkt = ds.GetProjection()
                    
                    if not gt or not srs_wkt:
                        logging.warning(f"文件 {i+1} 缺少地理信息: {os.path.basename(file_path)}")
                        ds = None
                        continue
                    
                    # 解析坐标系
                    srs = osr.SpatialReference()
                    srs.ImportFromWkt(srs_wkt)
                    
                    # 记录像元大小用于分辨率计算
                    if srs.IsGeographic():
                        pixel_sizes.append(abs(gt[1]))  # 地理坐标系直接使用度
                    else:
                        pixel_sizes.append(abs(gt[1]) / 111320.0)  # 投影坐标系转换为度
                    
                    # 计算当前文件的边界
                    min_x = gt[0]
                    max_y = gt[3]
                    max_x = min_x + gt[1] * ds.RasterXSize
                    min_y = max_y + gt[5] * ds.RasterYSize
                    
                    # 如果不是地理坐标系，转换到WGS84
                    if not srs.IsGeographic():
                        transform = osr.CoordinateTransformation(srs, target_srs)
                        
                        # 转换四个角点
                        corners = [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]
                        transformed_corners = []
                        
                        for x, y in corners:
                            try:
                                point = transform.TransformPoint(x, y)
                                transformed_corners.append((point[0], point[1]))
                            except Exception as e:
                                logging.warning(f"坐标转换失败 {file_path}: {str(e)}")
                                break
                        
                        if len(transformed_corners) == 4:
                            lons = [corner[0] for corner in transformed_corners]
                            lats = [corner[1] for corner in transformed_corners]
                            min_x, max_x = min(lons), max(lons)
                            min_y, max_y = min(lats), max(lats)
                        else:
                            logging.warning(f"跳过坐标转换失败的文件: {os.path.basename(file_path)}")
                            ds = None
                            continue
                    
                    # 记录边界
                    all_bounds.append([min_x, min_y, max_x, max_y])
                    logging.info(f"文件 {i+1} ({os.path.basename(file_path)}): 边界 [{min_x:.6f}, {min_y:.6f}, {max_x:.6f}, {max_y:.6f}]")
                    
                    ds = None
                    
                except Exception as e:
                    logging.warning(f"处理文件 {i+1} 时出错: {str(e)}")
                    continue
            
            # 计算统一的分辨率（使用中位数避免异常值影响）
            if pixel_sizes:
                pixel_sizes.sort()
                median_idx = len(pixel_sizes) // 2
                if len(pixel_sizes) % 2 == 0:
                    x_res = y_res = (pixel_sizes[median_idx-1] + pixel_sizes[median_idx]) / 2
                else:
                    x_res = y_res = pixel_sizes[median_idx]
                logging.info(f"统一分辨率（中位数）: {x_res:.8f} 度 (约 {x_res*111320:.1f} 米)")
            
            # 计算所有区域的联合边界
            if all_bounds:
                # 找到所有区域的最大范围
                min_x_global = min(bounds[0] for bounds in all_bounds)
                min_y_global = min(bounds[1] for bounds in all_bounds)
                max_x_global = max(bounds[2] for bounds in all_bounds)
                max_y_global = max(bounds[3] for bounds in all_bounds)
                
                logging.info(f"所有UTM区域联合边界: [{min_x_global:.6f}, {min_y_global:.6f}, {max_x_global:.6f}, {max_y_global:.6f}]")
                
                # 使用统一网格对齐策略，减少UTM区域间的偏移
                # 选择一个全球统一的网格原点（比如0,0）
                grid_origin_x = 0.0
                grid_origin_y = 0.0
                
                # 将边界对齐到统一网格
                min_x_aligned = grid_origin_x + np.floor((min_x_global - grid_origin_x) / x_res) * x_res
                max_x_aligned = grid_origin_x + np.ceil((max_x_global - grid_origin_x) / x_res) * x_res
                min_y_aligned = grid_origin_y + np.floor((min_y_global - grid_origin_y) / y_res) * y_res
                max_y_aligned = grid_origin_y + np.ceil((max_y_global - grid_origin_y) / y_res) * y_res
                
                output_bounds = [min_x_aligned, min_y_aligned, max_x_aligned, max_y_aligned]
                
                logging.info(f"网格对齐后的输出边界: {output_bounds}")
                logging.info(f"输出范围: 经度 {max_x_aligned-min_x_aligned:.6f}°, 纬度 {max_y_aligned-min_y_aligned:.6f}°")
                
                # 估算输出尺寸
                width_pixels = int((max_x_aligned - min_x_aligned) / x_res)
                height_pixels = int((max_y_aligned - min_y_aligned) / y_res)
                logging.info(f"预估输出尺寸: {width_pixels} x {height_pixels} 像素")
                
            else:
                logging.error("未能获取任何有效的边界信息")
                
        except Exception as e:
            logging.warning(f"边界计算过程中出错，使用默认设置: {str(e)}")
        
        # 构建gdalwarp选项 - 针对多UTM区域无缝镶嵌优化
        warp_options_dict = {
            'format': 'GTiff',
            'dstSRS': 'EPSG:4326',  # 目标投影：WGS84地理坐标系
            'xRes': x_res,  # 输出经度方向分辨率（度）
            'yRes': y_res,  # 输出纬度方向分辨率（度）
            'resampleAlg': 'nearest',  # 最近邻重采样，保持分类值
            'outputType': gdal.GDT_Byte,  # 输出数据类型为uint8（Byte）
            'srcNodata': 0,  # 源数据nodata值
            'dstNodata': 0,  # 目标nodata值
            'creationOptions': [
                'COMPRESS=LZW',  # LZW压缩
                'TILED=YES',  # 瓦片存储
                'BLOCKXSIZE=1024',
                'BLOCKYSIZE=1024',
                'NUM_THREADS=8',  # 使用8个线程（降低）
                'BIGTIFF=IF_SAFER',
                'SPARSE_OK=TRUE'
            ],
            'multithread': True,  # 启用多线程
            'warpMemoryLimit': 512,  # 512MB内存限制（降低）
            'callback': gdal.TermProgress_nocb  # 显示进度
        }
        
        # 如果计算出了输出边界，添加到选项中以确保网格对齐
        if output_bounds:
            warp_options_dict['outputBounds'] = output_bounds
            logging.info("启用输出边界对齐以减少UTM区域间的空间偏移")
            logging.info(f"使用统一网格边界: {output_bounds}")
        else:
            logging.warning("未设置输出边界，可能导致UTM区域间存在空间不连续")
        
        # 创建WarpOptions对象
        warp_options = gdal.WarpOptions(**warp_options_dict)
        
        logging.info("开始多UTM区域无缝镶嵌处理...")
        logging.info(f"镶嵌参数: 分辨率={x_res:.8f}°, 重采样=最近邻, 压缩=LZW")
        
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
        # 复制颜色映射表（如果存在），确保输出镶嵌结果保留色带
        try:
            src_ds = gdal.Open(input_files[0])  # 打开第一个输入文件，作为颜色表来源
            if src_ds is not None:  # 如果成功打开源数据
                src_band = src_ds.GetRasterBand(1)  # 获取源数据第一波段
                color_table = src_band.GetColorTable()  # 获取源数据的颜色映射表
                if color_table is not None:  # 如果源数据存在颜色映射表
                    out_band = ds.GetRasterBand(1)  # 获取输出数据第一波段
                    out_band.SetColorTable(color_table)  # 将颜色映射表复制到输出镶嵌结果
                    logging.info("已从输入文件复制颜色映射表到镶嵌结果")  # 记录成功复制颜色表
                src_ds = None  # 关闭源数据集，释放资源
        except Exception as e:
            logging.warning(f"复制颜色映射表失败: {str(e)}")  # 记录复制失败的警告信息
        
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
        
        # 为镶嵌结果应用颜色映射表并构建金字塔
        logging.info("开始为镶嵌结果构建金字塔...")
        pyramid_success = apply_color_table_and_build_pyramids(output_path)
        
        if pyramid_success:
            logging.info("镶嵌结果金字塔构建成功")
        else:
            logging.warning("镶嵌结果金字塔构建失败，但镶嵌本身成功")
        
        return True
        
    except ImportError:
        logging.error("GDAL未安装，无法使用镶嵌功能")
        return False
    except Exception as e:
        logging.error(f"创建镶嵌时出错: {str(e)}")
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
        logging.info("使用多线程转换，充分利用8核CPU（保守配置）")
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
                'NUM_THREADS=8',  # 使用8个线程（降低）
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

def create_batch_mosaic(input_files, output_path, batch_size=10):
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
        for batch_file in batch_files:
            if os.path.exists(batch_file):
                os.remove(batch_file)
        
        # 删除临时目录
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        return final_success
        
    except Exception as e:
        logging.error(f"分批镶嵌时出错: {str(e)}")
        return False

def list_zone_dirs(input_root):
    """
    列出输入根目录下的所有zone目录（zone1-zone10），按编号排序
    """
    zones = []  # 用于存放zone目录的完整路径列表
    for name in os.listdir(input_root):  # 遍历根目录下的所有条目
        full_path = os.path.join(input_root, name)  # 构造完整路径
        if os.path.isdir(full_path) and name.lower().startswith('zone'):  # 判断是否为目录且名称以zone开头
            zones.append(full_path)  # 收集该zone目录
    # 按zone编号排序，确保zone1..zone10的顺序
    def zone_key(p):  # 定义排序键函数
        base = os.path.basename(p)  # 提取目录名
        try:
            num = int(''.join(ch for ch in base if ch.isdigit()))  # 提取目录名中的数字并转为整数
        except ValueError:
            num = 9999  # 若未提取到数字，则放到最后
        return num  # 返回排序用的数字
    zones.sort(key=zone_key)  # 按编号进行排序
    return zones  # 返回排序后的zone目录列表

def find_year_zone_files(input_root, year):
    """
    查找指定年份在所有zone目录下的tif影像（排除.ovr），优先匹配包含“添加颜色映射表”的文件
    """
    year_files = []  # 用于保存该年份的所有zone影像完整路径
    zones = list_zone_dirs(input_root)  # 获取所有zone目录
    logging.info(f"检测到 {len(zones)} 个zone目录用于年份 {year} 的镶嵌")  # 记录zone数量
    missing_zones = []  # 用于记录未找到影像的zone
    for zdir in zones:  # 遍历每个zone目录
        found = None  # 当前zone找到的文件路径占位
        for fname in os.listdir(zdir):  # 遍历zone目录下的文件
            if not fname.lower().endswith('.tif'):  # 只处理.tif文件
                continue  # 跳过非tif
            if '.ovr' in fname.lower():  # 跳过金字塔.ovr文件
                continue  # 跳过
            if str(year) not in fname:  # 文件名需包含年份
                continue  # 跳过不含年份的文件
            if '添加颜色映射表' in fname:  # 优先选择带颜色表标记的文件
                found = os.path.join(zdir, fname)  # 记录找到的文件完整路径
                break  # 找到优先文件后立即退出循环
        # 如果未找到带“添加颜色映射表”的，尝试任何包含年份的tif
        if found is None:  # 若未找到优先文件
            for fname in os.listdir(zdir):  # 再次遍历以兜底
                if not fname.lower().endswith('.tif'):  # 只处理tif
                    continue  # 跳过
                if '.ovr' in fname.lower():  # 跳过.ovr
                    continue  # 跳过
                if str(year) in fname:  # 包含年份即可
                    found = os.path.join(zdir, fname)  # 记录兜底文件
                    break  # 找到即可退出
        # 根据是否找到文件进行记录
        if found is not None:  # 找到文件
            year_files.append(found)  # 添加到年份文件列表
        else:
            missing_zones.append(os.path.basename(zdir))  # 记录缺失影像的zone名
    # 输出缺失zone的警告信息
    if missing_zones:  # 如果存在缺失
        logging.warning(f"年份 {year} 缺失 {len(missing_zones)} 个zone影像: {', '.join(missing_zones)}")  # 记录警告
    logging.info(f"年份 {year} 找到 {len(year_files)} 个zone影像用于镶嵌")  # 记录找到的数量
    return year_files  # 返回该年份的所有zone影像

def mosaic_one_year(input_root, output_dir, year):
    """
    对指定年份进行跨zone的镶嵌合并，输出统一投影的结果
    """
    # 构造输出文件名，保持与现有命名规范一致
    output_file = os.path.join(
        output_dir,  # 目标输出目录
        f"人工林提取结果_每个区域16000个样本点_{year}年_变化概率_筛选样本_逐年分类.tif"  # 输出文件名
    )
    files = find_year_zone_files(input_root, year)  # 获取该年份的所有zone影像
    if not files:  # 若未找到任何文件
        logging.error(f"年份 {year} 未找到任何用于镶嵌的tif文件")  # 记录错误
        return False, output_file  # 返回失败及预期输出路径
    # 对10个zone进行直接统一镶嵌（数量少，无需分批）
    logging.info(f"开始年份 {year} 的镶嵌，文件数: {len(files)}")  # 记录开始信息
    ok = create_unified_mosaic(files, output_file)  # 调用统一镶嵌函数
    return ok, output_file  # 返回处理结果与输出路径

def main():
    """
    主函数
    """
    # 定义路径
    input_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型"  # 输入根目录，包含zone1..zone10
    output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\4.GEE导出结果_结果合并_马尔可夫模型_逐年合并"  # 输出目录（逐年合并）
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]  # 准备逐年处理的年份列表
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(output_dir)
    logging.info(f"开始ESRI数据镶嵌处理（按年跨zone合并）")
    logging.info(f"输入根目录: {input_dir}")
    logging.info(f"目标输出目录: {output_dir}")
    logging.info(f"日志文件: {log_path}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logging.error(f"输入目录不存在: {input_dir}")
        return
    
    # 逐年开始镶嵌
    all_start = time.time()  # 记录总开始时间
    try:
        for y in years:  # 遍历每个年份
            y_start = time.time()  # 记录该年份的开始时间
            ok, out_path = mosaic_one_year(input_dir, output_dir, y)  # 执行该年份的镶嵌
            y_end = time.time()  # 记录该年份的结束时间
            y_time = y_end - y_start  # 计算该年份耗时
            if ok and os.path.exists(out_path):  # 若成功并且输出文件存在
                file_size = os.path.getsize(out_path) / (1024 * 1024 * 1024)  # 转换为GB
                logging.info("\n" + "-"*48)
                logging.info(f"年份 {y} 镶嵌完成统计:")
                logging.info(f"输出文件: {out_path}")
                logging.info(f"文件大小: {file_size:.2f} GB")
                logging.info(f"耗时: {y_time:.2f} 秒")
                if y_time > 0:  # 避免除零
                    logging.info(f"处理速度: {file_size/y_time*60:.2f} GB/分钟")
            else:
                logging.error(f"年份 {y} 镶嵌失败或输出文件不存在: {out_path}")
        # 总结统计
        all_end = time.time()  # 记录总结束时间
        all_time = all_end - all_start  # 计算总耗时
        logging.info("\n" + "="*50)
        logging.info("逐年镶嵌全部完成")
        logging.info(f"总耗时: {all_time:.2f} 秒")
    except Exception as e:
        logging.error(f"逐年镶嵌过程中出错: {str(e)}")  # 记录总流程错误
    logging.info("="*50)  # 结束分隔线

if __name__ == "__main__":
    main()