#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码目的：对裁剪后的ESRI土地覆盖数据tif文件进行高效镶嵌合并
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
        
        # 设置GDAL配置以优化性能和内存使用
        gdal.SetConfigOption('GDAL_CACHEMAX', '4096')  # 4GB缓存
        gdal.SetConfigOption('GDAL_NUM_THREADS', '25')  # 使用25个线程
        gdal.SetConfigOption('VSI_CACHE', 'TRUE')  # 启用VSI缓存
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')  # 优化文件打开
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 使用gdalwarp进行重投影和镶嵌
        warp_options = gdal.WarpOptions(
            format='GTiff',
            dstSRS='EPSG:4326',  # 目标投影：WGS84地理坐标系
            xRes=0.0001,  # 约10米分辨率（地理坐标）
            yRes=0.0001,
            resampleAlg='nearest',  # 最快的重采样方法
            srcNodata=0,  # 源数据nodata值
            dstNodata=0,  # 目标nodata值
            creationOptions=[
                'COMPRESS=LZW',  # LZW压缩
                'TILED=YES',  # 瓦片存储
                'BLOCKXSIZE=1024',
                'BLOCKYSIZE=1024',
                'NUM_THREADS=25',  # 使用25个线程
                'BIGTIFF=IF_SAFER',
                'SPARSE_OK=TRUE'
            ],
            multithread=True,  # 启用多线程
            warpMemoryLimit=2048,  # 2GB内存限制
            callback=gdal.TermProgress_nocb  # 显示进度
        )
        
        logging.info("开始重投影和镶嵌处理...")
        ds = gdal.Warp(output_path, input_files, options=warp_options)
        if ds is None:
            logging.error("gdalwarp镶嵌失败")
            return False
        
        # 保留第一个输入文件的颜色映射表
        try:
            first_ds = gdal.Open(input_files[0])
            if first_ds is not None:
                first_band = first_ds.GetRasterBand(1)
                color_table = first_band.GetColorTable()
                if color_table is not None:
                    output_band = ds.GetRasterBand(1)
                    output_band.SetColorTable(color_table)
                    logging.info("已保留原始颜色映射表")
                first_ds = None
        except Exception as e:
            logging.warning(f"保留颜色映射表时出错: {str(e)}")
        
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
        logging.info("使用多线程转换，充分利用25核CPU")
        
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
                'NUM_THREADS=25',  # 使用25个线程
                'BIGTIFF=IF_SAFER',  # 大文件自动使用BigTIFF
                'SPARSE_OK=TRUE'  # 稀疏文件优化，对大量nodata有效
            ],
            outputType=gdal.GDT_Byte,  # uint8类型
            noData=0,  # nodata值
            stats=False,  # 跳过统计计算以提高速度
            callback=gdal.TermProgress_nocb  # 显示进度但不回调
        )
        
        # 执行转换
        logging.info("开始VRT到TIFF转换...")
        ds = gdal.Translate(output_path, vrt_path, options=translate_options)
        if ds is None:
            logging.error("VRT转换失败")
            return False
        
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

def main():
    """
    主函数
    """
    # 定义路径
    input_dir = r"K:\数据\全球人工林产品数据\筛选出东南亚区域_裁剪"
    output_dir = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\全球人工林产品数据"
    output_file = os.path.join(output_dir, "southeast_asia_PlantTree_2021_mosaic.tif")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(output_dir)
    logging.info(f"开始ESRI数据镶嵌处理")
    logging.info(f"输入目录: {input_dir}")
    logging.info(f"输出文件: {output_file}")
    logging.info(f"日志文件: {log_path}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logging.error(f"输入目录不存在: {input_dir}")
        return
    
    # 开始镶嵌
    start_time = time.time()
    
    # 镶嵌参数设置
    mosaic_method = 'first'  # 重叠区域使用第一个值
    use_vrt_technology = True  # 优先使用VRT技术
    
    logging.info(f"镶嵌方法: {mosaic_method}")
    logging.info(f"使用VRT技术: {use_vrt_technology}")
    
    try:
        # 获取所有tif文件
        tif_files = []
        for file in os.listdir(input_dir):
            if file.lower().endswith('.tif'):
                file_path = os.path.join(input_dir, file)
                tif_files.append(file_path)
        
        if not tif_files:
            logging.error("未找到任何tif文件")
            return
        
        logging.info(f"找到 {len(tif_files)} 个tif文件")
        
        # 使用统一投影镶嵌函数
        success = create_unified_mosaic(tif_files, output_file)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if success:
            # 获取输出文件信息
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024 * 1024)  # GB
                logging.info("\n" + "="*50)
                logging.info("镶嵌完成统计:")
                logging.info(f"输出文件: {output_file}")
                logging.info(f"文件大小: {file_size:.2f} GB")
                logging.info(f"总耗时: {total_time:.2f} 秒")
                logging.info(f"处理速度: {file_size/total_time*60:.2f} GB/分钟")
                logging.info("ESRI数据镶嵌处理成功完成!")
            else:
                logging.error("镶嵌完成但输出文件不存在")
        else:
            logging.error("镶嵌处理失败")
            
    except Exception as e:
        logging.error(f"镶嵌过程中出错: {str(e)}")
    
    logging.info("="*50)

if __name__ == "__main__":
    main()