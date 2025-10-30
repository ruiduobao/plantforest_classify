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
        
        # 设置GDAL配置以优化性能
        gdal.SetConfigOption('GDAL_CACHEMAX', '4096')  # 4GB缓存
        gdal.SetConfigOption('GDAL_NUM_THREADS', '25')  # 使用25个线程
        gdal.SetConfigOption('VSI_CACHE', 'TRUE')  # 启用VSI缓存
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(vrt_path), exist_ok=True)
        
        # 使用gdalbuildvrt创建VRT
        vrt_options = gdal.BuildVRTOptions(
            srcNodata=0,  # 源数据nodata值
            VRTNodata=0,  # VRT nodata值
            resolution='highest',  # 使用最高分辨率
            resampleAlg='nearest',  # 最快的重采样方法
            addAlpha=False,  # 不添加alpha通道
            hideNodata=True,  # 隐藏nodata值
            callback=gdal.TermProgress_nocb  # 显示进度
        )
        
        logging.info("开始创建VRT虚拟镶嵌...")
        vrt_ds = gdal.BuildVRT(vrt_path, input_files, options=vrt_options)
        if vrt_ds is None:
            logging.error("VRT创建失败")
            return False
        
        # 保留第一个输入文件的颜色映射表
        try:
            first_ds = gdal.Open(input_files[0])
            if first_ds is not None:
                first_band = first_ds.GetRasterBand(1)
                color_table = first_band.GetColorTable()
                if color_table is not None:
                    vrt_band = vrt_ds.GetRasterBand(1)
                    vrt_band.SetColorTable(color_table)
                    logging.info("已保留原始颜色映射表到VRT")
                first_ds = None
        except Exception as e:
            logging.warning(f"保留颜色映射表到VRT时出错: {str(e)}")
        
        # 获取VRT信息
        logging.info(f"VRT尺寸: {vrt_ds.RasterXSize} x {vrt_ds.RasterYSize}")
        logging.info(f"VRT波段数: {vrt_ds.RasterCount}")
        
        vrt_ds = None  # 关闭文件
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

def process_single_year_mosaic(year, base_input_dir, base_output_dir):
    """
    处理单个年份的ESRI数据镶嵌
    
    参数:
        year: 年份 (int)
        base_input_dir: 基础输入目录路径
        base_output_dir: 基础输出目录路径
    
    返回:
        tuple: (年份, 是否成功, 处理时间, 输出文件大小GB)
    """
    year_start_time = time.time()
    
    # 构建年份相关路径
    input_dir = os.path.join(base_input_dir, f"{year}年_分块裁剪")
    output_file = os.path.join(base_output_dir, f"southeast_asia_landcover_{year}_mosaic_ESRI_10m.tif")
    
    logging.info(f"📂 {year}年 输入目录: {input_dir}")
    logging.info(f"📂 {year}年 输出文件: {output_file}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logging.warning(f"⚠️ {year}年输入目录不存在，跳过: {input_dir}")
        return (year, False, 0, 0)
    
    # 获取所有tif文件
    tif_files = []
    try:
        for file in os.listdir(input_dir):
            if file.lower().endswith('.tif'):
                file_path = os.path.join(input_dir, file)
                tif_files.append(file_path)
    except Exception as e:
        logging.error(f"✗ {year}年读取输入目录失败: {str(e)}")
        return (year, False, 0, 0)
    
    logging.info(f"📄 {year}年找到 {len(tif_files)} 个tif文件")
    
    if not tif_files:
        logging.warning(f"⚠️ {year}年未找到任何tif文件")
        return (year, False, 0, 0)
    
    # 计算输入文件总大小
    total_input_size_gb = sum(os.path.getsize(f) for f in tif_files) / (1024**3)
    logging.info(f"📊 {year}年输入数据总大小: {total_input_size_gb:.2f} GB")
    
    # 镶嵌参数设置
    mosaic_method = 'first'  # 重叠区域使用第一个值
    use_vrt_technology = True  # 优先使用VRT技术
    
    logging.info(f"🔧 {year}年镶嵌方法: {mosaic_method}")
    logging.info(f"🔧 {year}年使用VRT技术: {use_vrt_technology}")
    
    try:
        # 使用统一投影镶嵌函数
        logging.info(f"🔄 {year}年开始镶嵌处理...")
        success = create_unified_mosaic(tif_files, output_file)
        
        year_end_time = time.time()
        year_time = year_end_time - year_start_time
        
        # 获取输出文件大小
        output_size_gb = 0
        if success and os.path.exists(output_file):
            output_size_gb = os.path.getsize(output_file) / (1024 * 1024 * 1024)
            logging.info(f"✅ {year}年镶嵌成功完成")
            logging.info(f"📊 {year}年输出文件大小: {output_size_gb:.2f} GB")
            logging.info(f"⏱️ {year}年处理耗时: {year_time:.2f} 秒")
            logging.info(f"🚀 {year}年处理速度: {output_size_gb/year_time*60:.2f} GB/分钟")
        else:
            logging.error(f"✗ {year}年镶嵌处理失败")
        
        return (year, success, year_time, output_size_gb)
        
    except Exception as e:
        logging.error(f"✗ {year}年镶嵌过程中出错: {str(e)}")
        return (year, False, time.time() - year_start_time, 0)


def main():
    """
    主函数：循环处理2017-2023年的ESRI数据镶嵌
    """
    print("=" * 80)
    print("🌍 ESRI土地覆盖数据多年份批量镶嵌工具")
    print("=" * 80)
    
    # ==================== 配置区域 ====================
    # 用户可以根据实际情况修改以下路径配置
    
    # 基础输入目录（包含各年份裁剪后子文件夹的根目录）
    # 目录结构应为: base_input_dir/2017年_分块裁剪/, base_input_dir/2018年_分块裁剪/, ...
    # 注意：请根据实际数据存储位置修改此路径
    base_input_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\ESRI_2017_2023_分块裁剪"
    
    # 基础输出目录（镶嵌后的文件将保存在此目录下）
    # 输出文件命名格式: southeast_asia_landcover_YYYY_mosaic_ESRI_10m.tif
    # 注意：请根据实际输出位置修改此路径
    base_output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据"
    
    # 要处理的年份范围（可以修改起始和结束年份）
    start_year = 2019  # 起始年份
    end_year = 2023    # 结束年份
    years = list(range(start_year, end_year + 1))  # 生成年份列表
    
    # ==================== 配置区域结束 ====================
    
    print(f"📂 基础输入目录: {base_input_dir}")
    print(f"📂 基础输出目录: {base_output_dir}")
    print(f"📅 处理年份范围: {start_year}-{end_year} ({len(years)}个年份)")
    print("=" * 80)
    
    # 创建基础输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(base_output_dir)
    logging.info(f"开始ESRI数据多年份批量镶嵌处理")
    logging.info(f"基础输入目录: {base_input_dir}")
    logging.info(f"基础输出目录: {base_output_dir}")
    logging.info(f"处理年份: {years}")
    logging.info(f"日志文件: {log_path}")
    
    # 检查基础输入目录是否存在
    if not os.path.exists(base_input_dir):
        logging.error(f"基础输入目录不存在: {base_input_dir}")
        print(f"❌ 错误：基础输入目录不存在: {base_input_dir}")
        return
    
    # 开始循环处理各个年份
    total_start_time = time.time()
    all_results = []
    
    logging.info(f"\n{'='*80}")
    logging.info(f"开始循环处理 {len(years)} 个年份的镶嵌")
    logging.info(f"{'='*80}")
    
    for i, year in enumerate(years, 1):
        logging.info(f"\n🔄 总进度: {i}/{len(years)} - 开始处理 {year} 年镶嵌")
        
        # 处理单个年份镶嵌
        year_result = process_single_year_mosaic(year, base_input_dir, base_output_dir)
        all_results.append(year_result)
        
        # 显示当前年份处理结果
        year_num, success, year_time, output_size_gb = year_result
        if success:
            logging.info(f"✅ {year} 年镶嵌完成 - 成功, 输出: {output_size_gb:.2f}GB, 耗时: {year_time:.2f}秒")
        else:
            logging.info(f"❌ {year} 年镶嵌失败 - 耗时: {year_time:.2f}秒")
    
    # 所有年份处理完成，输出总体统计
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    logging.info(f"\n{'='*80}")
    logging.info("🎉 所有年份镶嵌处理完成！总体统计:")
    logging.info(f"{'='*80}")
    
    total_success = 0
    total_fail = 0
    total_output_size = 0
    
    for year_num, success, year_time, output_size_gb in all_results:
        if success:
            total_success += 1
            total_output_size += output_size_gb
            logging.info(f"{year_num}年: ✅ 成功, 输出 {output_size_gb:.2f} GB, 耗时 {year_time:.2f} 秒")
        else:
            total_fail += 1
            logging.info(f"{year_num}年: ❌ 失败, 耗时 {year_time:.2f} 秒")
    
    logging.info(f"\n📊 汇总统计:")
    logging.info(f"处理年份数: {len(years)}")
    logging.info(f"成功镶嵌年份数: {total_success}")
    logging.info(f"失败镶嵌年份数: {total_fail}")
    logging.info(f"总输出文件大小: {total_output_size:.2f} GB")
    logging.info(f"成功率: {(total_success/len(years)*100):.2f}%")
    logging.info(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    logging.info(f"平均每年耗时: {total_time/len(years):.2f} 秒")
    if total_output_size > 0:
        logging.info(f"平均处理速度: {total_output_size/total_time*60:.2f} GB/分钟")
    
    logging.info(f"\n📁 输出目录: {base_output_dir}")
    logging.info(f"📄 日志文件: {log_path}")
    logging.info("="*80)
    
    print(f"\n🎉 处理完成！")
    print(f"📊 总计处理了 {len(years)} 个年份")
    print(f"✅ 成功: {total_success} 个年份")
    print(f"❌ 失败: {total_fail} 个年份")
    print(f"📦 总输出: {total_output_size:.2f} GB")
    print(f"⏱️ 总耗时: {total_time/60:.2f} 分钟")

if __name__ == "__main__":
    main()