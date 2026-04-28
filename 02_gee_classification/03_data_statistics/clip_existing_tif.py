#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本功能：对指定文件夹下的所有TIF文件进行批量、多进程、分块裁剪。
作者：锐多宝 (ruiduobao)
创建时间：2025年1月
版本：2.0

核心功能:
1.  **多进程并行处理**: 利用所有可用的CPU核心，并行处理多个TIF文件，大幅提升裁剪效率。
2.  **分块裁剪 (Tiled Cropping)**: 使用GDAL的分块读取和写入机制，有效处理超大栅格文件，防止内存溢出。
3.  **属性完美继承**:
    -   保留原始栅格的**颜色映射表 (Color Table)**。
    -   维持**LZW无损压缩**。
    -   确保输出数据格式为**Int8 (Byte)**。
    -   将**NoData值设为0**。
4.  **自动金字塔生成**: 为每个裁剪后的TIF文件自动构建外置金字塔（.ovr），优化在GIS软件中的显示性能。
5.  **健壮的错误处理与日志记录**: 记录详细的运行日志，包括每个文件的处理状态、错误信息和性能统计。
"""

import os
import sys
import logging
import time
from datetime import datetime
from osgeo import gdal, gdalconst
import multiprocessing
from pathlib import Path

# --- 配置参数 ---

# 输入TIF文件夹
INPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\4.GEE导出结果_结果合并_马尔可夫模型_逐年合并"
# 输出裁剪后TIF的文件夹
OUTPUT_DIR = r"F:\地理所\论文\东南亚10m\数据\正式分类_10.29\5.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚"
# 用于裁剪的矢量文件（SHP）
CLIP_SHP_PATH = r"F:\地理所\论文\东南亚10m\数据\裁剪矢量\东南亚裁剪矢量.shp"
# 日志文件存放目录 (确保日志在脚本所在目录的'日志'文件夹下)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "日志")

# --- 全局设置 ---

# 启用GDAL异常处理
gdal.UseExceptions()

def setup_logging(log_dir):
    """
    配置日志记录系统。
    将日志同时输出到控制台和文件中。
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    log_filename = f"clip_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"日志系统初始化完成，日志将记录到: {log_path}")
    return log_path

def build_overviews(file_path):
    """
    为栅格文件构建外置金字塔（.ovr），优化显示性能。
    
    Args:
        file_path (str): 栅格文件路径。
    """
    try:
        logging.info(f"开始为文件构建金字塔: {Path(file_path).name}")
        dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
        if dataset is None:
            logging.error(f"无法打开文件以构建金字塔: {file_path}")
            return

        # 为分类数据选择最近邻重采样
        resampling_method = "NEAREST"
        overview_levels = [2, 4, 8, 16, 32, 64]

        # 设置金字塔压缩选项
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
        gdal.SetConfigOption('TILED_OVERVIEW', 'YES')
        gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'IF_SAFER')

        # 构建金字塔
        dataset.BuildOverviews(resampling_method, overview_levels)
        logging.info(f"成功为 {Path(file_path).name} 构建金字塔。")

    except Exception as e:
        logging.error(f"为 {Path(file_path).name} 构建金字塔时发生错误: {e}")
    finally:
        # 清理GDAL配置
        gdal.SetConfigOption('COMPRESS_OVERVIEW', None)
        gdal.SetConfigOption('TILED_OVERVIEW', None)
        gdal.SetConfigOption('BIGTIFF_OVERVIEW', None)
        if 'dataset' in locals():
            dataset = None

def clip_raster_task(task_params):
    """
    单个TIF文件的裁剪任务函数，由多进程池调用。
    
    Args:
        task_params (tuple): 包含输入文件路径和输出文件路径的元组。
    """
    # --- 优化GDAL性能 ---
    # 在每个子进程中独立设置GDAL块缓存大小（单位MB）。
    # 这有助于在处理大型栅格时优化IO性能和内存管理，作为warpMemoryLimit的补充。
    gdal.SetConfigOption('GDAL_CACHEMAX', '256')

    input_file, output_file = task_params
    process_name = multiprocessing.current_process().name
    logging.info(f"[{process_name}] 开始处理: {Path(input_file).name}")

    try:
        # 打开原始栅格，获取颜色表
        src_ds = gdal.Open(input_file, gdalconst.GA_ReadOnly)
        if not src_ds:
            logging.error(f"无法打开源文件: {input_file}")
            return False
            
        src_band = src_ds.GetRasterBand(1)
        color_table = src_band.GetColorTable()

        # 配置gdal.Warp选项
        warp_options = gdal.WarpOptions(
            format='GTiff',
            cutlineDSName=CLIP_SHP_PATH,  # 裁剪矢量
            cropToCutline=True,           # 裁剪到矢量范围
            dstNodata=0,                  # 设置输出NoData值
            outputType=gdalconst.GDT_Byte,# 输出数据类型为Int8
            resampleAlg='near',           # 最近邻采样，适用于分类数据
            creationOptions=[
                'COMPRESS=LZW',           # LZW压缩
                'TILED=YES',              # 分块存储
                'BIGTIFF=IF_SAFER'        # 如果需要，使用BigTIFF
            ],
            # 内存管理，防止溢出
            warpMemoryLimit=1024,  # 每个线程分配2048MB内存
            multithread=False, # 禁用内部多线程，因为外部已使用多进程
            callback=gdal.TermProgress_nocb # 显示进度条
        )

        # 执行裁剪
        logging.info(f"正在裁剪 {Path(input_file).name}...")
        clipped_ds = gdal.Warp(output_file, src_ds, options=warp_options)
        
        if not clipped_ds:
            logging.error(f"裁剪失败: {Path(input_file).name}")
            return False

        # 获取裁剪后的波段并应用颜色表
        clipped_band = clipped_ds.GetRasterBand(1)
        if color_table:
            clipped_band.SetColorTable(color_table)
            logging.info(f"已将颜色表应用于: {Path(output_file).name}")
        
        # 设置颜色解释
        clipped_band.SetColorInterpretation(gdalconst.GCI_PaletteIndex)

        # 释放数据集
        src_ds = None
        clipped_ds = None

        # 构建金字塔
        build_overviews(output_file)

        logging.info(f"[{process_name}] ✓ 处理完成: {Path(output_file).name}")
        return True

    except Exception as e:
        logging.error(f"[{process_name}] ✗ 处理失败: {Path(input_file).name}。错误: {e}")
        # 如果出错，尝试删除可能已创建的不完整输出文件
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError as oe:
                logging.warning(f"无法删除不完整的输出文件 {output_file}: {oe}")
        return False

def main():
    """
    主函数，负责启动多进程任务。
    """
    # 设置日志
    setup_logging(LOG_DIR)
    
    logging.info("=" * 60)
    logging.info("TIF批量多进程裁剪脚本启动")
    logging.info(f"作者: 锐多宝 (ruiduobao)")
    logging.info(f"输入目录: {INPUT_DIR}")
    logging.info(f"输出目录: {OUTPUT_DIR}")
    logging.info(f"裁剪矢量: {CLIP_SHP_PATH}")
    logging.info("=" * 60)

    # 检查路径是否存在
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"输入目录不存在: {INPUT_DIR}")
        return
    if not os.path.exists(CLIP_SHP_PATH):
        logging.error(f"裁剪矢量文件不存在: {CLIP_SHP_PATH}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集所有待处理的TIF文件
    tasks = []
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.tif'):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            # 如果输出文件已存在，则跳过
            if os.path.exists(output_path):
                logging.info(f"文件已存在，跳过: {filename}")
                continue
            
            tasks.append((input_path, output_path))

    if not tasks:
        logging.info("没有需要处理的新文件。")
        return

    logging.info(f"发现 {len(tasks)} 个TIF文件待处理。")

    # 启动多进程池
    # 将进程数限制在更合理的范围内，以避免内存溢出
    # 例如，限制为4个或CPU核心数的一半
    cpu_cores = multiprocessing.cpu_count()
    num_processes = min(cpu_cores, 4) 
    logging.info(f"启动 {num_processes} 个进程进行并行处理 (总核心数: {cpu_cores})...")
    
    start_time = time.time()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(clip_raster_task, tasks)

    end_time = time.time()
    
    # 统计结果
    success_count = sum(1 for r in results if r)
    failed_count = len(tasks) - success_count
    
    logging.info("=" * 60)
    logging.info("批量裁剪处理完成！")
    logging.info(f"总耗时: {end_time - start_time:.2f} 秒")
    logging.info(f"成功处理: {success_count} 个文件")
    if failed_count > 0:
        logging.warning(f"失败: {failed_count} 个文件 (详情请查看日志)")
    logging.info("=" * 60)

if __name__ == '__main__':
    # 在Windows上，多进程需要这个保护
    multiprocessing.freeze_support()
    main()



