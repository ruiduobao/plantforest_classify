#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
林地样本点生成脚本（优化版本）

功能：
1. 从土地覆盖分类数据中生成林地样本点
2. 采用分块处理和多进程并行，提高处理效率
3. 支持内存监控和分批保存，避免内存溢出
4. 生成详细的处理日志和统计报告

优化特性：
- 多进程并行处理，充分利用CPU资源
- 分块读取栅格数据，减少内存占用
- 实时内存监控，动态调整处理策略
- 分批保存结果，避免大数据集内存问题
- 进度条显示，实时监控处理进度

作者：锐多宝 (ruiduobao)
日期：2024年
"""

import os
import sys
import time
import logging
import random
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import gc
import psutil
import warnings
warnings.filterwarnings('ignore')

# 配置参数
LANDCOVER_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\土地覆盖数据\southeast_asia_landcover_2024_mosaic.tif"
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\3.人工林样本点\1.林地样本点"
FOREST_VALUE = 2  # 森林像素值
SAMPLING_RATE = 5000  # 每500个像素选择1个
CHUNK_SIZE = 5000  # 分块处理大小
NUM_PROCESSES = min(6, cpu_count())  # 多进程数量
BATCH_SIZE = 50000  # 分批保存的样本点数量
MEMORY_THRESHOLD = 80  # 内存使用率阈值（百分比）

def setup_logging(output_dir):
    """
    设置日志记录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"forest_sampling_log_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def check_input_files():
    """
    检查输入文件是否存在
    """
    if not os.path.exists(LANDCOVER_FILE):
        raise FileNotFoundError(f"土地覆盖数据文件不存在: {LANDCOVER_FILE}")
    
    logging.info(f"输入文件检查完成:")
    logging.info(f"  土地覆盖数据: {LANDCOVER_FILE}")

def get_memory_usage():
    """
    获取当前内存使用率
    """
    return psutil.virtual_memory().percent

def save_batch_results(points_batch, output_dir, batch_num, crs):
    """
    保存批次结果到临时文件
    """
    if not points_batch:
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成批次文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = os.path.join(output_dir, f"temp_batch_{batch_num:04d}_{timestamp}.gpkg")
    
    # 创建GeoDataFrame并保存
    gdf_batch = gpd.GeoDataFrame(points_batch, crs=crs)
    gdf_batch.to_file(batch_file, driver='GPKG')
    
    logging.info(f"批次 {batch_num} 已保存: {len(points_batch)} 个点 -> {batch_file}")
    logging.info(f"当前内存使用率: {get_memory_usage():.1f}%")
    
    return batch_file

def merge_batch_files(batch_files, output_dir):
    """
    合并所有批次文件为最终结果
    """
    if not batch_files:
        return None
    
    logging.info(f"开始合并 {len(batch_files)} 个批次文件...")
    
    # 读取第一个文件获取结构
    gdf_final = gpd.read_file(batch_files[0])
    
    # 逐个读取并合并其他文件
    for i, batch_file in enumerate(batch_files[1:], 1):
        logging.info(f"合并进度: {i}/{len(batch_files)-1}")
        gdf_batch = gpd.read_file(batch_file)
        gdf_final = pd.concat([gdf_final, gdf_batch], ignore_index=True)
        
        # 删除临时文件
        try:
            os.remove(batch_file)
        except:
            pass
    
    # 删除第一个临时文件
    try:
        os.remove(batch_files[0])
    except:
        pass
    
    # 生成最终输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"nonforest_sample_points_{timestamp}.gpkg")
    
    # 保存最终结果
    gdf_final.to_file(output_file, driver='GPKG')
    
    logging.info(f"合并完成，最终文件: {output_file}")
    logging.info(f"总样本点数: {len(gdf_final)}")
    
    return gdf_final, output_file

def get_raster_info(raster_path):
    """
    获取栅格数据信息
    """
    with rasterio.open(raster_path) as src:
        info = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'nodata': src.nodata
        }
    return info

def process_chunk(args):
    """
    处理单个数据块的函数（用于多进程）
    """
    chunk_id, window, raster_path, transform, crs, sampling_rate, forest_value = args
    
    try:
        # 读取数据块
        with rasterio.open(raster_path) as src:
            data = src.read(1, window=window)
        
        # 获取窗口的行列偏移
        row_off, col_off = window.row_off, window.col_off
        
        # 筛选林地像素（值为2的森林像素）
        valid_mask = (data == forest_value) & (~np.isnan(data)) & (data != src.nodata)
        
        if not np.any(valid_mask):
            return []
        
        # 获取有效像素的行列索引
        rows, cols = np.where(valid_mask)
        values = data[rows, cols]
        
        # 随机采样
        total_pixels = len(rows)
        sample_size = max(1, total_pixels // sampling_rate)
        
        if sample_size > 0:
            # 随机选择样本索引
            sample_indices = random.sample(range(total_pixels), min(sample_size, total_pixels))
            
            # 获取采样点的行列和值
            sample_rows = rows[sample_indices] + row_off
            sample_cols = cols[sample_indices] + col_off
            sample_values = values[sample_indices]
            
            # 转换为地理坐标
            points = []
            for i in range(len(sample_rows)):
                # 计算像素中心的地理坐标
                x, y = rasterio.transform.xy(transform, sample_rows[i], sample_cols[i])
                points.append({
                    'geometry': Point(x, y),
                    'value': int(sample_values[i]),
                    'row': int(sample_rows[i]),
                    'col': int(sample_cols[i])
                })
            
            logging.info(f"块 {chunk_id}: 处理了 {total_pixels} 个林地像素，采样了 {len(points)} 个点")
            return points
        
        return []
        
    except Exception as e:
        logging.error(f"处理块 {chunk_id} 时出错: {str(e)}")
        return []

def generate_forest_sample_points():
    """
    生成林地样本点（优化版本，支持进度显示和分批保存）
    """
    logging.info("开始生成林地样本点...")
    
    # 获取栅格信息
    raster_info = get_raster_info(LANDCOVER_FILE)
    logging.info(f"栅格信息:")
    logging.info(f"  CRS: {raster_info['crs']}")
    logging.info(f"  尺寸: {raster_info['width']} x {raster_info['height']}")
    logging.info(f"  范围: {raster_info['bounds']}")
    logging.info(f"  NoData值: {raster_info['nodata']}")
    
    # 计算分块参数
    width, height = raster_info['width'], raster_info['height']
    
    # 创建处理任务列表
    tasks = []
    chunk_id = 0
    
    for row_start in range(0, height, CHUNK_SIZE):
        for col_start in range(0, width, CHUNK_SIZE):
            # 计算当前块的大小
            row_size = min(CHUNK_SIZE, height - row_start)
            col_size = min(CHUNK_SIZE, width - col_start)
            
            # 创建窗口
            window = Window(col_start, row_start, col_size, row_size)
            
            # 添加任务
            task = (
                chunk_id,
                window,
                LANDCOVER_FILE,
                raster_info['transform'],
                raster_info['crs'],
                SAMPLING_RATE,
                FOREST_VALUE
            )
            tasks.append(task)
            chunk_id += 1
    
    logging.info(f"总共创建了 {len(tasks)} 个处理块")
    logging.info(f"初始内存使用率: {get_memory_usage():.1f}%")
    
    # 初始化变量
    all_points = []
    batch_files = []
    batch_num = 0
    processed_chunks = 0
    
    # 使用进度条显示处理进度
    with tqdm(total=len(tasks), desc="处理进度", unit="块") as pbar:
        # 多进程处理
        with Pool(processes=NUM_PROCESSES) as pool:
            # 分批处理任务以控制内存使用
            batch_size = max(1, len(tasks) // 10)  # 将任务分成10批
            
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i+batch_size]
                
                # 处理当前批次
                results = pool.map(process_chunk, batch_tasks)
                
                # 收集结果
                for result in results:
                    if result:
                        all_points.extend(result)
                    
                    processed_chunks += 1
                    pbar.update(1)
                    
                    # 更新进度信息
                    progress = (processed_chunks / len(tasks)) * 100
                    memory_usage = get_memory_usage()
                    pbar.set_postfix({
                        '进度': f'{progress:.1f}%',
                        '内存': f'{memory_usage:.1f}%',
                        '样本点': len(all_points)
                    })
                
                # 检查是否需要保存批次结果
                if (len(all_points) >= BATCH_SIZE or 
                    get_memory_usage() > MEMORY_THRESHOLD or
                    processed_chunks == len(tasks)):
                    
                    if all_points:
                        # 保存当前批次
                        batch_file = save_batch_results(
                            all_points, OUTPUT_DIR, batch_num, raster_info['crs']
                        )
                        if batch_file:
                            batch_files.append(batch_file)
                        
                        # 清空内存中的点数据
                        all_points = []
                        batch_num += 1
                        
                        # 强制垃圾回收
                        gc.collect()
    
    logging.info(f"数据处理完成，共生成 {len(batch_files)} 个批次文件")
    
    # 合并所有批次文件
    if batch_files:
        gdf, output_file = merge_batch_files(batch_files, OUTPUT_DIR)
        return gdf
    else:
        logging.warning("没有生成任何样本点！")
        return None

def save_results(gdf, output_dir):
    """
    保存结果到文件（已由分批保存功能替代，保留用于生成统计报告）
    """
    if gdf is None or len(gdf) == 0:
        logging.error("没有数据可保存")
        return None
    
    # 统计各类别的样本点数量
    value_counts = gdf['value'].value_counts().sort_index()
    logging.info("各土地覆盖类型的样本点数量:")
    for value, count in value_counts.items():
        logging.info(f"  类型 {value}: {count} 个点 ({count/len(gdf)*100:.2f}%)")
    
    # 生成统计报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"forest_sampling_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("林地样本点生成报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入数据: {LANDCOVER_FILE}\n")
        f.write(f"采样率: 每{SAMPLING_RATE}个像素选择1个\n")
        f.write(f"总样本点数: {len(gdf)}\n")
        f.write(f"分批大小: {BATCH_SIZE}\n")
        f.write(f"内存阈值: {MEMORY_THRESHOLD}%\n\n")
        
        f.write("各土地覆盖类型统计:\n")
        for value, count in value_counts.items():
            f.write(f"  类型 {value}: {count} 个点 ({count/len(gdf)*100:.2f}%)\n")
    
    logging.info(f"统计报告已保存到: {report_file}")
    
    return report_file

def main():
    """
    主函数
    """
    start_time = time.time()
    
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 设置日志
    log_file = setup_logging(OUTPUT_DIR)
    
    logging.info("=" * 60)
    logging.info("非林地样本点生成脚本")
    logging.info("=" * 60)
    logging.info(f"配置参数:")
    logging.info(f"  土地覆盖数据: {LANDCOVER_FILE}")
    logging.info(f"  输出目录: {OUTPUT_DIR}")
    logging.info(f"  森林像素值: {FOREST_VALUE}")
    logging.info(f"  采样率: 每{SAMPLING_RATE}个像素选择1个")
    logging.info(f"  分块大小: {CHUNK_SIZE}")
    logging.info(f"  进程数: {NUM_PROCESSES}")
    logging.info(f"  分批大小: {BATCH_SIZE}")
    logging.info(f"  内存阈值: {MEMORY_THRESHOLD}%")
    
    try:
        # 检查输入文件
        check_input_files()
        
        # 生成非林地样本点
        gdf = generate_nonforest_sample_points()
        
        # 保存结果
        if gdf is not None:
            report_file = save_results(gdf, OUTPUT_DIR)
            
            # 计算处理时间
            elapsed_time = time.time() - start_time
            logging.info(f"处理完成！总耗时: {elapsed_time:.2f} 秒")
            
            if len(gdf) > 0:
                processing_speed = len(gdf) / elapsed_time
                logging.info(f"处理速度: {processing_speed:.0f} 点/秒")
                logging.info(f"最终内存使用率: {get_memory_usage():.1f}%")
        else:
            logging.error("样本点生成失败")
            
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # 设置日志
        log_file = setup_logging(OUTPUT_DIR)
        logging.info(f"日志文件: {log_file}")
        
        # 记录配置信息
        logging.info("=" * 60)
        logging.info("林地样本点生成脚本启动")
        logging.info("=" * 60)
        logging.info(f"输入文件: {LANDCOVER_FILE}")
        logging.info(f"输出目录: {OUTPUT_DIR}")
        logging.info(f"采样率: 每{SAMPLING_RATE}个像素选择1个")
        logging.info(f"林地值: {FOREST_VALUE}")
        logging.info(f"分块大小: {CHUNK_SIZE}x{CHUNK_SIZE}")
        logging.info(f"进程数: {NUM_PROCESSES}")
        logging.info(f"批次大小: {BATCH_SIZE}")
        logging.info(f"内存阈值: {MEMORY_THRESHOLD}%")
        
        # 生成样本点
        start_time = time.time()
        gdf = generate_forest_sample_points()
        end_time = time.time()
        
        if gdf is not None:
            logging.info(f"样本点生成完成！")
            logging.info(f"总耗时: {end_time - start_time:.2f} 秒")
            logging.info(f"总样本点数: {len(gdf)}")
            logging.info(f"平均处理速度: {len(gdf)/(end_time - start_time):.2f} 点/秒")
            
            # 生成统计报告
            save_results(gdf, OUTPUT_DIR)
            
        else:
            logging.error("样本点生成失败！")
            
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise