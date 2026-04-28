#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的等面积投影统计功能
只处理zone1的一个文件来快速验证修复效果
"""

import os
import rasterio
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

# 配置参数
TIF_DIR_PATH = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型"
OUTPUT_DIR = r"f:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\3.分析\输出结果"

# 像素面积（平方米）
PIXEL_AREA_M2 = 100.0

# 分类值映射
CLASS_VALUES = {
    1: "人工林",
    2: "自然林", 
    3: "其他"
}

def setup_logging():
    """设置日志配置"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUTPUT_DIR, f"测试修复后统计_{timestamp}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"日志文件: {log_file}")
    return log_file

def monitor_memory():
    """监控内存使用情况"""
    memory = psutil.virtual_memory()
    return memory.percent

def process_single_tif_file(file_path):
    """
    处理单个等面积投影TIF文件，测试修复后的像素统计
    """
    try:
        logging.info(f"开始处理: {os.path.basename(file_path)}")
        
        # 初始化像素计数器（使用64位整数避免溢出）
        pixel_counts = {class_val: np.int64(0) for class_val in CLASS_VALUES.keys()}
        total_pixels = np.int64(0)
        
        # 打开TIF文件
        with rasterio.open(file_path) as src:
            logging.info(f"文件尺寸: {src.width} x {src.height}")
            logging.info(f"数据类型: {src.dtypes[0]}")
            logging.info(f"NoData值: {src.nodata}")
            
            # 分块处理以节省内存
            chunk_size = 1000
            processed_chunks = 0
            total_chunks = ((src.height + chunk_size - 1) // chunk_size) * ((src.width + chunk_size - 1) // chunk_size)
            
            for row_start in range(0, src.height, chunk_size):
                for col_start in range(0, src.width, chunk_size):
                    # 计算当前块的边界
                    row_end = min(row_start + chunk_size, src.height)
                    col_end = min(col_start + chunk_size, src.width)
                    
                    # 定义读取窗口
                    window = rasterio.windows.Window(
                        col_start, row_start, 
                        col_end - col_start, row_end - row_start
                    )
                    
                    # 读取数据块
                    chunk_data = src.read(1, window=window)
                    
                    # 统计各类别像素数量
                    for class_val in CLASS_VALUES.keys():
                        mask = (chunk_data == class_val)
                        pixel_counts[class_val] += np.int64(np.sum(mask))
                        del mask  # 立即释放内存
                    
                    # 统计总像素数
                    valid_mask = chunk_data != src.nodata if src.nodata is not None else np.ones_like(chunk_data, dtype=bool)
                    total_pixels += np.int64(np.sum(valid_mask))
                    del valid_mask  # 立即释放内存
                    
                    # 删除数据块并执行垃圾回收
                    del chunk_data
                    
                    processed_chunks += 1
                    
                    # 每处理100个块输出一次进度
                    if processed_chunks % 100 == 0:
                        progress = (processed_chunks / total_chunks) * 100
                        memory_percent = monitor_memory()
                        logging.info(f"进度: {progress:.1f}% ({processed_chunks}/{total_chunks}), 内存: {memory_percent:.1f}%")
                        
                        # 强制垃圾回收
                        gc.collect()
        
        # 计算面积
        areas_m2 = {}
        areas_ha = {}
        areas_km2 = {}
        
        for class_val, class_name in CLASS_VALUES.items():
            areas_m2[class_name] = float(pixel_counts[class_val]) * PIXEL_AREA_M2
            areas_ha[class_name] = areas_m2[class_name] / 10000.0
            areas_km2[class_name] = areas_ha[class_name] / 100.0
        
        # 计算总面积
        total_area_m2 = float(total_pixels) * PIXEL_AREA_M2
        total_area_ha = total_area_m2 / 10000.0
        total_area_km2 = total_area_ha / 100.0
        
        # 验证像素数一致性
        calculated_total_pixels = sum(pixel_counts.values())
        
        logging.info(f"处理完成: {os.path.basename(file_path)}")
        logging.info(f"总像素数: {total_pixels:,}")
        logging.info(f"各类像素之和: {calculated_total_pixels:,}")
        logging.info(f"像素数差异: {calculated_total_pixels - total_pixels:,}")
        logging.info(f"总面积: {total_area_ha:.2f} 公顷")
        
        return {
            'file_path': file_path,
            'pixel_counts': pixel_counts,
            'total_pixels': int(total_pixels),
            'calculated_total_pixels': int(calculated_total_pixels),
            'areas_ha': areas_ha,
            'total_area_ha': total_area_ha,
            'areas_km2': areas_km2,
            'total_area_km2': total_area_km2
        }
        
    except Exception as e:
        logging.error(f"处理文件失败 {file_path}: {str(e)}")
        return None

def main():
    """主函数"""
    start_time = datetime.now()
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置日志
    log_file = setup_logging()
    
    logging.info("开始测试修复后的等面积投影统计")
    logging.info(f"像素面积: {PIXEL_AREA_M2} 平方米/像素")
    
    # 监控内存
    memory_percent = monitor_memory()
    logging.info(f"初始内存使用率: {memory_percent:.1f}%")
    
    # 找一个zone1的等面积投影文件进行测试
    zone1_dir = os.path.join(TIF_DIR_PATH, "zone1")
    test_files = []
    
    if os.path.exists(zone1_dir):
        for file in os.listdir(zone1_dir):
            if file.endswith("_等面积投影.tif"):
                test_files.append(os.path.join(zone1_dir, file))
                break  # 只测试一个文件
    
    if not test_files:
        logging.error("未找到测试文件")
        return
    
    logging.info(f"找到测试文件: {len(test_files)} 个")
    
    # 处理测试文件
    result = process_single_tif_file(test_files[0])
    
    if result:
        logging.info("测试结果:")
        logging.info(f"  文件: {os.path.basename(result['file_path'])}")
        logging.info(f"  总像素数: {result['total_pixels']:,}")
        logging.info(f"  各类像素之和: {result['calculated_total_pixels']:,}")
        logging.info(f"  像素数差异: {result['calculated_total_pixels'] - result['total_pixels']:,}")
        logging.info(f"  人工林像素: {result['pixel_counts'][1]:,}")
        logging.info(f"  自然林像素: {result['pixel_counts'][2]:,}")
        logging.info(f"  其他像素: {result['pixel_counts'][3]:,}")
        logging.info(f"  总面积: {result['total_area_ha']:.2f} 公顷")
        
        # 检查是否还有负数
        if result['total_area_ha'] < 0:
            logging.error("警告: 总面积仍为负数!")
        else:
            logging.info("✓ 总面积为正数，修复成功!")
    
    # 计算总耗时
    end_time = datetime.now()
    total_time = end_time - start_time
    logging.info(f"测试完成，总耗时: {total_time}")
    
    # 最终内存使用情况
    final_memory = monitor_memory()
    logging.info(f"最终内存使用率: {final_memory:.1f}%")

if __name__ == "__main__":
    main()