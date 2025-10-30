#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计人工林和自然林变化情况脚本
功能：统计2017-2024年间人工林(1)、自然林(2)、其他(3)类别的面积变化
作者：锐多宝 (ruiduobao)
创建时间：2025年1月
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import json
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

def setup_logging(output_dir):
    """
    设置日志记录
    
    参数:
    - output_dir: 输出目录
    
    返回:
    - logger: 日志记录器
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"forest_change_statistics_{timestamp}.log")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存至: {log_file}")
    
    return logger

def get_raster_statistics(raster_path, categories=[1, 2, 3]):
    """
    计算单个栅格文件的统计信息（使用分块读取处理大文件）
    
    参数:
    - raster_path: 栅格文件路径
    - categories: 需要统计的类别列表 [人工林(1), 自然林(2), 其他(3)]
    
    返回:
    - dict: 包含各类别像素数量和面积的字典
    """
    # 在子进程中创建本地logger
    local_logger = logging.getLogger(__name__)
    
    try:
        with rasterio.open(raster_path) as src:
            # 获取像素分辨率
            pixel_size_x, pixel_size_y = src.res
            crs = src.crs
            
            # 调试信息：输出像素大小和坐标系信息
            local_logger.info(f"文件: {raster_path}")
            local_logger.info(f"坐标系: {src.crs}")
            local_logger.info(f"像素大小: x={pixel_size_x}, y={pixel_size_y}")
            local_logger.info(f"Transform: {src.transform}")
            
            # 计算每个像素的面积
            if crs and crs.to_string() == 'EPSG:4326':
                # 地理坐标系，需要考虑纬度影响
                # 在东南亚地区（约北纬10度），1度约等于111km
                # 0.0001度 ≈ 11.1米
                pixel_area = abs(pixel_size_x * pixel_size_y) * (111000 ** 2)  # 转换为平方米
                local_logger.info(f"地理坐标系像素面积: {pixel_area} 平方米")
            else:
                # 投影坐标系，直接使用分辨率
                # 使用 abs() 确保面积为正数，因为 transform[4] 通常为负数
                pixel_area = abs(pixel_size_x * pixel_size_y)  # 像素面积（平方米）
                local_logger.info(f"投影坐标系像素面积: {pixel_area} 平方米")
            
            # 初始化统计计数器（使用64位整数防止溢出）
            category_counts = {category: np.int64(0) for category in categories}
            total_pixels = np.int64(0)
            valid_pixels = np.int64(0)
            
            # 分块读取数据以节省内存
            block_size = 1024  # 每次读取1024x1024像素块
            
            for row_start in range(0, src.height, block_size):
                for col_start in range(0, src.width, block_size):
                    # 计算当前块的窗口
                    row_end = min(row_start + block_size, src.height)
                    col_end = min(col_start + block_size, src.width)
                    
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    
                    # 读取当前块的数据
                    block_data = src.read(1, window=window)
                    
                    # 统计当前块（使用64位整数）
                    block_total = np.int64(block_data.size)
                    block_valid = np.int64(np.sum(~np.isnan(block_data)))
                    
                    total_pixels += block_total
                    valid_pixels += block_valid
                    
                    # 统计各类别像素数量（使用64位整数）
                    for category in categories:
                        category_counts[category] += np.int64(np.sum(block_data == category))
            
            # 构建统计结果
            stats = {}
            for category in categories:
                pixel_count = int(category_counts[category])  # 转换为Python int
                area_sqm = float(pixel_count * pixel_area)  # 确保使用浮点数计算
                area_sqkm = area_sqm / 1000000.0  # 转换为平方公里
                
                # 调试信息：输出每个类别的计算过程
                local_logger.info(f"类别 {category}: 像素数={pixel_count}, 面积(平方米)={area_sqm}, 面积(平方公里)={area_sqkm}")
                
                stats[f'category_{category}'] = {
                    'pixel_count': pixel_count,
                    'area_sqm': area_sqm,
                    'area_sqkm': area_sqkm,
                    'percentage': float(pixel_count / valid_pixels * 100) if valid_pixels > 0 else 0.0
                }
            
            # 添加总体统计信息
            stats['total_info'] = {
                'total_pixels': int(total_pixels),
                'valid_pixels': int(valid_pixels),
                'pixel_area_sqm': float(pixel_area),
                'crs': str(src.crs),
                'bounds': src.bounds,
                'transform': list(src.transform)
            }
            
            return stats
            
    except Exception as e:
        raise Exception(f"处理文件 {raster_path} 时出错: {str(e)}")

def process_single_file(args):
    """
    处理单个文件的包装函数（用于多进程）
    
    参数:
    - args: 包含文件路径、年份和类别的元组
    
    返回:
    - tuple: (年份, 统计结果, 文件路径)
    """
    file_path, year, categories = args
    try:
        stats = get_raster_statistics(file_path, categories)
        return year, stats, file_path, None
    except Exception as e:
        return year, None, file_path, str(e)

def find_classification_files(base_dir, years, file_pattern="*.tif"):
    """
    查找指定年份的分类结果文件
    
    参数:
    - base_dir: 基础目录路径
    - years: 年份列表
    - file_pattern: 文件匹配模式
    
    返回:
    - dict: {年份: 文件路径列表}
    """
    file_dict = {}
    
    # 直接在基础目录中查找所有匹配的文件
    pattern_path = os.path.join(base_dir, file_pattern)
    print(pattern_path)
    all_files = glob.glob(pattern_path)
    
    for year in years:
        year_files = []
        
        # 查找包含年份的文件
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            # 检查文件名中是否包含年份
            if str(year) in filename:
                year_files.append(file_path)
        
        if year_files:
            file_dict[year] = year_files
        else:
            print(f"警告: 未找到 {year} 年的分类文件")
    
    return file_dict

def calculate_change_statistics(yearly_stats):
    """
    计算年际变化统计
    
    参数:
    - yearly_stats: 年度统计数据字典
    
    返回:
    - dict: 变化统计结果
    """
    change_stats = {}
    years = sorted(yearly_stats.keys())
    
    # 计算相邻年份的变化
    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]
        change_key = f"{year1}_to_{year2}"
        
        change_stats[change_key] = {}
        
        for category in [1, 2, 3]:
            cat_key = f'category_{category}'
            
            if (cat_key in yearly_stats[year1] and 
                cat_key in yearly_stats[year2]):
                
                area1 = yearly_stats[year1][cat_key]['area_sqkm']
                area2 = yearly_stats[year2][cat_key]['area_sqkm']
                
                change_area = area2 - area1
                change_percent = (change_area / area1 * 100) if area1 > 0 else 0
                
                change_stats[change_key][cat_key] = {
                    'area_change_sqkm': float(change_area),
                    'percent_change': float(change_percent),
                    'initial_area': float(area1),
                    'final_area': float(area2)
                }
    
    # 计算总体变化（2017到2024）
    if len(years) >= 2:
        first_year, last_year = years[0], years[-1]
        total_change_key = f"{first_year}_to_{last_year}_total"
        
        change_stats[total_change_key] = {}
        
        for category in [1, 2, 3]:
            cat_key = f'category_{category}'
            
            if (cat_key in yearly_stats[first_year] and 
                cat_key in yearly_stats[last_year]):
                
                area1 = yearly_stats[first_year][cat_key]['area_sqkm']
                area2 = yearly_stats[last_year][cat_key]['area_sqkm']
                
                change_area = area2 - area1
                change_percent = (change_area / area1 * 100) if area1 > 0 else 0
                annual_rate = change_percent / (last_year - first_year)
                
                change_stats[total_change_key][cat_key] = {
                    'area_change_sqkm': float(change_area),
                    'percent_change': float(change_percent),
                    'annual_change_rate': float(annual_rate),
                    'initial_area': float(area1),
                    'final_area': float(area2),
                    'years_span': int(last_year - first_year)
                }
    
    return change_stats

def export_results_to_excel(yearly_stats, change_stats, output_path):
    """
    将统计结果导出到Excel文件
    
    参数:
    - yearly_stats: 年度统计数据
    - change_stats: 变化统计数据
    - output_path: 输出文件路径
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # 1. 年度面积统计表
        yearly_data = []
        for year in sorted(yearly_stats.keys()):
            row = {'年份': year}
            for category in [1, 2, 3]:
                cat_key = f'category_{category}'
                if cat_key in yearly_stats[year]:
                    stats = yearly_stats[year][cat_key]
                    category_name = {1: '人工林', 2: '自然林', 3: '其他'}[category]
                    row[f'{category_name}_面积(km²)'] = stats['area_sqkm']
                    row[f'{category_name}_占比(%)'] = stats['percentage']
            yearly_data.append(row)
        
        yearly_df = pd.DataFrame(yearly_data)
        
        # 格式化数值列
        for col in yearly_df.columns:
            if '面积(km²)' in col:
                yearly_df[col] = yearly_df[col].round(2)
            elif '占比(%)' in col:
                yearly_df[col] = yearly_df[col].round(2)
        
        yearly_df.to_excel(writer, sheet_name='年度面积统计', index=False)
        
        # 2. 年际变化统计表
        change_data = []
        for change_key, change_info in change_stats.items():
            if 'total' not in change_key:  # 排除总体变化
                years = change_key.split('_to_')
                row = {'变化期间': f"{years[0]}-{years[1]}"}
                
                for category in [1, 2, 3]:
                    cat_key = f'category_{category}'
                    if cat_key in change_info:
                        stats = change_info[cat_key]
                        category_name = {1: '人工林', 2: '自然林', 3: '其他'}[category]
                        row[f'{category_name}_变化面积(km²)'] = stats['area_change_sqkm']
                        row[f'{category_name}_变化率(%)'] = stats['percent_change']
                
                change_data.append(row)
        
        change_df = pd.DataFrame(change_data)
        change_df.to_excel(writer, sheet_name='年际变化统计', index=False)
        
        # 3. 总体变化统计表
        total_change_data = []
        for change_key, change_info in change_stats.items():
            if 'total' in change_key:
                years = change_key.replace('_total', '').split('_to_')
                row = {'变化期间': f"{years[0]}-{years[1]}"}
                
                for category in [1, 2, 3]:
                    cat_key = f'category_{category}'
                    if cat_key in change_info:
                        stats = change_info[cat_key]
                        category_name = {1: '人工林', 2: '自然林', 3: '其他'}[category]
                        row[f'{category_name}_总变化面积(km²)'] = stats['area_change_sqkm']
                        row[f'{category_name}_总变化率(%)'] = stats['percent_change']
                        row[f'{category_name}_年均变化率(%)'] = stats['annual_change_rate']
                
                total_change_data.append(row)
        
        if total_change_data:
            total_change_df = pd.DataFrame(total_change_data)
            total_change_df.to_excel(writer, sheet_name='总体变化统计', index=False)

def main():
    """
    主函数：执行人工林和自然林变化统计
    """
    # 配置参数
    BASE_DATA_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\第三次分类_样本筛选_每年逐年分类\GEE下载_逐年镶嵌_马尔科夫模型"  # 分类结果基础目录
    OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\第三次分类_样本筛选_每年逐年分类\GEE下载_逐年镶嵌_马尔科夫模型\1.统计人工林和自然林的变化情况_变化检测分类"  # 输出目录
    YEARS = list(range(2017, 2025))  # 2017-2024年
    CATEGORIES = [1, 2, 3]  # 人工林(1), 自然林(2), 其他(3)
    FILE_PATTERN = "optimized_landcover_*.tif"  # 文件匹配模式，匹配合并后的分类结果
    
    # 设置日志
    logger = setup_logging(OUTPUT_DIR)
    logger.info("开始执行人工林和自然林变化统计分析")
    logger.info(f"分析年份范围: {YEARS[0]}-{YEARS[-1]}")
    logger.info(f"统计类别: 人工林(1), 自然林(2), 其他(3)")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    try:
        # 查找分类文件
        logger.info("正在查找分类结果文件...")
        file_dict = find_classification_files(BASE_DATA_DIR, YEARS, FILE_PATTERN)
        
        if not file_dict:
            logger.error("未找到任何分类文件，请检查数据目录和文件路径")
            return
        
        logger.info(f"找到 {len(file_dict)} 个年份的分类文件")
        for year, files in file_dict.items():
            logger.info(f"  {year}年: {len(files)} 个文件")
        
        # 准备多进程处理的参数
        process_args = []
        for year, files in file_dict.items():
            for file_path in files:
                process_args.append((file_path, year, CATEGORIES))
        
        # 多进程处理文件
        logger.info(f"开始多进程处理 {len(process_args)} 个文件...")
        yearly_stats = {}
        failed_files = []
        
        # 使用进程池处理
        max_workers = min(mp.cpu_count(), len(process_args))
        logger.info(f"使用 {max_workers} 个进程进行并行处理")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(process_single_file, args): args for args in process_args}
            
            for future in as_completed(future_to_args):
                year, stats, file_path, error = future.result()
                
                if error:
                    logger.error(f"处理文件失败 {file_path}: {error}")
                    failed_files.append((file_path, error))
                else:
                    if year not in yearly_stats:
                        yearly_stats[year] = {}
                        for cat in CATEGORIES:
                            yearly_stats[year][f'category_{cat}'] = {
                                'pixel_count': 0,
                                'area_sqm': 0.0,
                                'area_sqkm': 0.0,
                                'percentage': 0.0
                            }
                    
                    # 累加统计结果
                    for cat in CATEGORIES:
                        cat_key = f'category_{cat}'
                        if cat_key in stats:
                            yearly_stats[year][cat_key]['pixel_count'] += stats[cat_key]['pixel_count']
                            yearly_stats[year][cat_key]['area_sqm'] += stats[cat_key]['area_sqm']
                            yearly_stats[year][cat_key]['area_sqkm'] += stats[cat_key]['area_sqkm']
                    
                    logger.info(f"完成处理: {os.path.basename(file_path)} ({year}年)")
        
        # 重新计算百分比（只考虑正面积）
        for year in yearly_stats:
            # 只计算正面积的总和
            positive_areas = []
            for cat in CATEGORIES:
                cat_key = f'category_{cat}'
                area = yearly_stats[year][cat_key]['area_sqkm']
                if area > 0:  # 只考虑正面积
                    positive_areas.append(area)
            
            total_positive_area = sum(positive_areas)
            
            for cat in CATEGORIES:
                cat_key = f'category_{cat}'
                area = yearly_stats[year][cat_key]['area_sqkm']
                if total_positive_area > 0 and area > 0:
                    yearly_stats[year][cat_key]['percentage'] = (area / total_positive_area * 100)
                else:
                    yearly_stats[year][cat_key]['percentage'] = 0.0
            
            # 记录负面积警告
            for cat in CATEGORIES:
                cat_key = f'category_{cat}'
                area = yearly_stats[year][cat_key]['area_sqkm']
                if area < 0:
                    logger.warning(f"{year}年 {cat_key} 面积为负值: {area:.2f} km²，可能存在数据问题")
        
        # 计算变化统计
        logger.info("计算年际变化统计...")
        change_stats = calculate_change_statistics(yearly_stats)
        
        # 生成输出文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存详细JSON结果
        json_output = os.path.join(OUTPUT_DIR, f"forest_change_statistics_{timestamp}.json")
        results = {
            'yearly_statistics': yearly_stats,
            'change_statistics': change_stats,
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'years_analyzed': YEARS,
                'categories': {1: '人工林', 2: '自然林', 3: '其他'},
                'total_files_processed': len(process_args) - len(failed_files),
                'failed_files': len(failed_files)
            }
        }
        
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存至: {json_output}")
        
        # 2. 导出Excel报告
        excel_output = os.path.join(OUTPUT_DIR, f"forest_change_report_{timestamp}.xlsx")
        export_results_to_excel(yearly_stats, change_stats, excel_output)
        logger.info(f"Excel报告已保存至: {excel_output}")
        
        # 3. 输出摘要统计
        logger.info("\n=== 统计结果摘要 ===")
        for year in sorted(yearly_stats.keys()):
            logger.info(f"\n{year}年统计结果:")
            for cat in CATEGORIES:
                cat_name = {1: '人工林', 2: '自然林', 3: '其他'}[cat]
                cat_key = f'category_{cat}'
                if cat_key in yearly_stats[year]:
                    stats = yearly_stats[year][cat_key]
                    logger.info(f"  {cat_name}: {stats['area_sqkm']:.2f} km² ({stats['percentage']:.2f}%)")
        
        # 输出总体变化
        for change_key, change_info in change_stats.items():
            if 'total' in change_key:
                years = change_key.replace('_total', '').split('_to_')
                logger.info(f"\n{years[0]}-{years[1]}年总体变化:")
                for cat in CATEGORIES:
                    cat_name = {1: '人工林', 2: '自然林', 3: '其他'}[cat]
                    cat_key = f'category_{cat}'
                    if cat_key in change_info:
                        stats = change_info[cat_key]
                        logger.info(f"  {cat_name}: {stats['area_change_sqkm']:+.2f} km² "
                                  f"({stats['percent_change']:+.2f}%, 年均{stats['annual_change_rate']:+.2f}%)")
        
        if failed_files:
            logger.warning(f"\n处理失败的文件数量: {len(failed_files)}")
            for file_path, error in failed_files[:5]:  # 只显示前5个
                logger.warning(f"  {file_path}: {error}")
        
        logger.info(f"\n统计分析完成！结果已保存至: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()