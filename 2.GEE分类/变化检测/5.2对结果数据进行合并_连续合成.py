#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于基准年份影像和多年变化检测结果的连续影像合并脚本
功能：从基准年份开始，逐年向前应用变化检测结果，生成连续多年的分类结果
支持批量处理GEE导出的分块变化检测结果文件
作者：锐多宝 (ruiduobao)
日期：2025年1月
"""

import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
import rasterio
from rasterio import windows
from rasterio.enums import Resampling
from tqdm import tqdm
import gc
import glob
from pathlib import Path

def scan_change_detection_files(input_dir, pattern="*.tif"):
    """
    扫描指定目录下的变化检测分块文件
    
    参数:
    - input_dir: 输入目录路径
    - pattern: 文件匹配模式，默认为"*.tif"
    
    返回:
    - 文件路径列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 搜索所有tif文件
    tif_files = list(input_path.glob(pattern))
    
    if not tif_files:
        raise FileNotFoundError(f"在目录 {input_dir} 中未找到匹配的文件: {pattern}")
    
    # 按文件名排序，确保处理顺序一致
    tif_files.sort()
    
    logging.info(f"在目录 {input_dir} 中找到 {len(tif_files)} 个变化检测文件")
    for i, file_path in enumerate(tif_files, 1):
        logging.info(f"  {i:2d}. {file_path.name}")
    
    return [str(f) for f in tif_files]

def get_raster_bounds_info(raster_path):
    """
    获取栅格文件的边界信息
    
    参数:
    - raster_path: 栅格文件路径
    
    返回:
    - 边界信息字典
    """
    with rasterio.open(raster_path) as src:
        return {
            'bounds': src.bounds,
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'res': src.res
        }

def setup_logging(output_dir):
    """
    设置日志记录
    """
    log_filename = f"merge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def check_raster_compatibility(base_path, change_path):
    """
    检查两个栅格文件的兼容性（投影、分辨率、范围等）
    """
    with rasterio.open(base_path) as base_src, rasterio.open(change_path) as change_src:
        # 检查投影系统
        if base_src.crs != change_src.crs:
            logging.warning(f"投影系统不匹配: 底图={base_src.crs}, 变化检测={change_src.crs}")
            return False
        
        # 检查分辨率
        if base_src.res != change_src.res:
            logging.warning(f"分辨率不匹配: 底图={base_src.res}, 变化检测={change_src.res}")
            return False
        
        # 检查影像范围
        base_bounds = base_src.bounds
        change_bounds = change_src.bounds
        
        logging.info(f"底图范围: {base_bounds}")
        logging.info(f"变化检测范围: {change_bounds}")
        
        # 计算重叠区域
        overlap_left = max(base_bounds.left, change_bounds.left)
        overlap_bottom = max(base_bounds.bottom, change_bounds.bottom)
        overlap_right = min(base_bounds.right, change_bounds.right)
        overlap_top = min(base_bounds.top, change_bounds.top)
        
        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            logging.error("两个影像没有重叠区域！")
            return False
        
        logging.info(f"重叠区域: left={overlap_left}, bottom={overlap_bottom}, right={overlap_right}, top={overlap_top}")
        return True

def apply_change_patch_to_base(base_path, change_path, base_data_cache=None, base_transform=None):
    """
    将单个变化检测分块应用到底图的对应区域
    
    参数:
    - base_path: 底图文件路径
    - change_path: 变化检测分块文件路径
    - base_data_cache: 底图数据缓存（可选，用于内存优化）
    - base_transform: 底图的地理变换参数
    
    返回:
    - 修改的像素统计信息
    """
    
    logging.info(f"应用变化检测分块: {os.path.basename(change_path)}")
    
    # 检查文件存在性
    if not os.path.exists(change_path):
        raise FileNotFoundError(f"变化检测文件不存在: {change_path}")
    
    start_time = time.time()
    
    with rasterio.open(base_path, 'r+') as base_src, rasterio.open(change_path) as change_src:
        
        # 检查基本兼容性
        if base_src.crs != change_src.crs:
            logging.warning(f"投影系统不匹配: 底图={base_src.crs}, 变化检测={change_src.crs}")
        
        # 计算变化检测分块在底图中的窗口位置
        change_bounds = change_src.bounds
        base_window = windows.from_bounds(
            change_bounds.left, change_bounds.bottom,
            change_bounds.right, change_bounds.top,
            base_src.transform
        )
        
        # 确保窗口为整数并在底图范围内
        base_window = windows.Window(
            max(0, int(round(base_window.col_off))),
            max(0, int(round(base_window.row_off))),
            min(base_src.width - max(0, int(round(base_window.col_off))), int(round(base_window.width))),
            min(base_src.height - max(0, int(round(base_window.row_off))), int(round(base_window.height)))
        )
        
        if base_window.width <= 0 or base_window.height <= 0:
            logging.warning(f"变化检测分块 {os.path.basename(change_path)} 与底图没有重叠区域，跳过")
            return {'applied': False, 'reason': 'no_overlap', 'changed_pixels': 0}
        
        logging.info(f"应用窗口: col={base_window.col_off}, row={base_window.row_off}, "
                    f"width={base_window.width}, height={base_window.height}")
        
        # 读取底图对应区域的数据
        base_data = base_src.read(1, window=base_window)
        
        # 读取变化检测数据，重采样到与底图窗口匹配的大小
        change_data = change_src.read(1, out_shape=(base_window.height, base_window.width))
        
        # 确保数据类型一致
        if base_data.dtype != change_data.dtype:
            change_data = change_data.astype(base_data.dtype)
        
        # 应用变化：变化检测非0值覆盖底图
        change_mask = change_data != 0
        original_data = base_data.copy()
        base_data[change_mask] = change_data[change_mask]
        
        # 将修改后的数据写回底图
        base_src.write(base_data, 1, window=base_window)
        
        # 统计修改的像素数
        changed_pixels = np.sum(change_mask)
        
        # 处理完成统计
        end_time = time.time()
        processing_time = end_time - start_time
        
        stats = {
            'applied': True,
            'file_name': os.path.basename(change_path),
            'processing_time': processing_time,
            'changed_pixels': changed_pixels,
            'total_patch_pixels': base_data.size,
            'change_percentage': (changed_pixels / base_data.size) * 100 if base_data.size > 0 else 0
        }
        
        logging.info(f"完成应用 {os.path.basename(change_path)}: "
                    f"修改像素 {changed_pixels:,}/{base_data.size:,} "
                    f"({stats['change_percentage']:.2f}%), 耗时 {processing_time:.2f}秒")
        
        return stats

def batch_apply_changes_to_base_image_sequential(base_image_path, change_detection_base_dir, 
                                              output_dir, base_year, start_year, end_year,
                                              file_pattern_template="*.tif"):
    """
    连续多年批量将变化检测分块应用到底图上，生成多年连续分类结果
    
    参数:
    - base_image_path: 基准年份底图文件路径
    - change_detection_base_dir: 变化检测分块文件根目录（包含各年份子目录）
    - output_dir: 输出目录
    - base_year: 基准年份（如2024）
    - start_year: 开始处理年份（如2023）
    - end_year: 结束年份（如2017）
    - file_pattern_template: 文件匹配模式模板，匹配每个年份文件夹下的文件
    
    返回:
    - 所有年份的处理统计信息
    """
    
    # 设置日志
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info("开始连续多年变化检测分块批量合并")
    logging.info(f"基准底图: {base_image_path} ({base_year}年)")
    logging.info(f"变化检测根目录: {change_detection_base_dir}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"处理年份范围: {start_year} → {end_year}")
    logging.info(f"文件匹配模式: {file_pattern_template}")
    logging.info("="*80)
    
    # 检查基准底图文件
    if not os.path.exists(base_image_path):
        raise FileNotFoundError(f"基准底图文件不存在: {base_image_path}")
    
    # 生成年份列表（从start_year到end_year，递减）
    years_to_process = list(range(start_year, end_year - 1, -1))
    logging.info(f"将按顺序处理年份: {years_to_process}")
    
    # 总体统计信息
    all_years_stats = {
        'base_year': base_year,
        'processed_years': [],
        'failed_years': [],
        'total_processing_time': 0,
        'yearly_results': {}
    }
    
    # 当前工作底图路径（初始为基准底图）
    current_base_path = base_image_path
    
    start_time = time.time()
    
    try:
        for year in years_to_process:
            logging.info(f"\n{'='*60}")
            logging.info(f"开始处理 {year} 年变化检测")
            logging.info(f"{'='*60}")
            
            # 构建当前年份的变化检测目录和文件模式
            year_change_dir = os.path.join(change_detection_base_dir, str(year))
            year_file_pattern = file_pattern_template  # 直接使用模板，不需要格式化年份
            
            # 检查年份目录是否存在
            if not os.path.exists(year_change_dir):
                logging.warning(f"年份目录不存在，跳过: {year_change_dir}")
                all_years_stats['failed_years'].append({
                    'year': year,
                    'reason': 'directory_not_found',
                    'path': year_change_dir
                })
                continue
            
            # 生成当前年份的输出文件路径
            output_filename = f"merged_zone5_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
            year_output_path = os.path.join(output_dir, output_filename)
            
            try:
                # 应用当前年份的变化检测
                year_result = batch_apply_changes_to_base_image(
                    base_image_path=current_base_path,
                    change_detection_dir=year_change_dir,
                    output_path=year_output_path,
                    file_pattern=year_file_pattern
                )
                
                # 记录成功处理的年份
                all_years_stats['processed_years'].append(year)
                all_years_stats['yearly_results'][year] = {
                    'result': year_result,
                    'output_file': year_output_path,
                    'input_base': current_base_path
                }
                
                # 更新当前工作底图为刚生成的结果
                current_base_path = year_output_path
                
                logging.info(f"✅ {year}年处理完成")
                logging.info(f"   输出文件: {year_output_path}")
                logging.info(f"   成功处理: {year_result['processed_files']} 个文件")
                logging.info(f"   变化比例: {year_result['overall_change_percentage']:.2f}%")
                
            except Exception as e:
                logging.error(f"❌ {year}年处理失败: {str(e)}")
                all_years_stats['failed_years'].append({
                    'year': year,
                    'reason': 'processing_error',
                    'error': str(e)
                })
                # 处理失败时不更新current_base_path，继续使用上一个成功的结果
                continue
        
        # 计算总体处理时间
        total_time = time.time() - start_time
        all_years_stats['total_processing_time'] = total_time
        
        # 输出最终统计
        logging.info("\n" + "="*80)
        logging.info("连续多年变化检测合并完成统计:")
        logging.info(f"基准年份: {base_year}")
        logging.info(f"成功处理年份: {all_years_stats['processed_years']}")
        logging.info(f"失败年份数: {len(all_years_stats['failed_years'])}")
        logging.info(f"总处理时间: {total_time:.2f}秒")
        
        if all_years_stats['failed_years']:
            logging.info("\n失败年份详情:")
            for failed in all_years_stats['failed_years']:
                logging.info(f"  - {failed['year']}年: {failed['reason']}")
        
        logging.info("="*80)
        
        # 保存总体统计结果
        stats_file = os.path.join(output_dir, f"sequential_merge_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            import json
            
            def convert_numpy_types(obj):
                """递归转换numpy数据类型和rasterio对象为Python原生类型"""
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'to_string'):  # rasterio CRS对象
                    return obj.to_string()
                elif hasattr(obj, '__dict__'):  # 其他复杂对象转为字符串
                    return str(obj)
                else:
                    return obj
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                serializable_stats = convert_numpy_types(all_years_stats.copy())
                json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
            logging.info(f"总体统计结果已保存到: {stats_file}")
        except Exception as e:
            logging.warning(f"保存总体统计文件失败: {str(e)}")
        
        return all_years_stats
        
    except Exception as e:
        logging.error(f"连续处理过程中发生严重错误: {str(e)}")
        raise e

def batch_apply_changes_to_base_image(base_image_path, change_detection_dir, output_path, 
                                    file_pattern="*.tif"):
    """
    批量将变化检测分块应用到底图上，输出一张完整的修改后底图
    
    参数:
    - base_image_path: 2024年底图文件路径
    - change_detection_dir: 变化检测分块文件所在目录
    - output_path: 输出修改后的完整底图路径
    - file_pattern: 文件匹配模式，默认为"*.tif"
    
    返回:
    - 批量处理统计信息
    """
    
    # 设置日志
    output_dir = os.path.dirname(output_path)
    logger = setup_logging(output_dir)
    
    logging.info("="*80)
    logging.info("开始批量应用变化检测分块到底图")
    logging.info(f"底图路径: {base_image_path}")
    logging.info(f"变化检测目录: {change_detection_dir}")
    logging.info(f"输出路径: {output_path}")
    logging.info(f"文件匹配模式: {file_pattern}")
    logging.info("="*80)
    
    # 检查底图文件
    if not os.path.exists(base_image_path):
        raise FileNotFoundError(f"底图文件不存在: {base_image_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 扫描变化检测文件
    change_files = scan_change_detection_files(change_detection_dir, file_pattern)
    
    if not change_files:
        logging.warning(f"在目录 {change_detection_dir} 中未找到匹配 {file_pattern} 的文件")
        return {'success': False, 'message': 'No files found'}
    
    logging.info(f"找到 {len(change_files)} 个变化检测分块文件")
    
    # 复制底图作为工作副本
    import shutil
    temp_base_path = output_path.replace('.tif', '_temp.tif')
    
    try:
        # 复制底图到临时文件
        logging.info("复制底图到临时工作文件...")
        shutil.copy2(base_image_path, temp_base_path)
        
        # 批量处理统计
        batch_stats = {
            'total_files': len(change_files),
            'processed_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_processing_time': 0,
            'total_changed_pixels': 0,
            'file_results': [],
            'failed_files_list': [],
            'skipped_files_list': []
        }
        
        start_time = time.time()
        
        # 逐个应用变化检测分块
        for i, change_file in enumerate(change_files, 1):
            try:
                logging.info(f"\n处理进度: {i}/{len(change_files)} - {os.path.basename(change_file)}")
                
                # 应用变化检测分块到底图
                result = apply_change_patch_to_base(temp_base_path, change_file)
                
                if result['applied']:
                    batch_stats['processed_files'] += 1
                    batch_stats['total_processing_time'] += result['processing_time']
                    batch_stats['total_changed_pixels'] += result['changed_pixels']
                    batch_stats['file_results'].append(result)
                    
                    logging.info(f"✓ 成功应用: {result['file_name']}, "
                               f"修改像素: {result['changed_pixels']:,}")
                else:
                    batch_stats['skipped_files'] += 1
                    batch_stats['skipped_files_list'].append({
                        'file': change_file,
                        'reason': result.get('reason', 'unknown')
                    })
                    logging.warning(f"跳过文件: {os.path.basename(change_file)} - {result.get('reason', 'unknown')}")
                
            except Exception as e:
                batch_stats['failed_files'] += 1
                batch_stats['failed_files_list'].append({
                    'file': change_file,
                    'error': str(e)
                })
                logging.error(f"处理文件失败 {os.path.basename(change_file)}: {str(e)}")
                continue
        
        # 将临时文件重命名为最终输出文件
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_base_path, output_path)
        
        # 计算总体统计
        total_time = time.time() - start_time
        batch_stats['total_batch_time'] = total_time
        
        # 计算整体变化比例
        with rasterio.open(output_path) as src:
            total_pixels = src.width * src.height
        batch_stats['overall_change_percentage'] = (batch_stats['total_changed_pixels'] / total_pixels) * 100 if total_pixels > 0 else 0
        
        # 获取输出文件信息
        output_info = get_raster_bounds_info(output_path)
        batch_stats['output_file_info'] = output_info
        
        # 输出最终统计
        logging.info("\n" + "="*80)
        logging.info("批量应用变化检测完成统计:")
        logging.info(f"总文件数: {batch_stats['total_files']}")
        logging.info(f"成功处理: {batch_stats['processed_files']}")
        logging.info(f"跳过文件: {batch_stats['skipped_files']}")
        logging.info(f"失败文件: {batch_stats['failed_files']}")
        logging.info(f"总处理时间: {batch_stats['total_batch_time']:.2f}秒")
        logging.info(f"平均处理时间: {batch_stats['total_processing_time']/max(1, batch_stats['processed_files']):.2f}秒/文件")
        logging.info(f"总修改像素: {batch_stats['total_changed_pixels']:,}")
        logging.info(f"整体变化比例: {batch_stats['overall_change_percentage']:.2f}%")
        logging.info(f"输出文件: {output_path}")
        
        if batch_stats['failed_files_list']:
            logging.info("\n失败文件列表:")
            for failed in batch_stats['failed_files_list']:
                logging.info(f"  - {os.path.basename(failed['file'])}: {failed['error']}")
        
        if batch_stats['skipped_files_list']:
            logging.info("\n跳过文件列表:")
            for skipped in batch_stats['skipped_files_list']:
                logging.info(f"  - {os.path.basename(skipped['file'])}: {skipped['reason']}")
        
        logging.info("="*80)
        
        # 保存统计结果到JSON文件
        stats_file = os.path.join(output_dir, f"batch_apply_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            import json
            
            def convert_numpy_types(obj):
                """递归转换numpy数据类型和rasterio对象为Python原生类型"""
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'to_string'):  # rasterio CRS对象
                    return obj.to_string()
                elif hasattr(obj, '__dict__'):  # 其他复杂对象转为字符串
                    return str(obj)
                else:
                    return obj
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                # 转换不可序列化的对象
                serializable_stats = convert_numpy_types(batch_stats.copy())
                for result in serializable_stats['file_results']:
                    if 'processing_time' in result:
                        result['processing_time'] = round(result['processing_time'], 2)
                
                json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
            logging.info(f"统计结果已保存到: {stats_file}")
        except Exception as e:
            logging.warning(f"保存统计文件失败: {str(e)}")
        
        return batch_stats
        
    except Exception as e:
        # 清理临时文件
        if os.path.exists(temp_base_path):
            try:
                os.remove(temp_base_path)
            except:
                pass
        
        logging.error(f"批量处理过程中发生错误: {str(e)}")
        raise e

# 主程序入口
if __name__ == "__main__":
    # ==================== 连续多年处理配置参数 ====================
    # 基准年份底图路径（2024年分类结果）
    BASE_IMAGE_PATH = r"D:\地理所\论文\东南亚10m人工林提取\数据\第二次分类\merged_zone5_2019_20251024_160615.tif"
    
    # 变化检测文件根目录（包含各年份子目录）
    CHANGE_DETECTION_BASE_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\GEE分类\第二次变化检测方法分类\原始GEE导出影像"
    
    # 输出目录
    OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\第二次分类"
    
    # 年份控制参数
    BASE_YEAR = 2019       # 基准年份
    START_YEAR = 2018       # 开始处理年份
    END_YEAR = 2017         # 结束年份
    
    # 文件匹配模式模板（匹配该年份文件夹下的所有tif文件）
    FILE_PATTERN_TEMPLATE = "*.tif"
    
    print("="*80)
    print("GEE变化检测连续多年批量合并工具")
    print("功能：基于基准年份影像，连续应用多年变化检测结果")
    print("="*80)
    print(f"基准底图: {BASE_IMAGE_PATH} ({BASE_YEAR}年)")
    print(f"变化检测根目录: {CHANGE_DETECTION_BASE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"处理年份范围: {START_YEAR} → {END_YEAR}")
    print(f"文件匹配模式: {FILE_PATTERN_TEMPLATE}")
    print("="*80)
    
    try:
        # 执行连续多年处理
        results = batch_apply_changes_to_base_image_sequential(
            base_image_path=BASE_IMAGE_PATH,
            change_detection_base_dir=CHANGE_DETECTION_BASE_DIR,
            output_dir=OUTPUT_DIR,
            base_year=BASE_YEAR,
            start_year=START_YEAR,
            end_year=END_YEAR,
            file_pattern_template=FILE_PATTERN_TEMPLATE
        )
        
        print("\n" + "="*80)
        print("连续多年处理完成摘要:")
        print(f"基准年份: {results['base_year']}")
        print(f"成功处理年份: {results['processed_years']}")
        print(f"失败年份数: {len(results['failed_years'])}")
        print(f"总耗时: {results['total_processing_time']:.2f}秒")
        print("="*80)
        
        if results['processed_years']:
            print(f"\n✅ 成功处理 {len(results['processed_years'])} 个年份")
            print("📁 生成的文件:")
            for year in results['processed_years']:
                output_file = results['yearly_results'][year]['output_file']
                change_pct = results['yearly_results'][year]['result']['overall_change_percentage']
                print(f"   {year}年: {os.path.basename(output_file)} (变化比例: {change_pct:.2f}%)")
            print(f"📁 输出目录: {OUTPUT_DIR}")
        else:
            print("\n❌ 没有成功处理任何年份，请检查输入参数和文件路径")
            
        if results['failed_years']:
            print(f"\n⚠️  失败年份:")
            for failed in results['failed_years']:
                print(f"   {failed['year']}年: {failed['reason']}")
        
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()