#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于2024年底图和2023年变化检测结果的影像合并脚本
功能：将2023年变化检测结果（非0值）覆盖到2024年底图对应位置，生成2023年分类结果
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
    """
    主程序入口 - 批量合并变化检测分块文件示例
    """
    
    # 配置参数
    BASE_IMAGE_PATH = r"D:\地理所\论文\东南亚10m人工林提取\数据\ZONE5_2024年单独分类\Zone5_2024_带颜色映射表_单独分类.tif"
    CHANGE_DETECTION_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\GEE分类\分类结果\多年分类结果\ZONE5_2023年_变化概率\分块"
    OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\ZONE5_2023年_变化概率"
    
    # 文件匹配模式（根据用户提供的文件名格式）
    FILE_PATTERN = "zone5_change_classification_2023_threshold420*.tif"
    
    # 分块大小（像素）
    BLOCK_SIZE = 1024
    
    print("="*80)
    print("GEE变化检测分块文件批量合并工具")
    print("功能：将2023年变化检测结果（非0值）覆盖到2024年底图上")
    print("="*80)
    print(f"底图文件: {BASE_IMAGE_PATH}")
    print(f"变化检测目录: {CHANGE_DETECTION_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"文件匹配模式: {FILE_PATTERN}")
    print(f"分块大小: {BLOCK_SIZE}x{BLOCK_SIZE} 像素")
    print("="*80)
    
    try:
        # 生成输出文件路径
        output_filename = f"merged_zone5_2023_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 执行批量应用变化检测
        results = batch_apply_changes_to_base_image(
            base_image_path=BASE_IMAGE_PATH,
            change_detection_dir=CHANGE_DETECTION_DIR,
            output_path=output_path,
            file_pattern=FILE_PATTERN
        )
        
        # 输出最终结果摘要
        print("\n" + "="*80)
        print("处理完成摘要:")
        print(f"总文件数: {results['total_files']}")
        print(f"成功处理: {results['processed_files']}")
        print(f"跳过文件: {results['skipped_files']}")
        print(f"失败文件: {results['failed_files']}")
        print(f"总耗时: {results['total_batch_time']:.2f}秒")
        print(f"整体变化比例: {results['overall_change_percentage']:.2f}%")
        print("="*80)
        
        if results['processed_files'] > 0:
            print(f"\n✅ 成功处理 {results['processed_files']} 个文件")
            print(f"📁 输出目录: {OUTPUT_DIR}")
        else:
            print("\n❌ 没有成功处理任何文件，请检查输入参数和文件路径")
            
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()