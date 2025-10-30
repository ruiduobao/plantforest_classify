#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码目的：使用矢量文件批量裁剪ESRI土地覆盖数据的tif文件
功能：
1. 读取指定文件夹中的所有tif文件
2. 使用东南亚国家边界矢量文件进行裁剪
3. 保持原栅格的空值、颜色映射等属性
4. 使用多进程加速处理
5. 输出裁剪后的栅格到指定文件夹

作者：锐多宝 (ruiduobao)
日期：2025年1月
"""

import os
import sys
import logging
import time
from datetime import datetime
# from multiprocessing import Pool, cpu_count  # 改为单进程处理，不再需要
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np

# 设置日志
def setup_logging(output_dir):
    """
    设置日志记录
    """
    log_filename = f"esri_clip_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

def clip_single_raster(args):
    """
    裁剪单个栅格文件的函数（用于多进程）
    
    参数:
        args: 包含(input_file, output_file, clip_gdf)的元组
    
    返回:
        tuple: (文件名, 是否成功, 错误信息)
    """
    input_file, output_file, clip_gdf = args
    filename = os.path.basename(input_file)
    
    try:
        print(f"开始处理: {filename}")
        
        # 读取原始栅格
        with rasterio.open(input_file) as src:
            print(f"{filename}: 读取栅格完成，坐标系: {src.crs}")
            
            # 获取栅格的坐标系
            raster_crs = src.crs
            
            # 将矢量重投影到栅格坐标系
            if clip_gdf.crs != raster_crs:
                print(f"{filename}: 重投影矢量从 {clip_gdf.crs} 到 {raster_crs}")
                clip_gdf_reproj = clip_gdf.to_crs(raster_crs)
            else:
                print(f"{filename}: 矢量坐标系匹配，无需重投影")
                clip_gdf_reproj = clip_gdf
            
            # 生成栅格的矢量范围
            print(f"{filename}: 生成栅格矢量范围")
            from shapely.geometry import box
            raster_bounds = src.bounds
            raster_geometry = box(raster_bounds.left, raster_bounds.bottom, 
                                raster_bounds.right, raster_bounds.top)
            
            # 创建栅格范围的GeoDataFrame
            raster_gdf = gpd.GeoDataFrame([1], geometry=[raster_geometry], crs=raster_crs)
            
            # 计算栅格范围与东南亚边界的交集
            print(f"{filename}: 计算交集")
            intersection_gdf = gpd.overlay(raster_gdf, clip_gdf_reproj, how='intersection')
            
            if intersection_gdf.empty:
                print(f"{filename}: 警告 - 栅格与裁剪区域无交集，跳过处理")
                return (os.path.basename(input_file), False, "栅格与裁剪区域无交集")
            
            # 合并交集几何体并转换为字典格式
            print(f"{filename}: 合并交集几何体")
            intersection_geometry = intersection_gdf.unary_union
            clip_geometry_dict = mapping(intersection_geometry)
            
            # 执行裁剪操作
            print(f"{filename}: 开始裁剪操作")
            out_image, out_transform = mask(
                src, 
                [clip_geometry_dict], 
                crop=True,  # 裁剪到几何体的边界框
                nodata=src.nodata,  # 保持原始的nodata值
                filled=False  # 不填充masked区域
            )
            print(f"{filename}: 裁剪完成")
            
            # 更新元数据
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": "uint8",  # ESRI土地覆盖数据使用uint8类型（0-255）
                "compress": "lzw",  # 使用LZW压缩
                "tiled": True,  # 使用瓦片存储
                "blockxsize": 512,
                "blockysize": 512
            })
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 写入裁剪后的栅格
            with rasterio.open(output_file, "w", **out_meta) as dest:
                # 确保数据类型为uint8
                out_image_uint8 = out_image.astype(np.uint8)
                dest.write(out_image_uint8)
                
                # 复制颜色映射表（如果存在）
                if src.colormap(1) is not None:
                    dest.write_colormap(1, src.colormap(1))
                
                # 复制其他属性
                for i in range(1, src.count + 1):
                    if src.tags(i):
                        dest.update_tags(i, **src.tags(i))
        
        return (os.path.basename(input_file), True, None)
        
    except Exception as e:
        error_msg = f"处理文件 {os.path.basename(input_file)} 时出错: {str(e)}"
        return (os.path.basename(input_file), False, error_msg)

def process_single_year(year, base_input_dir, base_output_dir, clip_shapefile, clip_gdf):
    """
    处理单个年份的数据
    
    参数:
        year: 年份 (如 2018)
        base_input_dir: 基础输入目录
        base_output_dir: 基础输出目录
        clip_shapefile: 裁剪矢量文件路径
        clip_gdf: 已读取的裁剪矢量数据
    
    返回:
        tuple: (年份, 成功文件数, 失败文件数, 总耗时)
    """
    # 构建年份相关路径
    input_dir = os.path.join(base_input_dir, f"{year}年")
    output_dir = os.path.join(base_output_dir, f"{year}年_分块裁剪")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"开始处理 {year} 年数据")
    logging.info(f"输入目录: {input_dir}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"{'='*60}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logging.error(f"{year}年输入目录不存在: {input_dir}")
        return (year, 0, 0, 0)
    
    # 获取所有tif文件
    tif_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.tif'):
            tif_files.append(file)
    
    logging.info(f"{year}年找到 {len(tif_files)} 个tif文件")
    
    if not tif_files:
        logging.warning(f"{year}年未找到任何tif文件")
        return (year, 0, 0, 0)
    
    # 准备处理参数
    process_args = []
    for tif_file in tif_files:
        input_path = os.path.join(input_dir, tif_file)
        output_path = os.path.join(output_dir, tif_file)
        process_args.append((input_path, output_path, clip_gdf))
    
    # 开始处理
    start_time = time.time()
    successful_files = []
    failed_files = []
    
    try:
        # 单进程逐个处理文件
        for i, args in enumerate(process_args, 1):
            logging.info(f"{year}年处理进度: {i}/{len(process_args)} - {args[0].split(os.sep)[-1]}")
            filename, success, error_msg = clip_single_raster(args)
            
            if success:
                successful_files.append(filename)
                logging.info(f"✓ {year}年成功处理: {filename}")
            else:
                failed_files.append(filename)
                logging.error(f"✗ {year}年处理失败: {error_msg}")
    
    except Exception as e:
        logging.error(f"{year}年处理过程中出错: {str(e)}")
        return (year, len(successful_files), len(failed_files), time.time() - start_time)
    
    # 处理完成统计
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info(f"\n{year}年处理完成统计:")
    logging.info(f"总文件数: {len(tif_files)}")
    logging.info(f"成功处理: {len(successful_files)}")
    logging.info(f"处理失败: {len(failed_files)}")
    logging.info(f"耗时: {total_time:.2f} 秒")
    logging.info(f"平均每文件耗时: {total_time/len(tif_files):.2f} 秒")
    
    if failed_files:
        logging.info(f"{year}年失败文件列表: {', '.join(failed_files)}")
    
    return (year, len(successful_files), len(failed_files), total_time)

def process_single_year(year, base_input_dir, base_output_dir, clip_shapefile, clip_gdf):
    """
    处理单个年份的ESRI数据裁剪
    
    参数:
        year: 年份 (int)
        base_input_dir: 基础输入目录路径
        base_output_dir: 基础输出目录路径  
        clip_shapefile: 裁剪矢量文件路径
        clip_gdf: 已读取的裁剪矢量GeoDataFrame
    
    返回:
        tuple: (年份, 成功数量, 失败数量, 处理时间)
    """
    year_start_time = time.time()
    
    # 构建年份相关路径
    input_dir = os.path.join(base_input_dir, f"{year}年")
    output_dir = os.path.join(base_output_dir, f"{year}年_分块裁剪")
    
    logging.info(f"📂 {year}年 输入目录: {input_dir}")
    logging.info(f"📂 {year}年 输出目录: {output_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logging.warning(f"⚠️ {year}年输入目录不存在，跳过: {input_dir}")
        return (year, 0, 0, 0)
    
    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"✓ {year}年输出目录创建成功: {output_dir}")
    except Exception as e:
        logging.error(f"✗ {year}年输出目录创建失败: {str(e)}")
        return (year, 0, 0, 0)
    
    # 获取所有tif文件
    tif_files = []
    try:
        for file in os.listdir(input_dir):
            if file.lower().endswith('.tif'):
                tif_files.append(file)
    except Exception as e:
        logging.error(f"✗ {year}年读取输入目录失败: {str(e)}")
        return (year, 0, 0, 0)
    
    logging.info(f"📄 {year}年找到 {len(tif_files)} 个tif文件")
    
    if not tif_files:
        logging.warning(f"⚠️ {year}年未找到任何tif文件")
        return (year, 0, 0, 0)
    
    # 准备处理参数
    process_args = []
    for tif_file in tif_files:
        input_path = os.path.join(input_dir, tif_file)
        output_path = os.path.join(output_dir, tif_file)
        process_args.append((input_path, output_path, clip_gdf))
    
    # 使用单进程处理避免内存问题
    logging.info(f"🔧 {year}年使用单进程进行处理（避免内存竞争问题）")
    
    # 开始处理
    successful_files = []
    failed_files = []
    
    try:
        # 单进程逐个处理文件
        for i, args in enumerate(process_args, 1):
            logging.info(f"🔄 {year}年处理进度: {i}/{len(process_args)}")
            filename, success, error_msg = clip_single_raster(args)
            
            if success:
                successful_files.append(filename)
                logging.info(f"✓ {year}年成功处理: {filename}")
            else:
                failed_files.append(filename)
                logging.error(f"✗ {year}年处理失败: {error_msg}")
    
    except Exception as e:
        logging.error(f"✗ {year}年处理过程中出错: {str(e)}")
        return (year, len(successful_files), len(failed_files), time.time() - year_start_time)
    
    # 年份处理完成统计
    year_end_time = time.time()
    year_time = year_end_time - year_start_time
    
    logging.info(f"\n📊 {year}年处理统计:")
    logging.info(f"总文件数: {len(tif_files)}")
    logging.info(f"成功处理: {len(successful_files)}")
    logging.info(f"处理失败: {len(failed_files)}")
    logging.info(f"耗时: {year_time:.2f} 秒")
    logging.info(f"平均每文件耗时: {year_time/len(tif_files):.2f} 秒")
    
    if failed_files:
        logging.warning(f"⚠️ {year}年失败文件列表: {', '.join(failed_files)}")
    
    return (year, len(successful_files), len(failed_files), year_time)


def main():
    """
    主函数：循环处理2017-2023年的ESRI数据分块裁剪
    """
    print("=" * 80)
    print("🌍 ESRI土地覆盖数据多年份批量裁剪工具")
    print("作者：锐多宝 (ruiduobao)")
    print("功能：循环处理2017-2023年的ESRI土地覆盖数据")
    print("=" * 80)
    
    # ==================== 配置区域 ====================
    # 用户可以根据实际情况修改以下路径配置
    
    # 基础输入目录（包含各年份子文件夹的根目录）
    # 目录结构应为: base_input_dir/2017年/, base_input_dir/2018年/, ...
    base_input_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\ESRI_2017_2023"
    
    # 基础输出目录（裁剪后的文件将保存在此目录下的年份子文件夹中）
    # 输出结构将为: base_output_dir/2017年_分块裁剪/, base_output_dir/2018年_分块裁剪/, ...
    base_output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\ESRI_2017_2023_分块裁剪"
    
    # 裁剪矢量文件路径（用于裁剪的shapefile文件）
    clip_shapefile = r"D:\地理所\论文\东南亚10m人工林提取\数据\东南亚国家\southeast_asia_combine.shp"
    
    # 要处理的年份范围（可以修改起始和结束年份）
    start_year = 2017  # 起始年份
    end_year = 2023    # 结束年份
    years = list(range(start_year, end_year + 1))  # 生成年份列表
    
    # ==================== 配置区域结束 ====================
    
    print(f"📂 基础输入目录: {base_input_dir}")
    print(f"📂 基础输出目录: {base_output_dir}")
    print(f"🗺️ 裁剪矢量文件: {clip_shapefile}")
    print(f"📅 处理年份范围: {start_year}-{end_year} ({len(years)}个年份)")
    print("=" * 80)
    
    # 创建基础输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(base_output_dir)
    logging.info(f"开始ESRI数据多年份批量裁剪处理")
    logging.info(f"基础输入目录: {base_input_dir}")
    logging.info(f"基础输出目录: {base_output_dir}")
    logging.info(f"裁剪矢量: {clip_shapefile}")
    logging.info(f"处理年份: {years}")
    logging.info(f"日志文件: {log_path}")
    
    # 检查基础输入目录是否存在
    if not os.path.exists(base_input_dir):
        logging.error(f"基础输入目录不存在: {base_input_dir}")
        print(f"❌ 错误：基础输入目录不存在: {base_input_dir}")
        return
    
    # 检查裁剪矢量文件是否存在
    if not os.path.exists(clip_shapefile):
        logging.error(f"裁剪矢量文件不存在: {clip_shapefile}")
        print(f"❌ 错误：裁剪矢量文件不存在: {clip_shapefile}")
        return
    
    # 读取裁剪矢量（只读取一次，供所有年份使用）
    logging.info("读取裁剪矢量文件...")
    try:
        clip_gdf = gpd.read_file(clip_shapefile)
        logging.info(f"裁剪矢量坐标系: {clip_gdf.crs}")
        logging.info(f"裁剪矢量要素数量: {len(clip_gdf)}")
        
    except Exception as e:
        logging.error(f"读取裁剪矢量文件失败: {str(e)}")
        return
    
    # 开始循环处理各个年份
    total_start_time = time.time()
    all_results = []
    
    logging.info(f"\n{'='*80}")
    logging.info(f"开始循环处理 {len(years)} 个年份的数据")
    logging.info(f"{'='*80}")
    
    for i, year in enumerate(years, 1):
        logging.info(f"\n🔄 总进度: {i}/{len(years)} - 开始处理 {year} 年")
        
        # 处理单个年份
        year_result = process_single_year(year, base_input_dir, base_output_dir, clip_shapefile, clip_gdf)
        all_results.append(year_result)
        
        # 显示当前年份处理结果
        year_num, success_count, fail_count, year_time = year_result
        logging.info(f"✅ {year} 年处理完成 - 成功: {success_count}, 失败: {fail_count}, 耗时: {year_time:.2f}秒")
    
    # 所有年份处理完成，输出总体统计
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    logging.info(f"\n{'='*80}")
    logging.info("🎉 所有年份处理完成！总体统计:")
    logging.info(f"{'='*80}")
    
    total_success = 0
    total_fail = 0
    
    for year_num, success_count, fail_count, year_time in all_results:
        total_success += success_count
        total_fail += fail_count
        logging.info(f"{year_num}年: 成功 {success_count} 个, 失败 {fail_count} 个, 耗时 {year_time:.2f} 秒")
    
    logging.info(f"\n📊 汇总统计:")
    logging.info(f"处理年份数: {len(years)}")
    logging.info(f"总成功文件数: {total_success}")
    logging.info(f"总失败文件数: {total_fail}")
    logging.info(f"总处理文件数: {total_success + total_fail}")
    logging.info(f"成功率: {(total_success/(total_success + total_fail)*100):.2f}%" if (total_success + total_fail) > 0 else "0%")
    logging.info(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    logging.info(f"平均每年耗时: {total_time/len(years):.2f} 秒")
    
    logging.info(f"\n📁 输出目录: {base_output_dir}")
    logging.info(f"📄 日志文件: {log_path}")
    logging.info("="*80)
    
    print(f"\n🎉 处理完成！")
    print(f"📊 总计处理了 {len(years)} 个年份，{total_success + total_fail} 个文件")
    print(f"✅ 成功: {total_success} 个文件")
    print(f"❌ 失败: {total_fail} 个文件")
    print(f"⏱️ 总耗时: {total_time/60:.2f} 分钟")

if __name__ == "__main__":
    main()