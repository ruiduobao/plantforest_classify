#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
脚本目的：根据分区网格数据，对三种类型的样本点数据进行分zone裁剪
功能：
1. 读取分区网格数据（面要素）
2. 对每个zone，裁剪三种点要素数据（人工林、非林地、自然林）
3. 输出每个zone的三种点要素矢量文件到指定文件夹
4. 使用多进程提高处理速度
5. 记录处理日志
6. 详细打印每个zone的样本类型个数

作者：锐多宝 (ruiduobao)
创建时间：2025年
"""

import os
import sys
import time
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

def setup_logging(output_dir):
    """
    设置日志记录功能
    
    Args:
        output_dir (str): 输出目录路径
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"vector_clipping_log_{timestamp}.txt")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 输出到文件
            logging.StreamHandler(sys.stdout)  # 输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_file}")
    return logger

def load_vector_data(file_path, logger):
    """
    加载矢量数据
    
    Args:
        file_path (str): 矢量文件路径
        logger: 日志记录器
    
    Returns:
        geopandas.GeoDataFrame: 加载的矢量数据
    """
    try:
        logger.info(f"正在加载矢量数据: {file_path}")
        
        # 根据文件扩展名选择读取方式
        if file_path.endswith('.shp'):
            gdf = gpd.read_file(file_path, encoding='utf-8')
        elif file_path.endswith('.gpkg'):
            gdf = gpd.read_file(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        logger.info(f"成功加载数据，共 {len(gdf)} 个要素")
        logger.info(f"数据坐标系: {gdf.crs}")
        
        return gdf
    
    except Exception as e:
        logger.error(f"加载矢量数据失败: {file_path}, 错误: {str(e)}")
        return None

def clip_points_by_zone(args):
    """
    对单个zone进行点要素裁剪的函数（用于多进程处理）
    
    Args:
        args (tuple): 包含处理参数的元组
            - zone_row: zone的几何和属性信息
            - point_datasets: 三种点数据的字典
            - output_base_dir: 输出基础目录
            - zone_id: zone的ID
    
    Returns:
        dict: 处理结果统计
    """
    zone_row, point_datasets, output_base_dir, zone_id = args
    
    # 获取zone几何
    zone_geom = zone_row['geometry']
    zone_name = str(zone_row.get('zone', f'zone_{zone_id}'))
    
    # 创建zone专用的输出目录
    zone_output_dir = os.path.join(output_base_dir, f"zone_{zone_name}")
    os.makedirs(zone_output_dir, exist_ok=True)
    
    results = {
        'zone_id': zone_id,
        'zone_name': zone_name,
        'processed_files': [],
        'point_counts': {},
        'errors': []
    }
    
    # 打印zone开始处理信息
    print(f"\n{'='*60}")
    print(f"开始处理 Zone {zone_name}")
    print(f"{'='*60}")
    
    try:
        # 首先输出zone区域要素本身
        zone_gdf = gpd.GeoDataFrame([zone_row], crs=zone_row.geometry.crs if hasattr(zone_row.geometry, 'crs') else 'EPSG:4326')
        
        # 根据zone编号决定文件命名格式
        zone_num = int(zone_name)
        if zone_num >= 6:
            # zone 6-10: 全大写，字母前后顺序调换 (BOUNDARY_ZONE_X)
            zone_boundary_filename = f"BOUNDARY_ZONE_{zone_name}.shp"
        else:
            # zone 1-5: 保持原格式 (zone_X_boundary)
            zone_boundary_filename = f"zone_{zone_name}_boundary.shp"
            
        zone_output_path = os.path.join(zone_output_dir, zone_boundary_filename)
        zone_gdf.to_file(zone_output_path, encoding='utf-8')
        results['processed_files'].append(zone_output_path)
        print(f"  Zone边界要素: 已保存到 {zone_boundary_filename}")
        
        # 对每种点数据进行裁剪
        for data_type, (points_gdf, output_prefix) in point_datasets.items():
            
            # 执行空间裁剪
            clipped_points = gpd.clip(points_gdf, zone_geom)
            
            # 记录裁剪后的点数量
            point_count = len(clipped_points)
            results['point_counts'][data_type] = point_count
            
            # 详细打印每种类型的点数量
            data_type_names = {
                'plantation': '人工林样本点',
                'nonforest': '非林地样本点', 
                'natural_forest': '自然林样本点'
            }
            
            print(f"  {data_type_names.get(data_type, data_type)}: {point_count:,} 个点")
            
            if point_count > 0:
                # 根据zone编号决定文件命名格式
                if zone_num >= 6:
                    # zone 6-10: 全大写，字母前后顺序调换
                    if output_prefix == 'plantation_points':
                        output_filename = f"POINTS_PLANTATION_ZONE_{zone_name}.shp"
                    elif output_prefix == 'nonforest_points':
                        output_filename = f"POINTS_NONFOREST_ZONE_{zone_name}.shp"
                    elif output_prefix == 'natural_forest_points':
                        output_filename = f"POINTS_NATURAL_FOREST_ZONE_{zone_name}.shp"
                else:
                    # zone 1-5: 保持原格式
                    output_filename = f"{output_prefix}_zone_{zone_name}.shp"
                
                output_path = os.path.join(zone_output_dir, output_filename)
                
                # 保存裁剪后的数据
                clipped_points.to_file(output_path, encoding='utf-8')
                results['processed_files'].append(output_path)
                
                print(f"    → 已保存到: {output_filename}")
            else:
                print(f"    → 该zone内没有找到相交的点要素")
        
        # 计算zone总点数
        total_points = sum(results['point_counts'].values())
        print(f"\nZone {zone_name} 总计: {total_points:,} 个样本点")
        print(f"{'='*60}")
    
    except Exception as e:
        error_msg = f"处理Zone {zone_name}时发生错误: {str(e)}"
        results['errors'].append(error_msg)
        print(f"❌ {error_msg}")
    
    return results

def main():
    """
    主函数：执行矢量裁剪任务
    """
    # 定义输入文件路径
    grid_file = r"K:\地理所\论文\东南亚10m人工林提取\数据\分类网格数据\分类的分区网格设置_上传.shp"
    
    # 三种点数据文件路径
    plantation_file = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\1.人工林样本点\2.2.根据产品数据和土地覆盖数据筛选_2017-2024年土地覆盖数据筛选\sdpt_filtered_points_20251011_220536_5倍数量_Sample_filtered.shp"
    nonforest_file = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\3.非林地样本点\3.根据网格分配1000个样本_多年土地覆盖过滤\points_OTHERS_LandcoverFilter_每个网格1000个分配.shp"
    natural_forest_file = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\2.自然林样本点\5.按照网格筛选1000个点_再经过8年的土地覆盖数据筛选\natural_forest_points_sampled_AfterCDD_AfterLandcoverFilter.shp"
    
    # 输出目录
    output_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志记录
    logger = setup_logging(output_dir)
    
    logger.info("="*60)
    logger.info("开始执行矢量裁剪任务")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # 1. 加载分区网格数据
        logger.info("步骤1: 加载分区网格数据")
        grid_gdf = load_vector_data(grid_file, logger)
        if grid_gdf is None:
            logger.error("无法加载分区网格数据，程序退出")
            return
        
        logger.info(f"分区网格数据包含 {len(grid_gdf)} 个zone")
        
        # 2. 加载三种点数据
        logger.info("步骤2: 加载三种点数据")
        
        # 加载人工林样本点
        plantation_gdf = load_vector_data(plantation_file, logger)
        if plantation_gdf is None:
            logger.error("无法加载人工林样本点数据")
            return
        
        # 加载非林地样本点
        nonforest_gdf = load_vector_data(nonforest_file, logger)
        if nonforest_gdf is None:
            logger.error("无法加载非林地样本点数据")
            return
        
        # 加载自然林样本点
        natural_forest_gdf = load_vector_data(natural_forest_file, logger)
        if natural_forest_gdf is None:
            logger.error("无法加载自然林样本点数据")
            return
        
        # 3. 确保所有数据使用相同的坐标系
        logger.info("步骤3: 检查和统一坐标系")
        target_crs = grid_gdf.crs
        
        if plantation_gdf.crs != target_crs:
            logger.info(f"转换人工林数据坐标系从 {plantation_gdf.crs} 到 {target_crs}")
            plantation_gdf = plantation_gdf.to_crs(target_crs)
        
        if nonforest_gdf.crs != target_crs:
            logger.info(f"转换非林地数据坐标系从 {nonforest_gdf.crs} 到 {target_crs}")
            nonforest_gdf = nonforest_gdf.to_crs(target_crs)
        
        if natural_forest_gdf.crs != target_crs:
            logger.info(f"转换自然林数据坐标系从 {natural_forest_gdf.crs} 到 {target_crs}")
            natural_forest_gdf = natural_forest_gdf.to_crs(target_crs)
        
        # 4. 准备多进程处理的数据
        logger.info("步骤4: 准备多进程处理")
        
        # 构建点数据字典
        point_datasets = {
            'plantation': (plantation_gdf, 'plantation_points'),
            'nonforest': (nonforest_gdf, 'nonforest_points'),
            'natural_forest': (natural_forest_gdf, 'natural_forest_points')
        }
        
        # 准备多进程参数
        process_args = []
        for idx, (_, zone_row) in enumerate(grid_gdf.iterrows()):
            args = (zone_row, point_datasets, output_dir, idx)
            process_args.append(args)
        
        # 5. 执行多进程裁剪
        logger.info("步骤5: 开始多进程裁剪处理")
        
        # 确定进程数量（使用CPU核心数，但不超过zone数量）
        num_processes = min(cpu_count(), len(grid_gdf))
        logger.info(f"使用 {num_processes} 个进程进行并行处理")
        
        print(f"\n🚀 开始处理 {len(grid_gdf)} 个zone的样本点裁剪...")
        print(f"📊 原始数据统计:")
        print(f"   - 人工林样本点: {len(plantation_gdf):,} 个")
        print(f"   - 非林地样本点: {len(nonforest_gdf):,} 个") 
        print(f"   - 自然林样本点: {len(natural_forest_gdf):,} 个")
        print(f"   - 总计: {len(plantation_gdf) + len(nonforest_gdf) + len(natural_forest_gdf):,} 个样本点")
        
        # 创建进程池并执行任务
        with Pool(processes=num_processes) as pool:
            results = pool.map(clip_points_by_zone, process_args)
        
        # 6. 统计处理结果
        logger.info("步骤6: 统计处理结果")
        
        total_processed = 0
        total_errors = 0
        summary_stats = {
            'plantation': {'total_points': 0, 'zones_with_data': 0},
            'nonforest': {'total_points': 0, 'zones_with_data': 0},
            'natural_forest': {'total_points': 0, 'zones_with_data': 0}
        }
        
        # 创建详细统计表格
        zone_details = []
        
        for result in results:
            if result['errors']:
                total_errors += len(result['errors'])
                for error in result['errors']:
                    logger.error(error)
            else:
                total_processed += 1
                
                # 收集zone详细信息
                zone_detail = {
                    'zone_name': result['zone_name'],
                    'plantation': result['point_counts'].get('plantation', 0),
                    'nonforest': result['point_counts'].get('nonforest', 0),
                    'natural_forest': result['point_counts'].get('natural_forest', 0)
                }
                zone_detail['total'] = zone_detail['plantation'] + zone_detail['nonforest'] + zone_detail['natural_forest']
                zone_details.append(zone_detail)
                
                # 统计各类型点数据
                for data_type, count in result['point_counts'].items():
                    summary_stats[data_type]['total_points'] += count
                    if count > 0:
                        summary_stats[data_type]['zones_with_data'] += 1
        
        # 7. 输出详细的zone统计表格
        print(f"\n📋 各Zone样本点详细统计:")
        print(f"{'='*80}")
        print(f"{'Zone':<8} {'人工林':<12} {'非林地':<12} {'自然林':<12} {'总计':<12}")
        print(f"{'-'*80}")
        
        for detail in sorted(zone_details, key=lambda x: int(x['zone_name'])):
            print(f"{detail['zone_name']:<8} {detail['plantation']:<12,} {detail['nonforest']:<12,} {detail['natural_forest']:<12,} {detail['total']:<12,}")
        
        print(f"{'-'*80}")
        
        # 计算总计
        total_plantation = sum(d['plantation'] for d in zone_details)
        total_nonforest = sum(d['nonforest'] for d in zone_details)
        total_natural = sum(d['natural_forest'] for d in zone_details)
        grand_total = total_plantation + total_nonforest + total_natural
        
        print(f"{'总计':<8} {total_plantation:<12,} {total_nonforest:<12,} {total_natural:<12,} {grand_total:<12,}")
        print(f"{'='*80}")
        
        # 8. 输出最终统计结果
        logger.info("="*60)
        logger.info("处理完成统计")
        logger.info("="*60)
        logger.info(f"总共处理的zone数量: {total_processed}")
        logger.info(f"处理失败的zone数量: {total_errors}")
        
        for data_type, stats in summary_stats.items():
            data_type_names = {
                'plantation': '人工林',
                'nonforest': '非林地', 
                'natural_forest': '自然林'
            }
            logger.info(f"{data_type_names.get(data_type, data_type)}:")
            logger.info(f"  - 总点数: {stats['total_points']:,}")
            logger.info(f"  - 有数据的zone数: {stats['zones_with_data']}")
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"总处理时间: {total_time:.2f} 秒")
        
        # 保存统计结果到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(output_dir, f"clipping_statistics_{timestamp}.txt")
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("矢量裁剪处理统计结果\n")
            f.write("="*50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总处理时间: {total_time:.2f} 秒\n")
            f.write(f"成功处理的zone数量: {total_processed}\n")
            f.write(f"处理失败的zone数量: {total_errors}\n\n")
            
            # 写入详细的zone统计
            f.write("各Zone样本点详细统计:\n")
            f.write("="*80 + "\n")
            f.write(f"{'Zone':<8} {'人工林':<12} {'非林地':<12} {'自然林':<12} {'总计':<12}\n")
            f.write("-"*80 + "\n")
            
            for detail in sorted(zone_details, key=lambda x: int(x['zone_name'])):
                f.write(f"{detail['zone_name']:<8} {detail['plantation']:<12,} {detail['nonforest']:<12,} {detail['natural_forest']:<12,} {detail['total']:<12,}\n")
            
            f.write("-"*80 + "\n")
            f.write(f"{'总计':<8} {total_plantation:<12,} {total_nonforest:<12,} {total_natural:<12,} {grand_total:<12,}\n")
            f.write("="*80 + "\n\n")
            
            for data_type, stats in summary_stats.items():
                data_type_names = {
                    'plantation': '人工林',
                    'nonforest': '非林地', 
                    'natural_forest': '自然林'
                }
                f.write(f"{data_type_names.get(data_type, data_type)}统计:\n")
                f.write(f"  总点数: {stats['total_points']:,}\n")
                f.write(f"  有数据的zone数: {stats['zones_with_data']}\n\n")
        
        logger.info(f"统计结果已保存到: {stats_file}")
        logger.info("矢量裁剪任务全部完成！")
        
        print(f"\n✅ 处理完成！")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 统计文件: {stats_file}")
        
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()