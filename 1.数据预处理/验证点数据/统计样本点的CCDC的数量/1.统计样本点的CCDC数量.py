"""
代码目的：统计300万个样本点在两个CCDC栅格上的值，并添加到矢量属性中
使用分块处理和多进程技术来提高处理速度，防止内存溢出
作者：锐多宝 (ruiduobao)
"""

import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import os
import logging
from datetime import datetime
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置日志
def setup_logging(output_dir):
    """设置日志记录"""
    log_filename = f"ccdc_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_path

def extract_raster_values_chunk(args):
    """
    为点矢量块提取栅格值的函数（用于多进程）
    
    参数:
    args: 包含(points_chunk, raster1_path, raster2_path, chunk_id)的元组
    
    返回:
    包含栅格值的DataFrame
    """
    points_chunk, raster1_path, raster2_path, chunk_id = args
    
    try:
        # 打开栅格文件
        with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
            # 获取点坐标
            coords = [(point.x, point.y) for point in points_chunk.geometry]
            
            # 提取栅格值
            raster1_values = list(src1.sample(coords))
            raster2_values = list(src2.sample(coords))
            
            # 转换为一维数组
            raster1_values = [val[0] if len(val) > 0 else np.nan for val in raster1_values]
            raster2_values = [val[0] if len(val) > 0 else np.nan for val in raster2_values]
            
            # 创建结果DataFrame
            result_df = points_chunk.copy()
            result_df['nBreaks_value'] = raster1_values  # nBreaks_mosaic.tif的值
            result_df['raster_mosaic_value'] = raster2_values  # raster_mosaic_uint16.tif的值
            
            logging.info(f"块 {chunk_id} 处理完成，包含 {len(result_df)} 个点")
            return result_df
            
    except Exception as e:
        logging.error(f"处理块 {chunk_id} 时出错: {str(e)}")
        return None

def process_points_in_batches(points_gdf, raster1_path, raster2_path, output_dir, batch_size=1000000, chunk_size=50000, n_processes=None):
    """
    分批处理点矢量数据，每批保存一份结果，最后拼接所有结果
    
    参数:
    points_gdf: 点矢量GeoDataFrame
    raster1_path: 第一个栅格文件路径
    raster2_path: 第二个栅格文件路径
    output_dir: 输出目录
    batch_size: 每批处理的点数量（默认100万）
    chunk_size: 每块的点数量（用于多进程）
    n_processes: 进程数量，默认为CPU核心数的一半
    
    返回:
    最终拼接的完整GeoDataFrame文件路径
    """
    
    if n_processes is None:
        n_processes = max(1, cpu_count() // 2)  # 使用CPU核心数的一半，减少内存压力
    
    total_points = len(points_gdf)
    total_batches = (total_points + batch_size - 1) // batch_size
    
    logging.info(f"开始分批处理 {total_points} 个点")
    logging.info(f"分为 {total_batches} 批，每批 {batch_size} 个点")
    logging.info(f"使用 {n_processes} 个进程，每块 {chunk_size} 个点")
    
    batch_files = []  # 存储每批结果文件路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 分批处理
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_points)
        
        logging.info(f"="*30 + f" 处理第 {batch_idx + 1}/{total_batches} 批 " + "="*30)
        logging.info(f"处理点范围: {start_idx} - {end_idx-1} (共 {end_idx - start_idx} 个点)")
        
        # 获取当前批次的数据
        batch_gdf = points_gdf.iloc[start_idx:end_idx].copy()
        
        # 处理当前批次
        batch_result = process_single_batch(batch_gdf, raster1_path, raster2_path, chunk_size, n_processes, batch_idx + 1)
        
        if batch_result is not None:
            # 保存当前批次结果
            batch_filename = f"batch_{batch_idx + 1:03d}_points_with_ccdc_values_{timestamp}_NATURETREE.gpkg"
            batch_filepath = os.path.join(output_dir, batch_filename)
            
            logging.info(f"保存第 {batch_idx + 1} 批结果...")
            batch_result.to_file(batch_filepath, driver='GPKG')
            batch_files.append(batch_filepath)
            logging.info(f"第 {batch_idx + 1} 批结果已保存: {batch_filepath}")
            
            # 释放内存
            del batch_result, batch_gdf
            import gc
            gc.collect()
        else:
            logging.error(f"第 {batch_idx + 1} 批处理失败")
            return None
    
    # 拼接所有批次结果
    if batch_files:
        logging.info("="*30 + " 开始拼接所有批次结果 " + "="*30)
        final_filepath = merge_batch_files(batch_files, output_dir, timestamp)
        return final_filepath
    else:
        logging.error("没有成功处理的批次")
        return None

def process_single_batch(batch_gdf, raster1_path, raster2_path, chunk_size, n_processes, batch_id):
    """
    处理单个批次的点矢量数据
    
    参数:
    batch_gdf: 当前批次的点矢量GeoDataFrame
    raster1_path: 第一个栅格文件路径
    raster2_path: 第二个栅格文件路径
    chunk_size: 每块的点数量
    n_processes: 进程数量
    batch_id: 批次ID
    
    返回:
    包含栅格值的GeoDataFrame
    """
    
    # 分块
    chunks = []
    total_chunks = (len(batch_gdf) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(batch_gdf), chunk_size):
        chunk = batch_gdf.iloc[i:i+chunk_size].copy()
        chunk_id = f"{batch_id}-{i // chunk_size + 1}"
        chunks.append((chunk, raster1_path, raster2_path, chunk_id))
    
    logging.info(f"第 {batch_id} 批数据分为 {len(chunks)} 块")
    
    # 多进程处理
    start_time = time.time()
    
    with Pool(processes=n_processes) as pool:
        # 使用tqdm显示进度条
        results = []
        with tqdm(total=len(chunks), desc=f"批次 {batch_id} 处理进度") as pbar:
            for result in pool.imap(extract_raster_values_chunk, chunks):
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    # 合并当前批次结果
    if results:
        batch_result = pd.concat(results, ignore_index=True)
        logging.info(f"第 {batch_id} 批处理完成，总共处理了 {len(batch_result)} 个点")
        
        processing_time = time.time() - start_time
        logging.info(f"第 {batch_id} 批处理时间: {processing_time:.2f} 秒")
        logging.info(f"第 {batch_id} 批处理速度: {len(batch_result)/processing_time:.2f} 点/秒")
        
        return batch_result
    else:
        logging.error(f"第 {batch_id} 批所有块处理失败")
        return None

def merge_batch_files(batch_files, output_dir, timestamp):
    """
    合并所有批次文件为最终结果
    
    参数:
    batch_files: 批次文件路径列表
    output_dir: 输出目录
    timestamp: 时间戳
    
    返回:
    最终合并文件的路径
    """
    
    logging.info(f"开始合并 {len(batch_files)} 个批次文件...")
    
    # 读取并合并所有批次文件
    all_batches = []
    for i, batch_file in enumerate(batch_files):
        logging.info(f"读取第 {i+1}/{len(batch_files)} 个批次文件: {os.path.basename(batch_file)}")
        batch_gdf = gpd.read_file(batch_file)
        all_batches.append(batch_gdf)
    
    # 合并所有批次
    logging.info("合并所有批次数据...")
    final_result = pd.concat(all_batches, ignore_index=True)
    
    # 保存最终结果
    final_filename = f"points_with_ccdc_values_final_{timestamp}.gpkg"
    final_filepath = os.path.join(output_dir, final_filename)
    
    logging.info("保存最终合并结果...")
    final_result.to_file(final_filepath, driver='GPKG')
    
    logging.info(f"最终结果已保存: {final_filepath}")
    logging.info(f"最终结果包含 {len(final_result)} 个点")
    
    # 可选：删除临时批次文件以节省空间
    # for batch_file in batch_files:
    #     os.remove(batch_file)
    #     logging.info(f"已删除临时文件: {batch_file}")
    
    return final_filepath

def main():
    """主函数"""
    
    # 输入文件路径 - 修改为用户指定的文件
    points_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\3.自然林样本点\2.筛选自然林样本点\natural_forest_points_selected_20251010_180425.gpkg"
    raster1_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\CCDC_1999-2019\nBreaks_mosaic.tif"
    raster2_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\CCDC_1999-2019\raster_mosaic_uint16.tif"
    
    # 输出目录 - 修改为对应的输出目录
    output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\3.自然林样本点\2.筛选自然林样本点"
    
    # 设置日志
    log_path = setup_logging(output_dir)
    logging.info("="*50)
    logging.info("开始CCDC栅格值提取任务")
    logging.info(f"日志文件: {log_path}")
    
    try:
        # 检查输入文件是否存在
        for file_path in [points_path, raster1_path, raster2_path]:
            if not os.path.exists(file_path):
                logging.error(f"文件不存在: {file_path}")
                return
        
        logging.info("所有输入文件检查通过")
        
        # 读取点矢量数据
        logging.info("正在读取点矢量数据...")
        start_time = time.time()
        points_gdf = gpd.read_file(points_path)
        read_time = time.time() - start_time
        logging.info(f"点矢量数据读取完成，包含 {len(points_gdf)} 个点，耗时 {read_time:.2f} 秒")
        
        # 检查坐标系
        logging.info(f"点矢量坐标系: {points_gdf.crs}")
        
        # 检查栅格信息
        with rasterio.open(raster1_path) as src1:
            logging.info(f"栅格1 ({os.path.basename(raster1_path)}) 信息:")
            logging.info(f"  - 尺寸: {src1.width} x {src1.height}")
            logging.info(f"  - 坐标系: {src1.crs}")
            logging.info(f"  - 数据类型: {src1.dtypes[0]}")
        
        with rasterio.open(raster2_path) as src2:
            logging.info(f"栅格2 ({os.path.basename(raster2_path)}) 信息:")
            logging.info(f"  - 尺寸: {src2.width} x {src2.height}")
            logging.info(f"  - 坐标系: {src2.crs}")
            logging.info(f"  - 数据类型: {src2.dtypes[0]}")
        
        # 处理数据 - 使用分批处理
        logging.info("开始分批提取栅格值...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        final_output_path = process_points_in_batches(
            points_gdf, 
            raster1_path, 
            raster2_path,
            output_dir,
            batch_size=1000000,  # 每批100万个点
            chunk_size=50000,    # 每块5万个点，可根据内存情况调整
            n_processes=None     # 自动检测CPU核心数
        )
        
        if final_output_path:
            logging.info(f"最终结果已保存到: {final_output_path}")
            
            # 读取最终结果进行统计
            logging.info("正在读取最终结果进行统计...")
            result_gdf = gpd.read_file(final_output_path)
            
            # # 统计信息
            # logging.info("="*30 + " 统计信息 " + "="*30)
            # logging.info(f"总点数: {len(result_gdf)}")
            
            # # nBreaks值统计
            # nbreaks_valid = result_gdf['nBreaks_value'].notna().sum()
            # nbreaks_stats = result_gdf['nBreaks_value'].describe()
            # logging.info(f"nBreaks_mosaic.tif 有效值数量: {nbreaks_valid}")
            # logging.info(f"nBreaks_mosaic.tif 统计: 最小值={nbreaks_stats['min']:.2f}, 最大值={nbreaks_stats['max']:.2f}, 平均值={nbreaks_stats['mean']:.2f}")
            
            # # raster_mosaic值统计
            # raster_valid = result_gdf['raster_mosaic_value'].notna().sum()
            # raster_stats = result_gdf['raster_mosaic_value'].describe()
            # logging.info(f"raster_mosaic_uint16.tif 有效值数量: {raster_valid}")
            # logging.info(f"raster_mosaic_uint16.tif 统计: 最小值={raster_stats['min']:.2f}, 最大值={raster_stats['max']:.2f}, 平均值={raster_stats['mean']:.2f}")
            
            # # 保存统计信息到CSV
            # stats_filename = f"ccdc_extraction_stats_{timestamp}.csv"
            # stats_path = os.path.join(output_dir, stats_filename)
            
            # stats_data = {
            #     '指标': ['总点数', 'nBreaks有效值数量', 'nBreaks最小值', 'nBreaks最大值', 'nBreaks平均值',
            #             'raster_mosaic有效值数量', 'raster_mosaic最小值', 'raster_mosaic最大值', 'raster_mosaic平均值'],
            #     '数值': [len(result_gdf), nbreaks_valid, nbreaks_stats['min'], nbreaks_stats['max'], nbreaks_stats['mean'],
            #             raster_valid, raster_stats['min'], raster_stats['max'], raster_stats['mean']]
            # }
            
            # stats_df = pd.DataFrame(stats_data)
            # stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
            # logging.info(f"统计信息已保存到: {stats_path}")
            
            # logging.info("="*50)
            # logging.info("任务完成！")
            
        else:
            logging.error("处理失败")
            
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()