#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
林地样本点生成脚本（无矢量版本）

功能说明：
本脚本直接基于土地覆盖栅格数据生成林地样本点，无需矢量数据：
1. 将整个土地覆盖栅格分块处理，筛选森林像素（值为2）
2. 按指定采样率在每个栅格块内随机生成样本点
3. 使用PlantTree人工林产品数据进行二次筛选，保留人工林像素（值为2）的点
4. 输出原始林地样本点和经过PlantTree筛选后的人工林样本点

优化特性：
- 栅格分块处理，直接在土地覆盖数据中生成样本点，无需矢量裁剪
- 多进程并行处理，充分利用CPU资源
- 分块读取栅格数据，减少内存占用
- 实时内存监控，动态调整处理策略
- 分批保存结果，避免大数据集内存问题
- 详细的进度监控和日志记录

输入数据：
- 土地覆盖数据（ESRI 10m分辨率）
- PlantTree人工林30米产品栅格数据

输出结果：
- 原始林地样本点数据（GPKG格式）
- 经PlantTree筛选的人工林样本点数据（GPKG格式）
- 详细的处理统计报告

作者：锐多宝 (ruiduobao)
创建时间：2025年
"""

import os  # 导入os用于路径和文件操作
import sys  # 导入sys用于系统交互
import time  # 导入time用于计时
import logging  # 导入logging用于日志记录
from datetime import datetime  # 导入datetime用于时间戳
import random  # 导入random用于设置随机种子（保证可复现）
import numpy as np  # 导入numpy用于数值计算
import pandas as pd  # 导入pandas用于数据拼接与统计
import geopandas as gpd  # 导入geopandas用于地理数据处理
from shapely.geometry import Point  # 导入Point用于创建点几何
import rasterio  # 导入rasterio用于栅格读取与采样
from rasterio.windows import Window  # 导入Window用于栅格分块读取
from multiprocessing import Pool, cpu_count  # 导入多进程库提高并行处理效率
from tqdm import tqdm  # 导入tqdm用于进度条显示
import gc  # 导入垃圾回收模块用于内存管理
import psutil  # 导入psutil用于内存监控
import warnings  # 导入warnings用于警告控制
warnings.filterwarnings('ignore')  # 忽略警告信息

# ===================== 配置参数（可根据需要修改） =====================
# 土地覆盖数据文件路径
LANDCOVER_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\土地覆盖数据\southeast_asia_landcover_2024_mosaic.tif"  # 指定土地覆盖栅格
# PlantTree人工林产品栅格文件路径
PLANTTREE_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\全球人工林产品数据\southeast_asia_PlantTree_2021_mosaic.tif"  # 指定人工林产品栅格
# 输出目录路径
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\1.人工林样本点\3.根据栅格数据进行补充_全东南亚"  # 指定输出目录
# 森林像素值（土地覆盖数据中的森林类别）
FOREST_VALUE = 2  # 土地覆盖数据中森林的像素值
# 人工林像素值（PlantTree数据中的人工林类别）
PLANTATION_VALUE = 2  # PlantTree数据中人工林的像素值（根据之前的修正，2代表人工林）
# 采样率（每多少个像素选择1个点）
SAMPLING_RATE = 2000  # 每2000个森林像素选择1个点
# 进程数量（根据机器情况调整，避免过高导致内存压力）
NUM_PROCESSES = min(6, cpu_count())  # 设置最多6个进程，兼顾性能与稳定性
# 分块处理大小（栅格读取的分块大小）
CHUNK_SIZE = 4000  # 分块大小，平衡内存使用和处理效率
# 分批保存的样本点数量
BATCH_SIZE = 90000  # 每批保存的样本点数量
# 内存使用率阈值（百分比）
MEMORY_THRESHOLD = 85  # 内存使用率超过85%时触发分批保存
# PlantTree采样的分块大小（在worker内一次采样的点数）
PLANTTREE_CHUNK_SIZE = 500000  # 单次采样的点数上限，平衡IO与内存

# ===================== 日志设置函数 =====================

def setup_logging(output_dir):  # 定义日志设置函数
    """设置日志记录，输出到文件和控制台"""  # 函数说明
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    log_file = os.path.join(output_dir, f"plantation_point_generation_{timestamp}.log")  # 构造日志文件路径
    
    # 配置日志格式和输出
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 输出到文件
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ]
    )
    logging.info(f"日志文件: {log_file}")  # 记录日志文件路径
    return log_file  # 返回日志文件路径，便于后续记录到报告

# ===================== 通用工具函数 =====================

def check_input_files():  # 定义输入文件检查函数
    """检查所有输入文件是否存在"""  # 函数说明
    files_to_check = [
        ("土地覆盖数据", LANDCOVER_FILE),
        ("PlantTree数据", PLANTTREE_FILE)
    ]  # 需要检查的文件列表
    
    for name, filepath in files_to_check:  # 遍历检查每个文件
        if not os.path.exists(filepath):  # 检查文件是否存在
            raise FileNotFoundError(f"{name}文件不存在: {filepath}")  # 如果不存在抛出异常
        logging.info(f"✓ {name}: {filepath}")  # 记录文件检查通过

def get_memory_usage():  # 定义内存使用率获取函数
    """获取当前内存使用率"""  # 函数说明
    return psutil.virtual_memory().percent  # 返回内存使用百分比

def get_raster_info(raster_path):  # 定义栅格信息获取函数
    """获取栅格数据的基本信息"""  # 函数说明
    with rasterio.open(raster_path) as src:  # 打开栅格文件
        info = {
            'crs': src.crs,  # 坐标系
            'transform': src.transform,  # 仿射变换参数
            'width': src.width,  # 宽度（列数）
            'height': src.height,  # 高度（行数）
            'bounds': src.bounds,  # 边界范围
            'nodata': src.nodata  # NoData值
        }
    return info  # 返回栅格信息字典

# ===================== 栅格分块处理函数 =====================

def process_raster_chunk(args):  # 定义处理单个栅格块的函数（用于多进程）
    """处理单个栅格块，生成林地样本点"""  # 函数说明
    chunk_id, window, landcover_path, forest_value, sampling_rate = args  # 解包参数
    
    try:  # 开始异常捕获
        # 打开土地覆盖栅格
        with rasterio.open(landcover_path) as src:  # 打开土地覆盖栅格
            # 读取窗口内的栅格数据
            chunk_data = src.read(1, window=window)  # 读取第一个波段的窗口数据
            chunk_transform = rasterio.windows.transform(window, src.transform)  # 获取窗口的仿射变换
            
            # 筛选森林像素
            valid_mask = (chunk_data == forest_value) & (chunk_data != src.nodata) & (~np.isnan(chunk_data))  # 创建有效像素掩膜
            
            if not np.any(valid_mask):  # 如果没有有效像素
                logging.info(f"块 {chunk_id}: 无森林像素")  # 记录信息
                return []  # 返回空列表
            
            # 获取有效像素的行列索引（相对于窗口）
            rows, cols = np.where(valid_mask)  # 获取有效像素的行列坐标
            values = chunk_data[rows, cols]  # 获取对应的像素值
            
            # 随机采样
            total_pixels = len(rows)  # 总的有效像素数
            sample_size = max(1, total_pixels // sampling_rate)  # 计算采样数量
            
            if sample_size > 0:  # 如果需要采样
                # 随机选择样本索引
                sample_indices = random.sample(range(total_pixels), min(sample_size, total_pixels))  # 随机选择索引
                
                # 获取采样点的行列和值
                sample_rows = rows[sample_indices]  # 采样点的行坐标（窗口内）
                sample_cols = cols[sample_indices]  # 采样点的列坐标（窗口内）
                sample_values = values[sample_indices]  # 采样点的像素值
                
                # 转换为地理坐标
                points = []  # 初始化点列表
                for i in range(len(sample_rows)):  # 遍历每个采样点
                    # 计算像素中心的地理坐标（使用窗口的仿射变换）
                    x, y = rasterio.transform.xy(chunk_transform, sample_rows[i], sample_cols[i])  # 转换为地理坐标
                    points.append({
                        'geometry': Point(x, y),  # 点几何
                        'landcover_value': int(sample_values[i]),  # 土地覆盖值
                        'chunk_id': chunk_id,  # 所属块的ID
                        'row': int(sample_rows[i]),  # 栅格行坐标（窗口内）
                        'col': int(sample_cols[i])  # 栅格列坐标（窗口内）
                    })
                
                logging.info(f"块 {chunk_id}: 处理了 {total_pixels} 个森林像素，采样了 {len(points)} 个点")  # 记录处理结果
                return points  # 返回采样点列表
            
            return []  # 如果不需要采样，返回空列表
            
    except Exception as e:  # 捕获异常
        logging.error(f"处理块 {chunk_id} 时出错: {str(e)}")  # 记录错误日志
        return []  # 返回空列表

# ===================== PlantTree过滤函数 =====================

def sample_planttree_chunk(args):  # 定义PlantTree采样函数（用于多进程）
    """在子进程中对一批坐标进行PlantTree栅格采样，返回保留标记列表"""  # 函数说明
    raster_path, coords, plantation_value = args  # 解包参数（栅格路径、坐标列表、人工林值）
    keep_flags = []  # 初始化保留标记列表
    try:  # 开始异常捕获
        with rasterio.open(raster_path) as src:  # 在子进程内打开栅格，避免句柄跨进程问题
            nodata = src.nodata  # 记录栅格NoData值
            for i in range(0, len(coords), PLANTTREE_CHUNK_SIZE):  # 采用子分块采样，避免一次性读太多
                sub = coords[i:i+PLANTTREE_CHUNK_SIZE]  # 取当前子块坐标
                values = list(src.sample(sub))  # 对子块坐标进行采样，返回各波段值数组
                for val in values:  # 遍历每个采样结果
                    v = val[0] if isinstance(val, (list, tuple, np.ndarray)) else val  # 提取第一个波段的值
                    # 对于人工林样本点筛选，只保留PlantTree值为2（人工林）的点
                    # 剔除值为1（自然林）和NoData值的点
                    if v is None:  # 若为None（越界或出错），剔除
                        keep_flags.append(False)  # 记录剔除
                    elif np.isnan(v):  # 若为NaN，视为NoData，剔除
                        keep_flags.append(False)  # 记录剔除
                    elif nodata is not None and v == nodata:  # 若等于栅格定义的NoData值，剔除
                        keep_flags.append(False)  # 记录剔除
                    elif v == plantation_value:  # 值为2（人工林），保留
                        keep_flags.append(True)  # 记录保留
                    else:  # 其他数值（包括值为1的自然林），剔除
                        keep_flags.append(False)  # 记录剔除
        return keep_flags  # 返回当前批次的保留标记
    except Exception as e:  # 捕获异常
        logging.error(f"子进程采样PlantTree失败: {str(e)}")  # 记录错误日志
        return [False] * len(coords)  # 为稳妥起见，发生错误时默认剔除

# ===================== 批次保存函数 =====================

def save_batch_results(points_batch, output_dir, batch_num, crs, file_prefix):  # 定义批次保存函数
    """保存批次结果到临时文件"""  # 函数说明
    if not points_batch:  # 如果批次为空
        return None  # 返回None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    
    # 生成批次文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    batch_file = os.path.join(output_dir, f"temp_{file_prefix}_batch_{batch_num:04d}_{timestamp}.gpkg")  # 构造批次文件路径
    
    # 创建GeoDataFrame并保存
    gdf_batch = gpd.GeoDataFrame(points_batch, crs=crs)  # 创建GeoDataFrame
    gdf_batch.to_file(batch_file, driver='GPKG')  # 保存为GPKG格式
    
    logging.info(f"批次 {batch_num} 已保存: {len(points_batch)} 个点 -> {batch_file}")  # 记录保存信息
    logging.info(f"当前内存使用率: {get_memory_usage():.1f}%")  # 记录内存使用率
    
    return batch_file  # 返回批次文件路径

def merge_batch_files(batch_files, output_dir, final_filename):  # 定义批次文件合并函数
    """合并所有批次文件为最终结果"""  # 函数说明
    if not batch_files:  # 如果没有批次文件
        return None  # 返回None
    
    logging.info(f"开始合并 {len(batch_files)} 个批次文件...")  # 记录合并开始
    
    # 读取第一个文件获取结构
    gdf_final = gpd.read_file(batch_files[0])  # 读取第一个批次文件
    
    # 逐个读取并合并其他文件
    for i, batch_file in enumerate(batch_files[1:], 1):  # 遍历其他批次文件
        logging.info(f"合并进度: {i}/{len(batch_files)-1}")  # 记录合并进度
        gdf_batch = gpd.read_file(batch_file)  # 读取批次文件
        gdf_final = pd.concat([gdf_final, gdf_batch], ignore_index=True)  # 合并数据
        
        # 删除临时文件
        try:  # 尝试删除临时文件
            os.remove(batch_file)  # 删除文件
        except:  # 捕获删除异常
            pass  # 忽略删除失败
    
    # 删除第一个临时文件
    try:  # 尝试删除第一个临时文件
        os.remove(batch_files[0])  # 删除文件
    except:  # 捕获删除异常
        pass  # 忽略删除失败
    
    # 生成最终输出文件路径
    output_file = os.path.join(output_dir, final_filename)  # 构造最终文件路径
    
    # 保存最终结果
    gdf_final.to_file(output_file, driver='GPKG')  # 保存最终结果
    
    logging.info(f"合并完成，最终文件: {output_file}")  # 记录合并完成
    logging.info(f"总样本点数: {len(gdf_final)}")  # 记录总点数
    
    return gdf_final, output_file  # 返回最终数据和文件路径

# ===================== 主处理流程 =====================

def generate_forest_sample_points():  # 定义林地样本点生成函数
    """生成林地样本点的主函数"""  # 函数说明
    logging.info("开始生成林地样本点...")  # 记录开始信息
    
    # 获取土地覆盖栅格信息
    landcover_info = get_raster_info(LANDCOVER_FILE)  # 获取栅格信息
    logging.info(f"土地覆盖栅格信息:")  # 记录栅格信息标题
    logging.info(f"  CRS: {landcover_info['crs']}")  # 记录坐标系
    logging.info(f"  尺寸: {landcover_info['width']} x {landcover_info['height']}")  # 记录尺寸
    logging.info(f"  范围: {landcover_info['bounds']}")  # 记录范围
    logging.info(f"  NoData值: {landcover_info['nodata']}")  # 记录NoData值
    
    # 创建栅格分块任务
    with rasterio.open(LANDCOVER_FILE) as src:  # 打开土地覆盖栅格
        # 计算分块数量
        chunk_rows = (src.height + CHUNK_SIZE - 1) // CHUNK_SIZE  # 计算行方向分块数
        chunk_cols = (src.width + CHUNK_SIZE - 1) // CHUNK_SIZE  # 计算列方向分块数
        
        tasks = []  # 初始化任务列表
        chunk_id = 0  # 初始化块ID
        
        for row_chunk in range(chunk_rows):  # 遍历行分块
            for col_chunk in range(chunk_cols):  # 遍历列分块
                # 计算当前块的窗口
                row_start = row_chunk * CHUNK_SIZE  # 起始行
                col_start = col_chunk * CHUNK_SIZE  # 起始列
                row_end = min(row_start + CHUNK_SIZE, src.height)  # 结束行
                col_end = min(col_start + CHUNK_SIZE, src.width)  # 结束列
                
                # 创建窗口对象
                window = rasterio.windows.Window(
                    col_start, row_start, 
                    col_end - col_start, row_end - row_start
                )  # 创建读取窗口
                
                # 创建任务
                task = (
                    chunk_id,  # 块ID
                    window,  # 读取窗口
                    LANDCOVER_FILE,  # 土地覆盖栅格路径
                    FOREST_VALUE,  # 森林像素值
                    SAMPLING_RATE  # 采样率
                )
                tasks.append(task)  # 添加任务到列表
                chunk_id += 1  # 增加块ID
    
    logging.info(f"总共创建了 {len(tasks)} 个栅格分块任务")  # 记录任务数量
    logging.info(f"栅格分块: {chunk_rows} 行 × {chunk_cols} 列")  # 记录分块信息
    logging.info(f"初始内存使用率: {get_memory_usage():.1f}%")  # 记录初始内存使用率
    
    # 初始化变量
    all_points = []  # 所有点的列表
    batch_files = []  # 批次文件列表
    batch_num = 0  # 批次编号
    processed_tasks = 0  # 已处理任务数
    
    # 使用进度条显示处理进度
    with tqdm(total=len(tasks), desc="处理进度", unit="块") as pbar:  # 创建进度条
        # 多进程处理
        with Pool(processes=NUM_PROCESSES) as pool:  # 创建进程池
            # 分批处理任务以控制内存使用
            batch_size = max(1, len(tasks) // 5)  # 将任务分成5批
            
            for i in range(0, len(tasks), batch_size):  # 分批处理
                batch_tasks = tasks[i:i+batch_size]  # 当前批次的任务
                
                # 处理当前批次
                results = pool.map(process_raster_chunk, batch_tasks)  # 并行处理批次任务
                
                # 收集结果
                for result in results:  # 遍历处理结果
                    if result:  # 如果结果不为空
                        all_points.extend(result)  # 添加到总点列表
                    
                    processed_tasks += 1  # 增加已处理任务数
                    pbar.update(1)  # 更新进度条
                    
                    # 更新进度信息
                    progress = (processed_tasks / len(tasks)) * 100  # 计算进度百分比
                    memory_usage = get_memory_usage()  # 获取内存使用率
                    pbar.set_postfix({
                        '进度': f'{progress:.1f}%',
                        '内存': f'{memory_usage:.1f}%',
                        '样本点': len(all_points)
                    })  # 设置进度条后缀信息
                
                # 检查是否需要保存批次结果
                if (len(all_points) >= BATCH_SIZE or 
                    get_memory_usage() > MEMORY_THRESHOLD or
                    processed_tasks == len(tasks)):  # 满足保存条件
                    
                    if all_points:  # 如果有点数据
                        # 保存当前批次
                        batch_file = save_batch_results(
                            all_points, OUTPUT_DIR, batch_num, landcover_info['crs'], "forest"
                        )  # 保存批次
                        if batch_file:  # 如果保存成功
                            batch_files.append(batch_file)  # 添加到批次文件列表
                        
                        # 清空内存中的点数据
                        all_points = []  # 清空点列表
                        batch_num += 1  # 增加批次编号
                        
                        # 强制垃圾回收
                        gc.collect()  # 执行垃圾回收
    
    logging.info(f"林地样本点生成完成，共生成 {len(batch_files)} 个批次文件")  # 记录生成完成
    
    # 合并所有批次文件
    if batch_files:  # 如果有批次文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
        final_filename = f"forest_sample_points_{timestamp}.gpkg"  # 构造最终文件名
        gdf, output_file = merge_batch_files(batch_files, OUTPUT_DIR, final_filename)  # 合并批次文件
        return gdf, output_file  # 返回最终数据和文件路径
    else:  # 如果没有批次文件
        logging.warning("没有生成任何林地样本点！")  # 记录警告
        return None, None  # 返回None

def filter_points_by_planttree(points_gdf):  # 定义PlantTree过滤函数
    """使用PlantTree数据过滤样本点，保留人工林区域的点"""  # 函数说明
    logging.info("开始使用PlantTree数据过滤样本点...")  # 记录开始信息
    
    # 准备坐标序列（经纬度对），供栅格采样使用
    coords = [(geom.x, geom.y) for geom in points_gdf.geometry]  # 从点几何提取坐标元组
    
    # 将坐标分块，准备并行处理
    coord_chunks = [coords[i:i+BATCH_SIZE] for i in range(0, len(coords), BATCH_SIZE)]  # 生成坐标分块列表
    logging.info(f"PlantTree采样分块数: {len(coord_chunks)}")  # 记录分块数量
    
    # 使用进程池并行采样PlantTree
    plant_keep_flags = []  # 初始化保留标记总列表
    with Pool(processes=NUM_PROCESSES) as pool:  # 创建进程池
        # 为每个分块准备参数
        tasks = [(PLANTTREE_FILE, chunk, PLANTATION_VALUE) for chunk in coord_chunks]  # 构造任务参数列表
        for idx, flags in enumerate(pool.imap(sample_planttree_chunk, tasks)):  # 以imap流式获取结果，避免一次性返回
            plant_keep_flags.extend(flags)  # 累加保留标记
            if (idx + 1) % 5 == 0 or (idx + 1) == len(coord_chunks):  # 每处理5块或最后一块记录一次进度
                logging.info(f"PlantTree采样进度: {idx + 1}/{len(coord_chunks)} 块")  # 输出进度日志
    
    # 将PlantTree保留标记转换为布尔Series
    plant_keep_series = np.array(plant_keep_flags, dtype=bool)  # 转换为布尔数组
    kept_by_plant = np.count_nonzero(plant_keep_series)  # 统计被PlantTree条件保留的数量
    dropped_by_plant = len(points_gdf) - kept_by_plant  # 统计被PlantTree条件剔除的数量
    logging.info(f"PlantTree筛选后保留(值为2的人工林): {kept_by_plant}, 剔除(值为1的自然林和NoData): {dropped_by_plant}")  # 记录筛选结果
    
    # 应用PlantTree筛选
    filtered_points_gdf = points_gdf.loc[plant_keep_series].copy()  # 根据保留标记筛选点
    filtered_points_gdf.reset_index(drop=True, inplace=True)  # 重置索引
    
    return filtered_points_gdf  # 返回筛选后的点数据

def save_results_and_report(forest_gdf, plantation_gdf, forest_file, output_dir):  # 定义结果保存和报告生成函数
    """保存最终结果并生成统计报告"""  # 函数说明
    
    # 保存经过PlantTree筛选的人工林样本点
    if plantation_gdf is not None and len(plantation_gdf) > 0:  # 如果有人工林样本点
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
        plantation_file = os.path.join(output_dir, f"plantation_tree_sample_points_{timestamp}.gpkg")  # 构造人工林样本点文件路径
        plantation_gdf.to_file(plantation_file, driver='GPKG')  # 保存人工林样本点
        logging.info(f"人工林样本点已保存到: {plantation_file}")  # 记录保存信息
    else:  # 如果没有人工林样本点
        plantation_file = None  # 设置为None
        logging.warning("没有人工林样本点可保存")  # 记录警告
    
    # 生成统计报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    report_file = os.path.join(output_dir, f"sample_point_generation_report_{timestamp}.txt")  # 构造报告文件路径
    
    with open(report_file, 'w', encoding='utf-8') as f:  # 打开报告文件
        f.write("没有SDPT国家的林地样本点生成报告\n")  # 写入标题
        f.write("=" * 60 + "\n")  # 写入分隔线
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 写入生成时间
        f.write(f"输入数据:\n")  # 写入输入数据标题
        f.write(f"  土地覆盖数据: {LANDCOVER_FILE}\n")  # 写入土地覆盖数据路径
        f.write(f"  PlantTree数据: {PLANTTREE_FILE}\n")  # 写入PlantTree数据路径
        f.write(f"处理参数:\n")  # 写入处理参数标题
        f.write(f"  森林像素值: {FOREST_VALUE}\n")  # 写入森林像素值
        f.write(f"  人工林像素值: {PLANTATION_VALUE}\n")  # 写入人工林像素值
        f.write(f"  采样率: 每{SAMPLING_RATE}个像素选择1个\n")  # 写入采样率
        f.write(f"  进程数: {NUM_PROCESSES}\n")  # 写入进程数
        f.write(f"  分批大小: {BATCH_SIZE}\n")  # 写入分批大小
        f.write(f"  内存阈值: {MEMORY_THRESHOLD}%\n\n")  # 写入内存阈值
        
        f.write("处理结果:\n")  # 写入处理结果标题
        if forest_gdf is not None:  # 如果有林地样本点
            f.write(f"  原始林地样本点数: {len(forest_gdf)}\n")  # 写入原始林地样本点数
            f.write(f"  原始林地样本点文件: {forest_file}\n")  # 写入原始林地样本点文件路径
        else:  # 如果没有林地样本点
            f.write(f"  原始林地样本点数: 0\n")  # 写入0
            
        if plantation_gdf is not None:  # 如果有人工林样本点
            f.write(f"  人工林样本点数: {len(plantation_gdf)}\n")  # 写入人工林样本点数
            f.write(f"  人工林样本点文件: {plantation_file}\n")  # 写入人工林样本点文件路径
            if forest_gdf is not None and len(forest_gdf) > 0:  # 如果有原始林地样本点
                retention_rate = len(plantation_gdf) / len(forest_gdf) * 100  # 计算保留率
                f.write(f"  PlantTree筛选保留率: {retention_rate:.2f}%\n")  # 写入保留率
        else:  # 如果没有人工林样本点
            f.write(f"  人工林样本点数: 0\n")  # 写入0
            f.write(f"  PlantTree筛选保留率: 0.00%\n")  # 写入0%保留率
    
    logging.info(f"统计报告已保存到: {report_file}")  # 记录报告保存信息
    
    return report_file  # 返回报告文件路径

# ===================== 主函数 =====================

def main():  # 定义主函数
    """主流程：读取范围矢量->分面生成林地样本点->PlantTree筛选->保存与统计"""  # 函数说明
    start_time = time.time()  # 记录开始时间

    # 固定随机种子，保证复现
    random.seed(42)  # 设置Python随机种子
    np.random.seed(42)  # 设置Numpy随机种子

    # 设置日志系统
    log_file = setup_logging(OUTPUT_DIR)  # 初始化日志，返回日志文件路径
    logging.info("没有SDPT国家的林地样本点生成脚本启动")  # 记录启动信息
    logging.info(f"土地覆盖数据: {LANDCOVER_FILE}")  # 记录土地覆盖数据路径
    logging.info(f"PlantTree数据: {PLANTTREE_FILE}")  # 记录PlantTree数据路径
    logging.info(f"输出目录: {OUTPUT_DIR}")  # 记录输出目录
    logging.info(f"森林像素值: {FOREST_VALUE}, 人工林像素值: {PLANTATION_VALUE}")  # 记录像素值
    logging.info(f"采样率: 每{SAMPLING_RATE}个像素选择1个")  # 记录采样率
    logging.info(f"进程数: {NUM_PROCESSES}, 分块大小: {CHUNK_SIZE}, 分批大小: {BATCH_SIZE}")  # 记录处理参数

    try:  # 开始异常捕获
        # 检查输入文件
        check_input_files()  # 检查所有输入文件是否存在
        
        # 生成林地样本点
        forest_gdf, forest_file = generate_forest_sample_points()  # 生成林地样本点
        
        if forest_gdf is not None and len(forest_gdf) > 0:  # 如果成功生成林地样本点
            logging.info(f"成功生成 {len(forest_gdf)} 个林地样本点")  # 记录成功信息
            
            # 使用PlantTree数据过滤样本点
            plantation_gdf = filter_points_by_planttree(forest_gdf)  # 进行PlantTree过滤
            
            if len(plantation_gdf) > 0:  # 如果有人工林样本点
                logging.info(f"PlantTree筛选后保留 {len(plantation_gdf)} 个人工林样本点")  # 记录筛选结果
            else:  # 如果没有人工林样本点
                logging.warning("PlantTree筛选后没有保留任何样本点")  # 记录警告
            
            # 保存结果并生成报告
            report_file = save_results_and_report(forest_gdf, plantation_gdf, forest_file, OUTPUT_DIR)  # 保存结果和生成报告
            
            # 计算处理时间
            elapsed_time = time.time() - start_time  # 计算总耗时
            logging.info(f"处理完成！总耗时: {elapsed_time:.2f} 秒")  # 记录完成信息
            
            if len(forest_gdf) > 0:  # 如果有林地样本点
                processing_speed = len(forest_gdf) / elapsed_time  # 计算处理速度
                logging.info(f"处理速度: {processing_speed:.0f} 点/秒")  # 记录处理速度
                logging.info(f"最终内存使用率: {get_memory_usage():.1f}%")  # 记录最终内存使用率
        else:  # 如果没有生成林地样本点
            logging.error("林地样本点生成失败")  # 记录错误
            
    except Exception as e:  # 捕获异常
        logging.error(f"程序执行出错: {str(e)}")  # 记录错误日志
        raise  # 重新抛出异常

if __name__ == "__main__":  # 主程序入口
    main()  # 调用主函数