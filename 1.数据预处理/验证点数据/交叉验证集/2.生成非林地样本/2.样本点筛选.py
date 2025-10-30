#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本点筛选脚本 - 基于PlantTree和SDPT数据筛选非林地样本点
目的：从土地覆盖数据生成的非林地样本点中，筛选出真正的非林地点
主要功能：
1. 基于PlantTree栅格数据筛选（去除人工林区域的点）
2. 基于SDPT矢量数据筛选（去除SDPT范围内的点）
3. 使用0.5度网格优化空间筛选性能
4. 支持多进程并行处理提升效率
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
from shapely.strtree import STRtree  # 导入STRtree用于空间索引优化
import rasterio  # 导入rasterio用于栅格读取与采样
from rasterio.transform import Affine  # 导入仿射变换类型
from multiprocessing import Pool, cpu_count  # 导入多进程库提高并行处理效率
import gc  # 导入垃圾回收模块用于内存管理

# ===================== 配置参数（可根据需要修改） =====================
# 非林地样本点的输入目录（上一步输出的目录）
INPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\2.非林地样本点\1.土地覆盖数据生成"  # 指定上一步的输出目录
# 人工林30米产品栅格文件路径
PLANTTREE_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\全球人工林产品数据\southeast_asia_PlantTree_2021_mosaic.tif"  # 指定人工林产品栅格
# SDPT范围矢量文件路径
SDPT_PATH = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\按国家筛选\sdpt_southeast_asia_20251006_181147.gpkg"  # 指定SDPT矢量范围
# 0.5度网格文件路径（用于分块优化SDPT筛选）
GRID_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\分类网格数据\southeast_asia_grid_0.5deg.shp"  # 指定0.5度网格文件
# 输出目录路径
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\2.非林地样本点\2.样本点筛选"  # 指定输出目录
# 进程数量（根据机器情况调整，避免过高导致内存压力）
NUM_PROCESSES = min(8, cpu_count())  # 设置最多8个进程，兼顾性能与稳定性
# 点数据分块大小（点数量大，分块处理更稳健）
POINT_CHUNK_SIZE = 200_000  # 每次处理20万点，避免一次性占用过多内存
# 采样栅格的分块大小（在worker内一次采样的点数）
SAMPLE_CHUNK_SIZE = 100_000  # 单次采样的点数上限，平衡IO与内存
# PlantTree产品中"表示非林地"的像素判断规则（NoData表示非林地）
# 人工林产品数据编码：1=人工林，2=自然林，NoData=非林地
# 对于非林地样本点筛选，只保留NoData值的点

# ===================== 日志设置函数 =====================

def setup_logging(output_dir):  # 定义日志设置函数
    """设置日志记录到文件和控制台"""  # 函数说明
    os.makedirs(output_dir, exist_ok=True)  # 若输出目录不存在则创建
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳字符串
    log_file = os.path.join(output_dir, f"nonforest_point_filter_log_{timestamp}.txt")  # 日志文件路径
    logging.basicConfig(  # 配置日志基本设置
        level=logging.INFO,  # 设置日志级别为INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        handlers=[  # 设置日志输出到文件和控制台
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件处理器，UTF-8编码
            logging.StreamHandler(sys.stdout)  # 控制台输出处理器
        ]
    )
    return log_file  # 返回日志文件路径，便于后续记录到报告

# ===================== 通用工具函数 =====================

def find_latest_input_file(input_dir):  # 定义查找最新输入GPKG的函数
    """在输入目录中查找最新的非林地样本GPKG文件"""  # 函数说明
    if not os.path.exists(input_dir):  # 检查输入目录是否存在
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")  # 如果不存在抛出异常
    candidates = []  # 初始化候选文件列表
    for name in os.listdir(input_dir):  # 遍历目录中文件名
        if name.lower().endswith('.gpkg') and 'nonforest_sample_points_' in name:  # 筛选匹配命名模式的gpkg
            candidates.append(os.path.join(input_dir, name))  # 将符合条件的文件加入列表
    if not candidates:  # 如果没有候选文件
        raise FileNotFoundError("输入目录中未找到非林地样本点GPKG文件 (nonforest_sample_points_*.gpkg)")  # 抛出异常
    latest = max(candidates, key=os.path.getmtime)  # 选取按修改时间最新的文件
    return latest  # 返回最新文件路径

# ===================== PlantTree采样（并行） =====================

def sample_planttree_chunk(args):  # 定义在worker中执行的采样函数
    """在子进程中对一批坐标进行PlantTree栅格采样，返回保留标记列表"""  # 函数说明
    raster_path, coords = args  # 解包参数（栅格路径、坐标列表）
    keep_flags = []  # 初始化保留标记列表
    try:  # 开始异常捕获
        with rasterio.open(raster_path) as src:  # 在子进程内打开栅格，避免句柄跨进程问题
            nodata = src.nodata  # 记录栅格NoData值
            for i in range(0, len(coords), SAMPLE_CHUNK_SIZE):  # 采用子分块采样，避免一次性读太多
                sub = coords[i:i+SAMPLE_CHUNK_SIZE]  # 取当前子块坐标
                values = list(src.sample(sub))  # 对子块坐标进行采样，返回各波段值数组
                for val in values:  # 遍历每个采样结果
                    v = val[0] if isinstance(val, (list, tuple, np.ndarray)) else val  # 提取第一个波段的值
                    # 对于非林地样本点筛选，只保留NoData值的点（非林地）
                    # 剔除值为1（人工林）和值为2（自然林）的点
                    if v is None:  # 若为None（越界或出错），视为应保留（非林地）
                        keep_flags.append(True)  # 记录保留
                    elif np.isnan(v):  # 若为NaN，视为NoData，保留
                        keep_flags.append(True)  # 记录保留
                    elif nodata is not None and v == nodata:  # 若等于栅格定义的NoData值，保留
                        keep_flags.append(True)  # 记录保留
                    else:  # 正常数值（1或2），需要剔除
                        keep_flags.append(False)  # 记录剔除
        return keep_flags  # 返回当前批次的保留标记
    except Exception as e:  # 捕获异常
        logging.error(f"子进程采样PlantTree失败: {str(e)}")  # 记录错误日志
        return [True] * len(coords)  # 为稳妥起见，发生错误时默认保留（避免误删），可在报告中提示

# ===================== SDPT空间过滤（基于网格分块优化） =====================

def filter_outside_sdpt_by_grid_optimized(points_gdf, sdpt_gdf, grid_gdf):
    """
    基于网格的SDPT空间筛选函数（性能优化版）
    使用空间索引和批量处理技术提升筛选速度
    
    参数:
        points_gdf: 待筛选的点数据
        sdpt_gdf: SDPT矢量数据
        grid_gdf: 0.5度网格数据
    
    返回:
        筛选后的点数据（SDPT范围外的点）
    """
    logging.info("开始基于网格的SDPT空间筛选（性能优化版）...")
    start_time = time.time()
    
    # 1. 为SDPT数据创建空间索引，大幅提升空间查询速度
    logging.info("为SDPT数据创建空间索引...")
    sdpt_tree = STRtree(sdpt_gdf.geometry)  # 创建SDPT的空间索引树
    
    # 2. 点与网格的空间连接，获取每个点所属的网格ID
    logging.info("执行点与网格的空间连接...")
    points_with_grid = gpd.sjoin(points_gdf, grid_gdf[['grid_id', 'geometry']], how='left', predicate='within')
    
    # 3. 获取包含点的网格列表
    grids_with_points = points_with_grid['grid_id'].dropna().unique()
    logging.info(f"共有 {len(grids_with_points)} 个网格包含样本点")
    
    # 4. 批量处理网格，减少循环开销
    filtered_chunks = []
    batch_size = 50  # 每批处理50个网格，平衡内存和性能
    
    for batch_start in range(0, len(grids_with_points), batch_size):
        batch_end = min(batch_start + batch_size, len(grids_with_points))
        batch_grids = grids_with_points[batch_start:batch_end]
        
        logging.info(f"处理网格批次 {batch_start//batch_size + 1}/{(len(grids_with_points)-1)//batch_size + 1}: 网格 {batch_start+1}-{batch_end}")
        
        # 批量获取当前批次所有网格的点
        batch_points = points_with_grid[points_with_grid['grid_id'].isin(batch_grids)].copy()
        batch_points = batch_points.drop(columns=['index_right'], errors='ignore')  # 清理可能存在的index_right列
        
        # 批量获取当前批次所有网格的几何范围
        batch_grid_geoms = grid_gdf[grid_gdf['grid_id'].isin(batch_grids)]
        
        # 使用空间索引快速查找与批次网格相交的SDPT
        batch_sdpt_indices = []
        for geom in batch_grid_geoms.geometry:
            intersecting_indices = list(sdpt_tree.query(geom))
            batch_sdpt_indices.extend(intersecting_indices)
        
        # 去重并获取相关的SDPT数据
        batch_sdpt_indices = list(set(batch_sdpt_indices))
        if batch_sdpt_indices:
            batch_sdpt = sdpt_gdf.iloc[batch_sdpt_indices]
        else:
            batch_sdpt = gpd.GeoDataFrame(columns=sdpt_gdf.columns, crs=sdpt_gdf.crs)
        
        # 对批次内的点进行SDPT筛选
        if len(batch_sdpt) == 0:
            # 批次内无SDPT，所有点都保留
            batch_points_clean = batch_points.drop(columns=['grid_id'], errors='ignore')
            filtered_chunks.append(batch_points_clean)
            logging.info(f"批次 {batch_start//batch_size + 1}: {len(batch_points)} 点，无SDPT面，全部保留")
        else:
            # 使用空间索引进行快速筛选
            points_outside = []
            for idx, point_row in batch_points.iterrows():
                point_geom = point_row.geometry
                # 使用空间索引快速查找相交的SDPT
                intersecting_sdpt = list(sdpt_tree.query(point_geom))
                
                # 检查是否真正相交（空间索引返回的是候选项）
                is_inside_sdpt = False
                for sdpt_idx in intersecting_sdpt:
                    if point_geom.intersects(sdpt_gdf.iloc[sdpt_idx].geometry):
                        is_inside_sdpt = True
                        break
                
                if not is_inside_sdpt:
                    points_outside.append(point_row)
            
            if points_outside:
                outside_gdf = gpd.GeoDataFrame(points_outside, crs=batch_points.crs)
                outside_gdf = outside_gdf.drop(columns=['grid_id'], errors='ignore')
                filtered_chunks.append(outside_gdf)
            
            logging.info(f"批次 {batch_start//batch_size + 1}: {len(batch_points)} 点，{len(batch_sdpt)} SDPT面，保留 {len(points_outside)} 点")
        
        # 强制垃圾回收，释放内存
        del batch_points, batch_grid_geoms, batch_sdpt
        gc.collect()
    
    # 5. 处理没有网格ID的点（可能在网格范围外）
    points_no_grid = points_with_grid[points_with_grid['grid_id'].isna()]
    if len(points_no_grid) > 0:
        logging.info(f"发现 {len(points_no_grid)} 个点位于网格范围外，使用空间索引进行全局SDPT筛选")
        points_no_grid = points_no_grid.drop(columns=['index_right'], errors='ignore')
        
        # 使用空间索引进行快速全局筛选
        global_outside = []
        for idx, point_row in points_no_grid.iterrows():
            point_geom = point_row.geometry
            intersecting_sdpt = list(sdpt_tree.query(point_geom))
            
            is_inside_sdpt = False
            for sdpt_idx in intersecting_sdpt:
                if point_geom.intersects(sdpt_gdf.iloc[sdpt_idx].geometry):
                    is_inside_sdpt = True
                    break
            
            if not is_inside_sdpt:
                global_outside.append(point_row)
        
        if global_outside:
            global_outside_gdf = gpd.GeoDataFrame(global_outside, crs=points_no_grid.crs)
            global_outside_gdf = global_outside_gdf.drop(columns=['grid_id'], errors='ignore')
            filtered_chunks.append(global_outside_gdf)
        
        logging.info(f"网格外点筛选完成，保留 {len(global_outside)} 点")
    
    # 6. 合并所有筛选结果
    if filtered_chunks:
        result_gdf = pd.concat(filtered_chunks, ignore_index=True)
        result_gdf = gpd.GeoDataFrame(result_gdf, crs=points_gdf.crs)
    else:
        result_gdf = gpd.GeoDataFrame(columns=points_gdf.columns, crs=points_gdf.crs)
    
    elapsed_time = time.time() - start_time
    logging.info(f"基于网格的SDPT筛选完成，用时 {elapsed_time:.2f} 秒")
    logging.info(f"筛选前: {len(points_gdf)} 点，筛选后: {len(result_gdf)} 点，保留率: {len(result_gdf)/len(points_gdf)*100:.2f}%")
    
    return result_gdf

def filter_outside_sdpt(points_chunk_gdf, sdpt_gdf):  # 保留原有的分块内SDPT范围外过滤函数（备用）
    """对一个点分块执行空间连接，保留与SDPT不相交的点（原有方法，作为备用）"""  # 函数说明
    # 使用GeoPandas的sjoin进行空间连接，predicate='intersects'表示相交判断
    joined = gpd.sjoin(points_chunk_gdf, sdpt_gdf[['geometry']], how='left', predicate='intersects')  # 左连接加上SDPT相交信息
    mask_outside = joined['index_right'].isna()  # index_right为空表示未与任何SDPT面相交
    return points_chunk_gdf.loc[mask_outside]  # 返回SDPT范围外的点

# ===================== 主处理流程 =====================

def main():  # 定义主函数
    """主流程：读取点数据->栅格采样筛PlantTree->矢量sjoin筛SDPT->保存与统计"""  # 函数说明
    start_time = time.time()  # 记录开始时间

    # 固定随机种子，保证复现（虽然此处主要为顺序一致性）
    random.seed(42)  # 设置Python随机种子
    np.random.seed(42)  # 设置Numpy随机种子

    # 设置日志系统
    log_file = setup_logging(OUTPUT_DIR)  # 初始化日志，返回日志文件路径
    logging.info("样本点筛选脚本启动")  # 记录启动信息
    logging.info(f"输入目录: {INPUT_DIR}")  # 记录输入目录
    logging.info(f"PlantTree栅格: {PLANTTREE_FILE}")  # 记录PlantTree栅格路径
    logging.info(f"SDPT矢量: {SDPT_PATH}")  # 记录SDPT矢量路径
    logging.info(f"网格文件: {GRID_FILE}")  # 记录网格文件路径
    logging.info(f"输出目录: {OUTPUT_DIR}")  # 记录输出目录
    logging.info(f"进程数: {NUM_PROCESSES}, 点分块: {POINT_CHUNK_SIZE}, 采样子块: {SAMPLE_CHUNK_SIZE}")  # 记录处理参数

    # 查找最新的输入点文件
    input_points_file = find_latest_input_file(INPUT_DIR)  # 获取最新的非林地样本点文件
    logging.info(f"将读取最新的非林地样本点文件: {input_points_file}")  # 记录选择的文件

    # 读取点数据
    points_gdf = gpd.read_file(input_points_file)  # 使用GeoPandas读取GPKG点数据
    logging.info(f"已读取点数据，共 {len(points_gdf)} 条记录，CRS: {points_gdf.crs}")  # 记录点数量与坐标系

    # 读取SDPT矢量数据
    sdpt_gdf = gpd.read_file(SDPT_PATH)  # 读取SDPT矢量范围
    logging.info(f"已读取SDPT矢量，共 {len(sdpt_gdf)} 个面，CRS: {sdpt_gdf.crs}")  # 记录面数量与坐标系
    
    # 读取0.5度网格数据
    grid_gdf = gpd.read_file(GRID_FILE)  # 读取0.5度网格数据
    logging.info(f"已读取0.5度网格，共 {len(grid_gdf)} 个网格，CRS: {grid_gdf.crs}")  # 记录网格数量与坐标系
    _ = sdpt_gdf.sindex  # 预先构建并缓存空间索引，避免在每次sjoin时重复构建

    # 坐标系一致性处理：统一到EPSG:4326（假设PlantTree与SDPT均为4326，如不一致则转投影）
    target_crs = 'EPSG:4326'  # 设定目标坐标系
    if points_gdf.crs is None or str(points_gdf.crs) != target_crs:  # 判断点坐标系是否为目标
        points_gdf = points_gdf.to_crs(target_crs)  # 转换点坐标系
        logging.info(f"点数据已重投影到 {target_crs}")  # 记录重投影
    if sdpt_gdf.crs is None or str(sdpt_gdf.crs) != target_crs:  # 判断SDPT坐标系是否为目标
        sdpt_gdf = sdpt_gdf.to_crs(target_crs)  # 转换SDPT坐标系
        logging.info(f"SDPT已重投影到 {target_crs}")  # 记录重投影
    if grid_gdf.crs is None or str(grid_gdf.crs) != target_crs:  # 判断网格坐标系是否为目标
        grid_gdf = grid_gdf.to_crs(target_crs)  # 转换网格坐标系
        logging.info(f"网格数据已重投影到 {target_crs}")  # 记录重投影
    
    # ================= PlantTree过滤（并行采样） =================
    # 准备坐标序列（经纬度对），供栅格采样使用
    coords = [(geom.x, geom.y) for geom in points_gdf.geometry]  # 从点几何提取坐标元组

    # 将坐标分块，准备并行处理
    coord_chunks = [coords[i:i+POINT_CHUNK_SIZE] for i in range(0, len(coords), POINT_CHUNK_SIZE)]  # 生成坐标分块列表
    logging.info(f"PlantTree采样分块数: {len(coord_chunks)}")  # 记录分块数量

    # 使用进程池并行采样PlantTree
    plant_keep_flags = []  # 初始化保留标记总列表
    with Pool(processes=NUM_PROCESSES) as pool:  # 创建进程池
        # 为每个分块准备参数（移除threshold参数）
        tasks = [(PLANTTREE_FILE, chunk) for chunk in coord_chunks]  # 构造任务参数列表
        for idx, flags in enumerate(pool.imap(sample_planttree_chunk, tasks)):  # 以imap流式获取结果，避免一次性返回
            plant_keep_flags.extend(flags)  # 累加保留标记
            if (idx + 1) % 5 == 0 or (idx + 1) == len(coord_chunks):  # 每处理5块或最后一块记录一次进度
                logging.info(f"PlantTree采样进度: {idx + 1}/{len(coord_chunks)} 块")  # 输出进度日志

    # 将PlantTree保留标记转换为布尔Series
    plant_keep_series = np.array(plant_keep_flags, dtype=bool)  # 转换为布尔数组
    kept_by_plant = np.count_nonzero(plant_keep_series)  # 统计被PlantTree条件保留的数量
    dropped_by_plant = len(points_gdf) - kept_by_plant  # 统计被PlantTree条件剔除的数量
    logging.info(f"PlantTree筛选后保留(NoData值，非林地): {kept_by_plant}, 剔除(值1和2，人工林和自然林): {dropped_by_plant}")  # 输出筛选统计

    # 先对点数据应用PlantTree筛选，减少后续SDPT运算成本
    points_after_plant = points_gdf.loc[plant_keep_series].copy()  # 保留PlantTree条件通过的点
    logging.info(f"进入SDPT筛选的点数: {len(points_after_plant)}")  # 记录进入SDPT筛选的点数

    # ===================== 第二步：SDPT筛选（基于网格优化） =====================
    logging.info("=" * 60)  # 分隔线
    logging.info("第二步：基于SDPT数据筛选（去除SDPT范围内的点）")  # 记录第二步开始
    
    # 调用基于网格优化的SDPT筛选函数（性能优化版）
    final_gdf = filter_outside_sdpt_by_grid_optimized(points_after_plant, sdpt_gdf, grid_gdf)  # 调用性能优化版的筛选函数
    logging.info(f"SDPT筛选后保留点数: {len(final_gdf)}")  # 记录SDPT筛选后的点数

    # ================= 保存结果与统计报告 =================
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 若输出目录不存在则创建
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    output_file = os.path.join(OUTPUT_DIR, f"nonforest_points_selected_{timestamp}.gpkg")  # 构造输出文件名

    # 将最终结果写出为GPKG
    final_gdf.to_file(output_file, driver='GPKG')  # 写出GPKG文件
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)  # 计算文件大小（MB）
    logging.info(f"结果已保存: {output_file}, 大小: {file_size_mb:.2f} MB, 记录数: {len(final_gdf)}")  # 记录保存信息

    # 生成统计报告
    report_path = os.path.join(OUTPUT_DIR, f"nonforest_point_filter_report_{timestamp}.txt")  # 报告文件路径
    with open(report_path, 'w', encoding='utf-8') as f:  # 打开报告文件进行写入
        f.write("非林地样本点筛选报告\n")  # 报告标题
        f.write("=" * 60 + "\n")  # 分隔线
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 写入生成时间
        f.write(f"输入点文件: {input_points_file}\n")  # 写入输入文件
        f.write(f"PlantTree栅格: {PLANTTREE_FILE}\n")  # 写入栅格路径
        f.write(f"SDPT矢量: {SDPT_PATH}\n")  # 写入SDPT路径
        f.write(f"网格文件: {GRID_FILE}\n")  # 写入网格文件路径
        f.write(f"进程数: {NUM_PROCESSES}, 点分块: {POINT_CHUNK_SIZE}, 采样子块: {SAMPLE_CHUNK_SIZE}\n\n")  # 写入参数
        f.write(f"原始点数: {len(points_gdf)}\n")  # 写入原始点数
        f.write(f"PlantTree筛选后保留(NoData值，非林地): {kept_by_plant}, 剔除(值1和2，人工林和自然林): {dropped_by_plant}\n")  # 写入PlantTree筛选统计
        f.write(f"进入SDPT筛选点数: {len(points_after_plant)}\n")  # 写入进入SDPT筛选点数
        f.write(f"SDPT筛选后保留点数: {len(final_gdf)}\n")  # 写入最终保留点数
        if 'value' in final_gdf.columns:  # 若存在原始土地覆盖值字段
            counts = final_gdf['value'].value_counts().sort_index()  # 统计各值的点数
            f.write("\n按土地覆盖值统计:\n")  # 写入统计标题
            for v, c in counts.items():  # 遍历各类别
                f.write(f"  值 {v}: {c} 点 ({c/len(final_gdf)*100:.2f}%)\n")  # 写入每类统计
    logging.info(f"统计报告已保存: {report_path}")  # 记录报告保存路径

    # 输出总耗时与速度
    elapsed = time.time() - start_time  # 计算总耗时
    logging.info(f"处理完成，总耗时: {elapsed:.2f} 秒")  # 记录耗时
    if elapsed > 0:  # 防止除零
        logging.info(f"平均处理速度: {len(points_gdf)/elapsed:.0f} 点/秒（包含并行与分块开销）")  # 记录速度估算

# 程序入口
if __name__ == '__main__':  # 判断是否为主模块执行
    try:  # 捕获整体执行异常
        main()  # 调用主函数
    except Exception as e:  # 捕获异常
        logging.error(f"程序运行出错: {str(e)}")  # 记录错误日志
        raise  # 抛出异常，方便外部定位问题