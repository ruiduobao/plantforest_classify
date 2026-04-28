#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
天然林样本点网格随机抽样脚本

功能说明：
本脚本实现基于区域网格的天然林样本点随机抽样功能：
1. 读取区域网格数据和天然林样本点数据
2. 将样本点与网格进行空间连接，确定每个点所属的网格
3. 用户输入每个网格需要保留的样本点数量
4. 在每个网格内进行随机抽样，如果点数不够则保留全部点
5. 输出抽样后的样本点数据

处理特点：
- 单进程处理，简单高效
- 基于空间连接的网格分组
- 灵活的抽样数量控制
- 详细的统计信息输出

输入数据：
- 区域网格数据（Shapefile格式）
- 天然林样本点数据（GPKG格式）

输出结果：
- 抽样后的天然林样本点数据（GPKG格式）
- 详细的抽样统计报告

作者：锐多宝 (ruiduobao)
创建时间：2024年
"""

import os  # 导入os用于路径和文件操作
import sys  # 导入sys用于系统交互
import logging  # 导入logging用于日志记录
from datetime import datetime  # 导入datetime用于时间戳
import random  # 导入random用于随机抽样
import numpy as np  # 导入numpy用于数值计算
import pandas as pd  # 导入pandas用于数据处理
import geopandas as gpd  # 导入geopandas用于地理数据处理
from shapely.geometry import Point  # 导入Point用于点几何处理
import warnings  # 导入warnings用于警告控制
warnings.filterwarnings('ignore')  # 忽略警告信息

# ===================== 配置参数 =====================
# 区域网格数据文件路径
GRID_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\分类网格数据\southeast_asia_grid_0.5deg.shp"  # 指定网格数据路径
# 天然林样本点数据文件路径
POINTS_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\2.非林地样本点\2.样本点筛选\nonforest_points_selected_20251009_190037.gpkg"  # 指定点数据路径
# 输出目录路径
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\2.非林地样本点\3.根据网格分配1000个样本"  # 指定输出目录
# 随机种子（保证结果可重现）
RANDOM_SEED = 42  # 设置随机种子

# ===================== 日志设置函数 =====================

def setup_logging(output_dir):  # 定义日志设置函数
    """设置日志记录，输出到文件和控制台"""  # 函数说明
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    log_file = os.path.join(output_dir, f"plantation_sample_points_grid_sampling_{timestamp}.log")  # 构造日志文件路径
    
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
    return log_file  # 返回日志文件路径

# ===================== 输入文件检查函数 =====================

def check_input_files():  # 定义输入文件检查函数
    """检查所有输入文件是否存在"""  # 函数说明
    files_to_check = [
        ("区域网格数据", GRID_FILE),
        ("天然林样本点数据", POINTS_FILE)
    ]  # 需要检查的文件列表
    
    for name, filepath in files_to_check:  # 遍历检查每个文件
        if not os.path.exists(filepath):  # 检查文件是否存在
            raise FileNotFoundError(f"{name}文件不存在: {filepath}")  # 如果不存在抛出异常
        logging.info(f"✓ {name}: {filepath}")  # 记录文件检查通过

# ===================== 数据读取函数 =====================

def load_data():  # 定义数据读取函数
    """读取网格数据和样本点数据"""  # 函数说明
    logging.info("开始读取数据...")  # 记录开始读取
    
    # 读取网格数据
    logging.info("读取区域网格数据...")  # 记录读取网格数据
    grid_gdf = gpd.read_file(GRID_FILE)  # 读取网格数据
    logging.info(f"网格数据读取完成，共 {len(grid_gdf)} 个网格，CRS: {grid_gdf.crs}")  # 记录网格数据信息
    
    # 读取样本点数据
    logging.info("读取landcover样本点数据...")  # 记录读取点数据
    points_gdf = gpd.read_file(POINTS_FILE)  # 读取点数据
    logging.info(f"样本点数据读取完成，共 {len(points_gdf)} 个点，CRS: {points_gdf.crs}")  # 记录点数据信息
    
    # 检查value属性是否存在
    if 'value' not in points_gdf.columns:  # 检查value列是否存在
        raise ValueError("样本点数据中缺少'value'属性列，请检查数据")  # 抛出异常
    
    # 统计地物类型分布
    logging.info("统计地物类型分布...")  # 记录统计开始
    landcover_stats = points_gdf['value'].value_counts().sort_index()  # 统计各地物类型数量
    logging.info("地物类型分布统计:")  # 记录统计标题
    for landcover_type, count in landcover_stats.items():  # 遍历统计结果
        percentage = (count / len(points_gdf)) * 100  # 计算百分比
        logging.info(f"  地物类型 {landcover_type}: {count} 个点 ({percentage:.2f}%)")  # 记录统计信息
    
    # 坐标系一致性检查
    if grid_gdf.crs != points_gdf.crs:  # 检查坐标系是否一致
        logging.info(f"坐标系不一致，将样本点从 {points_gdf.crs} 转换为 {grid_gdf.crs}")  # 记录坐标系转换
        points_gdf = points_gdf.to_crs(grid_gdf.crs)  # 转换点数据坐标系
        logging.info("坐标系转换完成")  # 记录转换完成
    
    return grid_gdf, points_gdf  # 返回网格和点数据

# ===================== 空间连接函数 =====================

def spatial_join_points_to_grids(grid_gdf, points_gdf):  # 定义空间连接函数
    """将样本点与网格进行空间连接，确定每个点所属的网格"""  # 函数说明
    logging.info("开始进行空间连接...")  # 记录开始空间连接
    
    # 进行空间连接，将点与网格关联
    joined_gdf = gpd.sjoin(points_gdf, grid_gdf, how='left', predicate='within')  # 空间连接，点在网格内
    
    # 检查是否有点没有匹配到网格
    unmatched_points = joined_gdf[joined_gdf['index_right'].isna()]  # 找出未匹配的点
    if len(unmatched_points) > 0:  # 如果有未匹配的点
        logging.warning(f"有 {len(unmatched_points)} 个点没有匹配到网格，这些点将被排除")  # 记录警告
        joined_gdf = joined_gdf.dropna(subset=['index_right'])  # 删除未匹配的点
    
    logging.info(f"空间连接完成，{len(joined_gdf)} 个点成功匹配到网格")  # 记录连接结果
    
    return joined_gdf  # 返回连接后的数据

# ===================== 用户输入函数 =====================

def get_sampling_parameters(grid_gdf, joined_gdf):  # 定义获取抽样参数函数
    """获取用户输入的抽样参数，按每个网格内地物类型比例分配"""  # 函数说明
    logging.info("统计每个网格的样本点数量和地物类型分布...")  # 记录统计开始
    
    # 统计每个网格内的点数和地物类型分布
    grid_stats = []  # 初始化网格统计列表
    
    for grid_idx in joined_gdf['index_right'].unique():  # 遍历每个网格
        if pd.isna(grid_idx):  # 跳过空值
            continue
        
        grid_points = joined_gdf[joined_gdf['index_right'] == grid_idx]  # 获取该网格的所有点
        grid_id = int(grid_idx)  # 网格ID
        total_points = len(grid_points)  # 该网格总点数
        
        # 统计该网格内各地物类型分布
        landcover_stats = grid_points['value'].value_counts().sort_index()  # 统计各地物类型数量
        
        grid_info = {
            'grid_id': grid_id,
            'total_points': total_points,
            'landcover_distribution': landcover_stats.to_dict(),
            'landcover_proportions': (landcover_stats / total_points).to_dict()
        }
        grid_stats.append(grid_info)
    
    # 显示网格统计信息
    print("\n" + "="*100)  # 打印分隔线
    print("网格样本点统计信息:")  # 打印标题
    print("="*100)  # 打印分隔线
    print(f"{'网格ID':<8} {'总点数':<8} {'地物类型分布':<50} {'建议抽样数':<12}")  # 打印表头
    print("-"*100)  # 打印分隔线
    
    for grid_info in grid_stats:  # 遍历每个网格
        grid_id = grid_info['grid_id']  # 网格ID
        total_points = grid_info['total_points']  # 总点数
        landcover_dist = grid_info['landcover_distribution']  # 地物类型分布
        
        # 构建地物类型分布字符串
        dist_str = ", ".join([f"{lc}:{count}" for lc, count in landcover_dist.items()])  # 地物类型分布字符串
        if len(dist_str) > 45:  # 如果字符串太长
            dist_str = dist_str[:42] + "..."  # 截断
        
        suggested_sample = min(total_points, max(10, total_points // 10))  # 建议抽样数（总点数的10%，至少10个）
        print(f"{grid_id:<8} {total_points:<8} {dist_str:<50} {suggested_sample:<12}")  # 打印网格信息
    
    print("="*100)  # 打印分隔线
    print(f"总网格数: {len(grid_stats)}")  # 打印总网格数
    print(f"总样本点数: {len(joined_gdf)}")  # 打印总点数
    print("="*100)  # 打印分隔线
    
    # 获取用户输入
    print("\n请选择抽样方式:")  # 提示用户选择
    print("1. 自动模式 - 每个网格抽样10%的点（最少10个，最多不超过该网格总点数）")  # 选项1
    print("2. 统一模式 - 每个网格抽样相同数量的点")  # 选项2
    print("3. 自定义模式 - 为每个网格单独设置抽样数量")  # 选项3
    
    mode = input("请选择模式 (1/2/3): ").strip()  # 获取用户选择
    
    sampling_dict = {}  # 初始化抽样字典
    
    if mode == "1":  # 自动模式
        logging.info("选择自动模式")  # 记录选择
        for grid_info in grid_stats:  # 遍历每个网格
            grid_id = grid_info['grid_id']  # 网格ID
            total_points = grid_info['total_points']  # 总点数
            target_count = min(total_points, max(10, total_points // 10))  # 目标抽样数
            
            # 按该网格内地物类型比例分配
            grid_sampling = {}  # 该网格的抽样分配
            landcover_proportions = grid_info['landcover_proportions']  # 地物类型比例
            
            allocated_total = 0  # 已分配总数
            for landcover_type, proportion in landcover_proportions.items():  # 遍历地物类型比例
                # 按比例计算该地物类型的目标数量
                proportional_target = max(1, int(target_count * proportion))  # 按比例计算，至少1个
                available_count = grid_info['landcover_distribution'][landcover_type]  # 该类型可用数量
                final_target = min(proportional_target, available_count)  # 不超过可用数量
                
                grid_sampling[landcover_type] = final_target  # 设置目标数量
                allocated_total += final_target  # 累加已分配数量
            
            # 如果分配总数超过目标，按比例缩减
            if allocated_total > target_count:  # 如果超过目标
                scale_factor = target_count / allocated_total  # 计算缩减比例
                for landcover_type in grid_sampling:  # 遍历地物类型
                    original_target = grid_sampling[landcover_type]  # 原始目标
                    adjusted_target = max(1, int(original_target * scale_factor))  # 调整后目标，至少1个
                    available_count = grid_info['landcover_distribution'][landcover_type]  # 可用数量
                    grid_sampling[landcover_type] = min(adjusted_target, available_count)  # 更新目标
            
            sampling_dict[grid_id] = grid_sampling  # 设置该网格的抽样分配
            
            logging.info(f"网格 {grid_id}: 总点数 {total_points}, 目标抽样 {target_count}, "
                        f"分配: {grid_sampling}")  # 记录分配信息
    
    elif mode == "2":  # 统一模式
        uniform_count = int(input("请输入每个网格的抽样数量: ").strip())  # 获取统一抽样数量
        logging.info(f"选择统一模式，每个网格抽样 {uniform_count} 个点")  # 记录选择
        
        for grid_info in grid_stats:  # 遍历每个网格
            grid_id = grid_info['grid_id']  # 网格ID
            total_points = grid_info['total_points']  # 总点数
            target_count = min(uniform_count, total_points)  # 目标抽样数，不超过总点数
            
            # 按该网格内地物类型比例分配
            grid_sampling = {}  # 该网格的抽样分配
            landcover_proportions = grid_info['landcover_proportions']  # 地物类型比例
            
            allocated_total = 0  # 已分配总数
            for landcover_type, proportion in landcover_proportions.items():  # 遍历地物类型比例
                # 按比例计算该地物类型的目标数量
                proportional_target = max(1, int(target_count * proportion))  # 按比例计算，至少1个
                available_count = grid_info['landcover_distribution'][landcover_type]  # 该类型可用数量
                final_target = min(proportional_target, available_count)  # 不超过可用数量
                
                grid_sampling[landcover_type] = final_target  # 设置目标数量
                allocated_total += final_target  # 累加已分配数量
            
            # 如果分配总数超过目标，按比例缩减
            if allocated_total > target_count:  # 如果超过目标
                scale_factor = target_count / allocated_total  # 计算缩减比例
                for landcover_type in grid_sampling:  # 遍历地物类型
                    original_target = grid_sampling[landcover_type]  # 原始目标
                    adjusted_target = max(1, int(original_target * scale_factor))  # 调整后目标，至少1个
                    available_count = grid_info['landcover_distribution'][landcover_type]  # 可用数量
                    grid_sampling[landcover_type] = min(adjusted_target, available_count)  # 更新目标
            
            sampling_dict[grid_id] = grid_sampling  # 设置该网格的抽样分配
            
            logging.info(f"网格 {grid_id}: 总点数 {total_points}, 目标抽样 {target_count}, "
                        f"分配: {grid_sampling}")  # 记录分配信息
    
    else:  # 自定义模式
        logging.info("选择自定义模式")  # 记录选择
        for grid_info in grid_stats:  # 遍历每个网格
            grid_id = grid_info['grid_id']  # 网格ID
            total_points = grid_info['total_points']  # 总点数
            
            print(f"\n网格 {grid_id} (总点数: {total_points})")  # 打印网格信息
            print(f"地物类型分布: {grid_info['landcover_distribution']}")  # 打印地物类型分布
            
            while True:  # 循环获取有效输入
                try:  # 尝试获取输入
                    target_count = int(input(f"请输入网格 {grid_id} 的抽样数量 (最大 {total_points}): ").strip())  # 获取目标数量
                    if 0 < target_count <= total_points:  # 检查输入有效性
                        break  # 跳出循环
                    else:  # 输入无效
                        print(f"请输入1到{total_points}之间的数字")  # 提示重新输入
                except ValueError:  # 捕获输入错误
                    print("请输入有效的数字")  # 提示重新输入
            
            # 按该网格内地物类型比例分配
            grid_sampling = {}  # 该网格的抽样分配
            landcover_proportions = grid_info['landcover_proportions']  # 地物类型比例
            
            allocated_total = 0  # 已分配总数
            for landcover_type, proportion in landcover_proportions.items():  # 遍历地物类型比例
                # 按比例计算该地物类型的目标数量
                proportional_target = max(1, int(target_count * proportion))  # 按比例计算，至少1个
                available_count = grid_info['landcover_distribution'][landcover_type]  # 该类型可用数量
                final_target = min(proportional_target, available_count)  # 不超过可用数量
                
                grid_sampling[landcover_type] = final_target  # 设置目标数量
                allocated_total += final_target  # 累加已分配数量
            
            # 如果分配总数超过目标，按比例缩减
            if allocated_total > target_count:  # 如果超过目标
                scale_factor = target_count / allocated_total  # 计算缩减比例
                for landcover_type in grid_sampling:  # 遍历地物类型
                    original_target = grid_sampling[landcover_type]  # 原始目标
                    adjusted_target = max(1, int(original_target * scale_factor))  # 调整后目标，至少1个
                    available_count = grid_info['landcover_distribution'][landcover_type]  # 可用数量
                    grid_sampling[landcover_type] = min(adjusted_target, available_count)  # 更新目标
            
            sampling_dict[grid_id] = grid_sampling  # 设置该网格的抽样分配
            
            logging.info(f"网格 {grid_id}: 总点数 {total_points}, 目标抽样 {target_count}, "
                        f"分配: {grid_sampling}")  # 记录分配信息
    
    return sampling_dict  # 返回抽样字典

# ===================== 随机抽样函数 =====================

def perform_landcover_sampling(joined_gdf, sampling_dict):  # 定义按网格内地物类型分层抽样函数
    """
    按网格内地物类型进行分层随机抽样
    
    参数:
    - joined_gdf: 包含网格信息的样本点数据
    - sampling_dict: 每个网格的地物类型抽样分配字典 {grid_id: {landcover_type: target_count}}
    
    返回:
    - sampled_gdf: 抽样后的数据
    - sampling_stats: 抽样统计信息
    """  # 函数说明
    logging.info(f"开始按网格内地物类型进行分层随机抽样，随机种子: {RANDOM_SEED}")  # 记录抽样开始
    
    # 设置随机种子
    random.seed(RANDOM_SEED)  # 设置随机种子
    np.random.seed(RANDOM_SEED)  # 设置numpy随机种子
    
    sampled_points = []  # 初始化抽样结果列表
    sampling_stats = {  # 初始化抽样统计信息
        'total_original_points': len(joined_gdf),  # 原始总点数
        'total_sampled_points': 0,  # 抽样总点数
        'grid_stats': {},  # 网格统计信息
        'landcover_stats': {},  # 地物类型统计信息
        'random_seed': RANDOM_SEED  # 随机种子
    }
    
    # 按网格进行抽样
    for grid_id, landcover_targets in sampling_dict.items():  # 遍历每个网格的抽样分配
        logging.info(f"处理网格 {grid_id}，地物类型目标: {landcover_targets}")  # 记录当前处理网格
        
        # 获取该网格的所有点
        grid_points = joined_gdf[joined_gdf['index_right'] == grid_id].copy()  # 获取网格内所有点
        
        if len(grid_points) == 0:  # 如果网格内没有点
            logging.warning(f"网格 {grid_id} 内没有样本点")  # 记录警告
            sampling_stats['grid_stats'][grid_id] = {  # 记录网格统计
                'original_points': 0,
                'sampled_points': 0,
                'landcover_distribution': {},
                'sampling_details': {}
            }
            continue  # 跳过该网格
        
        grid_sampled_points = []  # 该网格的抽样结果
        grid_sampling_details = {}  # 该网格的抽样详情
        
        # 按地物类型进行抽样
        for landcover_type, target_count in landcover_targets.items():  # 遍历地物类型目标
            # 获取该地物类型的所有点
            landcover_points = grid_points[grid_points['value'] == landcover_type]  # 获取该地物类型的点
            available_count = len(landcover_points)  # 可用点数
            
            if available_count == 0:  # 如果该地物类型没有点
                logging.warning(f"网格 {grid_id} 中地物类型 {landcover_type} 没有样本点")  # 记录警告
                grid_sampling_details[landcover_type] = {  # 记录抽样详情
                    'available': 0,
                    'target': target_count,
                    'sampled': 0,
                    'sampling_rate': 0.0
                }
                continue  # 跳过该地物类型
            
            # 确定实际抽样数量
            actual_sample_count = min(target_count, available_count)  # 实际抽样数量
            
            if actual_sample_count == available_count:  # 如果目标数量大于等于可用数量
                # 保留所有点
                selected_points = landcover_points.copy()  # 保留所有点
                logging.info(f"网格 {grid_id} 地物类型 {landcover_type}: 保留所有 {available_count} 个点")  # 记录保留信息
            else:  # 如果需要随机抽样
                # 随机抽样
                selected_indices = np.random.choice(  # 随机选择索引
                    landcover_points.index, 
                    size=actual_sample_count, 
                    replace=False
                )
                selected_points = landcover_points.loc[selected_indices].copy()  # 获取选中的点
                logging.info(f"网格 {grid_id} 地物类型 {landcover_type}: 从 {available_count} 个点中随机抽样 {actual_sample_count} 个")  # 记录抽样信息
            
            grid_sampled_points.append(selected_points)  # 添加到网格抽样结果
            
            # 记录抽样详情
            sampling_rate = actual_sample_count / available_count if available_count > 0 else 0  # 计算抽样率
            grid_sampling_details[landcover_type] = {  # 记录抽样详情
                'available': available_count,
                'target': target_count,
                'sampled': actual_sample_count,
                'sampling_rate': sampling_rate
            }
            
            # 更新地物类型统计
            if landcover_type not in sampling_stats['landcover_stats']:  # 如果地物类型不在统计中
                sampling_stats['landcover_stats'][landcover_type] = {  # 初始化地物类型统计
                    'original_points': 0,
                    'sampled_points': 0
                }
            sampling_stats['landcover_stats'][landcover_type]['original_points'] += available_count  # 累加原始点数
            sampling_stats['landcover_stats'][landcover_type]['sampled_points'] += actual_sample_count  # 累加抽样点数
        
        # 合并该网格的抽样结果
        if grid_sampled_points:  # 如果有抽样结果
            grid_sampled_gdf = pd.concat(grid_sampled_points, ignore_index=True)  # 合并抽样结果
            sampled_points.append(grid_sampled_gdf)  # 添加到总抽样结果
            
            # 记录网格统计信息
            sampling_stats['grid_stats'][grid_id] = {  # 记录网格统计
                'original_points': len(grid_points),
                'sampled_points': len(grid_sampled_gdf),
                'landcover_distribution': grid_points['value'].value_counts().to_dict(),
                'sampling_details': grid_sampling_details
            }
            
            logging.info(f"网格 {grid_id} 抽样完成: 原始 {len(grid_points)} 个点，抽样 {len(grid_sampled_gdf)} 个点")  # 记录网格抽样完成
        else:  # 如果没有抽样结果
            logging.warning(f"网格 {grid_id} 没有抽样到任何点")  # 记录警告
            sampling_stats['grid_stats'][grid_id] = {  # 记录网格统计
                'original_points': len(grid_points),
                'sampled_points': 0,
                'landcover_distribution': grid_points['value'].value_counts().to_dict(),
                'sampling_details': grid_sampling_details
            }
    
    # 合并所有抽样结果
    if sampled_points:  # 如果有抽样结果
        final_sampled_gdf = pd.concat(sampled_points, ignore_index=True)  # 合并所有抽样结果
        sampling_stats['total_sampled_points'] = len(final_sampled_gdf)  # 更新抽样总点数
        
        # 统计最终抽样结果中各地物类型的分布
        final_landcover_distribution = final_sampled_gdf['value'].value_counts().sort_index()  # 统计最终地物类型分布
        sampling_stats['final_landcover_distribution'] = final_landcover_distribution.to_dict()  # 记录最终分布
        
        logging.info(f"抽样完成: 原始 {len(joined_gdf)} 个点，抽样 {len(final_sampled_gdf)} 个点")  # 记录抽样完成
        logging.info(f"最终地物类型分布: {sampling_stats['final_landcover_distribution']}")  # 记录最终分布
        
        # 统计最终结果中各地物类型的分布
        final_landcover_stats = final_sampled_gdf['value'].value_counts().sort_index()  # 统计最终各地物类型数量
        logging.info("最终抽样结果中各地物类型分布:")  # 记录最终分布标题
        for landcover_type, count in final_landcover_stats.items():  # 遍历最终分布
            percentage = (count / len(final_sampled_gdf)) * 100  # 计算百分比
            logging.info(f"  地物类型 {landcover_type}: {count} 个点 ({percentage:.2f}%)")  # 记录最终分布
        
        return final_sampled_gdf, sampling_stats  # 返回抽样结果和统计信息
    else:  # 如果没有抽样结果
        logging.error("没有抽样到任何点")  # 记录错误
        return gpd.GeoDataFrame(), sampling_stats  # 返回空结果

# ===================== 结果保存函数 =====================

def save_results(sampled_gdf, sampling_stats, output_dir):  # 定义结果保存函数
    """保存抽样结果和统计报告"""  # 函数说明
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    
    # 保存抽样后的样本点数据
    if len(sampled_gdf) > 0:  # 如果有抽样结果
        output_points_file = os.path.join(output_dir, f"landcover_points_sampled_{timestamp}.gpkg")  # 构造输出文件路径
        sampled_gdf.to_file(output_points_file, driver='GPKG')  # 保存为GPKG格式
        logging.info(f"抽样结果已保存: {output_points_file}")  # 记录保存完成
    else:  # 如果没有抽样结果
        output_points_file = None  # 设置为None
        logging.warning("没有抽样结果可保存")  # 记录警告
    
    # 保存统计报告
    stats_file = os.path.join(output_dir, f"landcover_sampling_statistics_{timestamp}.txt")  # 构造统计文件路径
    with open(stats_file, 'w', encoding='utf-8') as f:  # 打开统计文件
        f.write("Landcover样本点网格内分层随机抽样统计报告\n")  # 写入标题
        f.write("="*80 + "\n")  # 写入分隔线
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 写入处理时间
        f.write(f"随机种子: {sampling_stats['random_seed']}\n")  # 写入随机种子
        f.write(f"原始样本点总数: {sampling_stats['total_original_points']:,}\n")  # 写入原始总数
        f.write(f"抽样后样本点总数: {sampling_stats['total_sampled_points']:,}\n")  # 写入抽样总数
        
        # 计算总体抽样率
        overall_sampling_rate = (sampling_stats['total_sampled_points'] / 
                               sampling_stats['total_original_points'] * 100) if sampling_stats['total_original_points'] > 0 else 0
        f.write(f"总体抽样率: {overall_sampling_rate:.2f}%\n\n")  # 写入总体抽样率
        
        # 网格统计信息
        f.write("="*80 + "\n")  # 写入分隔线
        f.write("各网格抽样统计\n")  # 写入网格统计标题
        f.write("="*80 + "\n")  # 写入分隔线
        f.write(f"总网格数: {len(sampling_stats['grid_stats'])}\n\n")  # 写入总网格数
        
        # 详细网格信息
        for grid_id, grid_stat in sampling_stats['grid_stats'].items():  # 遍历网格统计
            f.write(f"网格 {grid_id}:\n")  # 写入网格ID
            f.write(f"  原始点数: {grid_stat['original_points']:,}\n")  # 写入原始点数
            f.write(f"  抽样点数: {grid_stat['sampled_points']:,}\n")  # 写入抽样点数
            
            # 计算网格抽样率
            grid_sampling_rate = (grid_stat['sampled_points'] / grid_stat['original_points'] * 100) if grid_stat['original_points'] > 0 else 0
            f.write(f"  抽样率: {grid_sampling_rate:.2f}%\n")  # 写入抽样率
            
            # 地物类型分布
            f.write(f"  地物类型分布:\n")  # 写入地物类型分布标题
            for landcover_type, count in grid_stat['landcover_distribution'].items():  # 遍历地物类型分布
                percentage = (count / grid_stat['original_points'] * 100) if grid_stat['original_points'] > 0 else 0
                f.write(f"    地物类型 {landcover_type}: {count:,} 个点 ({percentage:.2f}%)\n")  # 写入地物类型信息
            
            # 抽样详情
            f.write(f"  抽样详情:\n")  # 写入抽样详情标题
            for landcover_type, detail in grid_stat['sampling_details'].items():  # 遍历抽样详情
                f.write(f"    地物类型 {landcover_type}:\n")  # 写入地物类型
                f.write(f"      可用点数: {detail['available']:,}\n")  # 写入可用点数
                f.write(f"      目标点数: {detail['target']:,}\n")  # 写入目标点数
                f.write(f"      抽样点数: {detail['sampled']:,}\n")  # 写入抽样点数
                f.write(f"      抽样率: {detail['sampling_rate']*100:.2f}%\n")  # 写入抽样率
            f.write("\n")  # 写入空行
        
        # 全局地物类型统计
        f.write("="*80 + "\n")  # 写入分隔线
        f.write("全局地物类型统计\n")  # 写入全局统计标题
        f.write("="*80 + "\n")  # 写入分隔线
        f.write(f"地物类型数: {len(sampling_stats['landcover_stats'])}\n\n")  # 写入地物类型数
        
        for landcover_type, stats in sampling_stats['landcover_stats'].items():  # 遍历地物类型统计
            original_points = stats['original_points']  # 原始点数
            sampled_points = stats['sampled_points']  # 抽样点数
            sampling_rate = (sampled_points / original_points * 100) if original_points > 0 else 0  # 抽样率
            
            f.write(f"地物类型 {landcover_type}:\n")  # 写入地物类型
            f.write(f"  原始点数: {original_points:,}\n")  # 写入原始点数
            f.write(f"  抽样点数: {sampled_points:,}\n")  # 写入抽样点数
            f.write(f"  抽样率: {sampling_rate:.2f}%\n\n")  # 写入抽样率
        
        # 最终抽样结果分布
        f.write("="*80 + "\n")  # 写入分隔线
        f.write("最终抽样结果中各地物类型分布\n")  # 写入最终分布标题
        f.write("="*80 + "\n")  # 写入分隔线
        
        if 'final_landcover_distribution' in sampling_stats:  # 如果有最终分布统计
            for landcover_type, count in sampling_stats['final_landcover_distribution'].items():  # 遍历最终分布
                percentage = (count / sampling_stats['total_sampled_points'] * 100) if sampling_stats['total_sampled_points'] > 0 else 0
                f.write(f"地物类型 {landcover_type}: {count:,} 个点 ({percentage:.2f}%)\n")  # 写入最终分布信息
        
        f.write("\n" + "="*80 + "\n")  # 写入结束分隔线
        f.write("报告生成完成\n")  # 写入完成信息
        f.write("="*80 + "\n")  # 写入分隔线
    
    logging.info(f"统计报告已保存: {stats_file}")  # 记录统计报告保存完成
    
    return output_points_file, stats_file  # 返回输出文件路径

# ===================== 主函数 =====================

def main():  # 定义主函数
    """主处理流程"""  # 函数说明
    try:  # 开始异常捕获
        # 设置日志
        log_file = setup_logging(OUTPUT_DIR)  # 设置日志
        
        logging.info("开始Landcover样本点网格内分层抽样处理...")  # 记录开始处理
        logging.info(f"随机种子: {RANDOM_SEED}")  # 记录随机种子
        
        # 检查输入文件
        check_input_files()  # 检查输入文件
        
        # 读取数据
        grid_gdf, points_gdf = load_data()  # 读取数据
        
        # 空间连接
        joined_gdf = spatial_join_points_to_grids(grid_gdf, points_gdf)  # 空间连接
        
        if len(joined_gdf) == 0:  # 如果没有连接结果
            logging.error("没有样本点与网格匹配，请检查数据")  # 记录错误
            return  # 退出函数
        
        # 获取抽样参数
        sampling_dict = get_sampling_parameters(grid_gdf, joined_gdf)  # 获取抽样参数
        
        # 进行抽样
        sampled_gdf, sampling_stats = perform_landcover_sampling(joined_gdf, sampling_dict)  # 进行抽样
        
        # 保存结果
        output_points_file, stats_file = save_results(sampled_gdf, sampling_stats, OUTPUT_DIR)  # 保存结果
        
        # 输出完成信息
        print("\n" + "="*60)  # 打印分隔线
        print("处理完成!")  # 打印完成信息
        print("="*60)  # 打印分隔线
        if output_points_file:  # 如果有输出文件
            print(f"抽样结果文件: {output_points_file}")  # 打印输出文件路径
        print(f"统计报告文件: {stats_file}")  # 打印统计文件路径
        print(f"日志文件: {log_file}")  # 打印日志文件路径
        print("="*60)  # 打印分隔线
        
        logging.info("Landcover样本点网格内分层抽样处理完成!")  # 记录处理完成
        
    except Exception as e:  # 捕获异常
        logging.error(f"处理过程中发生错误: {str(e)}")  # 记录错误
        raise  # 重新抛出异常

if __name__ == "__main__":  # 如果是主程序
    main()  # 调用主函数