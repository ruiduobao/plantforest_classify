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
POINTS_FILE = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\3.自然林样本点\4.经过了FD和CCDCyear的筛选的自然林\2800万个自然林点_经过了breaks的筛选.gpkg"  # 指定点数据路径
# 输出目录路径
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\交叉验证集\3.自然林样本点\5.按照网格筛选200个点"  # 指定输出目录
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
    logging.info("读取天然林样本点数据...")  # 记录读取点数据
    points_gdf = gpd.read_file(POINTS_FILE)  # 读取点数据
    logging.info(f"样本点数据读取完成，共 {len(points_gdf)} 个点，CRS: {points_gdf.crs}")  # 记录点数据信息
    
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
    """获取用户输入的抽样参数"""  # 函数说明
    logging.info("统计每个网格的样本点数量...")  # 记录统计开始
    
    # 统计每个网格内的点数
    grid_point_counts = joined_gdf.groupby('index_right').size().reset_index(name='point_count')  # 按网格统计点数
    grid_point_counts['grid_id'] = grid_point_counts['index_right']  # 添加网格ID列
    
    # 显示网格统计信息
    print("\n" + "="*60)  # 打印分隔线
    print("网格样本点统计信息:")  # 打印标题
    print("="*60)  # 打印分隔线
    print(f"{'网格ID':<10} {'当前点数':<10} {'建议抽样数':<12}")  # 打印表头
    print("-"*60)  # 打印分隔线
    
    for _, row in grid_point_counts.iterrows():  # 遍历每个网格
        grid_id = int(row['grid_id'])  # 网格ID
        point_count = int(row['point_count'])  # 当前点数
        suggested_sample = min(point_count, max(1, point_count // 2))  # 建议抽样数（当前点数的一半，至少1个）
        print(f"{grid_id:<10} {point_count:<10} {suggested_sample:<12}")  # 打印网格信息
    
    print("="*60)  # 打印分隔线
    print(f"总网格数: {len(grid_point_counts)}")  # 打印总网格数
    print(f"总样本点数: {len(joined_gdf)}")  # 打印总点数
    print("="*60)  # 打印分隔线
    
    # 获取用户输入
    print("\n请输入每个网格需要保留的样本点数量:")  # 提示用户输入
    print("格式: 网格ID:数量,网格ID:数量,... (例如: 1:10,2:15,3:8)")  # 输入格式说明
    print("或者输入 'auto' 使用建议的抽样数量")  # 自动选项说明
    print("或者输入一个数字作为所有网格的统一抽样数量")  # 统一数量选项说明
    
    user_input = input("请输入: ").strip()  # 获取用户输入
    
    sampling_dict = {}  # 初始化抽样字典
    
    if user_input.lower() == 'auto':  # 如果选择自动模式
        logging.info("使用自动建议的抽样数量")  # 记录自动模式
        for _, row in grid_point_counts.iterrows():  # 遍历每个网格
            grid_id = int(row['grid_id'])  # 网格ID
            point_count = int(row['point_count'])  # 当前点数
            suggested_sample = min(point_count, max(1, point_count // 2))  # 建议抽样数
            sampling_dict[grid_id] = suggested_sample  # 添加到抽样字典
    elif user_input.isdigit():  # 如果输入的是数字
        uniform_count = int(user_input)  # 统一抽样数量
        logging.info(f"使用统一抽样数量: {uniform_count}")  # 记录统一数量
        for _, row in grid_point_counts.iterrows():  # 遍历每个网格
            grid_id = int(row['grid_id'])  # 网格ID
            sampling_dict[grid_id] = uniform_count  # 设置统一数量
    else:  # 如果是自定义输入
        try:  # 尝试解析输入
            pairs = user_input.split(',')  # 按逗号分割
            for pair in pairs:  # 遍历每个配对
                grid_id, count = pair.split(':')  # 按冒号分割
                sampling_dict[int(grid_id.strip())] = int(count.strip())  # 添加到抽样字典
            logging.info("使用用户自定义的抽样数量")  # 记录自定义模式
        except Exception as e:  # 捕获解析异常
            logging.error(f"输入格式错误: {e}")  # 记录错误
            raise ValueError("输入格式错误，请检查格式")  # 抛出异常
    
    return sampling_dict  # 返回抽样字典

# ===================== 随机抽样函数 =====================

def perform_grid_sampling(joined_gdf, sampling_dict):  # 定义网格抽样函数
    """在每个网格内进行随机抽样"""  # 函数说明
    logging.info("开始进行网格内随机抽样...")  # 记录开始抽样
    
    # 设置随机种子
    random.seed(RANDOM_SEED)  # 设置随机种子
    np.random.seed(RANDOM_SEED)  # 设置numpy随机种子
    
    sampled_points = []  # 初始化抽样结果列表
    sampling_stats = []  # 初始化统计信息列表
    
    # 按网格分组进行抽样
    for grid_id, group in joined_gdf.groupby('index_right'):  # 按网格分组
        grid_id = int(grid_id)  # 转换网格ID为整数
        current_count = len(group)  # 当前网格的点数
        target_count = sampling_dict.get(grid_id, current_count)  # 目标抽样数量
        
        if target_count >= current_count:  # 如果目标数量大于等于当前点数
            # 保留所有点
            sampled_group = group.copy()  # 复制所有点
            actual_count = current_count  # 实际抽样数等于当前点数
            logging.info(f"网格 {grid_id}: 目标 {target_count} 个点，当前 {current_count} 个点，保留全部")  # 记录保留全部
        else:  # 如果需要抽样
            # 随机抽样
            sampled_group = group.sample(n=target_count, random_state=RANDOM_SEED)  # 随机抽样
            actual_count = target_count  # 实际抽样数等于目标数量
            logging.info(f"网格 {grid_id}: 目标 {target_count} 个点，当前 {current_count} 个点，抽样 {actual_count} 个")  # 记录抽样结果
        
        sampled_points.append(sampled_group)  # 添加抽样结果
        
        # 记录统计信息
        sampling_stats.append({
            'grid_id': grid_id,  # 网格ID
            'original_count': current_count,  # 原始点数
            'target_count': target_count,  # 目标点数
            'sampled_count': actual_count,  # 实际抽样点数
            'sampling_rate': actual_count / current_count if current_count > 0 else 0  # 抽样率
        })
    
    # 合并所有抽样结果
    if sampled_points:  # 如果有抽样结果
        final_sampled_gdf = pd.concat(sampled_points, ignore_index=True)  # 合并所有抽样结果
        logging.info(f"抽样完成，总共保留 {len(final_sampled_gdf)} 个样本点")  # 记录抽样完成
    else:  # 如果没有抽样结果
        final_sampled_gdf = gpd.GeoDataFrame()  # 创建空的GeoDataFrame
        logging.warning("没有抽样结果")  # 记录警告
    
    return final_sampled_gdf, sampling_stats  # 返回抽样结果和统计信息

# ===================== 结果保存函数 =====================

def save_results(sampled_gdf, sampling_stats, output_dir):  # 定义结果保存函数
    """保存抽样结果和统计报告"""  # 函数说明
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    
    # 保存抽样后的样本点数据
    if len(sampled_gdf) > 0:  # 如果有抽样结果
        output_points_file = os.path.join(output_dir, f"natural_forest_points_sampled_{timestamp}.gpkg")  # 构造输出文件路径
        sampled_gdf.to_file(output_points_file, driver='GPKG')  # 保存为GPKG格式
        logging.info(f"抽样结果已保存: {output_points_file}")  # 记录保存完成
    else:  # 如果没有抽样结果
        output_points_file = None  # 设置为None
        logging.warning("没有抽样结果可保存")  # 记录警告
    
    # 保存统计报告
    stats_file = os.path.join(output_dir, f"sampling_statistics_{timestamp}.txt")  # 构造统计文件路径
    with open(stats_file, 'w', encoding='utf-8') as f:  # 打开统计文件
        f.write("天然林样本点网格随机抽样统计报告\n")  # 写入标题
        f.write("="*50 + "\n")  # 写入分隔线
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 写入处理时间
        f.write(f"随机种子: {RANDOM_SEED}\n")  # 写入随机种子
        f.write("\n")  # 写入空行
        
        # 写入总体统计
        total_original = sum(stat['original_count'] for stat in sampling_stats)  # 计算总原始点数
        total_sampled = sum(stat['sampled_count'] for stat in sampling_stats)  # 计算总抽样点数
        overall_rate = total_sampled / total_original if total_original > 0 else 0  # 计算总体抽样率
        
        f.write("总体统计:\n")  # 写入总体统计标题
        f.write(f"  总网格数: {len(sampling_stats)}\n")  # 写入总网格数
        f.write(f"  原始样本点总数: {total_original}\n")  # 写入原始点总数
        f.write(f"  抽样后样本点总数: {total_sampled}\n")  # 写入抽样点总数
        f.write(f"  总体抽样率: {overall_rate:.2%}\n")  # 写入总体抽样率
        f.write("\n")  # 写入空行
        
        # 写入详细统计
        f.write("各网格详细统计:\n")  # 写入详细统计标题
        f.write(f"{'网格ID':<8} {'原始点数':<8} {'目标点数':<8} {'抽样点数':<8} {'抽样率':<8}\n")  # 写入表头
        f.write("-" * 50 + "\n")  # 写入分隔线
        
        for stat in sampling_stats:  # 遍历统计信息
            f.write(f"{stat['grid_id']:<8} {stat['original_count']:<8} {stat['target_count']:<8} "
                   f"{stat['sampled_count']:<8} {stat['sampling_rate']:<8.2%}\n")  # 写入统计行
    
    logging.info(f"统计报告已保存: {stats_file}")  # 记录统计报告保存完成
    
    return output_points_file, stats_file  # 返回输出文件路径

# ===================== 主函数 =====================

def main():  # 定义主函数
    """主处理流程"""  # 函数说明
    try:  # 开始异常捕获
        # 设置日志
        log_file = setup_logging(OUTPUT_DIR)  # 设置日志
        
        logging.info("开始天然林样本点网格随机抽样处理...")  # 记录开始处理
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
        sampled_gdf, sampling_stats = perform_grid_sampling(joined_gdf, sampling_dict)  # 进行抽样
        
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
        
        logging.info("天然林样本点网格随机抽样处理完成!")  # 记录处理完成
        
    except Exception as e:  # 捕获异常
        logging.error(f"处理过程中发生错误: {str(e)}")  # 记录错误
        raise  # 重新抛出异常

if __name__ == "__main__":  # 如果是主程序
    main()  # 调用主函数