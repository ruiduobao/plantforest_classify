#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDPT数据根据国家属性筛选东南亚国家
目的：利用SDPT数据中的country字段直接筛选出东南亚国家的人工林数据
作者：锐多宝 (ruiduobao)
创建时间：2025年1月6日
"""

import geopandas as gpd
import pandas as pd
import os
import logging
from datetime import datetime
from collections import Counter
import numpy as np

def setup_logging(output_dir):
    """
    设置日志记录系统
    Args:
        output_dir: 输出目录路径
    Returns:
        logger: 配置好的日志记录器
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"sdpt_country_filter_log_{timestamp}.txt")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件输出
            logging.StreamHandler()  # 控制台输出
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存至: {log_file}")
    return logger

def explore_country_attributes(sdpt_path, logger):
    """
    探索SDPT数据中的国家属性值
    Args:
        sdpt_path: SDPT数据文件路径
        logger: 日志记录器
    Returns:
        gdf: 读取的GeoDataFrame
        country_stats: 国家统计信息
    """
    logger.info("开始读取SDPT数据...")
    logger.info(f"数据路径: {sdpt_path}")
    
    try:
        # 读取SDPT数据
        gdf = gpd.read_file(sdpt_path)
        logger.info(f"成功读取SDPT数据，共 {len(gdf)} 条记录")
        
        # 查看数据结构
        logger.info("数据列名:")
        for col in gdf.columns:
            logger.info(f"  - {col}")
        
        # 检查是否存在country字段
        if 'country' not in gdf.columns:
            logger.error("数据中未找到'country'字段！")
            # 查看是否有类似的字段
            country_like_cols = [col for col in gdf.columns if 'country' in col.lower() or 'nation' in col.lower()]
            if country_like_cols:
                logger.info(f"发现可能的国家字段: {country_like_cols}")
            return None, None
        
        # 统计country字段的值
        logger.info("\n开始分析country字段...")
        country_values = gdf['country'].dropna()  # 去除空值
        country_stats = Counter(country_values)
        
        logger.info(f"country字段统计（共 {len(country_stats)} 个不同国家）:")
        logger.info(f"总记录数: {len(gdf)}")
        logger.info(f"有效country记录数: {len(country_values)}")
        logger.info(f"空值记录数: {len(gdf) - len(country_values)}")
        
        # 按记录数排序显示所有国家
        logger.info("\n各国家记录数统计（按记录数降序排列）:")
        for country, count in country_stats.most_common():
            percentage = (count / len(country_values)) * 100
            logger.info(f"  {country}: {count:,} 条记录 ({percentage:.2f}%)")
        
        return gdf, country_stats
        
    except Exception as e:
        logger.error(f"读取SDPT数据时发生错误: {str(e)}")
        return None, None

def identify_southeast_asia_countries(country_stats, logger):
    """
    识别东南亚国家名称
    Args:
        country_stats: 国家统计信息
        logger: 日志记录器
    Returns:
        southeast_asia_countries: 东南亚国家列表
    """
    logger.info("\n开始识别东南亚国家...")
    
    # 定义东南亚国家的可能名称（英文）
    # 包括各种可能的拼写和缩写形式
    southeast_asia_reference = {
        'Indonesia': ['Indonesia', 'indonesia', 'INDONESIA', 'IDN', 'ID'],
        'Malaysia': ['Malaysia', 'malaysia', 'MALAYSIA', 'MYS', 'MY'],
        'Thailand': ['Thailand', 'thailand', 'THAILAND', 'THA', 'TH'],
        'Vietnam': ['Vietnam', 'vietnam', 'VIETNAM', 'VNM', 'VN', 'Viet Nam'],
        'Philippines': ['Philippines', 'philippines', 'PHILIPPINES', 'PHL', 'PH'],
        'Singapore': ['Singapore', 'singapore', 'SINGAPORE', 'SGP', 'SG'],
        'Myanmar': ['Myanmar', 'myanmar', 'MYANMAR', 'MMR', 'MM', 'Burma'],
        'Cambodia': ['Cambodia', 'cambodia', 'CAMBODIA', 'KHM', 'KH'],
        'Laos': ['Laos', 'laos', 'LAOS', 'LAO', 'LA', 'Lao PDR'],
        'Brunei': ['Brunei', 'brunei', 'BRUNEI', 'BRN', 'BN', 'Brunei Darussalam'],
        'Timor-Leste': ['Timor-Leste', 'timor-leste', 'TIMOR-LESTE', 'TLS', 'TL', 'East Timor']
    }
    
    # 从实际数据中找到匹配的国家名称
    found_countries = []
    available_countries = list(country_stats.keys())
    
    logger.info("在数据中搜索东南亚国家...")
    for standard_name, variants in southeast_asia_reference.items():
        found_variants = []
        for variant in variants:
            if variant in available_countries:
                found_variants.append(variant)
                found_countries.append(variant)
        
        if found_variants:
            total_records = sum(country_stats[variant] for variant in found_variants)
            logger.info(f"  {standard_name}: 找到 {found_variants} - {total_records:,} 条记录")
        else:
            logger.warning(f"  {standard_name}: 未在数据中找到")
    
    # 显示所有找到的东南亚国家
    logger.info(f"\n总共找到 {len(found_countries)} 个东南亚国家:")
    total_sea_records = sum(country_stats[country] for country in found_countries)
    for country in found_countries:
        count = country_stats[country]
        percentage = (count / sum(country_stats.values())) * 100
        logger.info(f"  - {country}: {count:,} 条记录 ({percentage:.2f}%)")
    
    logger.info(f"\n东南亚国家总记录数: {total_sea_records:,}")
    logger.info(f"占总数据比例: {(total_sea_records / sum(country_stats.values())) * 100:.2f}%")
    
    return found_countries

def filter_by_countries(gdf, target_countries, output_dir, logger):
    """
    根据国家列表筛选数据并保存
    Args:
        gdf: 原始GeoDataFrame
        target_countries: 目标国家列表
        output_dir: 输出目录
        logger: 日志记录器
    Returns:
        filtered_gdf: 筛选后的数据
    """
    logger.info(f"\n开始根据国家筛选数据...")
    logger.info(f"筛选条件: {target_countries}")
    
    try:
        # 筛选数据
        filtered_gdf = gdf[gdf['country'].isin(target_countries)].copy()
        logger.info(f"筛选完成，共 {len(filtered_gdf)} 条记录")
        
        # 统计筛选后各国家的记录数
        logger.info("\n筛选后各国家记录数:")
        filtered_stats = Counter(filtered_gdf['country'])
        for country, count in filtered_stats.most_common():
            percentage = (count / len(filtered_gdf)) * 100
            logger.info(f"  {country}: {count:,} 条记录 ({percentage:.2f}%)")
        
        # 保存筛选结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"sdpt_southeast_asia_{timestamp}.gpkg")
        
        logger.info(f"\n开始保存筛选结果到: {output_file}")
        filtered_gdf.to_file(output_file, driver='GPKG')
        
        # 获取文件大小
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"文件保存成功，大小: {file_size_mb:.2f} MB")
        
        return filtered_gdf
        
    except Exception as e:
        logger.error(f"筛选或保存数据时发生错误: {str(e)}")
        return None

def main():
    """
    主函数：执行完整的数据筛选流程
    """
    # 设置路径
    sdpt_path = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\1.数据预处理\SDPT下载\gfw_planted_forests_20250919_173056.gpkg"
    output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\按国家筛选"
    
    # 设置日志
    logger = setup_logging(output_dir)
    logger.info("="*60)
    logger.info("开始SDPT数据按国家筛选处理")
    logger.info("="*60)
    
    # 第一步：探索数据中的国家属性
    gdf, country_stats = explore_country_attributes(sdpt_path, logger)
    if gdf is None:
        logger.error("无法读取数据，程序终止")
        return
    
    # 第二步：识别东南亚国家
    southeast_asia_countries = identify_southeast_asia_countries(country_stats, logger)
    if not southeast_asia_countries:
        logger.error("未找到东南亚国家，程序终止")
        return
    
    # 第三步：根据国家筛选数据
    filtered_gdf = filter_by_countries(gdf, southeast_asia_countries, output_dir, logger)
    if filtered_gdf is None:
        logger.error("数据筛选失败，程序终止")
        return
    
    logger.info("="*60)
    logger.info("SDPT数据按国家筛选处理完成！")
    logger.info("="*60)

if __name__ == "__main__":
    main()