#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理shapefile文件，去掉除landcover和A开头字段以外的所有属性
功能：
1. 批量处理指定目录中的所有shapefile文件
2. 只保留landcover字段和A开头的字段（A00-A63等）
3. 按照zone和年份分别创建子文件夹存储结果
4. 从文件名中解析zone和年份信息

作者：锐多宝 (ruiduobao)
日期：2025年1月21日
"""

import geopandas as gpd
import pandas as pd
import os
import glob
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
# 输入目录
INPUT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并_剔除异常样本"

# 输出根目录
OUTPUT_ROOT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并_剔除异常样本_去掉不需要的属性"

# ==================== 核心函数 ====================

def parse_filename_info(filename):
    """
    从文件名中解析zone和年份信息
    例如：Zone1_2019_Merged_AllBand_Sample_filtered.shp -> zone='Zone1', year='2019'
    """
    # 移除文件扩展名
    basename = os.path.splitext(filename)[0]
    
    # 使用正则表达式提取zone和年份
    # 匹配模式：Zone数字_年份
    pattern = r'(Zone\d+)_(\d{4})'
    match = re.search(pattern, basename)
    
    if match:
        zone = match.group(1)
        year = match.group(2)
        return zone, year
    else:
        # 如果无法解析，尝试其他可能的模式
        # 尝试匹配年份（4位数字）
        year_pattern = r'(\d{4})'
        year_match = re.search(year_pattern, basename)
        year = year_match.group(1) if year_match else 'Unknown'
        
        # 尝试匹配zone
        zone_pattern = r'(Zone\d+)'
        zone_match = re.search(zone_pattern, basename)
        zone = zone_match.group(1) if zone_match else 'Unknown'
        
        return zone, year

def filter_columns(gdf):
    """
    筛选列，只保留landcover和A开头的字段
    """
    # 获取所有列名
    all_columns = gdf.columns.tolist()
    
    # 筛选需要保留的列
    keep_columns = []
    
    # 保留landcover字段
    if 'landcover' in all_columns:
        keep_columns.append('landcover')
    
    # 保留A开头的字段（如A00, A01, A02等）
    for col in all_columns:
        if col.startswith('A') and col != 'landcover':
            keep_columns.append(col)
    
    # 保留geometry列（如果存在）
    if 'geometry' in all_columns:
        keep_columns.append('geometry')
    
    print(f"    原始字段数: {len(all_columns)}")
    print(f"    保留字段数: {len(keep_columns)}")
    print(f"    保留的字段: {keep_columns[:10]}{'...' if len(keep_columns) > 10 else ''}")
    
    # 筛选数据
    filtered_gdf = gdf[keep_columns].copy()
    
    return filtered_gdf

def create_output_directory(zone, year, root_dir):
    """
    创建输出目录结构：root_dir/zone/year/
    """
    output_dir = os.path.join(root_dir, zone, year)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"    创建目录: {output_dir}")
    return output_dir

def process_single_shapefile(input_path, output_root_dir):
    """
    处理单个shapefile文件
    """
    try:
        filename = os.path.basename(input_path)
        print(f"\n处理文件: {filename}")
        
        # 解析文件名获取zone和年份信息
        zone, year = parse_filename_info(filename)
        print(f"  解析结果: Zone={zone}, Year={year}")
        
        # 读取shapefile
        print(f"  正在读取文件...")
        gdf = gpd.read_file(input_path)
        print(f"  读取到 {len(gdf)} 个要素")
        
        # 筛选字段
        print(f"  正在筛选字段...")
        filtered_gdf = filter_columns(gdf)
        
        # 创建输出目录
        output_dir = create_output_directory(zone, year, output_root_dir)
        
        # 生成输出文件路径
        output_filename = filename  # 保持原文件名
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存结果
        print(f"  正在保存到: {output_path}")
        filtered_gdf.to_file(output_path, driver='ESRI Shapefile')
        
        print(f"  处理完成！")
        
        return {
            'input_file': filename,
            'zone': zone,
            'year': year,
            'original_features': len(gdf),
            'original_columns': len(gdf.columns),
            'filtered_columns': len(filtered_gdf.columns),
            'output_path': output_path,
            'status': 'Success'
        }
        
    except Exception as e:
        print(f"  错误: {str(e)}")
        return {
            'input_file': os.path.basename(input_path),
            'zone': 'Unknown',
            'year': 'Unknown',
            'original_features': 0,
            'original_columns': 0,
            'filtered_columns': 0,
            'output_path': '',
            'status': f'Failed: {str(e)}'
        }

def get_shapefile_list(input_dir):
    """
    获取输入目录中的所有shapefile
    """
    pattern = os.path.join(input_dir, "*.shp")
    shp_files = glob.glob(pattern)
    return shp_files

def create_output_root_directory(output_dir):
    """
    创建输出根目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出根目录: {output_dir}")
    else:
        print(f"输出根目录已存在: {output_dir}")

def main():
    """
    主函数
    """
    print("=" * 80)
    print("批量处理shapefile文件 - 去掉不需要的属性")
    print("=" * 80)
    
    # 创建输出根目录
    create_output_root_directory(OUTPUT_ROOT_DIR)
    
    # 获取所有shapefile列表
    print(f"\n正在扫描输入目录: {INPUT_DIR}")
    shp_files = get_shapefile_list(INPUT_DIR)
    
    if not shp_files:
        print("错误：未找到shapefile文件")
        return
    
    print(f"找到 {len(shp_files)} 个shapefile文件")
    
    # 处理统计
    process_stats = []
    
    # 批量处理每个文件
    for i, input_path in enumerate(shp_files, 1):
        print(f"\n[{i}/{len(shp_files)}] " + "="*50)
        
        # 处理文件
        result = process_single_shapefile(input_path, OUTPUT_ROOT_DIR)
        process_stats.append(result)
    
    # 生成处理统计报告
    print("\n" + "=" * 80)
    print("批量处理统计报告")
    print("=" * 80)
    
    if process_stats:
        df_stats = pd.DataFrame(process_stats)
        
        # 显示统计信息
        print(f"\n处理文件总数: {len(process_stats)}")
        success_count = len(df_stats[df_stats['status'] == 'Success'])
        print(f"成功处理: {success_count}")
        print(f"处理失败: {len(process_stats) - success_count}")
        
        if success_count > 0:
            successful_df = df_stats[df_stats['status'] == 'Success']
            
            # 按zone和年份统计
            print(f"\n按Zone统计:")
            zone_stats = successful_df.groupby('zone').size()
            for zone, count in zone_stats.items():
                print(f"  {zone}: {count} 个文件")
            
            print(f"\n按年份统计:")
            year_stats = successful_df.groupby('year').size()
            for year, count in year_stats.items():
                print(f"  {year}: {count} 个文件")
            
            # 字段统计
            total_original_features = successful_df['original_features'].sum()
            avg_original_columns = successful_df['original_columns'].mean()
            avg_filtered_columns = successful_df['filtered_columns'].mean()
            
            print(f"\n字段统计:")
            print(f"  总要素数: {total_original_features:,}")
            print(f"  平均原始字段数: {avg_original_columns:.1f}")
            print(f"  平均筛选后字段数: {avg_filtered_columns:.1f}")
            print(f"  平均字段减少: {avg_original_columns - avg_filtered_columns:.1f}")
        
        # 保存统计报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(OUTPUT_ROOT_DIR, f"属性筛选统计报告_{timestamp}.csv")
        df_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"\n统计报告已保存到: {stats_file}")
        
        # 显示失败的文件
        failed_df = df_stats[df_stats['status'] != 'Success']
        if len(failed_df) > 0:
            print(f"\n处理失败的文件:")
            for _, row in failed_df.iterrows():
                print(f"  {row['input_file']}: {row['status']}")
    
    print(f"\n批量处理完成！")
    print(f"输出根目录: {OUTPUT_ROOT_DIR}")
    print(f"文件按照 Zone/Year 的目录结构进行组织")

if __name__ == "__main__":
    main()