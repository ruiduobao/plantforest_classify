#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并每个zone下多个年份的矢量点文件
功能：
1. 扫描每个zone目录下的所有年份文件夹
2. 合并每个zone下所有年份的shapefile数据
3. 为合并后的数据添加year属性记录原始年份
4. 输出合并后的文件，命名为"Zone{X}_Merged_AllBand_Sample_filtered.shp"

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
# 输入根目录
INPUT_ROOT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并_剔除异常样本_去掉不需要的属性"

# 输出目录
OUTPUT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并_剔除异常样本_去掉不需要的属性\合并后的zone数据"

# ==================== 核心函数 ====================

def get_zone_directories(root_dir):
    """
    获取所有zone目录
    """
    zone_dirs = []
    if os.path.exists(root_dir):
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path) and item.startswith('Zone'):
                zone_dirs.append(item_path)
    
    zone_dirs.sort()  # 按zone编号排序
    return zone_dirs

def get_year_directories(zone_dir):
    """
    获取zone目录下的所有年份目录
    """
    year_dirs = []
    if os.path.exists(zone_dir):
        for item in os.listdir(zone_dir):
            item_path = os.path.join(zone_dir, item)
            if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
                year_dirs.append((item, item_path))
    
    year_dirs.sort()  # 按年份排序
    return year_dirs

def get_shapefiles_in_directory(directory):
    """
    获取目录中的所有shapefile文件
    """
    pattern = os.path.join(directory, "*.shp")
    shp_files = glob.glob(pattern)
    return shp_files

def merge_zone_data(zone_dir, zone_name):
    """
    合并单个zone下所有年份的数据
    """
    print(f"\n处理Zone: {zone_name}")
    print(f"Zone目录: {zone_dir}")
    
    # 获取所有年份目录
    year_dirs = get_year_directories(zone_dir)
    
    if not year_dirs:
        print(f"  警告: {zone_name} 下未找到年份目录")
        return None
    
    print(f"  找到 {len(year_dirs)} 个年份目录: {[year for year, _ in year_dirs]}")
    
    # 存储所有年份的数据
    all_gdfs = []
    merge_stats = {
        'zone': zone_name,
        'years_processed': [],
        'files_processed': [],
        'total_features': 0,
        'year_feature_counts': {},
        'status': 'Success'
    }
    
    # 处理每个年份
    for year, year_dir in year_dirs:
        print(f"    处理年份: {year}")
        
        # 获取该年份目录下的所有shapefile
        shp_files = get_shapefiles_in_directory(year_dir)
        
        if not shp_files:
            print(f"      警告: {year} 年份目录下未找到shapefile文件")
            continue
        
        print(f"      找到 {len(shp_files)} 个shapefile文件")
        
        # 处理该年份的所有shapefile
        year_gdfs = []
        year_feature_count = 0
        
        for shp_file in shp_files:
            try:
                filename = os.path.basename(shp_file)
                print(f"        读取文件: {filename}")
                
                # 读取shapefile
                gdf = gpd.read_file(shp_file)
                
                # 添加year属性
                gdf['year'] = int(year)
                
                year_gdfs.append(gdf)
                year_feature_count += len(gdf)
                
                merge_stats['files_processed'].append(f"{year}/{filename}")
                
                print(f"          要素数: {len(gdf)}")
                
            except Exception as e:
                print(f"          错误: 读取文件 {filename} 失败 - {str(e)}")
                merge_stats['status'] = f'Partial Success: {str(e)}'
        
        # 合并该年份的所有数据
        if year_gdfs:
            year_combined = pd.concat(year_gdfs, ignore_index=True)
            all_gdfs.append(year_combined)
            
            merge_stats['years_processed'].append(year)
            merge_stats['year_feature_counts'][year] = year_feature_count
            merge_stats['total_features'] += year_feature_count
            
            print(f"      {year}年合计要素数: {year_feature_count}")
    
    # 合并所有年份的数据
    if not all_gdfs:
        print(f"  错误: {zone_name} 下没有成功读取任何数据")
        merge_stats['status'] = 'Failed: No data found'
        return merge_stats
    
    print(f"  正在合并所有年份数据...")
    final_gdf = pd.concat(all_gdfs, ignore_index=True)
    
    print(f"  合并完成！总要素数: {len(final_gdf)}")
    print(f"  数据字段: {list(final_gdf.columns)}")
    
    # 生成输出文件名
    output_filename = f"{zone_name}_Merged_AllBand_Sample_filtered.shp"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"  创建输出目录: {OUTPUT_DIR}")
    
    # 保存合并后的数据
    print(f"  正在保存到: {output_path}")
    final_gdf.to_file(output_path, driver='ESRI Shapefile')
    
    merge_stats['output_file'] = output_path
    merge_stats['final_feature_count'] = len(final_gdf)
    
    print(f"  {zone_name} 处理完成！")
    
    return merge_stats

def create_output_directory():
    """
    创建输出目录
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    else:
        print(f"输出目录已存在: {OUTPUT_DIR}")

def main():
    """
    主函数
    """
    print("=" * 80)
    print("合并每个zone下多个年份的矢量点文件")
    print("=" * 80)
    
    # 创建输出目录
    create_output_directory()
    
    # 获取所有zone目录
    print(f"\n正在扫描输入根目录: {INPUT_ROOT_DIR}")
    zone_dirs = get_zone_directories(INPUT_ROOT_DIR)
    
    if not zone_dirs:
        print("错误：未找到zone目录")
        return
    
    print(f"找到 {len(zone_dirs)} 个zone目录")
    
    # 处理统计
    all_merge_stats = []
    
    # 处理每个zone
    for i, zone_dir in enumerate(zone_dirs, 1):
        zone_name = os.path.basename(zone_dir)
        print(f"\n[{i}/{len(zone_dirs)}] " + "="*50)
        
        # 合并zone数据
        merge_result = merge_zone_data(zone_dir, zone_name)
        if merge_result:
            all_merge_stats.append(merge_result)
    
    # 生成处理统计报告
    print("\n" + "=" * 80)
    print("合并处理统计报告")
    print("=" * 80)
    
    if all_merge_stats:
        df_stats = pd.DataFrame(all_merge_stats)
        
        # 显示统计信息
        print(f"\n处理Zone总数: {len(all_merge_stats)}")
        success_count = len(df_stats[df_stats['status'] == 'Success'])
        print(f"成功处理: {success_count}")
        print(f"处理失败: {len(all_merge_stats) - success_count}")
        
        if success_count > 0:
            successful_df = df_stats[df_stats['status'] == 'Success']
            
            # 总体统计
            total_features = successful_df['total_features'].sum()
            total_files = sum(len(files) for files in successful_df['files_processed'])
            
            print(f"\n总体统计:")
            print(f"  总要素数: {total_features:,}")
            print(f"  总文件数: {total_files}")
            
            # 各zone统计
            print(f"\n各Zone统计:")
            for _, row in successful_df.iterrows():
                years_str = ', '.join(row['years_processed'])
                print(f"  {row['zone']}: {row['total_features']:,} 个要素, 年份: {years_str}")
        
        # 保存统计报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = os.path.join(OUTPUT_DIR, f"zone合并统计报告_{timestamp}.csv")
        
        # 准备保存的统计数据
        stats_for_save = []
        for stat in all_merge_stats:
            row = {
                'zone': stat['zone'],
                'status': stat['status'],
                'total_features': stat['total_features'],
                'years_processed': '; '.join(stat['years_processed']),
                'files_count': len(stat['files_processed']),
                'output_file': stat.get('output_file', ''),
                'year_feature_counts': str(stat['year_feature_counts'])
            }
            stats_for_save.append(row)
        
        pd.DataFrame(stats_for_save).to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"\n统计报告已保存到: {stats_file}")
        
        # 显示失败的zone
        failed_df = df_stats[df_stats['status'] != 'Success']
        if len(failed_df) > 0:
            print(f"\n处理失败的Zone:")
            for _, row in failed_df.iterrows():
                print(f"  {row['zone']}: {row['status']}")
    
    print(f"\n合并处理完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"每个zone的多年份数据已合并为单个文件，并添加了year属性")

if __name__ == "__main__":
    main()