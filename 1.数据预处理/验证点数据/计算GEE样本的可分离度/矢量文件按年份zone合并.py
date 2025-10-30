#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
矢量文件按年份和Zone合并脚本
功能：
将GEE结果目录中的矢量文件按照年份和zone进行合并
例如：zone_1_2017_2_nature_tree.shp, zone_1_2017_1_planted_tree.shp, zone_1_2017_3_others_tree.shp
合并为：Zone1_2017_Merged_AllBand_Sample.shp

作者：锐多宝 (ruiduobao)
日期：2025年1月21日
"""

import os
import glob
import re
from osgeo import ogr, osr
import pandas as pd
from datetime import datetime

# ==================== 配置参数 ====================
# 输入目录
INPUT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果"

# 输出目录
OUTPUT_DIR = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并"

# 类别映射
CLASS_MAPPING = {
    'nature_tree': 2,    # 自然林
    'planted_tree': 1,   # 人工林
    'others_tree': 3     # 其他
}

def parse_filename(filename):
    """
    解析文件名，提取zone、年份和类别信息
    例如：zone_1_2017_2_nature_tree.shp -> zone=1, year=2017, class=nature_tree
    """
    # 移除.shp扩展名
    basename = os.path.splitext(filename)[0]
    
    # 使用正则表达式匹配文件名模式
    pattern = r'zone_(\d+)_(\d{4})_\d+_(\w+_tree)'
    match = re.match(pattern, basename)
    
    if match:
        zone = int(match.group(1))
        year = int(match.group(2))
        class_type = match.group(3)
        return zone, year, class_type
    else:
        return None, None, None

def get_shapefile_list(input_dir):
    """
    获取输入目录中所有符合条件的shapefile列表
    """
    shp_files = []
    pattern = os.path.join(input_dir, "zone_*_*_*_*_tree.shp")
    
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        zone, year, class_type = parse_filename(filename)
        
        if zone is not None and year is not None and class_type is not None:
            shp_files.append({
                'filepath': filepath,
                'filename': filename,
                'zone': zone,
                'year': year,
                'class_type': class_type
            })
    
    return shp_files

def group_files_by_zone_year(shp_files):
    """
    按zone和年份分组文件
    """
    groups = {}
    
    for file_info in shp_files:
        key = (file_info['zone'], file_info['year'])
        if key not in groups:
            groups[key] = []
        groups[key].append(file_info)
    
    return groups

def merge_shapefiles(file_group, output_path):
    """
    合并同一zone和年份的所有shapefile
    """
    print(f"正在合并文件到: {output_path}")
    
    # 创建输出shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    
    out_ds = driver.CreateDataSource(output_path)
    
    # 获取第一个文件的空间参考系统和字段结构
    first_file = file_group[0]['filepath']
    first_ds = ogr.Open(first_file, 0)
    first_layer = first_ds.GetLayer(0)
    spatial_ref = first_layer.GetSpatialRef()
    layer_defn = first_layer.GetLayerDefn()
    
    # 创建输出图层
    out_layer = out_ds.CreateLayer("merged_samples", spatial_ref, ogr.wkbPoint)
    
    # 复制字段定义
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)
    
    # 添加landcover字段（如果不存在）
    landcover_field = ogr.FieldDefn("landcover", ogr.OFTInteger)
    out_layer.CreateField(landcover_field)
    
    first_ds = None
    
    total_features = 0
    
    # 遍历每个文件，复制要素
    for file_info in file_group:
        print(f"  处理文件: {file_info['filename']}")
        
        ds = ogr.Open(file_info['filepath'], 0)
        if ds is None:
            print(f"    警告：无法打开文件 {file_info['filepath']}")
            continue
        
        layer = ds.GetLayer(0)
        feature_count = layer.GetFeatureCount()
        print(f"    要素数量: {feature_count}")
        
        # 获取landcover值
        landcover_value = CLASS_MAPPING.get(file_info['class_type'], 0)
        
        layer.ResetReading()
        for feature in layer:
            # 创建新要素
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            
            # 复制几何
            geom = feature.GetGeometryRef()
            if geom:
                out_feature.SetGeometry(geom.Clone())
            
            # 复制属性
            for i in range(feature.GetFieldCount()):
                field_name = feature.GetFieldDefnRef(i).GetName()
                field_value = feature.GetField(i)
                out_feature.SetField(field_name, field_value)
            
            # 设置landcover值
            out_feature.SetField("landcover", landcover_value)
            
            # 添加要素到输出图层
            out_layer.CreateFeature(out_feature)
            out_feature = None
            total_features += 1
        
        ds = None
    
    out_ds = None
    print(f"  合并完成，总要素数: {total_features}")
    return total_features

def create_output_directory(output_dir):
    """
    创建输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("矢量文件按年份和Zone合并脚本")
    print("=" * 60)
    
    # 创建输出目录
    create_output_directory(OUTPUT_DIR)
    
    # 获取所有shapefile列表
    print(f"\n正在扫描输入目录: {INPUT_DIR}")
    shp_files = get_shapefile_list(INPUT_DIR)
    
    if not shp_files:
        print("错误：未找到符合条件的shapefile文件")
        return
    
    print(f"找到 {len(shp_files)} 个符合条件的shapefile文件")
    
    # 按zone和年份分组
    groups = group_files_by_zone_year(shp_files)
    print(f"共有 {len(groups)} 个zone-年份组合需要合并")
    
    # 统计信息
    merge_stats = []
    
    # 遍历每个组合进行合并
    for (zone, year), file_group in groups.items():
        print(f"\n处理Zone {zone}, 年份 {year}:")
        print(f"  包含文件数: {len(file_group)}")
        
        # 检查是否包含所有三个类别
        class_types = [f['class_type'] for f in file_group]
        print(f"  包含类别: {class_types}")
        
        # 生成输出文件名
        output_filename = f"Zone{zone}_{year}_Merged_AllBand_Sample.shp"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 合并文件
        try:
            total_features = merge_shapefiles(file_group, output_path)
            merge_stats.append({
                'zone': zone,
                'year': year,
                'input_files': len(file_group),
                'output_file': output_filename,
                'total_features': total_features,
                'status': 'Success'
            })
        except Exception as e:
            print(f"  错误：合并失败 - {str(e)}")
            merge_stats.append({
                'zone': zone,
                'year': year,
                'input_files': len(file_group),
                'output_file': output_filename,
                'total_features': 0,
                'status': f'Failed: {str(e)}'
            })
    
    # 生成统计报告
    print("\n" + "=" * 60)
    print("合并统计报告")
    print("=" * 60)
    
    df_stats = pd.DataFrame(merge_stats)
    print(df_stats.to_string(index=False))
    
    # 保存统计报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = os.path.join(OUTPUT_DIR, f"合并统计报告_{timestamp}.csv")
    df_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"\n统计报告已保存到: {stats_file}")
    
    # 成功统计
    success_count = len(df_stats[df_stats['status'] == 'Success'])
    total_count = len(df_stats)
    print(f"\n合并完成: {success_count}/{total_count} 个组合成功")

if __name__ == "__main__":
    main()