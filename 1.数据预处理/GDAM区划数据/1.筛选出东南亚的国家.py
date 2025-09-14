#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从GDAM全球数据中筛选出东南亚国家的矢量数据

目的：
1. 读取GDAM全球矢量数据
2. 筛选出东南亚国家
3. 输出到指定目录

作者：锐多宝 (ruiduobao)
日期：2025年
"""

import os
import logging
from datetime import datetime
import geopandas as gpd
import pandas as pd
from pathlib import Path

# 配置日志
def setup_logging(output_dir):
    """
    设置日志配置
    """
    log_filename = f"southeast_asia_filter_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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

def main():
    """
    主函数：从GDAM全球数据中筛选东南亚国家
    """
    # 输入输出路径配置
    input_gpkg = r"K:\数据\GDAM全球\gadm_410-gpkg\gadm_410.gpkg"  # GDAM全球数据路径
    output_dir = r"K:\数据\GDAM全球\东南亚国家"  # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(output_dir)
    
    logging.info("开始从GDAM全球数据中筛选东南亚国家")
    logging.info(f"输入文件: {input_gpkg}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"日志文件: {log_path}")
    
    try:
        # 定义东南亚国家列表（英文名称，与GDAM数据中的NAME_0字段对应）
        southeast_asia_countries = [
            'Brunei',           # 文莱
            'Cambodia',         # 柬埔寨
            'Indonesia',        # 印度尼西亚
            'Laos',            # 老挝
            'Malaysia',         # 马来西亚
            'Myanmar',          # 缅甸
            'Philippines',      # 菲律宾
            'Singapore',        # 新加坡
            'Thailand',         # 泰国
            'Vietnam',          # 越南
            'Timor-Leste'       # 东帝汶
        ]
        
        logging.info(f"目标东南亚国家数量: {len(southeast_asia_countries)}")
        logging.info(f"国家列表: {', '.join(southeast_asia_countries)}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_gpkg):
            raise FileNotFoundError(f"输入文件不存在: {input_gpkg}")
        
        # 读取GDAM全球数据
        logging.info("正在读取GDAM全球数据...")
        
        # 首先查看GPKG文件中的图层
        import fiona
        layers = fiona.listlayers(input_gpkg)
        logging.info(f"GPKG文件中的图层: {layers}")
        
        # 通常GDAM数据的国家级别数据在ADM_0图层中
        if 'ADM_0' in layers:
            layer_name = 'ADM_0'
        elif 'gadm_410-0' in layers:
            layer_name = 'gadm_410-0'
        else:
            # 如果没有找到标准图层名，使用第一个图层
            layer_name = layers[0]
            logging.warning(f"未找到标准的ADM_0图层，使用图层: {layer_name}")
        
        logging.info(f"使用图层: {layer_name}")
        
        # 读取所有行政级别的数据
        all_layers_data = []
        
        # 尝试读取不同的行政级别图层
        for level in range(6):  # GDAM通常有0-5级
            possible_layer_names = [f'ADM_{level}', f'gadm_410-{level}']
            
            for layer_candidate in possible_layer_names:
                if layer_candidate in layers:
                    logging.info(f"正在读取行政级别 {level} 的数据 (图层: {layer_candidate})...")
                    level_gdf = gpd.read_file(input_gpkg, layer=layer_candidate)
                    logging.info(f"级别 {level} 数据: {len(level_gdf)} 条记录")
                    all_layers_data.append(level_gdf)
                    break
        
        if not all_layers_data:
            # 如果没有找到标准的分级图层，使用原来的方法
            gdf_world = gpd.read_file(input_gpkg, layer=layer_name)
        else:
            # 合并所有级别的数据
            gdf_world = gpd.GeoDataFrame(pd.concat(all_layers_data, ignore_index=True))
        
        logging.info(f"成功读取全球数据，共 {len(gdf_world)} 个国家/地区")
        logging.info(f"数据坐标系: {gdf_world.crs}")
        
        # 查看数据的列名，找到国家名称字段
        logging.info(f"数据列名: {list(gdf_world.columns)}")
        
        # 常见的国家名称字段
        name_fields = ['NAME_0', 'COUNTRY', 'NAME', 'name', 'country']
        country_field = None
        
        for field in name_fields:
            if field in gdf_world.columns:
                country_field = field
                break
        
        if country_field is None:
            raise ValueError(f"未找到国家名称字段，可用字段: {list(gdf_world.columns)}")
        
        logging.info(f"使用国家名称字段: {country_field}")
        
        # 查看数据中的国家名称样例
        unique_countries = gdf_world[country_field].unique()
        logging.info(f"数据中的国家总数: {len(unique_countries)}")
        logging.info(f"前10个国家名称: {list(unique_countries[:10])}")
        
        # 筛选东南亚国家
        logging.info("正在筛选东南亚国家...")
        
        # 使用isin方法筛选
        southeast_asia_gdf = gdf_world[gdf_world[country_field].isin(southeast_asia_countries)]
        
        logging.info(f"筛选出的东南亚国家数量: {len(southeast_asia_gdf)}")
        
        # 显示筛选出的国家
        found_countries = southeast_asia_gdf[country_field].unique().tolist()
        logging.info(f"找到的东南亚国家: {found_countries}")
        
        # 检查哪些国家没有找到
        missing_countries = set(southeast_asia_countries) - set(found_countries)
        if missing_countries:
            logging.warning(f"未找到的国家: {list(missing_countries)}")
            
            # 尝试模糊匹配
            logging.info("尝试模糊匹配未找到的国家...")
            for missing in missing_countries:
                # 在所有国家名称中查找包含关键词的
                matches = [country for country in unique_countries 
                          if missing.lower() in country.lower() or country.lower() in missing.lower()]
                if matches:
                    logging.info(f"'{missing}' 的可能匹配: {matches}")
        
        if len(southeast_asia_gdf) == 0:
            raise ValueError("未筛选出任何东南亚国家，请检查国家名称匹配")
        
        # 输出目录准备
        # 不再需要预定义单一输出文件，因为会按级别分别输出
        
        # 按行政级别分别保存数据
        logging.info("正在按行政级别保存筛选结果...")
        
        # 确定可用的行政级别
        admin_levels = []
        for level in range(6):  # GDAM通常有0-5级
            gid_field = f'GID_{level}'
            name_field = f'NAME_{level}'
            if gid_field in southeast_asia_gdf.columns and name_field in southeast_asia_gdf.columns:
                # 检查该级别是否有非空数据
                non_null_count = southeast_asia_gdf[gid_field].notna().sum()
                if non_null_count > 0:
                    admin_levels.append(level)
        
        logging.info(f"可用的行政级别: {admin_levels}")
        
        # 为每个行政级别创建单独的数据集
        for level in admin_levels:
            gid_field = f'GID_{level}'
            name_field = f'NAME_{level}'
            
            # 筛选出该级别的有效数据：根据该级别的GID字段进行去重聚合
            level_data = southeast_asia_gdf[southeast_asia_gdf[gid_field].notna()].copy()
            
            if len(level_data) > 0:
                # 按照该级别的GID进行分组，每个行政区划只保留一条记录
                # 使用dissolve进行几何合并，这样可以得到该级别的真实边界
                try:
                    level_data = level_data.dissolve(by=gid_field, as_index=False)
                    logging.info(f"第{level}级通过dissolve合并后，共 {len(level_data)} 条记录")
                except Exception as e:
                    logging.warning(f"第{level}级dissolve合并失败: {e}，使用drop_duplicates方法")
                    # 如果dissolve失败，使用简单的去重
                    level_data = level_data.drop_duplicates(subset=[gid_field]).copy()
                    logging.info(f"第{level}级去重后，共 {len(level_data)} 条记录")
            
            if len(level_data) == 0:
                logging.warning(f"第{level}级行政区划没有找到纯粹的边界数据，跳过")
                continue
            
            # 根据级别命名
            level_names = {
                0: "国家级",
                1: "省级", 
                2: "市级",
                3: "县级",
                4: "乡级",
                5: "村级"
            }
            
            level_name_cn = level_names.get(level, f"第{level}级")
            level_name_en = f"ADM_{level}"
            
            logging.info(f"处理{level_name_cn}数据 (ADM_{level})，共 {len(level_data)} 条记录")
            
            # 输出文件路径
            output_shp = os.path.join(output_dir, f"southeast_asia_{level_name_en}.shp")
            output_gpkg_level = os.path.join(output_dir, f"southeast_asia_{level_name_en}.gpkg")
            
            # 保存为Shapefile
            level_data.to_file(output_shp, encoding='utf-8')
            logging.info(f"已保存{level_name_cn}Shapefile: {output_shp}")
            
            # 保存为GeoPackage
            level_data.to_file(output_gpkg_level, driver='GPKG')
            logging.info(f"已保存{level_name_cn}GeoPackage: {output_gpkg_level}")
            
            # 统计该级别的信息
            unique_entities = level_data[name_field].nunique()
            logging.info(f"{level_name_cn}包含 {unique_entities} 个不同的行政区划")
        

        
        # 输出统计信息
        logging.info("\n=== 筛选结果统计 ===")
        logging.info(f"筛选出的东南亚国家: {', '.join(found_countries)}")
        logging.info(f"总记录数: {len(southeast_asia_gdf)}")
        logging.info(f"按行政级别分布:")
        
        for level in admin_levels:
            gid_field = f'GID_{level}'
            name_field = f'NAME_{level}'
            level_count = southeast_asia_gdf[southeast_asia_gdf[gid_field].notna()].shape[0]
            unique_count = southeast_asia_gdf[southeast_asia_gdf[gid_field].notna()][name_field].nunique()
            level_names = {0: "国家级", 1: "省级", 2: "市级", 3: "县级", 4: "乡级", 5: "村级"}
            level_name = level_names.get(level, f"第{level}级")
            logging.info(f"  - {level_name} (ADM_{level}): {level_count} 条记录, {unique_count} 个行政区划")
        
        # 计算国家级总面积（如果数据有面积信息）
        if southeast_asia_gdf.crs and southeast_asia_gdf.crs.is_geographic and 0 in admin_levels:
            country_level_data = southeast_asia_gdf[southeast_asia_gdf['GID_0'].notna()]
            if len(country_level_data) > 0:
                # 转换为等积投影计算面积
                southeast_asia_projected = country_level_data.to_crs('EPSG:3857')  # Web Mercator
                total_area_km2 = southeast_asia_projected.geometry.area.sum() / 1e6  # 转换为平方公里
                logging.info(f"东南亚总面积: {total_area_km2:,.2f} 平方公里")
        
        logging.info("东南亚国家筛选任务完成！")
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()