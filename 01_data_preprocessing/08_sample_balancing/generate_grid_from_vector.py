#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据矢量数据生成网格数据
目的：基于东南亚国家边界矢量数据，生成0.5°×0.5°的规则网格，用于后续分类处理
作者：锐多宝 (ruiduobao)
创建时间：2024
"""

import os
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def create_grid(bounds, grid_size=0.5):
    """
    创建规则网格
    
    参数:
        bounds: 边界范围 (minx, miny, maxx, maxy)
        grid_size: 网格大小（度），默认0.5度
    
    返回:
        grid_polygons: 网格多边形列表
        grid_ids: 网格ID列表
    """
    minx, miny, maxx, maxy = bounds
    
    # 计算网格数量
    cols = int(np.ceil((maxx - minx) / grid_size))
    rows = int(np.ceil((maxy - miny) / grid_size))
    
    print(f"网格范围: {minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}")
    print(f"网格大小: {grid_size}°")
    print(f"网格数量: {cols} × {rows} = {cols * rows}")
    
    grid_polygons = []
    grid_ids = []
    
    # 生成网格
    for i in tqdm(range(rows), desc="生成网格"):
        for j in range(cols):
            # 计算网格边界
            left = minx + j * grid_size
            right = minx + (j + 1) * grid_size
            bottom = miny + i * grid_size
            top = miny + (i + 1) * grid_size
            
            # 创建网格多边形
            grid_polygon = Polygon([
                (left, bottom),
                (right, bottom),
                (right, top),
                (left, top),
                (left, bottom)
            ])
            
            grid_polygons.append(grid_polygon)
            grid_ids.append(f"grid_{i:04d}_{j:04d}")
    
    return grid_polygons, grid_ids

def main():
    """
    主函数：生成网格数据
    """
    # 输入参数设置
    reference_shapefile = r"K:\数据\GDAM全球\东南亚国家\southeast_asia_combine.shp"
    output_dir = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\分类网格数据"
    grid_size = 0.5  # 网格大小（度）
    
    print("=" * 60)
    print("东南亚地区网格数据生成程序")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(reference_shapefile):
        print(f"错误：参考矢量文件不存在: {reference_shapefile}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    try:
        # 读取参考矢量数据
        print("\n1. 读取参考矢量数据...")
        reference_gdf = gpd.read_file(reference_shapefile)
        print(f"   - 坐标系: {reference_gdf.crs}")
        print(f"   - 要素数量: {len(reference_gdf)}")
        
        # 检查几何有效性
        print("   - 检查几何有效性...")
        valid_mask = reference_gdf.geometry.is_valid & (~reference_gdf.geometry.is_empty)
        print(f"   - 有效几何数量: {valid_mask.sum()}/{len(reference_gdf)}")
        
        if valid_mask.sum() == 0:
            print("   错误：没有有效的几何数据")
            return
        
        # 只保留有效几何
        reference_gdf = reference_gdf[valid_mask].copy()
        
        # 修复无效几何
        invalid_geoms = ~reference_gdf.geometry.is_valid
        if invalid_geoms.any():
            print(f"   - 修复 {invalid_geoms.sum()} 个无效几何...")
            reference_gdf.loc[invalid_geoms, 'geometry'] = reference_gdf.loc[invalid_geoms, 'geometry'].buffer(0)
        
        # 转换到WGS84坐标系
        if reference_gdf.crs != 'EPSG:4326':
            print("   - 转换坐标系到WGS84...")
            reference_gdf = reference_gdf.to_crs('EPSG:4326')
        
        # 获取边界范围
        bounds = reference_gdf.total_bounds
        print(f"   - 边界范围: {bounds}")
        
        # 检查边界是否有效
        if np.any(np.isnan(bounds)):
            print("   错误：边界范围包含NaN值")
            return
        
        # 扩展边界以确保完全覆盖
        buffer = grid_size * 0.1  # 10%的缓冲区
        extended_bounds = (
            bounds[0] - buffer,  # minx
            bounds[1] - buffer,  # miny
            bounds[2] + buffer,  # maxx
            bounds[3] + buffer   # maxy
        )
        
        # 生成网格
        print("\n2. 生成网格数据...")
        grid_polygons, grid_ids = create_grid(extended_bounds, grid_size)
        
        # 创建GeoDataFrame
        print("\n3. 创建网格GeoDataFrame...")
        grid_gdf = gpd.GeoDataFrame({
            'grid_id': grid_ids,
            'geometry': grid_polygons
        }, crs='EPSG:4326')
        
        # 与参考数据求交，只保留有交集的网格
        print("\n4. 筛选有效网格（与参考区域相交）...")
        print(f"   - 原始网格数量: {len(grid_gdf)}")
        
        # 使用空间索引加速相交计算
        valid_grids = []
        reference_union = reference_gdf.unary_union
        
        for idx, grid in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc="筛选网格"):
            if grid.geometry.intersects(reference_union):
                valid_grids.append(idx)
        
        # 保留有效网格
        grid_gdf_filtered = grid_gdf.iloc[valid_grids].copy()
        grid_gdf_filtered.reset_index(drop=True, inplace=True)
        
        print(f"   - 有效网格数量: {len(grid_gdf_filtered)}")
        
        # 添加网格属性
        print("\n5. 添加网格属性...")
        grid_gdf_filtered['area_km2'] = grid_gdf_filtered.geometry.area * 111.32 * 111.32  # 近似面积(km²)
        grid_gdf_filtered['center_lon'] = grid_gdf_filtered.geometry.centroid.x
        grid_gdf_filtered['center_lat'] = grid_gdf_filtered.geometry.centroid.y
        
        # 保存结果
        output_file = os.path.join(output_dir, f"southeast_asia_grid_{grid_size}deg.shp")
        print(f"\n6. 保存网格数据到: {output_file}")
        
        grid_gdf_filtered.to_file(output_file, encoding='utf-8')
        
        # 输出统计信息
        print("\n" + "=" * 60)
        print("网格生成完成！")
        print("=" * 60)
        print(f"网格大小: {grid_size}° × {grid_size}°")
        print(f"网格数量: {len(grid_gdf_filtered)}")
        print(f"坐标系: WGS84 (EPSG:4326)")
        print(f"输出文件: {output_file}")
        print(f"平均网格面积: {grid_gdf_filtered['area_km2'].mean():.2f} km²")
        
        # 保存处理日志
        log_file = os.path.join(output_dir, "grid_generation_log.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("东南亚地区网格数据生成日志\n")
            f.write("=" * 40 + "\n")
            f.write(f"参考矢量文件: {reference_shapefile}\n")
            f.write(f"网格大小: {grid_size}° × {grid_size}°\n")
            f.write(f"网格数量: {len(grid_gdf_filtered)}\n")
            f.write(f"坐标系: WGS84 (EPSG:4326)\n")
            f.write(f"输出文件: {output_file}\n")
            f.write(f"边界范围: {bounds}\n")
            f.write(f"平均网格面积: {grid_gdf_filtered['area_km2'].mean():.2f} km²\n")
        
        print(f"处理日志已保存到: {log_file}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()