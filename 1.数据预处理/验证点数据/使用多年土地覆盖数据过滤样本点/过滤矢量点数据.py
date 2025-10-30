# 为了获取稳定的样本数据，过滤每一年的矢量点数据，只保留这些数据人工林、自然林分别在2017年-2024年中，矢量点值要这些年一直为2，过滤其他OTHERS点矢量，几年不能有一个值为2
import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

SHP_PATH = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\手工补充样本点"
# 为了获取稳定的样本数据，过滤每一年的矢量点数据，只保留这些数据人工林、自然林分别在2017年-2024年中，矢量点值要这些年一直为2，过滤其他OTHERS点矢量，几年不能有一个值为2
SHP_FILE =[
"手工补充人工林样本点.shp",]
# SHP_FILE =[
# "Zone5_2017_Merged_AllBand_Sample.shp",
# "Zone5_2018_Merged_AllBand_Sample.shp",
# "Zone5_2019_Merged_AllBand_Sample.shp",
# "Zone5_2020_Merged_AllBand_Sample.shp",
# "Zone5_2021_Merged_AllBand_Sample.shp",
# "Zone5_2022_Merged_AllBand_Sample.shp",
# "Zone5_2023_Merged_AllBand_Sample.shp",
# "Zone5_2024_Merged_AllBand_Sample.shp",]


TIF_PATH=r"D:\地理所\论文\东南亚10m人工林提取\数据"

# 土地覆盖数据中，林地为2，非林地为非2
TIF_NAME=[
"southeast_asia_landcover_2017_mosaic_ESRI_10m.tif",
"southeast_asia_landcover_2018_mosaic_ESRI_10m.tif",
"southeast_asia_landcover_2019_mosaic_ESRI_10m.tif",
"southeast_asia_landcover_2020_mosaic_ESRI_10m.tif",
"southeast_asia_landcover_2021_mosaic_ESRI_10m.tif",
"southeast_asia_landcover_2022_mosaic_ESRI_10m.tif",
"southeast_asia_landcover_2023_mosaic_ESRI_10m.tif",
"southeast_asia_landcover_2024_mosaic_ESRI_10m.tif",]

# 人工林CLASS为1，自然林CLASS为2，OTHERS的CLASS为3
SHP_PROPERCITY="class"

OUT_PUT_SHP_PATH=r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\手工补充样本点"

def extract_landcover_values(gdf, tif_path, chunk_size=10000):
    """
    从栅格数据中提取矢量点的土地覆盖值，使用分块处理防止内存溢出
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        包含点几何的GeoDataFrame
    tif_path : str
        栅格文件路径
    chunk_size : int
        每次处理的点数量，默认10000
    """
    print(f"  - 分块提取栅格值，每块处理 {chunk_size} 个点")
    
    # 获取点的坐标
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    total_points = len(coords)
    values = []
    
    with rasterio.open(tif_path) as src:
        # 分块处理
        for i in tqdm(range(0, total_points, chunk_size), desc="  - 处理分块"):
            chunk_coords = coords[i:i + chunk_size]
            chunk_values = []
            
            for coord in chunk_coords:
                try:
                    # 使用sample方法提取点值
                    value = list(src.sample([coord]))[0][0]
                    chunk_values.append(value)
                except:
                    chunk_values.append(np.nan)
            
            values.extend(chunk_values)
    
    return values

def filter_sample_points(chunk_size=10000):
    """
    过滤样本点数据
    
    Parameters:
    -----------
    chunk_size : int
        栅格读取时每次处理的点数量，默认10000
    """
    print("开始过滤样本点数据...")
    print(f"使用分块处理，每块处理 {chunk_size} 个点")
    
    # 创建输出目录
    os.makedirs(OUT_PUT_SHP_PATH, exist_ok=True)
    
    # 读取基础shapefile作为基础数据
    base_shp_path = os.path.join(SHP_PATH, SHP_FILE[0])
    if not os.path.exists(base_shp_path):
        print(f"错误: 基础文件不存在 {base_shp_path}")
        return
    
    result_gdf = gpd.read_file(base_shp_path)
    print(f"基础点数量: {len(result_gdf)}")
    
    # 为所有8年提取土地覆盖值
    for i, tif_file in enumerate(TIF_NAME):
        year = 2017 + i
        print(f"\n处理 {year} 年土地覆盖数据: {tif_file}")
        
        # 读取土地覆盖数据
        tif_path = os.path.join(TIF_PATH, tif_file)
        if not os.path.exists(tif_path):
            print(f"警告: 土地覆盖文件不存在 {tif_path}")
            # 如果文件不存在，用NaN填充该年份的数据
            result_gdf[f'landcover_{year}'] = np.nan
            continue
        
        # 提取土地覆盖值（使用分块处理）
        print("提取土地覆盖值...")
        landcover_values = extract_landcover_values(result_gdf, tif_path, chunk_size)
        result_gdf[f'landcover_{year}'] = landcover_values
    
    print("\n开始应用过滤规则...")
    
    # 检查实际存在的土地覆盖列
    available_landcover_columns = [col for col in result_gdf.columns if col.startswith('landcover_')]
    print(f"可用的土地覆盖列: {available_landcover_columns}")
    
    if len(available_landcover_columns) == 0:
        print("错误: 没有找到任何土地覆盖数据列")
        return
    
    # 创建土地覆盖值矩阵
    landcover_matrix = result_gdf[available_landcover_columns].values
    
    # 初始化过滤标记
    keep_mask = np.zeros(len(result_gdf), dtype=bool)
    
    # 遍历每个点应用过滤规则
    for idx, row in tqdm(result_gdf.iterrows(), total=len(result_gdf), desc="应用过滤规则"):
        class_value = row[SHP_PROPERCITY]
        landcover_values = landcover_matrix[idx]
        
        # 检查是否有缺失值
        if np.any(np.isnan(landcover_values)):
            continue
        
        if class_value == 1 or class_value == 2:  # 人工林或自然林
            # 保留所有年份都为2的点
            if np.all(landcover_values == 2):
                keep_mask[idx] = True
        elif class_value == 3:  # OTHERS
            # 过滤掉任何一年为2的点，即保留所有年份都不为2的点
            if np.all(landcover_values != 2):
                keep_mask[idx] = True
    
    # 应用过滤
    filtered_gdf = result_gdf[keep_mask].copy()
    
    print(f"\n过滤结果:")
    print(f"原始总点数: {len(result_gdf)}")
    print(f"过滤后点数: {len(filtered_gdf)}")
    print(f"保留比例: {len(filtered_gdf)/len(result_gdf)*100:.2f}%")
    
    # 按类别统计
    for class_val in [1, 2, 3]:
        original_count = len(result_gdf[result_gdf[SHP_PROPERCITY] == class_val])
        filtered_count = len(filtered_gdf[filtered_gdf[SHP_PROPERCITY] == class_val])
        class_name = {1: "人工林", 2: "自然林", 3: "OTHERS"}[class_val]
        if original_count > 0:
            print(f"{class_name} - 原始: {original_count}, 过滤后: {filtered_count}, 保留率: {filtered_count/original_count*100:.2f}%")
        else:
            print(f"{class_name} - 原始: {original_count}, 过滤后: {filtered_count}, 保留率: 0.00%")
    
    # 保存过滤后的结果
    # 1. 保存不包含土地覆盖列的过滤结果（只保留原始属性）
    output_file = os.path.join(OUT_PUT_SHP_PATH, "手工补充人工林样本点_LandcoverFilter.shp")
    output_gdf = filtered_gdf.drop(columns=available_landcover_columns).copy()
    output_gdf.to_file(output_file)
    print(f"保存过滤后数据: {output_file}")
    
    # # 2. 保存包含土地覆盖信息的完整数据（用于分析）
    # complete_output_file = os.path.join(OUT_PUT_SHP_PATH, "sdpt_filtered_points_20251011_220536_5倍数量_with_landcover_2017_2024.shp")
    # filtered_gdf.to_file(complete_output_file)
    # print(f"保存完整过滤数据(包含土地覆盖信息): {complete_output_file}")
    
    print("\n过滤完成!")

if __name__ == "__main__":
    # 可以调整chunk_size参数，根据内存情况设置
    # 如果内存较小，可以设置为5000或更小
    # 如果内存充足，可以设置为20000或更大
    filter_sample_points(chunk_size=150000)
