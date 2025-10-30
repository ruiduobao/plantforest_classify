# 根据10米分辨率的分类数据生成花生样本点（像素值为3的区域），确保空间分布均匀
# 读取分类结果数据,分块多进程加快速度
# 筛选规则为，上下左右7个像素点都为花生（像素值=3）时，该像素点为花生样本点
# 使用网格采样确保空间分布均匀
# 生成样本点文件，文件名称为花生样本点.shp（点矢量)
# 栅格位置：秋季作物结果数据

import os
import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import Point
import multiprocessing as mp
from tqdm import tqdm
import random
import time
import math

# 定义输入和输出文件路径
input_raster = r"E:\地理所\论文\河南农作物提取_2025.04\提交数据\农作物分类tif数据\秋季收获作物(玉米、花生、大豆、水稻)\2023年秋季作物结果数据.tif"
output_shapefile = r"F:\BaiduSyncdisk\论文\河南农作物提取\验证点\根据产品数据获得的样本点\根据产品数据获得的花生样本点.shp"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_shapefile), exist_ok=True)

# 定义块大小和重叠区域大小
BLOCK_SIZE = 2500  # 增大块大小减少进程间通信开销
OVERLAP = 7  # 上下左右需要检查7个像素点

# 定义目标样本点数量
TARGET_SAMPLES = 10000  # 目标1万个样本点

# 网格采样参数 - 用于确保均匀分布
GRID_SIZE = 50  # 网格大小（像素），用于均匀采样

def process_block(args):
    """处理一个数据块并返回符合条件的样本点，使用网格采样确保均匀分布"""
    row_start, col_start, raster_path, block_target_samples = args
    
    # 打开栅格文件
    with rasterio.open(raster_path) as src:
        # 考虑重叠区域读取数据块
        row_end = min(row_start + BLOCK_SIZE + OVERLAP, src.height)
        col_end = min(col_start + BLOCK_SIZE + OVERLAP, src.width)
        
        # 确保起始位置不小于0
        effective_row_start = max(0, row_start - OVERLAP)
        effective_col_start = max(0, col_start - OVERLAP)
        
        # 读取数据
        window = Window(effective_col_start, effective_row_start, 
                        col_end - effective_col_start, row_end - effective_row_start)
        data = src.read(1, window=window)
        
        # 初始化结果列表
        sample_points = []
        
        # 计算有效的处理区域（考虑边界）
        valid_row_start = OVERLAP if row_start >= OVERLAP else row_start
        valid_row_end = data.shape[0] - OVERLAP if row_end < src.height else data.shape[0]
        valid_col_start = OVERLAP if col_start >= OVERLAP else col_start
        valid_col_end = data.shape[1] - OVERLAP if col_end < src.width else data.shape[1]
        
        # 使用网格采样确保空间分布均匀
        # 计算网格数量
        grid_rows = math.ceil((valid_row_end - valid_row_start) / GRID_SIZE)
        grid_cols = math.ceil((valid_col_end - valid_col_start) / GRID_SIZE)
        total_grids = grid_rows * grid_cols
        
        if total_grids == 0:
            return sample_points
        
        # 计算每个网格应该采样的点数
        samples_per_grid = max(1, block_target_samples // total_grids)
        
        # 遍历每个网格
        for grid_row in range(grid_rows):
            for grid_col in range(grid_cols):
                # 计算当前网格的边界
                grid_start_row = valid_row_start + grid_row * GRID_SIZE
                grid_end_row = min(valid_row_start + (grid_row + 1) * GRID_SIZE, valid_row_end)
                grid_start_col = valid_col_start + grid_col * GRID_SIZE
                grid_end_col = min(valid_col_start + (grid_col + 1) * GRID_SIZE, valid_col_end)
                
                # 收集当前网格内所有满足条件的花生像素点
                grid_candidates = []
                
                for i in range(grid_start_row, grid_end_row):
                    for j in range(grid_start_col, grid_end_col):
                        if data[i, j] == 3:  # 只考虑花生像素（像素值=3）
                            # 使用NumPy的切片操作一次性检查周围像素
                            window_data = data[max(0, i-7):min(data.shape[0], i+8), 
                                              max(0, j-7):min(data.shape[1], j+8)]
                            
                            # 如果窗口大小不是15x15（边界情况），则跳过
                            if window_data.shape[0] < 15 or window_data.shape[1] < 15:
                                continue
                            
                            # 检查是否所有像素都是花生（像素值=3）
                            if np.all(window_data == 3):
                                grid_candidates.append((i, j))
                
                # 从当前网格的候选点中随机选择指定数量的点
                if grid_candidates:
                    selected_count = min(samples_per_grid, len(grid_candidates))
                    selected_pixels = random.sample(grid_candidates, selected_count)
                    
                    for i, j in selected_pixels:
                        # 计算实际地理坐标
                        x, y = src.xy(effective_row_start + i, effective_col_start + j)
                        sample_points.append((x, y))
                
                # 如果已经找到足够多的点，提前返回
                if len(sample_points) >= block_target_samples * 1.2:
                    return sample_points
        
        return sample_points

def main():
    start_time = time.time()
    
    # 打开栅格文件获取基本信息
    with rasterio.open(input_raster) as src:
        height = src.height
        width = src.width
        crs = src.crs
        nodata_value = src.nodata
    
    print(f"栅格大小: {width}x{height} 像素")
    print(f"NoData值: {nodata_value}")
    print(f"目标像素值: 3 (花生)")
    print(f"目标样本数: {TARGET_SAMPLES}")
    
    # 计算需要处理的块数
    num_blocks_y = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_x = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = num_blocks_x * num_blocks_y
    
    print(f"将栅格分成 {total_blocks} 个块进行处理")
    
    # 计算每个块的目标样本数（基于块的面积比例）
    total_area = height * width
    
    # 准备多进程参数
    args_list = []
    for i in range(0, height, BLOCK_SIZE):
        for j in range(0, width, BLOCK_SIZE):
            # 计算当前块的实际大小
            block_height = min(BLOCK_SIZE, height - i)
            block_width = min(BLOCK_SIZE, width - j)
            block_area = block_height * block_width
            
            # 根据块的面积比例计算目标样本数
            block_target_samples = int((block_area / total_area) * TARGET_SAMPLES * 1.5)  # 多采样50%以便后续选择
            block_target_samples = max(1, block_target_samples)  # 确保至少有1个样本
            
            args_list.append((i, j, input_raster, block_target_samples))
    
    # 使用多进程处理
    all_sample_points = []
    
    # 确定要使用的CPU核心数
    cpu_count = mp.cpu_count()
    process_count = max(1, cpu_count - 5)  # 留出5个核心给系统使用
    
    print(f"使用 {process_count} 个CPU核心进行处理")
    
    with mp.Pool(processes=process_count) as pool:
        # 使用imap_unordered可能会更快，因为不需要保持结果的顺序
        for result in tqdm(pool.imap_unordered(process_block, args_list), total=len(args_list), desc="处理数据块"):
            all_sample_points.extend(result)
    
    print(f"找到 {len(all_sample_points)} 个满足条件的样本点")
    
    # 如果样本点数量超过目标数量，使用空间均匀采样进行筛选
    if len(all_sample_points) > TARGET_SAMPLES:
        print(f"使用空间均匀采样选择 {TARGET_SAMPLES} 个样本点")
        
        # 将样本点转换为numpy数组便于处理
        points_array = np.array(all_sample_points)
        
        # 计算空间范围
        x_min, x_max = points_array[:, 0].min(), points_array[:, 0].max()
        y_min, y_max = points_array[:, 1].min(), points_array[:, 1].max()
        
        # 创建空间网格进行均匀采样
        grid_size = int(np.sqrt(TARGET_SAMPLES))  # 大致的网格大小
        x_step = (x_max - x_min) / grid_size if x_max != x_min else 1
        y_step = (y_max - y_min) / grid_size if y_max != y_min else 1
        
        selected_points = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 定义当前网格的边界
                x_start = x_min + i * x_step
                x_end = x_min + (i + 1) * x_step
                y_start = y_min + j * y_step
                y_end = y_min + (j + 1) * y_step
                
                # 找到当前网格内的所有点
                mask = ((points_array[:, 0] >= x_start) & (points_array[:, 0] < x_end) &
                        (points_array[:, 1] >= y_start) & (points_array[:, 1] < y_end))
                grid_points = points_array[mask]
                
                # 从当前网格随机选择一个点（如果有的话）
                if len(grid_points) > 0:
                    selected_idx = random.randint(0, len(grid_points) - 1)
                    selected_points.append(tuple(grid_points[selected_idx]))
                
                # 如果已经选择了足够的点，停止
                if len(selected_points) >= TARGET_SAMPLES:
                    break
            
            if len(selected_points) >= TARGET_SAMPLES:
                break
        
        # 如果网格采样得到的点数不够，随机补充
        if len(selected_points) < TARGET_SAMPLES:
            remaining_points = [tuple(point) for point in points_array if tuple(point) not in selected_points]
            needed = TARGET_SAMPLES - len(selected_points)
            if needed <= len(remaining_points):
                selected_points.extend(random.sample(remaining_points, needed))
            else:
                selected_points.extend(remaining_points)
        
        all_sample_points = selected_points[:TARGET_SAMPLES]
    
    # 创建GeoDataFrame并保存为shapefile
    if len(all_sample_points) > 0:
        geometries = [Point(x, y) for x, y in all_sample_points]
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=crs)
        
        # 添加字段
        gdf['id'] = range(1, len(gdf) + 1)
        gdf['crop_type'] = 'peanut'  # 标记为花生
        gdf['pixel_value'] = 3  # 记录像素值
        
        # 保存为shapefile
        gdf.to_file(output_shapefile)
        
        end_time = time.time()
        print(f"处理完成，耗时 {end_time - start_time:.2f} 秒")
        print(f"最终生成 {len(all_sample_points)} 个空间均匀分布的花生样本点")
        print(f"样本点已保存到 {output_shapefile}")
    else:
        print("未找到符合条件的花生样本点")

if __name__ == "__main__":
    main()



