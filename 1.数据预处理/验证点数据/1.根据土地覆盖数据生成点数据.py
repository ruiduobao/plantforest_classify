# 根据土地覆盖数据生成验证点数据
# 对每个矢量网格生成1200个林地随机点（土地覆盖值为2）
# 确保随机点间距至少10m（不在同一像元），避免内存溢出
# 使用分块处理和多进程加速

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Point
import multiprocessing as mp
from tqdm import tqdm
import random
import time
import logging
from datetime import datetime

# 配置参数
LANDCOVER_FILE = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\土地覆盖数据\southeast_asia_landcover_2024_mosaic.tif"
PLANTTREE_FILE = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\全球人工林产品数据\southeast_asia_PlantTree_2021_mosaic.tif"
GRID_FILE = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\分类网格数据\southeast_asia_grid_0.5deg.shp"
OUTPUT_DIR = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\验证点数据"
OUTPUT_GPKG = os.path.join(OUTPUT_DIR, "landcover_forest_sample_points.gpkg")
LOG_FILE = os.path.join(OUTPUT_DIR, f"sample_generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 采样参数
TARGET_POINTS_PER_GRID = 1200  # 每个网格目标点数
TARGET_POINTS_PER_TYPE = 600  # 每种人工林类型的目标点数（1和2各600个）
FOREST_VALUE = 2  # 林地像素值
MIN_DISTANCE_PIXELS = 1  # 最小间距（像素），10m分辨率下1像素=10m
BLOCK_SIZE = 1000  # 处理块大小，避免内存溢出

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_raster_info(raster_path):
    """获取栅格基本信息"""
    with rasterio.open(raster_path) as src:
        return {
            'width': src.width,
            'height': src.height,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'nodata': src.nodata
        }

def sample_points_in_grid(args):
    """在单个网格内采样点"""
    grid_id, grid_geom, landcover_path, planttree_path, target_points, raster_crs = args
    
    try:
        # 打开土地覆盖栅格文件
        with rasterio.open(landcover_path) as landcover_src, rasterio.open(planttree_path) as planttree_src:
            # 验证栅格坐标系与传递的坐标系是否一致
            if landcover_src.crs != raster_crs:
                logger.warning(f"网格 {grid_id} 栅格坐标系 {landcover_src.crs} 与预期坐标系 {raster_crs} 不一致")
                # 注意：此时grid_geom已经在主函数中转换为栅格坐标系
            
            # 裁剪土地覆盖栅格到当前网格
            try:
                landcover_masked, landcover_transform = mask(landcover_src, [grid_geom], crop=True, nodata=landcover_src.nodata)
                landcover_masked = landcover_masked[0]  # 取第一个波段
            except Exception as e:
                logger.warning(f"网格 {grid_id} 土地覆盖栅格裁剪失败: {e}")
                return grid_id, []
            
            # 裁剪人工林栅格到当前网格
            try:
                planttree_masked, planttree_transform = mask(planttree_src, [grid_geom], crop=True, nodata=planttree_src.nodata)
                planttree_masked = planttree_masked[0]  # 取第一个波段
            except Exception as e:
                logger.warning(f"网格 {grid_id} 人工林栅格裁剪失败: {e}")
                return grid_id, []
            
            # 检查是否有有效数据
            if landcover_masked.size == 0 or planttree_masked.size == 0:
                logger.warning(f"网格 {grid_id} 无有效数据")
                return grid_id, []
            
            # 确保两个栅格数据形状一致
            if landcover_masked.shape != planttree_masked.shape:
                logger.warning(f"网格 {grid_id} 栅格形状不一致: 土地覆盖{landcover_masked.shape} vs 人工林{planttree_masked.shape}")
                # 取较小的形状进行裁剪
                min_rows = min(landcover_masked.shape[0], planttree_masked.shape[0])
                min_cols = min(landcover_masked.shape[1], planttree_masked.shape[1])
                landcover_masked = landcover_masked[:min_rows, :min_cols]
                planttree_masked = planttree_masked[:min_rows, :min_cols]
                logger.info(f"网格 {grid_id} 已调整为统一形状: {landcover_masked.shape}")
            
            # 找到所有林地像素（土地覆盖值为2）
            forest_mask = (landcover_masked == FOREST_VALUE)
            
            # 分别处理人工林值为1和2的像素
            planttree_1_mask = forest_mask & (planttree_masked == 1)
            planttree_2_mask = forest_mask & (planttree_masked == 2)
            
            planttree_1_indices = np.where(planttree_1_mask)
            planttree_2_indices = np.where(planttree_2_mask)
            
            logger.info(f"网格 {grid_id} 找到人工林类型1: {len(planttree_1_indices[0])} 个像素")
            logger.info(f"网格 {grid_id} 找到人工林类型2: {len(planttree_2_indices[0])} 个像素")
            
            if len(planttree_1_indices[0]) == 0 and len(planttree_2_indices[0]) == 0:
                logger.warning(f"网格 {grid_id} 无符合条件的林地像素（土地覆盖为林地且人工林数据为1或2）")
                return grid_id, []
            
            # 分别为两种人工林类型生成候选点
            candidate_points_type1 = []
            candidate_points_type2 = []
            
            # 处理人工林类型1的候选点
            if len(planttree_1_indices[0]) > 0:
                candidate_points_type1 = generate_candidate_points(
                    planttree_1_indices, landcover_masked, planttree_masked, 
                    landcover_transform, grid_id, 1, TARGET_POINTS_PER_TYPE
                )
            
            # 处理人工林类型2的候选点
            if len(planttree_2_indices[0]) > 0:
                candidate_points_type2 = generate_candidate_points(
                    planttree_2_indices, landcover_masked, planttree_masked, 
                    landcover_transform, grid_id, 2, TARGET_POINTS_PER_TYPE
                )
            
            # 合并两种类型的候选点
            all_candidate_points = candidate_points_type1 + candidate_points_type2
            
            if len(all_candidate_points) == 0:
                logger.warning(f"网格 {grid_id} 无有效候选点")
                return grid_id, []
            
            # 如果候选点数量少于目标数量，返回所有点
            if len(all_candidate_points) <= target_points:
                selected_points = [(x, y, planttree_val) for x, y, _, _, planttree_val in all_candidate_points]
                logger.info(f"网格 {grid_id} 生成 {len(selected_points)} 个点（少于目标数量）")
                return grid_id, selected_points
            
            # 分别从两种类型中采样
            selected_type1 = spatial_sampling(candidate_points_type1, TARGET_POINTS_PER_TYPE, MIN_DISTANCE_PIXELS) if candidate_points_type1 else []
            selected_type2 = spatial_sampling(candidate_points_type2, TARGET_POINTS_PER_TYPE, MIN_DISTANCE_PIXELS) if candidate_points_type2 else []
            
            # 合并结果
            selected_points = selected_type1 + selected_type2
            
            logger.info(f"网格 {grid_id} 生成 {len(selected_points)} 个点 (类型1: {len(selected_type1)}, 类型2: {len(selected_type2)})")
            return grid_id, selected_points
            
    except Exception as e:
        logger.error(f"处理网格 {grid_id} 时出错: {e}")
        return grid_id, []

def generate_candidate_points(indices, landcover_masked, planttree_masked, landcover_transform, grid_id, planttree_type, target_points):
    """为指定人工林类型生成候选点"""
    candidate_points = []
    
    # 如果林地像素数量较多（超过目标点数的5倍），使用更严格的筛选条件
    # 要求周围7个像素都为林地（像素值=2）
    if len(indices[0]) > target_points * 5:
        logger.info(f"网格 {grid_id} 人工林类型{planttree_type} 像素较多，使用严格筛选（周围7像素都为林地）")
        
        for i, j in zip(indices[0], indices[1]):
            # 检查周围7个像素的窗口（15x15窗口）
            window_data = landcover_masked[max(0, i-7):min(landcover_masked.shape[0], i+8), 
                                         max(0, j-7):min(landcover_masked.shape[1], j+8)]
            
            # 如果窗口大小不是15x15（边界情况），则跳过
            if window_data.shape[0] < 15 or window_data.shape[1] < 15:
                continue
            
            # 检查是否所有像素都是林地（像素值=2）
            if np.all(window_data == FOREST_VALUE):
                # 获取对应位置的人工林数据值
                if i < planttree_masked.shape[0] and j < planttree_masked.shape[1]:
                    planttree_value = planttree_masked[i, j]
                    # 将像素坐标转换为地理坐标（像素中心点）
                    x, y = rasterio.transform.xy(landcover_transform, i, j, offset='center')
                    candidate_points.append((x, y, i, j, planttree_value))  # 包含像素坐标和人工林值
                
        logger.info(f"网格 {grid_id} 人工林类型{planttree_type} 严格筛选后剩余 {len(candidate_points)} 个候选点")
    else:
        # 林地像素数量较少时，使用所有林地像素作为候选点
        logger.info(f"网格 {grid_id} 人工林类型{planttree_type} 像素较少，使用所有像素作为候选点")
        for i, j in zip(indices[0], indices[1]):
            # 获取对应位置的人工林数据值
            if i < planttree_masked.shape[0] and j < planttree_masked.shape[1]:
                planttree_value = planttree_masked[i, j]
                # 将像素坐标转换为地理坐标（像素中心点）
                x, y = rasterio.transform.xy(landcover_transform, i, j, offset='center')
                candidate_points.append((x, y, i, j, planttree_value))  # 包含像素坐标和人工林值
    
    return candidate_points

def spatial_sampling(candidate_points, target_count, min_distance):
    """空间采样，确保点之间的最小距离"""
    if len(candidate_points) <= target_count:
        return [(x, y, planttree_val) for x, y, _, _, planttree_val in candidate_points]
    
    selected = []
    remaining = candidate_points.copy()
    
    # 随机选择第一个点
    first_idx = random.randint(0, len(remaining) - 1)
    selected.append(remaining.pop(first_idx))
    
    # 迭代选择其他点
    while len(selected) < target_count and remaining:
        # 随机选择一个候选点
        candidate_idx = random.randint(0, len(remaining) - 1)
        candidate = remaining[candidate_idx]
        
        # 检查与已选择点的距离
        too_close = False
        for selected_point in selected:
            # 使用像素坐标计算距离（更快）
            pixel_dist = max(abs(candidate[2] - selected_point[2]), 
                           abs(candidate[3] - selected_point[3]))
            if pixel_dist < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected.append(remaining.pop(candidate_idx))
        else:
            remaining.pop(candidate_idx)
        
        # 如果剩余候选点太少，随机选择剩余的点
        if len(remaining) < (target_count - len(selected)) * 2:
            break
    
    # 如果还没有足够的点，从剩余点中随机选择
    if len(selected) < target_count and remaining:
        needed = target_count - len(selected)
        additional = random.sample(remaining, min(needed, len(remaining)))
        selected.extend(additional)
    
    return [(x, y, planttree_val) for x, y, _, _, planttree_val in selected]

def main():
    """主函数"""
    start_time = time.time()
    
    logger.info("开始生成土地覆盖验证点数据")
    logger.info(f"土地覆盖文件: {LANDCOVER_FILE}")
    logger.info(f"网格文件: {GRID_FILE}")
    logger.info(f"输出文件: {OUTPUT_GPKG}")
    logger.info(f"每网格目标点数: {TARGET_POINTS_PER_GRID}")
    logger.info(f"每种人工林类型目标点数: {TARGET_POINTS_PER_TYPE}")
    logger.info(f"林地像素值: {FOREST_VALUE}")
    
    # 检查输入文件是否存在
    if not os.path.exists(LANDCOVER_FILE):
        logger.error(f"土地覆盖文件不存在: {LANDCOVER_FILE}")
        return
    
    if not os.path.exists(GRID_FILE):
        logger.error(f"网格文件不存在: {GRID_FILE}")
        return
    
    # 获取栅格信息
    raster_info = get_raster_info(LANDCOVER_FILE)
    logger.info(f"栅格大小: {raster_info['width']}x{raster_info['height']} 像素")
    logger.info(f"栅格坐标系: {raster_info['crs']}")
    
    # 读取网格数据
    logger.info("读取网格数据...")
    grid_gdf = gpd.read_file(GRID_FILE)
    logger.info(f"网格数量: {len(grid_gdf)}")
    logger.info(f"网格坐标系: {grid_gdf.crs}")
    
    # 确保坐标系一致
    if grid_gdf.crs != raster_info['crs']:
        logger.info(f"检测到坐标系不一致：")
        logger.info(f"  网格坐标系: {grid_gdf.crs}")
        logger.info(f"  栅格坐标系: {raster_info['crs']}")
        logger.info(f"正在转换网格坐标系...")
        grid_gdf = grid_gdf.to_crs(raster_info['crs'])
        logger.info(f"坐标系转换完成")
    else:
        logger.info(f"网格和栅格坐标系一致: {grid_gdf.crs}")
    
    # 准备多进程参数
    args_list = []
    for idx, row in grid_gdf.iterrows():
        grid_id = f"grid_{idx}"
        grid_geom = row.geometry
        args_list.append((grid_id, grid_geom, LANDCOVER_FILE, PLANTTREE_FILE, TARGET_POINTS_PER_GRID, raster_info['crs']))
    
    logger.info(f"准备处理 {len(args_list)} 个网格")
    
    # 使用多进程处理
    cpu_count = mp.cpu_count()
    # process_count = max(1, min(cpu_count - 15, len(args_list)))  # 留出2个核心给系统
    process_count = 9
    logger.info(f"使用 {process_count} 个CPU核心进行处理")
    
    all_points = []
    grid_stats = {}
    batch_counter = 0
    
    with mp.Pool(processes=process_count) as pool:
        # 使用imap显示进度
        results = list(tqdm(
            pool.imap(sample_points_in_grid, args_list),
            total=len(args_list),
            desc="处理网格"
        ))
    
    # 收集结果并每十个网格保存一次
    total_points = 0
    successful_grids = 0
    batch_points = []
    
    for i, (grid_id, points) in enumerate(results):
        if points:
            batch_points.extend(points)
            all_points.extend(points)
            grid_stats[grid_id] = len(points)
            total_points += len(points)
            successful_grids += 1
        else:
            grid_stats[grid_id] = 0
        
        # 每处理10个网格或处理完所有网格时保存一次
        if (i + 1) % 10 == 0 or (i + 1) == len(results):
            if batch_points:
                batch_counter += 1
                
                # 创建点几何和属性
                geometries = [Point(x, y) for x, y, _ in batch_points]
                planttree_values = [planttree_val for _, _, planttree_val in batch_points]
                
                # 创建GeoDataFrame
                points_gdf = gpd.GeoDataFrame({
                    'id': range(len(all_points) - len(batch_points) + 1, len(all_points) + 1),
                    'land_cover': 'forest',
                    'landcover_value': FOREST_VALUE,
                    'planttree_value': planttree_values,
                    'geometry': geometries
                }, crs=raster_info['crs'])
                
                # 保存批次文件
                batch_filename = os.path.join(OUTPUT_DIR, f"landcover_forest_sample_points_batch_{batch_counter:03d}.gpkg")
                points_gdf.to_file(batch_filename, driver='GPKG')
                
                logger.info(f"批次 {batch_counter} 已保存到: {batch_filename} (包含 {len(batch_points)} 个点)")
                batch_points = []  # 清空批次点列表
    
    logger.info(f"处理完成，共生成 {total_points} 个样本点，分 {batch_counter} 个批次保存")
    logger.info(f"成功处理网格数: {successful_grids}/{len(args_list)}")
    
    # 创建GeoDataFrame并保存完整文件
    if all_points:
        logger.info("创建并保存完整的样本点文件...")
        
        # 创建点几何和属性
        geometries = [Point(x, y) for x, y, _ in all_points]
        planttree_values = [planttree_val for _, _, planttree_val in all_points]
        
        # 创建GeoDataFrame
        points_gdf = gpd.GeoDataFrame({
            'id': range(1, len(geometries) + 1),
            'land_cover': 'forest',
            'landcover_value': FOREST_VALUE,
            'planttree_value': planttree_values,
            'geometry': geometries
        }, crs=raster_info['crs'])
        
        # 保存为GPKG格式
        points_gdf.to_file(OUTPUT_GPKG, driver='GPKG')
        
        logger.info(f"完整样本点文件已保存到: {OUTPUT_GPKG}")
        
        # 输出统计信息
        logger.info("\n网格统计:")
        for grid_id, count in grid_stats.items():
            if count > 0:
                logger.info(f"{grid_id}: {count} 个点")
        
        # 计算人工林类型统计
        type1_count = sum(1 for _, _, planttree_val in all_points if planttree_val == 1)
        type2_count = sum(1 for _, _, planttree_val in all_points if planttree_val == 2)
        
        # 计算统计
        points_per_grid = [count for count in grid_stats.values() if count > 0]
        if points_per_grid:
            logger.info(f"\n统计摘要:")
            logger.info(f"总点数: {len(all_points)}")
            logger.info(f"人工林类型1点数: {type1_count}")
            logger.info(f"人工林类型2点数: {type2_count}")
            logger.info(f"平均每网格点数: {np.mean(points_per_grid):.1f}")
            logger.info(f"最大每网格点数: {max(points_per_grid)}")
            logger.info(f"最小每网格点数: {min(points_per_grid)}")
            logger.info(f"达到目标点数的网格: {sum(1 for c in points_per_grid if c >= TARGET_POINTS_PER_GRID)}")
    
    else:
        logger.warning("未生成任何样本点")
    
    end_time = time.time()
    logger.info(f"\n总耗时: {end_time - start_time:.2f} 秒")
    logger.info("土地覆盖验证点数据生成完成!")

if __name__ == "__main__":
    main()