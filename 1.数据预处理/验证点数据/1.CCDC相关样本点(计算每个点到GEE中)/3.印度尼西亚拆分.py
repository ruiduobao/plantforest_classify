# 印度尼西亚验证点按经度拆分脚本
# 将印度尼西亚验证点数据按经度拆分为4份，确保每份点数尽量相似
# 输入：Indonesia_validation_points.shp
# 输出：4个按经度分割的验证点文件

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置参数
INPUT_DIR = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\验证点数据\按国家裁剪"
INPUT_FILE = os.path.join(INPUT_DIR, "Indonesia_validation_points.shp")
OUTPUT_DIR = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\验证点数据\印度尼西亚拆分"
LOG_FILE = os.path.join(OUTPUT_DIR, f"indonesia_split_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 拆分份数
NUM_SPLITS = 4

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

def check_file_exists(file_path, file_description):
    """检查文件是否存在"""
    if not os.path.exists(file_path):
        logger.error(f"{file_description}不存在: {file_path}")
        return False
    logger.info(f"{file_description}存在: {file_path}")
    return True

def load_indonesia_data():
    """加载印度尼西亚验证点数据"""
    logger.info("开始加载印度尼西亚验证点数据...")
    
    # 加载数据
    gdf = gpd.read_file(INPUT_FILE)
    logger.info(f"数据加载完成，共 {len(gdf)} 个点")
    logger.info(f"坐标系: {gdf.crs}")
    
    # 检查必要的列
    required_columns = ['geometry']
    missing_columns = [col for col in required_columns if col not in gdf.columns]
    if missing_columns:
        logger.error(f"缺少必要的列: {missing_columns}")
        return None
    
    # 提取经纬度坐标
    gdf['longitude'] = gdf.geometry.x
    gdf['latitude'] = gdf.geometry.y
    
    logger.info(f"经度范围: {gdf['longitude'].min():.6f} 到 {gdf['longitude'].max():.6f}")
    logger.info(f"纬度范围: {gdf['latitude'].min():.6f} 到 {gdf['latitude'].max():.6f}")
    
    return gdf

def calculate_longitude_splits(gdf, num_splits):
    """计算经度分割点，确保每份点数尽量相似"""
    logger.info(f"计算经度分割点，目标分割为 {num_splits} 份...")
    
    # 按经度排序
    gdf_sorted = gdf.sort_values('longitude').reset_index(drop=True)
    total_points = len(gdf_sorted)
    points_per_split = total_points // num_splits
    
    logger.info(f"总点数: {total_points}")
    logger.info(f"每份目标点数: {points_per_split}")
    
    # 计算分割点的索引
    split_indices = []
    for i in range(1, num_splits):
        split_index = i * points_per_split
        split_indices.append(split_index)
    
    # 获取分割点的经度值
    split_longitudes = []
    for idx in split_indices:
        # 取分割点前后的经度平均值，避免在同一经度上分割
        if idx < len(gdf_sorted):
            lon_before = gdf_sorted.iloc[idx-1]['longitude']
            lon_after = gdf_sorted.iloc[idx]['longitude']
            split_lon = (lon_before + lon_after) / 2
            split_longitudes.append(split_lon)
    
    logger.info(f"经度分割点: {split_longitudes}")
    
    return split_longitudes

def split_data_by_longitude(gdf, split_longitudes):
    """按经度分割数据"""
    logger.info("开始按经度分割数据...")
    
    splits = []
    split_names = []
    
    # 第一份：最小经度到第一个分割点
    min_lon = gdf['longitude'].min()
    if len(split_longitudes) > 0:
        first_split = gdf[gdf['longitude'] <= split_longitudes[0]].copy()
        splits.append(first_split)
        split_names.append(f"Indonesia_part1_lon_{min_lon:.3f}_to_{split_longitudes[0]:.3f}")
        logger.info(f"第1份: 经度 {min_lon:.6f} 到 {split_longitudes[0]:.6f}, 点数: {len(first_split)}")
    
    # 中间几份
    for i in range(len(split_longitudes) - 1):
        lon_start = split_longitudes[i]
        lon_end = split_longitudes[i + 1]
        split_data = gdf[(gdf['longitude'] > lon_start) & (gdf['longitude'] <= lon_end)].copy()
        splits.append(split_data)
        split_names.append(f"Indonesia_part{i+2}_lon_{lon_start:.3f}_to_{lon_end:.3f}")
        logger.info(f"第{i+2}份: 经度 {lon_start:.6f} 到 {lon_end:.6f}, 点数: {len(split_data)}")
    
    # 最后一份：最后一个分割点到最大经度
    if len(split_longitudes) > 0:
        max_lon = gdf['longitude'].max()
        last_split = gdf[gdf['longitude'] > split_longitudes[-1]].copy()
        splits.append(last_split)
        split_names.append(f"Indonesia_part{len(split_longitudes)+1}_lon_{split_longitudes[-1]:.3f}_to_{max_lon:.3f}")
        logger.info(f"第{len(split_longitudes)+1}份: 经度 {split_longitudes[-1]:.6f} 到 {max_lon:.6f}, 点数: {len(last_split)}")
    
    return splits, split_names

def save_split_data(splits, split_names):
    """保存分割后的数据"""
    logger.info("开始保存分割后的数据...")
    
    split_stats = {}
    
    for i, (split_data, split_name) in enumerate(zip(splits, split_names)):
        if len(split_data) == 0:
            logger.warning(f"第{i+1}份数据为空，跳过保存")
            continue
        
        # 重置索引并添加新的ID
        split_data = split_data.reset_index(drop=True)
        split_data['split_id'] = range(1, len(split_data) + 1)
        
        # 保存为GPKG格式
        gpkg_file = os.path.join(OUTPUT_DIR, f"{split_name}.gpkg")
        split_data.to_file(gpkg_file, driver='GPKG')
        
        # 保存为SHP格式
        shp_file = os.path.join(OUTPUT_DIR, f"{split_name}.shp")
        split_data.to_file(shp_file, driver='ESRI Shapefile', encoding='utf-8')
        
        # 统计信息
        if 'planttree_value' in split_data.columns:
            type1_count = len(split_data[split_data['planttree_value'] == 1])
            type2_count = len(split_data[split_data['planttree_value'] == 2])
        else:
            type1_count = 0
            type2_count = 0
        
        split_stats[f"Part_{i+1}"] = {
            'total_points': len(split_data),
            'type1_points': type1_count,
            'type2_points': type2_count,
            'longitude_range': f"{split_data['longitude'].min():.6f} - {split_data['longitude'].max():.6f}",
            'gpkg_file': gpkg_file,
            'shp_file': shp_file
        }
        
        logger.info(f"第{i+1}份已保存:")
        logger.info(f"  点数: {len(split_data)} (类型1: {type1_count}, 类型2: {type2_count})")
        logger.info(f"  经度范围: {split_data['longitude'].min():.6f} - {split_data['longitude'].max():.6f}")
        logger.info(f"  GPKG文件: {gpkg_file}")
        logger.info(f"  SHP文件: {shp_file}")
    
    return split_stats

def generate_summary_report(split_stats, original_total):
    """生成汇总报告"""
    logger.info("\n=== 印度尼西亚验证点拆分汇总报告 ===")
    
    total_splits = len(split_stats)
    total_split_points = sum(stats['total_points'] for stats in split_stats.values())
    total_type1_points = sum(stats['type1_points'] for stats in split_stats.values())
    total_type2_points = sum(stats['type2_points'] for stats in split_stats.values())
    
    logger.info(f"原始总点数: {original_total}")
    logger.info(f"拆分后总点数: {total_split_points}")
    logger.info(f"拆分份数: {total_splits}")
    logger.info(f"人工林类型1总点数: {total_type1_points}")
    logger.info(f"人工林类型2总点数: {total_type2_points}")
    
    logger.info("\n各部分详细统计:")
    for part_name, stats in split_stats.items():
        logger.info(f"  {part_name}: {stats['total_points']} 个点 (类型1: {stats['type1_points']}, 类型2: {stats['type2_points']})")
        logger.info(f"    经度范围: {stats['longitude_range']}")
    
    # 计算点数分布的均匀性
    point_counts = [stats['total_points'] for stats in split_stats.values()]
    mean_points = np.mean(point_counts)
    std_points = np.std(point_counts)
    cv = std_points / mean_points * 100  # 变异系数
    
    logger.info(f"\n点数分布统计:")
    logger.info(f"  平均点数: {mean_points:.1f}")
    logger.info(f"  标准差: {std_points:.1f}")
    logger.info(f"  变异系数: {cv:.2f}%")
    
    # 保存汇总统计到CSV文件
    summary_data = []
    for part_name, stats in split_stats.items():
        summary_data.append({
            'Part': part_name,
            'Total_Points': stats['total_points'],
            'Type1_Points': stats['type1_points'],
            'Type2_Points': stats['type2_points'],
            'Longitude_Range': stats['longitude_range'],
            'GPKG_File': os.path.basename(stats['gpkg_file']),
            'SHP_File': os.path.basename(stats['shp_file'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(OUTPUT_DIR, "indonesia_split_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    logger.info(f"\n汇总统计已保存到: {summary_csv}")

def main():
    """主函数"""
    start_time = datetime.now()
    
    logger.info("开始印度尼西亚验证点按经度拆分处理")
    logger.info(f"输入文件: {INPUT_FILE}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"拆分份数: {NUM_SPLITS}")
    
    # 检查输入文件是否存在
    if not check_file_exists(INPUT_FILE, "印度尼西亚验证点文件"):
        return
    
    try:
        # 加载数据
        gdf = load_indonesia_data()
        if gdf is None:
            return
        
        original_total = len(gdf)
        
        # 计算经度分割点
        split_longitudes = calculate_longitude_splits(gdf, NUM_SPLITS)
        
        # 按经度分割数据
        splits, split_names = split_data_by_longitude(gdf, split_longitudes)
        
        # 保存分割后的数据
        split_stats = save_split_data(splits, split_names)
        
        # 生成汇总报告
        generate_summary_report(split_stats, original_total)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"\n处理完成！总耗时: {duration}")
        logger.info(f"所有输出文件已保存到: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()