# 验证点按国家分配脚本
# 将验证点数据按照国家边界进行裁剪和分配
# 输入：验证点GPKG文件和国家边界矢量文件
# 输出：按国家分别保存的验证点文件

import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import logging
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置参数
INPUT_POINTS_FILE = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\验证点数据\gpkg\landcover_forest_sample_points.gpkg"
COUNTRY_BOUNDARY_FILE = r"K:\数据\GDAM全球\东南亚国家\southeast_asia_ADM_0.shp"
OUTPUT_DIR = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\数据\验证点数据\按国家裁剪"
LOG_FILE = os.path.join(OUTPUT_DIR, f"country_allocation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 国家名称属性字段
COUNTRY_NAME_FIELD = 'NAME_0'

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

def load_data():
    """加载验证点数据和国家边界数据"""
    logger.info("开始加载数据...")
    
    # 加载验证点数据
    logger.info("加载验证点数据...")
    points_gdf = gpd.read_file(INPUT_POINTS_FILE)
    logger.info(f"验证点数据加载完成，共 {len(points_gdf)} 个点")
    logger.info(f"验证点坐标系: {points_gdf.crs}")
    
    # 加载国家边界数据
    logger.info("加载国家边界数据...")
    countries_gdf = gpd.read_file(COUNTRY_BOUNDARY_FILE)
    logger.info(f"国家边界数据加载完成，共 {len(countries_gdf)} 个国家")
    logger.info(f"国家边界坐标系: {countries_gdf.crs}")
    
    # 检查国家名称字段是否存在
    if COUNTRY_NAME_FIELD not in countries_gdf.columns:
        logger.error(f"国家边界数据中不存在字段: {COUNTRY_NAME_FIELD}")
        logger.info(f"可用字段: {list(countries_gdf.columns)}")
        return None, None
    
    # 显示国家列表
    country_names = countries_gdf[COUNTRY_NAME_FIELD].tolist()
    logger.info(f"包含的国家: {country_names}")
    
    return points_gdf, countries_gdf

def ensure_same_crs(points_gdf, countries_gdf):
    """确保两个数据集使用相同的坐标系"""
    if points_gdf.crs != countries_gdf.crs:
        logger.info(f"坐标系不一致，将验证点数据从 {points_gdf.crs} 转换为 {countries_gdf.crs}")
        points_gdf = points_gdf.to_crs(countries_gdf.crs)
        logger.info("坐标系转换完成")
    else:
        logger.info(f"坐标系一致: {points_gdf.crs}")
    
    return points_gdf

def allocate_points_by_country(points_gdf, countries_gdf):
    """按国家分配验证点"""
    logger.info("开始按国家分配验证点...")
    
    # 执行空间连接，将点分配给对应的国家
    logger.info("执行空间连接...")
    points_with_country = gpd.sjoin(points_gdf, countries_gdf, how='left', predicate='within')
    
    # 统计结果
    total_points = len(points_with_country)
    assigned_points = len(points_with_country.dropna(subset=[COUNTRY_NAME_FIELD]))
    unassigned_points = total_points - assigned_points
    
    logger.info(f"空间连接完成:")
    logger.info(f"  总点数: {total_points}")
    logger.info(f"  已分配点数: {assigned_points}")
    logger.info(f"  未分配点数: {unassigned_points}")
    
    if unassigned_points > 0:
        logger.warning(f"有 {unassigned_points} 个点未能分配到任何国家")
    
    return points_with_country

def save_points_by_country(points_with_country):
    """按国家保存验证点数据"""
    logger.info("开始按国家保存验证点数据...")
    
    # 获取所有有效的国家名称
    valid_countries = points_with_country.dropna(subset=[COUNTRY_NAME_FIELD])
    country_names = valid_countries[COUNTRY_NAME_FIELD].unique()
    
    logger.info(f"需要保存的国家数量: {len(country_names)}")
    
    country_stats = {}
    
    # 为每个国家保存单独的文件
    for country_name in tqdm(country_names, desc="保存国家文件"):
        # 筛选属于当前国家的点
        country_points = points_with_country[points_with_country[COUNTRY_NAME_FIELD] == country_name].copy()
        
        if len(country_points) == 0:
            logger.warning(f"国家 {country_name} 没有验证点")
            continue
        
        # 清理列名，移除空间连接产生的重复列
        # 保留原始验证点的列，移除国家边界的其他属性列
        original_columns = ['id', 'land_cover', 'landcover_value', 'planttree_value', 'geometry']
        country_points = country_points[original_columns + [COUNTRY_NAME_FIELD]]
        
        # 重置索引
        country_points = country_points.reset_index(drop=True)
        country_points['country_id'] = range(1, len(country_points) + 1)
        
        # 生成文件名（处理特殊字符）
        safe_country_name = country_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        output_file = os.path.join(OUTPUT_DIR, f"{safe_country_name}_validation_points.gpkg")
        
        # 保存GPKG文件
        country_points.to_file(output_file, driver='GPKG')
        
        # 同时保存SHP文件
        shp_output_file = os.path.join(OUTPUT_DIR, f"{safe_country_name}_validation_points.shp")
        country_points.to_file(shp_output_file, driver='ESRI Shapefile', encoding='utf-8')
        
        # 统计信息
        type1_count = len(country_points[country_points['planttree_value'] == 1])
        type2_count = len(country_points[country_points['planttree_value'] == 2])
        
        country_stats[country_name] = {
            'total_points': len(country_points),
            'type1_points': type1_count,
            'type2_points': type2_count,
            'output_file': output_file
        }
        
        logger.info(f"国家 {country_name}: {len(country_points)} 个点 (类型1: {type1_count}, 类型2: {type2_count})")
        logger.info(f"  GPKG文件: {output_file}")
        logger.info(f"  SHP文件: {shp_output_file}")
    
    return country_stats

def save_unassigned_points(points_with_country):
    """保存未分配到任何国家的点"""
    unassigned_points = points_with_country[points_with_country[COUNTRY_NAME_FIELD].isna()].copy()
    
    if len(unassigned_points) > 0:
        logger.info(f"保存 {len(unassigned_points)} 个未分配的点...")
        
        # 保留原始列
        original_columns = ['id', 'land_cover', 'landcover_value', 'planttree_value', 'geometry']
        unassigned_points = unassigned_points[original_columns]
        unassigned_points = unassigned_points.reset_index(drop=True)
        
        output_file = os.path.join(OUTPUT_DIR, "unassigned_validation_points.gpkg")
        unassigned_points.to_file(output_file, driver='GPKG')
        
        # 同时保存SHP文件
        shp_output_file = os.path.join(OUTPUT_DIR, "unassigned_validation_points.shp")
        unassigned_points.to_file(shp_output_file, driver='ESRI Shapefile', encoding='utf-8')
        
        logger.info(f"未分配点已保存到: {output_file}")
        logger.info(f"未分配点SHP文件已保存到: {shp_output_file}")
        return output_file
    else:
        logger.info("所有点都已成功分配到国家")
        return None

def generate_summary_report(country_stats, unassigned_file):
    """生成汇总报告"""
    logger.info("\n=== 按国家分配汇总报告 ===")
    
    total_countries = len(country_stats)
    total_assigned_points = sum(stats['total_points'] for stats in country_stats.values())
    total_type1_points = sum(stats['type1_points'] for stats in country_stats.values())
    total_type2_points = sum(stats['type2_points'] for stats in country_stats.values())
    
    logger.info(f"处理的国家数量: {total_countries}")
    logger.info(f"已分配的总点数: {total_assigned_points}")
    logger.info(f"人工林类型1点数: {total_type1_points}")
    logger.info(f"人工林类型2点数: {total_type2_points}")
    
    logger.info("\n各国家详细统计:")
    for country_name, stats in sorted(country_stats.items()):
        logger.info(f"  {country_name}: {stats['total_points']} 个点 (类型1: {stats['type1_points']}, 类型2: {stats['type2_points']})")
    
    if unassigned_file:
        logger.info(f"\n未分配点文件: {unassigned_file}")
    
    # 保存汇总统计到CSV文件
    summary_data = []
    for country_name, stats in country_stats.items():
        summary_data.append({
            'Country': country_name,
            'Total_Points': stats['total_points'],
            'Type1_Points': stats['type1_points'],
            'Type2_Points': stats['type2_points'],
            'Output_File': os.path.basename(stats['output_file'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(OUTPUT_DIR, "country_allocation_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    logger.info(f"\n汇总统计已保存到: {summary_csv}")

def main():
    """主函数"""
    start_time = datetime.now()
    
    logger.info("开始验证点按国家分配处理")
    logger.info(f"输入验证点文件: {INPUT_POINTS_FILE}")
    logger.info(f"国家边界文件: {COUNTRY_BOUNDARY_FILE}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    # 检查输入文件是否存在
    if not check_file_exists(INPUT_POINTS_FILE, "验证点文件"):
        return
    
    if not check_file_exists(COUNTRY_BOUNDARY_FILE, "国家边界文件"):
        return
    
    try:
        # 加载数据
        points_gdf, countries_gdf = load_data()
        if points_gdf is None or countries_gdf is None:
            return
        
        # 确保坐标系一致
        points_gdf = ensure_same_crs(points_gdf, countries_gdf)
        
        # 按国家分配验证点
        points_with_country = allocate_points_by_country(points_gdf, countries_gdf)
        
        # 按国家保存验证点数据
        country_stats = save_points_by_country(points_with_country)
        
        # 保存未分配的点
        unassigned_file = save_unassigned_points(points_with_country)
        
        # 生成汇总报告
        generate_summary_report(country_stats, unassigned_file)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"\n处理完成！总耗时: {duration}")
        logger.info(f"所有输出文件已保存到: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()