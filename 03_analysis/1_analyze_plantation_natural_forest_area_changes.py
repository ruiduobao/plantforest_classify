
"""
多进程统计各个zone在每一年的人工林、自然林、其他地物的像素个数和面积
作者：锐多宝 (ruiduobao)
功能：统计东南亚10m人工林提取项目中各zone的土地覆盖类型面积变化
优化版本：使用分块读取避免内存溢出，复用像素面积计算提高速度
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import multiprocessing as mp
from multiprocessing import Pool
import logging
from datetime import datetime
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

# 配置参数
TIF_DIR_PATH = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型"
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\统计面积"

# zone 从zone1到zone10
ZONES = [f"zone{i}" for i in range(1, 11)]

# 年份范围
YEAR_START = 2017
YEAR_END = 2024
YEARS = list(range(YEAR_START, YEAR_END + 1))

# 像素值的类型定义
LAND_COVER_TYPES = {
    0: "空值",
    1: "自然林", 
    2: "人工林",
    3: "其他"
}

# 多进程数量（进一步减少以节省内存）
NUM_PROCESSES = 2

# 分块大小（像素）- 进一步减少以控制内存使用
CHUNK_SIZE = 1000

# 全局变量存储每个zone的像素面积（避免重复计算）
ZONE_PIXEL_AREAS = {}

def setup_logging():
    """设置日志记录"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    log_filename = os.path.join(OUTPUT_DIR, f"统计日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

def calculate_pixel_area_for_zone(zone_name, tif_file_path):
    """
    计算指定zone的平均像素面积（复用计算结果）
    
    Args:
        zone_name: zone名称
        tif_file_path: TIF文件路径
    
    Returns:
        float: 平均像素面积（平方米）
    """
    global ZONE_PIXEL_AREAS
    
    # 如果已经计算过该zone的像素面积，直接返回
    if zone_name in ZONE_PIXEL_AREAS:
        logging.info(f"复用zone {zone_name}的像素面积计算结果")
        return ZONE_PIXEL_AREAS[zone_name]
    
    logging.info(f"首次计算zone {zone_name}的像素面积")
    
    with rasterio.open(tif_file_path) as src:
        # 获取地理变换参数
        transform = src.transform
        bounds = src.bounds
        
        # 计算纬度范围中心点
        lat_center = (bounds.bottom + bounds.top) / 2.0
        
        # 在WGS84坐标系下，10m分辨率的像素面积计算
        pixel_size_x = abs(transform[0])  # 经度方向像素大小（度）
        pixel_size_y = abs(transform[4])  # 纬度方向像素大小（度）
        
        # 将度转换为米（使用中心纬度进行近似计算）
        # 1度纬度 ≈ 111320米
        # 1度经度 ≈ 111320 * cos(纬度)米
        lat_rad = np.radians(lat_center)
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(lat_rad)
        
        # 计算平均像素面积（平方米）
        pixel_area_m2 = (pixel_size_x * meters_per_degree_lon) * (pixel_size_y * meters_per_degree_lat)
        
        # 存储到全局变量中供复用
        ZONE_PIXEL_AREAS[zone_name] = pixel_area_m2
        
        logging.info(f"Zone {zone_name}像素面积计算完成，平均面积: {pixel_area_m2:.2f} 平方米")
        
        return pixel_area_m2


def process_tif_file_chunked(file_path):
    """
    使用分块方式处理单个TIF文件，避免内存溢出
    
    Args:
        file_path: TIF文件路径
    
    Returns:
        dict: 包含统计结果的字典
    """
    try:
        # 从文件路径提取zone和年份信息
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        zone_name = parts[1]  # 例如：zone1
        year = int(parts[2])  # 年份
        
        logging.info(f"开始处理文件: {filename}")
        
        # 初始化统计结果
        pixel_counts = {land_type: 0 for land_type in LAND_COVER_TYPES.values()}
        
        with rasterio.open(file_path) as src:
            height, width = src.height, src.width
            
            # 获取该zone的平均像素面积（复用计算）
            avg_pixel_area_m2 = calculate_pixel_area_for_zone(zone_name, file_path)
            
            # 分块处理
            total_chunks = 0
            processed_chunks = 0
            
            # 计算总块数
            for row in range(0, height, CHUNK_SIZE):
                for col in range(0, width, CHUNK_SIZE):
                    total_chunks += 1
            
            logging.info(f"文件 {filename} 将分为 {total_chunks} 个块处理")
            
            # 逐块处理
            for row in range(0, height, CHUNK_SIZE):
                for col in range(0, width, CHUNK_SIZE):
                    # 监控内存使用
                    memory_percent = monitor_memory()
                    if memory_percent > 90:
                        logging.warning(f"内存使用率过高 ({memory_percent:.1f}%)，强制垃圾回收")
                        gc.collect()
                    
                    # 计算当前块的窗口
                    window_height = min(CHUNK_SIZE, height - row)
                    window_width = min(CHUNK_SIZE, width - col)
                    window = Window(col, row, window_width, window_height)
                    
                    # 读取当前块的数据
                    chunk_data = src.read(1, window=window)
                    
                    # 统计每种类型的像素数量
                    for pixel_value, land_type in LAND_COVER_TYPES.items():
                        mask = (chunk_data == pixel_value)
                        pixel_count = np.sum(mask)
                        pixel_counts[land_type] += pixel_count
                        
                        # 立即删除mask以释放内存
                        del mask
                    
                    processed_chunks += 1
                    
                    # 每处理50个块输出一次进度（减少日志频率）
                    if processed_chunks % 50 == 0:
                        progress = (processed_chunks / total_chunks) * 100
                        logging.info(f"文件 {filename} 处理进度: {progress:.1f}% ({processed_chunks}/{total_chunks})")
                    
                    # 清理内存并强制垃圾回收
                    del chunk_data
                    gc.collect()
                    
                    # 每处理10个块强制垃圾回收一次
                    if processed_chunks % 10 == 0:
                        gc.collect()
                    
            # 计算面积（使用平均像素面积）
            areas_ha = {land_type: count * avg_pixel_area_m2 / 10000 for land_type, count in pixel_counts.items()}  # 转换为公顷
            areas_km2 = {land_type: area_ha / 100 for land_type, area_ha in areas_ha.items()}  # 转换为平方公里
            
            # 构建结果
            result = {
                'zone': zone_name,
                'year': year,
                'file_path': file_path,
                'pixel_counts': pixel_counts,
                'areas_ha': areas_ha,
                'areas_km2': areas_km2,
                'avg_pixel_area_m2': avg_pixel_area_m2
            }
            
            logging.info(f"文件 {filename} 处理完成")
            
            # 强制垃圾回收
            gc.collect()
            
            return result
            
    except Exception as e:
        logging.error(f"处理文件时出错 {file_path}: {str(e)}")
        return None


def monitor_memory():
    """监控内存使用情况"""
    memory = psutil.virtual_memory()
    logging.info(f"内存使用情况: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
    return memory.percent

def collect_tif_files():
    """
    收集所有需要处理的TIF文件路径
    
    Returns:
        list: TIF文件路径列表
    """
    tif_files = []
    
    for zone in ZONES:
        zone_dir = os.path.join(TIF_DIR_PATH, zone)
        
        if not os.path.exists(zone_dir):
            logging.warning(f"Zone目录不存在: {zone_dir}")
            continue
            
        for year in YEARS:
            # 构建文件名：optimized_zone3_2017_添加颜色映射表.tif
            filename = f"optimized_{zone}_{year}_添加颜色映射表.tif"
            file_path = os.path.join(zone_dir, filename)
            
            if os.path.exists(file_path):
                tif_files.append(file_path)
                logging.debug(f"找到文件: {file_path}")
            else:
                logging.warning(f"文件不存在: {file_path}")
    
    logging.info(f"总共收集到 {len(tif_files)} 个TIF文件")
    return tif_files


def create_detailed_statistics_table(results):
    """
    创建详细统计表（每个zone每年的数据）
    
    Args:
        results: 处理结果列表
    
    Returns:
        pandas.DataFrame: 详细统计表
    """
    detailed_data = []
    
    for result in results:
        row = {
            'Zone': result['zone'],
            'Year': result['year'],
            'Average_Pixel_Area_m2': result['avg_pixel_area_m2']
        }
        
        # 添加像素数量
        for land_type, count in result['pixel_counts'].items():
            row[f'{land_type}_像素数'] = count
        
        # 添加面积（公顷）
        for land_type, area in result['areas_ha'].items():
            row[f'{land_type}_面积_ha'] = round(area, 2)
        
        detailed_data.append(row)
    
    df = pd.DataFrame(detailed_data)
    
    # 按zone和year排序
    df = df.sort_values(['Zone', 'Year'])
    
    return df


def create_summary_statistics_table(results):
    """
    创建汇总统计表（所有zone的总计）
    
    Args:
        results: 处理结果列表
    
    Returns:
        pandas.DataFrame: 汇总统计表
    """
    # 按年份汇总
    year_summary = {}
    
    for result in results:
        year = result['year']
        
        if year not in year_summary:
            year_summary[year] = {land_type: {'pixels': 0, 'area_ha': 0} for land_type in LAND_COVER_TYPES.values()}
        
        # 累加像素数量和面积
        for land_type, count in result['pixel_counts'].items():
            year_summary[year][land_type]['pixels'] += count
        
        for land_type, area in result['areas_ha'].items():
            year_summary[year][land_type]['area_ha'] += area
    
    # 转换为DataFrame格式
    summary_data = []
    for year, data in sorted(year_summary.items()):
        row = {'Year': year}
        
        for land_type in LAND_COVER_TYPES.values():
            row[f'{land_type}_像素数_总计'] = data[land_type]['pixels']
            row[f'{land_type}_面积_ha_总计'] = round(data[land_type]['area_ha'], 2)
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    return df

def main():
    """主函数：执行多进程统计分析（优化版本）"""
    
    # 设置日志
    log_filename = setup_logging()
    logging.info("="*80)
    logging.info("开始执行多zone土地覆盖类型面积统计分析（优化版本）")
    logging.info(f"日志文件: {log_filename}")
    logging.info("="*80)
    
    # 监控初始内存使用
    monitor_memory()
    
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"创建输出目录: {OUTPUT_DIR}")
    
    # 收集所有TIF文件路径
    logging.info("开始收集TIF文件路径...")
    tif_files = collect_tif_files()
    
    if not tif_files:
        logging.error("未找到任何TIF文件！")
        return
    
    logging.info(f"共找到 {len(tif_files)} 个TIF文件")
    
    # 按zone分组显示文件信息
    zone_file_counts = {}
    for file_path in tif_files:
        filename = os.path.basename(file_path)
        zone_name = filename.split('_')[1]
        zone_file_counts[zone_name] = zone_file_counts.get(zone_name, 0) + 1
    
    for zone, count in sorted(zone_file_counts.items()):
        logging.info(f"{zone}: {count} 个文件")
    
    # 监控内存使用
    monitor_memory()
    
    # 使用多进程处理TIF文件
    logging.info(f"开始使用 {NUM_PROCESSES} 个进程处理TIF文件...")
    
    start_time = datetime.now()
    
    try:
        with Pool(processes=NUM_PROCESSES) as pool:
            # 使用新的分块处理函数
            results = pool.map(process_tif_file_chunked, tif_files)
        
        # 过滤掉None结果（处理失败的文件）
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            logging.error("所有文件处理都失败了！")
            return
        
        logging.info(f"成功处理了 {len(valid_results)} 个文件，失败 {len(tif_files) - len(valid_results)} 个文件")
        
        # 监控处理后内存使用
        monitor_memory()
        
        # 生成统计表格
        logging.info("开始生成统计表格...")
        
        # 1. 生成详细统计表（每个zone每年的数据）
        detailed_df = create_detailed_statistics_table(valid_results)
        detailed_output_path = os.path.join(OUTPUT_DIR, "详细统计_各zone各年土地覆盖面积.xlsx")
        detailed_df.to_excel(detailed_output_path, index=False)
        logging.info(f"详细统计表已保存到: {detailed_output_path}")
        
        # 2. 生成汇总统计表（所有zone的总计）
        summary_df = create_summary_statistics_table(valid_results)
        summary_output_path = os.path.join(OUTPUT_DIR, "汇总统计_所有zone土地覆盖面积.xlsx")
        summary_df.to_excel(summary_output_path, index=False)
        logging.info(f"汇总统计表已保存到: {summary_output_path}")
        
        # 计算总处理时间
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # 输出处理结果摘要
        logging.info("="*80)
        logging.info("处理完成！结果摘要：")
        logging.info(f"总处理时间: {total_time}")
        logging.info(f"处理文件数: {len(valid_results)}")
        logging.info(f"失败文件数: {len(tif_files) - len(valid_results)}")
        logging.info(f"详细统计表: {detailed_output_path}")
        logging.info(f"汇总统计表: {summary_output_path}")
        logging.info("="*80)
        
        # 打印部分结果到控制台
        print("\n" + "="*80)
        print("处理完成！")
        print(f"总处理时间: {total_time}")
        print(f"成功处理: {len(valid_results)} 个文件")
        print(f"详细统计表: {detailed_output_path}")
        print(f"汇总统计表: {summary_output_path}")
        print("="*80)
        
        # 显示汇总统计的前几行
        print("\n汇总统计表预览:")
        print(summary_df.head(10).to_string(index=False))
        
    except Exception as e:
        logging.error(f"多进程处理过程中出现错误: {str(e)}")
        raise



if __name__ == "__main__":
    main()