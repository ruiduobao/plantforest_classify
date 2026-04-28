"""
分析等面积投影TIF文件的人工林和自然林面积变化情况
作者：锐多宝 (ruiduobao)
功能：读取等面积投影的TIF文件，通过像素计数计算面积
优势：等面积投影下每个像素面积相同，只需统计像素数量即可
"""

import os
import rasterio
import numpy as np
import pandas as pd
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

# 多进程数量（保守设置）
NUM_PROCESSES = 7

# 分块大小（较小的块以节省内存）
CHUNK_SIZE = 1000

# 等面积投影下的像素面积（平方米）
# 10米分辨率 = 100平方米/像素
PIXEL_AREA_M2 = 100.0

# 分类值定义
CLASS_VALUES = {
    1: "人工林",
    2: "自然林",
    3: "其他"
}

def setup_logging():
    """设置日志记录"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    log_filename = os.path.join(OUTPUT_DIR, f"等面积投影面积统计日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

def monitor_memory():
    """监控内存使用情况"""
    memory = psutil.virtual_memory()
    logging.info(f"内存使用情况: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
    return memory.percent

def process_equal_area_tif_file(file_info):
    """
    处理单个等面积投影TIF文件，统计各类别像素数量
    
    Args:
        file_info: 包含文件路径和相关信息的元组 (file_path, zone, year)
    
    Returns:
        dict: 统计结果字典
    """
    file_path, zone, year = file_info
    
    try:
        logging.info(f"开始处理: {os.path.basename(file_path)}")
        
        # 监控内存使用
        memory_percent = monitor_memory()
        if memory_percent > 90:
            logging.warning(f"内存使用率过高 ({memory_percent:.1f}%)，强制垃圾回收")
            gc.collect()
        
        # 初始化像素计数器（使用64位整数避免溢出）
        pixel_counts = {class_val: np.int64(0) for class_val in CLASS_VALUES.keys()}
        total_pixels = np.int64(0)
        
        # 打开TIF文件
        with rasterio.open(file_path) as src:
            # 获取文件基本信息
            width, height = src.width, src.height
            total_file_pixels = width * height
            
            logging.info(f"文件尺寸: {width} x {height} = {total_file_pixels:,} 像素")
            
            # 分块读取文件以节省内存
            processed_chunks = 0
            total_chunks = ((height - 1) // CHUNK_SIZE + 1) * ((width - 1) // CHUNK_SIZE + 1)
            
            for row_start in range(0, height, CHUNK_SIZE):
                for col_start in range(0, width, CHUNK_SIZE):
                    # 计算当前块的边界
                    row_end = min(row_start + CHUNK_SIZE, height)
                    col_end = min(col_start + CHUNK_SIZE, width)
                    
                    # 定义读取窗口
                    window = rasterio.windows.Window(
                        col_start, row_start, 
                        col_end - col_start, row_end - row_start
                    )
                    
                    # 读取数据块
                    chunk_data = src.read(1, window=window)
                    
                    # 统计各类别像素数量
                    for class_val in CLASS_VALUES.keys():
                        mask = (chunk_data == class_val)
                        pixel_counts[class_val] += np.int64(np.sum(mask))
                        del mask  # 立即释放内存
                    
                    # 统计总像素数
                    valid_mask = chunk_data != src.nodata if src.nodata is not None else np.ones_like(chunk_data, dtype=bool)
                    total_pixels += np.int64(np.sum(valid_mask))
                    del valid_mask  # 立即释放内存
                    
                    # 删除数据块并执行垃圾回收
                    del chunk_data
                    
                    processed_chunks += 1
                    
                    # 每处理50个块输出一次进度
                    if processed_chunks % 50 == 0:
                        progress = (processed_chunks / total_chunks) * 100
                        logging.info(f"处理进度: {progress:.1f}% ({processed_chunks}/{total_chunks} 块)")
                    
                    # 每处理10个块强制执行一次垃圾回收
                    if processed_chunks % 10 == 0:
                        gc.collect()
        
        # 计算面积（平方米、公顷、平方公里）
        areas_m2 = {}
        areas_ha = {}
        areas_km2 = {}
        
        for class_val, class_name in CLASS_VALUES.items():
            area_m2 = pixel_counts[class_val] * PIXEL_AREA_M2
            areas_m2[class_name] = area_m2
            areas_ha[class_name] = area_m2 / 10000  # 转换为公顷
            areas_km2[class_name] = area_m2 / 1000000  # 转换为平方公里
        
        # 计算总面积
        total_area_m2 = total_pixels * PIXEL_AREA_M2
        total_area_ha = total_area_m2 / 10000
        total_area_km2 = total_area_m2 / 1000000
        
        # 构建结果字典
        result = {
            'zone': zone,
            'year': year,
            'file_path': file_path,
            'total_pixels': total_pixels,
            'total_area_m2': total_area_m2,
            'total_area_ha': total_area_ha,
            'total_area_km2': total_area_km2,
            'pixel_counts': pixel_counts,
            'areas_m2': areas_m2,
            'areas_ha': areas_ha,
            'areas_km2': areas_km2
        }
        
        logging.info(f"完成处理: {os.path.basename(file_path)}")
        logging.info(f"  人工林: {areas_ha['人工林']:.2f} 公顷")
        logging.info(f"  自然林: {areas_ha['自然林']:.2f} 公顷")
        
        # 强制垃圾回收
        gc.collect()
        
        return result
        
    except Exception as e:
        logging.error(f"处理文件时出错 {file_path}: {str(e)}")
        return None

def collect_equal_area_tif_files():
    """
    收集所有等面积投影TIF文件路径
    
    Returns:
        list: 文件信息列表 [(file_path, zone, year), ...]
    """
    file_infos = []
    
    for zone in ZONES:
        zone_dir = os.path.join(TIF_DIR_PATH, zone)
        
        if not os.path.exists(zone_dir):
            logging.warning(f"Zone目录不存在: {zone_dir}")
            continue
            
        for year in YEARS:
            # 构建等面积投影文件名
            filename = f"optimized_{zone}_{year}_添加颜色映射表_等面积投影.tif"
            file_path = os.path.join(zone_dir, filename)
            
            if os.path.exists(file_path):
                file_infos.append((file_path, zone, year))
                logging.debug(f"找到等面积投影文件: {file_path}")
            else:
                logging.warning(f"等面积投影文件不存在: {file_path}")
    
    logging.info(f"总共收集到 {len(file_infos)} 个等面积投影TIF文件")
    return file_infos

def create_detailed_statistics_table(results):
    """
    创建详细统计表
    
    Args:
        results: 处理结果列表
    
    Returns:
        pd.DataFrame: 详细统计表
    """
    detailed_data = []
    
    for result in results:
        if result is None:
            continue
            
        zone = result['zone']
        year = result['year']
        
        # 添加基本信息行
        row = {
            'Zone': zone,
            'Year': year,
            'Total_Area_Ha': result['total_area_ha'],
            'Total_Area_Km2': result['total_area_km2'],
            'Plantation_Forest_Ha': result['areas_ha']['人工林'],
            'Natural_Forest_Ha': result['areas_ha']['自然林'],
            'Other_Ha': result['areas_ha']['其他'],
            'Plantation_Forest_Km2': result['areas_km2']['人工林'],
            'Natural_Forest_Km2': result['areas_km2']['自然林'],
            'Other_Km2': result['areas_km2']['其他'],
            'Plantation_Forest_Pixels': result['pixel_counts'][1],
            'Natural_Forest_Pixels': result['pixel_counts'][2],
            'Other_Pixels': result['pixel_counts'][3],
            'Total_Pixels': result['total_pixels']
        }
        
        detailed_data.append(row)
    
    # 创建DataFrame并排序
    df = pd.DataFrame(detailed_data)
    if not df.empty:
        df = df.sort_values(['Zone', 'Year'])
    
    return df

def create_summary_statistics_table(results):
    """
    创建汇总统计表（按zone汇总）
    
    Args:
        results: 处理结果列表
    
    Returns:
        pd.DataFrame: 汇总统计表
    """
    # 按zone分组汇总
    zone_summaries = {}
    
    for result in results:
        if result is None:
            continue
            
        zone = result['zone']
        
        if zone not in zone_summaries:
            zone_summaries[zone] = {
                'years': [],
                'total_plantation_ha': 0,
                'total_natural_ha': 0,
                'total_other_ha': 0,
                'total_area_ha': 0,
                'avg_plantation_ha': 0,
                'avg_natural_ha': 0,
                'avg_other_ha': 0
            }
        
        zone_summaries[zone]['years'].append(result['year'])
        zone_summaries[zone]['total_plantation_ha'] += result['areas_ha']['人工林']
        zone_summaries[zone]['total_natural_ha'] += result['areas_ha']['自然林']
        zone_summaries[zone]['total_other_ha'] += result['areas_ha']['其他']
        zone_summaries[zone]['total_area_ha'] += result['total_area_ha']
    
    # 计算平均值
    summary_data = []
    for zone, summary in zone_summaries.items():
        year_count = len(summary['years'])
        if year_count > 0:
            summary_row = {
                'Zone': zone,
                'Year_Count': year_count,
                'Years': f"{min(summary['years'])}-{max(summary['years'])}",
                'Avg_Plantation_Forest_Ha': summary['total_plantation_ha'] / year_count,
                'Avg_Natural_Forest_Ha': summary['total_natural_ha'] / year_count,
                'Avg_Other_Ha': summary['total_other_ha'] / year_count,
                'Avg_Total_Area_Ha': summary['total_area_ha'] / year_count,
                'Total_Plantation_Forest_Ha': summary['total_plantation_ha'],
                'Total_Natural_Forest_Ha': summary['total_natural_ha'],
                'Total_Other_Ha': summary['total_other_ha'],
                'Total_Area_Ha': summary['total_area_ha']
            }
            summary_data.append(summary_row)
    
    # 创建DataFrame并排序
    df = pd.DataFrame(summary_data)
    if not df.empty:
        df = df.sort_values('Zone')
    
    return df

def main():
    """主函数：执行等面积投影文件的面积统计"""
    
    # 设置日志
    log_filename = setup_logging()
    logging.info("="*80)
    logging.info("开始执行等面积投影TIF文件面积统计")
    logging.info(f"日志文件: {log_filename}")
    logging.info(f"像素面积: {PIXEL_AREA_M2} 平方米/像素")
    logging.info("="*80)
    
    # 监控初始内存使用
    monitor_memory()
    
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"创建输出目录: {OUTPUT_DIR}")
    
    # 收集所有等面积投影TIF文件
    logging.info("开始收集等面积投影TIF文件...")
    file_infos = collect_equal_area_tif_files()
    
    if not file_infos:
        logging.error("未找到任何等面积投影TIF文件！请先运行等面积投影转换脚本。")
        return
    
    logging.info(f"共找到 {len(file_infos)} 个等面积投影TIF文件")
    
    # 按zone分组显示文件信息
    zone_file_counts = {}
    for file_path, zone, year in file_infos:
        zone_file_counts[zone] = zone_file_counts.get(zone, 0) + 1
    
    for zone, count in sorted(zone_file_counts.items()):
        logging.info(f"{zone}: {count} 个等面积投影文件")
    
    # 监控内存使用
    monitor_memory()
    
    # 使用多进程处理TIF文件
    logging.info(f"开始使用 {NUM_PROCESSES} 个进程进行面积统计...")
    
    start_time = datetime.now()
    
    try:
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(process_equal_area_tif_file, file_infos)
        
        # 过滤掉失败的结果
        successful_results = [r for r in results if r is not None]
        failed_count = len(file_infos) - len(successful_results)
        
        logging.info(f"处理完成！成功: {len(successful_results)} 个文件，失败: {failed_count} 个文件")
        
        if not successful_results:
            logging.error("没有成功处理的文件，无法生成统计表！")
            return
        
        # 生成详细统计表
        logging.info("生成详细统计表...")
        detailed_df = create_detailed_statistics_table(successful_results)
        
        # 生成汇总统计表
        logging.info("生成汇总统计表...")
        summary_df = create_summary_statistics_table(successful_results)
        
        # 保存结果到Excel文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = os.path.join(OUTPUT_DIR, f"等面积投影面积统计结果_{timestamp}.xlsx")
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            detailed_df.to_excel(writer, sheet_name='详细统计', index=False)
            summary_df.to_excel(writer, sheet_name='汇总统计', index=False)
        
        logging.info(f"统计结果已保存到: {excel_filename}")
        
        # 监控处理后内存使用
        monitor_memory()
        
        # 计算总处理时间
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # 输出处理结果摘要
        logging.info("="*80)
        logging.info("等面积投影面积统计完成！结果摘要：")
        logging.info(f"总处理时间: {total_time}")
        logging.info(f"成功处理文件数: {len(successful_results)}")
        logging.info(f"失败文件数: {failed_count}")
        logging.info(f"详细统计记录数: {len(detailed_df)}")
        logging.info(f"汇总统计记录数: {len(summary_df)}")
        logging.info(f"结果文件: {excel_filename}")
        logging.info(f"日志文件: {log_filename}")
        logging.info("="*80)
        
        # 打印部分结果到控制台
        print("\n" + "="*80)
        print("等面积投影面积统计完成！")
        print(f"总处理时间: {total_time}")
        print(f"成功处理: {len(successful_results)} 个文件")
        print(f"失败: {failed_count} 个文件")
        print(f"结果已保存到: {excel_filename}")
        print("="*80)
        
        # 显示部分统计结果
        if not summary_df.empty:
            print("\n汇总统计预览:")
            print(summary_df[['Zone', 'Year_Count', 'Avg_Plantation_Forest_Ha', 'Avg_Natural_Forest_Ha']].head(10))
        
    except Exception as e:
        logging.error(f"多进程处理过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()