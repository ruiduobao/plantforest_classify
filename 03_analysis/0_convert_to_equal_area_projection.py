"""
将所有zone的TIF文件转换为等面积投影，简化面积计算
作者：锐多宝 (ruiduobao)
功能：将WGS84坐标系的TIF文件转换为Albers等面积圆锥投影
目的：在等面积投影下，每个像素面积相同，只需统计像素数量即可计算面积
"""

import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
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
OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\等面积投影转换"

# zone 从zone1到zone10
ZONES = [f"zone{i}" for i in range(1, 11)]

# 年份范围
YEAR_START = 2017
YEAR_END = 2024
YEARS = list(range(YEAR_START, YEAR_END + 1))

# 多进程数量（适中以平衡速度和内存使用）
NUM_PROCESSES = 4

# 东南亚地区适用的Albers等面积圆锥投影参数
# 标准纬线设置为适合东南亚地区的纬度
EQUAL_AREA_CRS = CRS.from_proj4(
    "+proj=aea +lat_0=0 +lon_0=115 +lat_1=-5 +lat_2=15 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

# 目标分辨率（米）- 保持10米分辨率
TARGET_RESOLUTION = 10.0

def setup_logging():
    """设置日志记录"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    log_filename = os.path.join(OUTPUT_DIR, f"等面积投影转换日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
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

def reproject_to_equal_area(input_file_path):
    """
    将单个TIF文件重投影为等面积投影
    
    Args:
        input_file_path: 输入TIF文件路径
    
    Returns:
        str: 输出文件路径，如果成功的话
    """
    try:
        # 构建输出文件路径（在原文件夹中添加后缀）
        input_dir = os.path.dirname(input_file_path)
        input_filename = os.path.basename(input_file_path)
        
        # 添加等面积投影后缀
        name_without_ext = os.path.splitext(input_filename)[0]
        output_filename = f"{name_without_ext}_等面积投影.tif"
        output_file_path = os.path.join(input_dir, output_filename)
        
        # 如果输出文件已存在，跳过处理
        if os.path.exists(output_file_path):
            logging.info(f"文件已存在，跳过: {output_filename}")
            return output_file_path
        
        logging.info(f"开始处理: {input_filename}")
        
        # 监控内存使用
        memory_percent = monitor_memory()
        if memory_percent > 85:
            logging.warning(f"内存使用率较高 ({memory_percent:.1f}%)，强制垃圾回收")
            gc.collect()
        
        # 打开输入文件
        with rasterio.open(input_file_path) as src:
            # 计算重投影的变换参数
            transform, width, height = calculate_default_transform(
                src.crs, 
                EQUAL_AREA_CRS, 
                src.width, 
                src.height, 
                *src.bounds,
                resolution=TARGET_RESOLUTION
            )
            
            # 设置输出文件的元数据
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': EQUAL_AREA_CRS,
                'transform': transform,
                'width': width,
                'height': height,
                'compress': 'lzw',  # 使用LZW压缩减少文件大小
                'tiled': True,      # 使用瓦片格式提高读取效率
                'blockxsize': 512,
                'blockysize': 512
            })
            
            # 执行重投影
            with rasterio.open(output_file_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=EQUAL_AREA_CRS,
                    resampling=Resampling.nearest,  # 使用最近邻重采样保持分类值
                    num_threads=2  # 限制线程数以控制内存使用
                )
                
                # 复制颜色映射表（如果存在）
                if src.colormap(1) is not None:
                    dst.write_colormap(1, src.colormap(1))
        
        logging.info(f"完成处理: {output_filename}")
        
        # 强制垃圾回收
        gc.collect()
        
        return output_file_path
        
    except Exception as e:
        logging.error(f"处理文件时出错 {input_file_path}: {str(e)}")
        return None

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

def calculate_pixel_area():
    """
    计算等面积投影下的像素面积
    
    Returns:
        float: 像素面积（平方米）
    """
    # 在等面积投影下，像素面积 = 分辨率^2
    pixel_area_m2 = TARGET_RESOLUTION * TARGET_RESOLUTION
    logging.info(f"等面积投影下的像素面积: {pixel_area_m2} 平方米")
    return pixel_area_m2

def main():
    """主函数：执行等面积投影转换"""
    
    # 设置日志
    log_filename = setup_logging()
    logging.info("="*80)
    logging.info("开始执行TIF文件等面积投影转换")
    logging.info(f"日志文件: {log_filename}")
    logging.info(f"目标投影: {EQUAL_AREA_CRS}")
    logging.info(f"目标分辨率: {TARGET_RESOLUTION} 米")
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
    
    # 计算等面积投影下的像素面积
    pixel_area_m2 = calculate_pixel_area()
    
    # 监控内存使用
    monitor_memory()
    
    # 使用多进程处理TIF文件
    logging.info(f"开始使用 {NUM_PROCESSES} 个进程进行等面积投影转换...")
    
    start_time = datetime.now()
    
    try:
        with Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(reproject_to_equal_area, tif_files)
        
        # 统计处理结果
        successful_files = [r for r in results if r is not None]
        failed_count = len(tif_files) - len(successful_files)
        
        logging.info(f"转换完成！成功: {len(successful_files)} 个文件，失败: {failed_count} 个文件")
        
        # 监控处理后内存使用
        monitor_memory()
        
        # 计算总处理时间
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # 输出处理结果摘要
        logging.info("="*80)
        logging.info("等面积投影转换完成！结果摘要：")
        logging.info(f"总处理时间: {total_time}")
        logging.info(f"成功转换文件数: {len(successful_files)}")
        logging.info(f"失败文件数: {failed_count}")
        logging.info(f"等面积投影像素面积: {pixel_area_m2} 平方米")
        logging.info(f"日志文件: {log_filename}")
        logging.info("="*80)
        
        # 打印部分结果到控制台
        print("\n" + "="*80)
        print("等面积投影转换完成！")
        print(f"总处理时间: {total_time}")
        print(f"成功转换: {len(successful_files)} 个文件")
        print(f"失败: {failed_count} 个文件")
        print(f"等面积投影像素面积: {pixel_area_m2} 平方米")
        print("="*80)
        
        # 保存像素面积信息到文件
        pixel_area_info_path = os.path.join(OUTPUT_DIR, "等面积投影像素面积信息.txt")
        with open(pixel_area_info_path, 'w', encoding='utf-8') as f:
            f.write(f"等面积投影参数:\n")
            f.write(f"投影: {EQUAL_AREA_CRS}\n")
            f.write(f"分辨率: {TARGET_RESOLUTION} 米\n")
            f.write(f"像素面积: {pixel_area_m2} 平方米\n")
            f.write(f"像素面积（公顷）: {pixel_area_m2 / 10000} 公顷\n")
            f.write(f"像素面积（平方公里）: {pixel_area_m2 / 1000000} 平方公里\n")
        
        logging.info(f"像素面积信息已保存到: {pixel_area_info_path}")
        
    except Exception as e:
        logging.error(f"多进程处理过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()