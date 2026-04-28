"""
将指定的TIF文件转换为等面积投影，简化面积计算
作者：锐多宝 (ruiduobao)
功能：将WGS84坐标系的TIF文件转换为Albers等面积圆锥投影
目的：在等面积投影下，每个像素面积相同，只需统计像素数量即可计算面积
"""

import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import logging
from datetime import datetime
import warnings
import gc
import psutil
import multiprocessing as mp

warnings.filterwarnings('ignore')

# 输入TIF文件路径
tif_path = r"F:\人工林增长和自然林消失\output\plantation_growth_and_natural_forest_disappearance_zone1.tif"

# 输出TIF文件路径
TIF_OUT_PATH = r"F:\人工林增长和自然林消失\output\plantation_growth_and_natural_forest_disappearance_等面积投影.tif"

# 东南亚地区适用的Albers等面积圆锥投影参数
# 标准纬线设置为适合东南亚地区的纬度
EQUAL_AREA_CRS = CRS.from_proj4(
    "+proj=aea +lat_0=0 +lon_0=115 +lat_1=-5 +lat_2=15 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

# 目标分辨率（米）- 保持10米分辨率
TARGET_RESOLUTION = 10.0

def setup_logging(output_dir):
    """
    设置日志记录
    Args:
        output_dir (str): 日志文件存放目录
    """
    # 确保日志输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 设置日志文件名，包含时间戳
    log_filename = os.path.join(output_dir, f"等面积投影转换日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为INFO
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),  # 输出到文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return log_filename

def monitor_memory():
    """监控内存使用情况"""
    # 获取虚拟内存信息
    memory = psutil.virtual_memory()
    # 记录内存使用百分比和具体数值
    logging.info(f"内存使用情况: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
    return memory.percent

def reproject_to_equal_area(input_file_path, output_file_path):
    """
    将单个TIF文件重投影为等面积投影
    
    Args:
        input_file_path (str): 输入TIF文件路径
        output_file_path (str): 输出TIF文件路径
    
    Returns:
        str: 输出文件路径，如果成功的话
    """
    try:
        # 如果输出文件已存在，跳过处理
        if os.path.exists(output_file_path):
            logging.info(f"文件已存在，跳过: {os.path.basename(output_file_path)}")
            return output_file_path
        
        logging.info(f"开始处理: {os.path.basename(input_file_path)}")
        
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
                    num_threads=mp.cpu_count()  # 使用所有CPU核心
                )
                

        logging.info(f"完成处理: {os.path.basename(output_file_path)}")
        
        # 强制垃圾回收
        gc.collect()
        
        return output_file_path
        
    except Exception as e:
        logging.error(f"处理文件时出错 {input_file_path}: {str(e)}")
        return None

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
    
    # 从输出路径获取输出目录
    output_dir = os.path.dirname(TIF_OUT_PATH)
    
    # 设置日志
    log_filename = setup_logging(output_dir)
    logging.info("="*80)
    logging.info("开始执行TIF文件等面积投影转换")
    logging.info(f"日志文件: {log_filename}")
    logging.info(f"输入文件: {tif_path}")
    logging.info(f"输出文件: {TIF_OUT_PATH}")
    logging.info(f"目标投影: {EQUAL_AREA_CRS}")
    logging.info(f"目标分辨率: {TARGET_RESOLUTION} 米")
    logging.info("="*80)
    
    # 监控初始内存使用
    monitor_memory()
    
    # 检查输入文件是否存在
    if not os.path.exists(tif_path):
        logging.error(f"输入文件不存在: {tif_path}")
        return
        
    # 计算等面积投影下的像素面积
    pixel_area_m2 = calculate_pixel_area()
    
    # 监控内存使用
    monitor_memory()
    
    # 开始处理
    logging.info(f"开始进行等面积投影转换...")
    
    start_time = datetime.now()
    
    try:
        result = reproject_to_equal_area(tif_path, TIF_OUT_PATH)
        
        if result:
            logging.info(f"转换成功！输出文件位于: {result}")
        else:
            logging.error("转换失败！")

        # 监控处理后内存使用
        monitor_memory()
        
        # 计算总处理时间
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # 输出处理结果摘要
        logging.info("="*80)
        logging.info("等面积投影转换完成！结果摘要：")
        logging.info(f"总处理时间: {total_time}")
        if result:
            logging.info(f"成功转换文件: {os.path.basename(result)}")
        else:
            logging.info("转换失败")
        logging.info(f"等面积投影像素面积: {pixel_area_m2} 平方米")
        logging.info(f"日志文件: {log_filename}")
        logging.info("="*80)
        
        # 打印部分结果到控制台
        print("\n" + "="*80)
        print("等面积投影转换完成！")
        print(f"总处理时间: {total_time}")
        if result:
            print(f"成功转换文件: {os.path.basename(result)}")
        else:
            print("转换失败")
        print(f"等面积投影像素面积: {pixel_area_m2} 平方米")
        print("="*80)
        
        # 保存像素面积信息到文件
        pixel_area_info_path = os.path.join(output_dir, "等面积投影像素面积信息.txt")
        with open(pixel_area_info_path, 'w', encoding='utf-8') as f:
            f.write(f"等面积投影参数:\n")
            f.write(f"投影: {EQUAL_AREA_CRS}\n")
            f.write(f"分辨率: {TARGET_RESOLUTION} 米\n")
            f.write(f"像素面积: {pixel_area_m2} 平方米\n")
            f.write(f"像素面积（公顷）: {pixel_area_m2 / 10000} 公顷\n")
            f.write(f"像素面积（平方公里）: {pixel_area_m2 / 1000000} 平方公里\n")
        
        logging.info(f"像素面积信息已保存到: {pixel_area_info_path}")
        
    except Exception as e:
        logging.error(f"处理过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()

