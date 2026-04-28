"""
测试等面积投影TIF文件的面积统计功能
作者：锐多宝 (ruiduobao)
功能：测试读取等面积投影的TIF文件，验证像素计数统计功能
"""

import os
import rasterio
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

# 配置参数 - 只测试zone1
TIF_DIR_PATH = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型"
OUTPUT_DIR = r"f:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\3.分析\输出结果"

# 只测试zone1
ZONES = ["zone1"]

# 年份范围
YEAR_START = 2017
YEAR_END = 2024
YEARS = list(range(YEAR_START, YEAR_END + 1))

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
    
    log_filename = os.path.join(OUTPUT_DIR, f"测试_等面积投影统计_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
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
        
        # 初始化像素计数器
        pixel_counts = {class_val: 0 for class_val in CLASS_VALUES.keys()}
        total_pixels = 0
        
        # 打开TIF文件
        with rasterio.open(file_path) as src:
            # 获取文件基本信息
            width, height = src.width, src.height
            total_file_pixels = width * height
            
            logging.info(f"文件尺寸: {width} x {height} = {total_file_pixels:,} 像素")
            logging.info(f"投影信息: {src.crs}")
            
            # 读取整个文件（因为是测试，先简单处理）
            data = src.read(1)
            
            # 统计各类别像素数量
            for class_val in CLASS_VALUES.keys():
                mask = (data == class_val)
                pixel_counts[class_val] = np.sum(mask)
                logging.info(f"  {CLASS_VALUES[class_val]}: {pixel_counts[class_val]:,} 像素")
            
            # 统计总像素数（排除nodata）
            if src.nodata is not None:
                valid_mask = data != src.nodata
                total_pixels = np.sum(valid_mask)
            else:
                total_pixels = total_file_pixels
            
            logging.info(f"  有效像素总数: {total_pixels:,}")
        
        # 计算面积（平方米、公顷、平方公里）
        areas_m2 = {}
        areas_ha = {}
        areas_km2 = {}
        
        for class_val, class_name in CLASS_VALUES.items():
            area_m2 = pixel_counts[class_val] * PIXEL_AREA_M2
            areas_m2[class_name] = area_m2
            areas_ha[class_name] = area_m2 / 10000  # 转换为公顷
            areas_km2[class_name] = area_m2 / 1000000  # 转换为平方公里
            
            logging.info(f"  {class_name}面积: {areas_ha[class_name]:.2f} 公顷 ({areas_km2[class_name]:.4f} 平方公里)")
        
        # 计算总面积
        total_area_m2 = total_pixels * PIXEL_AREA_M2
        total_area_ha = total_area_m2 / 10000
        total_area_km2 = total_area_m2 / 1000000
        
        logging.info(f"  总面积: {total_area_ha:.2f} 公顷 ({total_area_km2:.4f} 平方公里)")
        
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

def main():
    """主函数：测试等面积投影文件的面积统计"""
    
    # 设置日志
    log_filename = setup_logging()
    logging.info("="*80)
    logging.info("开始测试等面积投影TIF文件面积统计")
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
    
    # 逐个处理文件（测试版本不使用多进程）
    logging.info("开始逐个处理文件...")
    
    start_time = datetime.now()
    results = []
    
    for i, file_info in enumerate(file_infos):
        logging.info(f"处理进度: {i+1}/{len(file_infos)}")
        result = process_equal_area_tif_file(file_info)
        if result is not None:
            results.append(result)
    
    # 统计处理结果
    successful_count = len(results)
    failed_count = len(file_infos) - successful_count
    
    logging.info(f"处理完成！成功: {successful_count} 个文件，失败: {failed_count} 个文件")
    
    if not results:
        logging.error("没有成功处理的文件！")
        return
    
    # 生成简单的统计表
    logging.info("生成统计表...")
    
    data_rows = []
    for result in results:
        row = {
            'Zone': result['zone'],
            'Year': result['year'],
            'Total_Area_Ha': result['total_area_ha'],
            'Plantation_Forest_Ha': result['areas_ha']['人工林'],
            'Natural_Forest_Ha': result['areas_ha']['自然林'],
            'Other_Ha': result['areas_ha']['其他'],
            'Total_Pixels': result['total_pixels'],
            'Plantation_Pixels': result['pixel_counts'][1],
            'Natural_Pixels': result['pixel_counts'][2],
            'Other_Pixels': result['pixel_counts'][3]
        }
        data_rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data_rows)
    df = df.sort_values(['Zone', 'Year'])
    
    # 保存结果到Excel文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_filename = os.path.join(OUTPUT_DIR, f"测试_等面积投影统计结果_{timestamp}.xlsx")
    
    df.to_excel(excel_filename, index=False)
    logging.info(f"统计结果已保存到: {excel_filename}")
    
    # 计算总处理时间
    end_time = datetime.now()
    total_time = end_time - start_time
    
    # 输出处理结果摘要
    logging.info("="*80)
    logging.info("测试完成！结果摘要：")
    logging.info(f"总处理时间: {total_time}")
    logging.info(f"成功处理文件数: {successful_count}")
    logging.info(f"失败文件数: {failed_count}")
    logging.info(f"结果文件: {excel_filename}")
    logging.info(f"日志文件: {log_filename}")
    logging.info("="*80)
    
    # 打印结果到控制台
    print("\n" + "="*80)
    print("等面积投影统计测试完成！")
    print(f"总处理时间: {total_time}")
    print(f"成功处理: {successful_count} 个文件")
    print(f"失败: {failed_count} 个文件")
    print(f"结果已保存到: {excel_filename}")
    print("="*80)
    
    # 显示统计结果
    if not df.empty:
        print("\n统计结果预览:")
        print(df[['Zone', 'Year', 'Plantation_Forest_Ha', 'Natural_Forest_Ha', 'Total_Area_Ha']])

if __name__ == "__main__":
    main()