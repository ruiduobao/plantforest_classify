# 建立年度转移计数矩阵 (Transition Count Matrix): 需要遍历2017年到2023年的所有像元，统计每一年t到下一年t+1的状态变化数量。
# 统计林地转为林地、非林地的个数；统计非林地转为林地、非林地的个数。
# 林地的值为2；空值为0；非林地为非2，为1\3\4...，都归为非林地
# 分块读取、临时文件保存、内存优化处理

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from tqdm import tqdm
import gc
import tempfile
import shutil
from datetime import datetime

# 栅格文件路径：
TIFS=r"K:\地理所\论文\东南亚10m人工林提取\数据\ESRI\ESRI_2017_2024"
YEARS = list(range(2017, 2025))  # 2017-2024年
FILE_PATTERN = "southeast_asia_landcover_*_mosaic_ESRI_10m.tif"  # 文件匹配模式，匹配合并后的分类结果

# TIFS=r"D:\地理所\论文\东南亚10m人工林提取\数据\第二次分类\合并"
# YEARS = list(range(2017, 2019))  # 2017-2024年
# FILE_PATTERN = "merged_zone5_*.tif"

OUTPUT_FILE=r"D:\地理所\论文\东南亚10m人工林提取\数据\第二次分类_马尔可夫模型_高性能\计算转移概率\transition_count_matrix_2017_2024.csv"

# 临时文件目录
TEMP_DIR = os.path.join(os.path.dirname(OUTPUT_FILE), "temp_blocks")

# 多进程设置
MAX_WORKERS = min(cpu_count() - 1, 8)  # 默认进程数，用户可以修改

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_file_paths():
    """获取所有年份的栅格文件路径"""
    file_paths = {}
    for year in YEARS:
        pattern = os.path.join(TIFS, FILE_PATTERN.replace('*', str(year)))
        files = glob.glob(pattern)
        if files:
            file_paths[year] = files[0]
            logger.info(f"找到 {year} 年文件: {files[0]}")
        else:
            logger.warning(f"未找到 {year} 年的文件，模式: {pattern}")
    return file_paths

def classify_pixel(value):
    """
    将像素值分类为林地(2)或非林地(其他值)
    林地的值为2；空值为0；非林地为非2，为1\3\4...，都归为非林地
    """
    if value == 0:  # 空值
        return 0
    elif value == 2:  # 林地
        return 2
    else:  # 非林地
        return 1

def process_single_block(args):
    """处理单个数据块并保存到临时文件 - 多进程版本"""
    file_paths, window, block_id, year_pairs = args
    
    try:
        logger.info(f"进程开始处理数据块 {block_id}")
        
        # 初始化转移计数
        transition_counts = {}
        for year_pair in year_pairs:
            transition_counts[year_pair] = {
                'forest_to_forest': 0,      # 林地转林地 (2->2)
                'forest_to_nonforest': 0,   # 林地转非林地 (2->1)
                'nonforest_to_forest': 0,   # 非林地转林地 (1->2)
                'nonforest_to_nonforest': 0 # 非林地转非林地 (1->1)
            }
        
        # 每个进程独立读取数据块（避免共享内存）
        year_data = {}
        for year in sorted(file_paths.keys()):
            with rasterio.open(file_paths[year]) as src:
                data = src.read(1, window=window)
                # 分类像素值
                classified_data = np.vectorize(classify_pixel)(data)
                year_data[year] = classified_data
                logger.debug(f"进程已加载 {year} 年数据块 {block_id}")
        
        # 计算年度间转移
        for year_pair in year_pairs:
            year1, year2 = year_pair
            if year1 in year_data and year2 in year_data:
                data1 = year_data[year1]
                data2 = year_data[year2]
                
                # 创建有效像素掩码（排除空值0）- 提前过滤
                valid_mask = (data1 != 0) & (data2 != 0)
                
                # 如果没有有效像素，直接跳过
                if not np.any(valid_mask):
                    continue
                
                # 只对有效像素进行计算，提高效率
                valid_data1 = data1[valid_mask]
                valid_data2 = data2[valid_mask]
                
                # 计算转移（只计算有效像素）
                # 林地转林地 (2->2)
                forest_to_forest = np.sum((valid_data1 == 2) & (valid_data2 == 2))
                
                # 林地转非林地 (2->1)
                forest_to_nonforest = np.sum((valid_data1 == 2) & (valid_data2 == 1))
                
                # 非林地转林地 (1->2)
                nonforest_to_forest = np.sum((valid_data1 == 1) & (valid_data2 == 2))
                
                # 非林地转非林地 (1->1)
                nonforest_to_nonforest = np.sum((valid_data1 == 1) & (valid_data2 == 1))
                
                transition_counts[year_pair]['forest_to_forest'] = int(forest_to_forest)
                transition_counts[year_pair]['forest_to_nonforest'] = int(forest_to_nonforest)
                transition_counts[year_pair]['nonforest_to_forest'] = int(nonforest_to_forest)
                transition_counts[year_pair]['nonforest_to_nonforest'] = int(nonforest_to_nonforest)
        
        # 保存到临时文件
        temp_file = os.path.join(TEMP_DIR, f"block_{block_id}.csv")
        save_block_results(transition_counts, temp_file, block_id)
        
        # 强制释放内存
        del year_data
        del transition_counts
        gc.collect()
        
        logger.info(f"进程完成数据块 {block_id} 处理，结果已保存到 {temp_file}")
        return temp_file
        
    except Exception as e:
        logger.error(f"进程处理数据块 {block_id} 时出错: {e}")
        return None

def save_block_results(transition_counts, temp_file, block_id):
    """保存单个数据块的结果到临时CSV文件"""
    results = []
    for year_pair, counts in transition_counts.items():
        year1, year2 = year_pair
        results.append({
            'block_id': block_id,
            '起始年份': year1,
            '结束年份': year2,
            '年度对': f"{year1}-{year2}",
            '林地转林地': counts['forest_to_forest'],
            '林地转非林地': counts['forest_to_nonforest'],
            '非林地转林地': counts['nonforest_to_forest'],
            '非林地转非林地': counts['nonforest_to_nonforest'],
            '总像素数': sum(counts.values())
        })
    
    df = pd.DataFrame(results)
    df.to_csv(temp_file, index=False, encoding='utf-8-sig')

def merge_temp_files():
    """合并所有临时文件"""
    logger.info("开始合并临时文件...")
    
    # 获取所有临时文件
    temp_files = glob.glob(os.path.join(TEMP_DIR, "block_*.csv"))
    
    if not temp_files:
        logger.error("未找到任何临时文件")
        return None
    
    logger.info(f"找到 {len(temp_files)} 个临时文件")
    
    # 初始化汇总结果
    year_pairs = []
    for i in range(len(YEARS) - 1):
        year_pairs.append((YEARS[i], YEARS[i + 1]))
    
    total_counts = {}
    for year_pair in year_pairs:
        total_counts[year_pair] = {
            'forest_to_forest': 0,
            'forest_to_nonforest': 0,
            'nonforest_to_forest': 0,
            'nonforest_to_nonforest': 0
        }
    
    # 逐个读取临时文件并累加
    for temp_file in tqdm(temp_files, desc="合并临时文件"):
        try:
            df = pd.read_csv(temp_file, encoding='utf-8-sig')
            
            for _, row in df.iterrows():
                year1 = int(row['起始年份'])
                year2 = int(row['结束年份'])
                year_pair = (year1, year2)
                
                if year_pair in total_counts:
                    total_counts[year_pair]['forest_to_forest'] += int(row['林地转林地'])
                    total_counts[year_pair]['forest_to_nonforest'] += int(row['林地转非林地'])
                    total_counts[year_pair]['nonforest_to_forest'] += int(row['非林地转林地'])
                    total_counts[year_pair]['nonforest_to_nonforest'] += int(row['非林地转非林地'])
            
        except Exception as e:
            logger.error(f"读取临时文件 {temp_file} 时出错: {e}")
    
    # 创建最终结果
    final_results = []
    for year_pair in year_pairs:
        year1, year2 = year_pair
        counts = total_counts[year_pair]
        
        final_results.append({
            '起始年份': year1,
            '结束年份': year2,
            '年度对': f"{year1}-{year2}",
            '林地转林地': counts['forest_to_forest'],
            '林地转非林地': counts['forest_to_nonforest'],
            '非林地转林地': counts['nonforest_to_forest'],
            '非林地转非林地': counts['nonforest_to_nonforest'],
            '总像素数': sum(counts.values())
        })
        
        logger.info(f"{year1}-{year2}: 林地→林地={counts['forest_to_forest']}, "
                   f"林地→非林地={counts['forest_to_nonforest']}, "
                   f"非林地→林地={counts['nonforest_to_forest']}, "
                   f"非林地→非林地={counts['nonforest_to_nonforest']}")
    
    return pd.DataFrame(final_results)

def calculate_transition_matrix():
    """计算转移概率矩阵 - 内存优化版本"""
    # 创建临时目录
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    try:
        # 获取文件路径
        file_paths = get_file_paths()
        
        if len(file_paths) < 2:
            logger.error("需要至少2年的数据来计算转移概率")
            return
        
        # 创建年度对
        year_pairs = []
        for i in range(len(YEARS) - 1):
            year1, year2 = YEARS[i], YEARS[i + 1]
            if year1 in file_paths and year2 in file_paths:
                year_pairs.append((year1, year2))
        
        logger.info(f"将计算以下年度对的转移: {year_pairs}")
        
        # 获取第一个文件的元数据来确定分块策略
        first_file = list(file_paths.values())[0]
        with rasterio.open(first_file) as src:
            height, width = src.height, src.width
            logger.info(f"栅格尺寸: {height} x {width}")
        
        # 设置较小的分块大小以减少内存使用
        block_size = 2000  # 减小分块大小
        
        # 创建分块窗口
        windows = []
        block_id = 0
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                window_height = min(block_size, height - row)
                window_width = min(block_size, width - col)
                window = Window(col, row, window_width, window_height)
                windows.append((window, block_id))
                block_id += 1
        
        logger.info(f"总共 {len(windows)} 个数据块需要处理")
        
        # 使用多进程处理数据块
        processed_files = []
        
        # 准备多进程参数
        process_args = []
        for window, block_id in windows:
            process_args.append((file_paths, window, block_id, year_pairs))
        
        logger.info(f"使用 {MAX_WORKERS} 个进程并行处理数据块")
        
        # 使用ProcessPoolExecutor进行多进程处理
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_block = {executor.submit(process_single_block, args): args[2] 
                             for args in process_args}
            
            # 使用tqdm显示进度
            with tqdm(total=len(future_to_block), desc="处理数据块") as pbar:
                for future in as_completed(future_to_block):
                    block_id = future_to_block[future]
                    try:
                        temp_file = future.result()
                        if temp_file:
                            processed_files.append(temp_file)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"数据块 {block_id} 处理失败: {e}")
                        pbar.update(1)
        
        logger.info(f"所有数据块处理完成，共生成 {len(processed_files)} 个临时文件")
        
        # 合并所有临时文件
        final_df = merge_temp_files()
        
        if final_df is not None:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            
            # 保存最终结果
            final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
            logger.info(f"转移计数矩阵已保存到: {OUTPUT_FILE}")
            
            # 打印汇总统计
            print("\n=== 转移计数矩阵汇总 ===")
            print(final_df.to_string(index=False))
            
            return final_df
        else:
            logger.error("合并临时文件失败")
            return None
            
    finally:
        # 清理临时文件
        if os.path.exists(TEMP_DIR):
            try:
                shutil.rmtree(TEMP_DIR)
                logger.info("临时文件已清理")
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {e}")

if __name__ == "__main__":
    logger.info("开始计算转移概率矩阵（内存优化版本）...")
    result_df = calculate_transition_matrix()
    if result_df is not None:
        logger.info("计算完成！")
    else:
        logger.error("计算失败！")