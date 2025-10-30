#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用S-CCD算法计算每个点从1985年到2024年的断点数量

本脚本的主要功能：
1. 读取Brunei_GEE_VALUE文件夹下的所有年份CSV数据
2. 对每个点的时间序列数据进行预处理和质量控制
3. 使用pyxccd库的S-CCD算法检测断点
4. 统计每个点的断点数量并保存结果
5. 支持多进程处理以提高计算效率
6. 输出详细的处理日志

作者: 锐多宝 (ruiduobao)
日期: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from pyxccd import cold_detect
try:
    from pyxccd import sccd_detect, cold_detect
except ImportError:
    print("错误：未找到pyxccd库，请先安装：pip install pyxccd")
    sys.exit(1)

# 配置参数
class Config:
    """配置类，包含所有处理参数"""
    # 数据路径配置
    DATA_DIR = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.CCDC计算FD\数据\Brunei_GEE_VALUE"
    OUTPUT_DIR = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.CCDC计算FD\结果"
    LOG_DIR = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.CCDC计算FD\日志"
    
    # 时间范围
    START_YEAR = 1985
    END_YEAR = 2024
    
    # 光谱波段名称（按pyxccd要求的顺序）
    BAND_NAMES = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
    
    # 算法选择
    ALGORITHMS = ['S-CCD', 'CCD']  # 支持的算法类型
    
    # 数据质量控制参数
    MIN_OBSERVATIONS = 50  # 每个点最少观测数量
    MAX_CLOUD_COVER = 80   # 最大云量阈值
    VALID_RANGE = {        # 各波段有效值范围
        'B': (0, 20000),
        'G': (0, 20000), 
        'R': (0, 20000),
        'NIR': (0, 30000),
        'SWIR1': (0, 20000),
        'SWIR2': (0, 20000),
        'THERMAL': (0, 10000)  # 热红外波段范围
    }
    
    # S-CCD算法参数
    SCCD_PARAMS = {
        'probability_threshold': 0.95,  # 降低变化检测概率阈值
        'min_days_conse': 3,           # 减少连续观测最小天数
        'min_num_c': 6,                # 增加最小观测数量
        'max_num_c': 12,               # 增加最大观测数量
        'num_c': 8,                    # 增加默认观测数量
        'tmask_b1_index': 3,           # NIR波段索引（用于时间掩膜）
        'tmask_b2_index': 4            # SWIR1波段索引（用于时间掩膜）
    }
    
    # 多进程配置
    N_PROCESSES = min(cpu_count() - 1, 8)  # 使用CPU核心数-1，最多8个进程
    CHUNK_SIZE = 100  # 每个进程处理的点数量

def setup_logging():
    """设置日志记录"""
    # 创建日志目录
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 设置日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(Config.LOG_DIR, f"sccd_calculation_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_all_data():
    """加载所有年份的数据并合并"""
    logger = logging.getLogger(__name__)
    logger.info("开始加载数据...")
    
    # 查找所有CSV文件
    csv_pattern = os.path.join(Config.DATA_DIR, "landsat_individual_7-10_pts_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        raise FileNotFoundError(f"在{Config.DATA_DIR}中未找到数据文件")
    
    logger.info(f"找到{len(csv_files)}个数据文件")
    
    # 读取并合并所有数据
    all_data = []
    for csv_file in sorted(csv_files):
        year = os.path.basename(csv_file).split('_')[-1].replace('.csv', '')
        logger.info(f"正在读取{year}年数据: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file)
            # 过滤掉空数据行
            df = df.dropna(subset=Config.BAND_NAMES)
            if len(df) > 0:
                all_data.append(df)
                logger.info(f"{year}年有效数据: {len(df)}条")
            else:
                logger.warning(f"{year}年无有效数据")
        except Exception as e:
            logger.error(f"读取{csv_file}时出错: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("没有找到有效的数据")
    
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"总共加载了{len(combined_data)}条有效观测数据")
    
    return combined_data

def preprocess_data(data):
    """数据预处理和质量控制"""
    logger = logging.getLogger(__name__)
    logger.info("开始数据预处理...")
    
    # 转换日期格式
    data['date'] = pd.to_datetime(data['image_date'])
    data['julian_day'] = data['date'].dt.dayofyear + (data['date'].dt.year - Config.START_YEAR) * 365
    
    # 数据质量过滤
    initial_count = len(data)
    
    # 1. 过滤云量过高的数据
    if 'cloud_cover' in data.columns:
        data = data[data['cloud_cover'] <= Config.MAX_CLOUD_COVER]
        logger.info(f"云量过滤后剩余: {len(data)}条 (移除了{initial_count - len(data)}条)")
    
    # 2. 过滤光谱值异常的数据
    for band in Config.BAND_NAMES:
        if band in data.columns:
            min_val, max_val = Config.VALID_RANGE[band]
            data = data[(data[band] >= min_val) & (data[band] <= max_val)]
    
    logger.info(f"光谱值过滤后剩余: {len(data)}条")
    
    # 3. 按点ID分组，过滤观测数量不足的点
    point_counts = data.groupby('id').size()
    valid_points = point_counts[point_counts >= Config.MIN_OBSERVATIONS].index
    data = data[data['id'].isin(valid_points)]
    
    logger.info(f"观测数量过滤后剩余: {len(data)}条数据，涉及{len(valid_points)}个点")
    
    return data, valid_points

def calculate_breakpoints_for_point(point_data, algorithm='S-CCD'):
    """为单个点计算断点（支持S-CCD和CCD两种算法）"""
    point_id = point_data['id'].iloc[0]
    
    try:
        # 按时间排序
        point_data = point_data.sort_values('julian_day')
        
        # 提取时间序列数据
        dates = point_data['julian_day'].values
        
        # 提取光谱数据（转换为整数，pyxccd要求）
        spectral_data = {}
        for band in Config.BAND_NAMES:
            if band in point_data.columns:
                spectral_data[band] = point_data[band].astype(int).values
            else:
                # 如果某个波段缺失，用0填充
                spectral_data[band] = np.zeros(len(dates), dtype=int)
        
        # 为CCD算法创建热红外波段（如果不存在则用NIR波段的值）
        if 'THERMAL' in point_data.columns:
            thermal_data = point_data['THERMAL'].astype(int).values
        else:
            # 如果没有热红外波段，使用NIR波段作为替代
            thermal_data = spectral_data['NIR'].copy()
        
        # 数据验证和清理
        min_obs = max(Config.MIN_OBSERVATIONS, Config.SCCD_PARAMS['min_num_c'], 12)  # 确保至少有12个观测
        if len(dates) < min_obs:
            return {
                'point_id': point_id,
                'algorithm': algorithm,
                'num_observations': len(dates),
                'num_breaks': -1,
                'break_dates': '',
                'time_span_years': 0,
                'status': f'error: insufficient observations ({len(dates)} < {min_obs})'
            }
        
        # 转换为numpy数组并确保数据类型正确（pyxccd要求int64类型）
        dates = np.array(dates, dtype=np.int64)
        
        # 验证日期数据的合理性
        if np.any(dates <= 0) or np.any(dates > 50000):  # 合理的日期范围
            return {
                'point_id': point_id,
                'algorithm': algorithm,
                'num_observations': len(dates),
                'num_breaks': -1,
                'break_dates': '',
                'time_span_years': 0,
                'status': 'error: invalid date values'
            }
        
        # 确保光谱数据是连续的数组
        for band in spectral_data:
            spectral_data[band] = np.ascontiguousarray(spectral_data[band], dtype=np.int64)
            # 检查数据是否有效
            if np.any(np.isnan(spectral_data[band])) or np.any(np.isinf(spectral_data[band])):
                spectral_data[band] = np.nan_to_num(spectral_data[band], nan=0, posinf=10000, neginf=-1000).astype(np.int64)
            # 确保光谱值在合理范围内
            spectral_data[band] = np.clip(spectral_data[band], -2000, 30000)
        
        thermal_data = np.ascontiguousarray(thermal_data, dtype=np.int64)
        if np.any(np.isnan(thermal_data)) or np.any(np.isinf(thermal_data)):
            thermal_data = np.nan_to_num(thermal_data, nan=0, posinf=10000, neginf=-1000).astype(np.int64)
        thermal_data = np.clip(thermal_data, -2000, 30000)
        
        # 创建质量评估数组（全部设为0，表示清晰像素）
        qas = np.zeros(len(dates), dtype=np.int64)
        # 确保QA值在合理范围内
        qas = np.clip(qas, 0, 255)
        
        # 根据算法类型调用相应的检测函数
        if algorithm == 'S-CCD':
            # 调用S-CCD算法，只使用支持的参数
            result = sccd_detect(
                dates=dates,
                ts_b=spectral_data['B'],
                ts_g=spectral_data['G'], 
                ts_r=spectral_data['R'],
                ts_n=spectral_data['NIR'],
                ts_s1=spectral_data['SWIR1'],
                ts_s2=spectral_data['SWIR2'],
                qas=qas,
                p_cg=Config.SCCD_PARAMS['probability_threshold'],
                conse=Config.SCCD_PARAMS['min_days_conse']
            )
        elif algorithm == 'CCD':
            # 调用CCD(COLD)算法
            # 将point_id转换为整数（如果是字符串的话）
            try:
                pos_value = int(point_id) if isinstance(point_id, str) else point_id
            except (ValueError, TypeError):
                pos_value = hash(str(point_id)) % 1000000  # 使用哈希值作为备选
            
            result = cold_detect(
                dates=dates,
                ts_b=spectral_data['B'],
                ts_g=spectral_data['G'], 
                ts_r=spectral_data['R'],
                ts_n=spectral_data['NIR'],
                ts_s1=spectral_data['SWIR1'],
                ts_s2=spectral_data['SWIR2'],
                ts_t=thermal_data,  # 添加热红外波段
                qas=qas,
                p_cg=Config.SCCD_PARAMS['probability_threshold'],
                conse=Config.SCCD_PARAMS['min_days_conse'],
                pos=pos_value  # 确保pos参数是整数
            )
        else:
            raise ValueError(f"不支持的算法类型: {algorithm}")
        
        # 解析结果，计算断点数量
        if result and hasattr(result, 'break_day'):
            # 过滤有效断点（在研究时间范围内）
            start_julian = (Config.START_YEAR - Config.START_YEAR) * 365
            end_julian = (Config.END_YEAR - Config.START_YEAR + 1) * 365
            
            valid_breaks = [
                break_day for break_day in result.break_day 
                if start_julian <= break_day <= end_julian and break_day > 0
            ]
            
            num_breaks = len(valid_breaks)
            
            # 转换断点日期为实际日期
            break_dates = []
            for break_day in valid_breaks:
                year = Config.START_YEAR + int(break_day // 365)
                day_of_year = int(break_day % 365)
                if day_of_year == 0:
                    day_of_year = 365
                    year -= 1
                try:
                    break_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                    break_dates.append(break_date.strftime('%Y-%m-%d'))
                except:
                    break_dates.append(f"{year}-{day_of_year:03d}")
            
            return {
                'point_id': point_id,
                'algorithm': algorithm,
                'num_observations': len(dates),
                'num_breaks': num_breaks,
                'break_dates': ';'.join(break_dates) if break_dates else '',
                'time_span_years': (dates.max() - dates.min()) / 365.25,
                'status': 'success'
            }
        else:
            return {
                'point_id': point_id,
                'algorithm': algorithm,
                'num_observations': len(dates),
                'num_breaks': 0,
                'break_dates': '',
                'time_span_years': (dates.max() - dates.min()) / 365.25,
                'status': 'no_breaks_detected'
            }
            
    except Exception as e:
        return {
            'point_id': point_id,
            'algorithm': algorithm,
            'num_observations': len(point_data) if 'point_data' in locals() else 0,
            'num_breaks': -1,
            'break_dates': '',
            'time_span_years': 0,
            'status': f'error: {str(e)}'
        }

def process_points_chunk(point_ids, all_data, algorithm='S-CCD'):
    """处理一批点的断点计算（支持S-CCD和CCD算法）"""
    results = []
    
    for point_id in point_ids:
        point_data = all_data[all_data['id'] == point_id]
        if len(point_data) > 0:
            result = calculate_breakpoints_for_point(point_data, algorithm)
            results.append(result)
    
    return results

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("="*60)
    logger.info("开始S-CCD和CCD断点检测计算")
    logger.info(f"数据目录: {Config.DATA_DIR}")
    logger.info(f"输出目录: {Config.OUTPUT_DIR}")
    logger.info(f"时间范围: {Config.START_YEAR}-{Config.END_YEAR}")
    logger.info(f"支持算法: {', '.join(Config.ALGORITHMS)}")
    logger.info(f"使用进程数: {Config.N_PROCESSES}")
    logger.info("="*60)
    
    try:
        # 创建输出目录
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # 1. 加载数据
        all_data = load_all_data()
        
        # 2. 数据预处理
        processed_data, valid_points = preprocess_data(all_data)
        
        logger.info(f"开始处理{len(valid_points)}个有效点")
        
        # 3. 准备多进程处理
        point_chunks = [valid_points[i:i + Config.CHUNK_SIZE] 
                       for i in range(0, len(valid_points), Config.CHUNK_SIZE)]
        
        logger.info(f"数据分为{len(point_chunks)}个批次进行处理")
        
        # 4. 对每种算法分别进行计算
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for algorithm in Config.ALGORITHMS:
            logger.info(f"\n开始{algorithm}算法计算...")
            all_results = []
            
            if Config.N_PROCESSES > 1:
                # 多进程处理
                with Pool(processes=Config.N_PROCESSES) as pool:
                    process_func = partial(process_points_chunk, all_data=processed_data, algorithm=algorithm)
                    
                    for i, chunk_results in enumerate(pool.map(process_func, point_chunks)):
                        all_results.extend(chunk_results)
                        logger.info(f"{algorithm} - 完成批次 {i+1}/{len(point_chunks)}，已处理{len(all_results)}个点")
            else:
                # 单进程处理
                for i, chunk in enumerate(point_chunks):
                    chunk_results = process_points_chunk(chunk, processed_data, algorithm)
                    all_results.extend(chunk_results)
                    logger.info(f"{algorithm} - 完成批次 {i+1}/{len(point_chunks)}，已处理{len(all_results)}个点")
            
            # 5. 保存当前算法的结果
            results_df = pd.DataFrame(all_results)
            
            # 生成输出文件名
            algorithm_name = algorithm.lower().replace('-', '')
            output_file = os.path.join(Config.OUTPUT_DIR, f"{algorithm_name}_breakpoints_{timestamp}.csv")
            
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"{algorithm}结果已保存到: {output_file}")
            
            # 6. 统计当前算法结果
            logger.info(f"\n{algorithm}算法处理结果统计:")
            logger.info(f"总处理点数: {len(results_df)}")
            logger.info(f"成功处理: {len(results_df[results_df['status'] == 'success'])}")
            logger.info(f"无断点检测: {len(results_df[results_df['status'] == 'no_breaks_detected'])}")
            logger.info(f"处理出错: {len(results_df[results_df['status'].str.contains('error', na=False)])}")
            
            # 断点数量统计
            valid_results = results_df[results_df['num_breaks'] >= 0]
            if len(valid_results) > 0:
                logger.info(f"\n{algorithm}断点数量统计:")
                logger.info(f"平均断点数: {valid_results['num_breaks'].mean():.2f}")
                logger.info(f"最大断点数: {valid_results['num_breaks'].max()}")
                logger.info(f"最小断点数: {valid_results['num_breaks'].min()}")
                
                # 断点数量分布
                break_counts = valid_results['num_breaks'].value_counts().sort_index()
                logger.info(f"\n{algorithm}断点数量分布:")
                for num_breaks, count in break_counts.items():
                    logger.info(f"  {num_breaks}个断点: {count}个点 ({count/len(valid_results)*100:.1f}%)")
            
            logger.info(f"{algorithm}算法计算完成！\n")
        
        logger.info("="*60)
        logger.info("所有算法的断点检测计算完成！")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()