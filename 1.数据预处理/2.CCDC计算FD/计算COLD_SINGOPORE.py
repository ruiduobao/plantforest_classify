#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLD算法实现 - 新加坡时间序列变化检测
基于pyxccd库实现COLD (Continuous Change Detection and Classification) 算法
处理1985-2024年Landsat时间序列数据

作者: AI Assistant
日期: 2025-01-16
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime, date
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 直接使用环境中已安装的pyxccd库
try:
    # 导入pyxccd的COLD算法模块
    from pyxccd.ccd import cold_detect
    from scipy.stats import chi2
    print("成功导入pyxccd COLD模块")
    PYXCCD_AVAILABLE = True
except ImportError as e:
    print(f"导入pyxccd失败: {e}")
    print("请确保pyxccd库已正确安装")
    PYXCCD_AVAILABLE = False
    cold_detect = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('日志/cold_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class COLDProcessor:
    """
    COLD算法处理器
    用于处理Landsat时间序列数据并进行变化检测
    """
    
    def __init__(self, data_dir, output_dir):
        """
        初始化COLD处理器
        
        Parameters:
        -----------
        data_dir : str
            输入数据目录路径
        output_dir : str
            输出结果目录路径
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # COLD算法参数 - 基于用户建议优化（仅使用pyxccd支持的参数）
        self.cold_params = {
            'p_cg': 0.1,            # 变化概率阈值 (进一步放宽至0.1)
            'conse': 2,             # 连续观测数 (降至最低2个)
            'b_c2': True,           # Collection 2数据标志
            'gap_days': 1095,       # 间隙天数 (3年，适应热带地区季节性云覆盖)
            'lam': 5                # 正则化参数 (进一步降低)
        }
        
        logger.info(f"COLD处理器初始化完成")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"COLD参数: {self.cold_params}")
    
    def load_data(self, start_year=1985, end_year=2024):
        """
        加载指定年份范围的CSV数据文件
        
        Parameters:
        -----------
        start_year : int
            开始年份
        end_year : int
            结束年份
            
        Returns:
        --------
        pd.DataFrame: 合并后的数据框
        """
        logger.info(f"开始加载{start_year}-{end_year}年数据文件...")
        
        all_data = []
        
        # 读取每年的数据文件
        for year in range(start_year, end_year + 1):
            file_path = self.data_dir / f"raw_data_{year}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                    logger.info(f"成功读取{year}年数据: {file_path.name}, 记录数: {len(df)}")
                except Exception as e:
                    logger.error(f"读取文件 {file_path} 失败: {e}")
                    continue
            else:
                logger.warning(f"文件不存在: {file_path}")
        
        if not all_data:
            raise ValueError("没有成功读取任何数据文件")
        
        # 合并数据
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"数据合并完成，总记录数: {len(combined_data)}")
        
        return combined_data
    
    def preprocess_data(self, df):
        """
        数据预处理
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始数据框
            
        Returns:
        --------
        pd.DataFrame: 预处理后的数据框
        """
        logger.info("开始数据预处理...")
        
        # 检查必需的列
        required_cols = ['id', 'date_str', 'days_since_1970', 'B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'THERMAL', 'QA']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        # 复制数据避免修改原始数据
        processed_df = df.copy()
        
        # 转换日期
        processed_df['date'] = pd.to_datetime(processed_df['date_str'])
        processed_df['ordinal_date'] = processed_df['date'].apply(lambda x: x.toordinal())
        
        # 处理无效值
        spectral_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'THERMAL']
        for band in spectral_bands:
            # 将空值和异常值设为NaN
            processed_df[band] = pd.to_numeric(processed_df[band], errors='coerce')
            # 处理Landsat的填充值
            processed_df.loc[processed_df[band] == 65535, band] = np.nan
            processed_df.loc[processed_df[band] < 0, band] = np.nan
            processed_df.loc[processed_df[band] > 65000, band] = np.nan
        
        # 处理QA波段
        processed_df['QA'] = pd.to_numeric(processed_df['QA'], errors='coerce')
        processed_df['QA'] = processed_df['QA'].fillna(0).astype(int)
        
        # 删除所有光谱波段都为空的记录
        valid_mask = processed_df[spectral_bands].notna().any(axis=1)
        processed_df = processed_df[valid_mask].copy()
        
        # 按点ID和日期排序
        processed_df = processed_df.sort_values(['id', 'ordinal_date']).reset_index(drop=True)
        
        logger.info(f"预处理完成，有效记录数: {len(processed_df)}")
        logger.info(f"唯一点数: {processed_df['id'].nunique()}")
        
        return processed_df
    
    def convert_qa_pixel_to_cfmask(self, qa_pixel):
        """
        将Landsat Collection 2的QA_PIXEL转换为cfmask格式
        基于用户建议优化位掩码处理，采用更精确的置信度评估
        
        QA_PIXEL位定义（Landsat Collection 2）:
        Bit 0: Fill (1=填充)
        Bit 1: Dilated Cloud (1=云扩展)
        Bit 2: Cirrus (1=卷云，L8/L9)
        Bit 3: Cloud (1=云)
        Bit 4: Cloud Shadow (1=云阴影)
        Bit 5: Snow (1=雪)
        Bit 6: Clear (1=清晰)
        Bit 7: Water (1=水体)
        Bit 8-9: Cloud Confidence (0=无, 1=低, 2=中, 3=高)
        Bit 10-11: Cloud Shadow Confidence
        Bit 12-13: Snow/Ice Confidence
        Bit 14-15: Cirrus Confidence
        
        Parameters:
        -----------
        qa_pixel : int
            QA_PIXEL值
            
        Returns:
        --------
        int: cfmask值 (0=clear, 1=water, 2=cloud_shadow, 3=snow, 4=cloud, 255=fill)
        """
        # 处理NaN值和非数值
        if pd.isna(qa_pixel) or not isinstance(qa_pixel, (int, float, np.integer, np.floating)):
            return 255  # fill
        
        # 转换为整数
        qa_pixel = int(qa_pixel)
        
        # 检查填充值 (bit 0)
        if qa_pixel & (1 << 0):
            return 255  # fill
        
        # 检查水体 (bit 7)
        if qa_pixel & (1 << 7):
            return 1  # water
        
        # 检查雪 (bit 5) - 热带地区罕见，但保留检测
        if qa_pixel & (1 << 5):
            return 3  # snow
        
        # 优化的云检测：结合标志位和置信度
        cloud_flag = qa_pixel & (1 << 3)
        cloud_conf = (qa_pixel >> 8) & 3  # Bit 8-9
        
        # 云检测策略：标志位为1且置信度>=2（中等以上）才标记为云
        if cloud_flag and cloud_conf >= 2:
            return 4  # cloud
        
        # 优化的云阴影检测
        shadow_flag = qa_pixel & (1 << 4)
        shadow_conf = (qa_pixel >> 10) & 3  # Bit 10-11
        
        # 阴影检测策略：标志位为1且置信度>=2才标记为阴影
        if shadow_flag and shadow_conf >= 2:
            return 2  # cloud shadow
        
        # 卷云检测（针对L8/L9）- 采用宽松策略
        cirrus_flag = qa_pixel & (1 << 2)  # Bit 2
        cirrus_conf = (qa_pixel >> 14) & 3  # Bit 14-15
        
        # 卷云策略：只有标志位为1且置信度为3（最高）才排除
        # 这样可以保留大部分薄卷云观测，因为它们对地表反射率影响较小
        if cirrus_flag and cirrus_conf == 3:
            return 4  # 将高置信度卷云标记为云
        
        # 检查Clear位（Bit 6）- 如果明确标记为清晰则优先保留
        clear_flag = qa_pixel & (1 << 6)
        if clear_flag:
            return 0  # clear
        
        # 对于未明确标记但也没有明显质量问题的像素，标记为清晰
        # 这种策略最大化数据利用率，适合热带地区云覆盖严重的情况
        return 0  # clear
    
    def prepare_point_data(self, point_df):
        """
        为单个点准备COLD算法输入数据
        
        Parameters:
        -----------
        point_df : pd.DataFrame
            单个点的时间序列数据
            
        Returns:
        --------
        tuple: (dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas)
        """
        # 按日期排序
        point_df = point_df.sort_values('ordinal_date').copy()
        
        # 过滤掉所有光谱波段都为空的记录
        spectral_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2']
        valid_mask = point_df[spectral_bands].notna().all(axis=1)
        point_df = point_df[valid_mask].copy()
        
        if len(point_df) == 0:
            raise ValueError("没有有效的光谱数据")
        
        # 转换QA_PIXEL为cfmask格式
        point_df['cfmask'] = point_df['QA'].apply(self.convert_qa_pixel_to_cfmask)
        
        # 过滤掉异常的QA值（保留有效范围0-255）
        valid_qa_mask = (point_df['cfmask'] >= 0) & (point_df['cfmask'] <= 255)
        point_df = point_df[valid_qa_mask].copy()
        
        if len(point_df) == 0:
            raise ValueError("没有有效的QA数据")
        
        # 提取时间序列数据，使用更合理的填充值
        dates = point_df['ordinal_date'].values.astype(np.int64)
        ts_b = point_df['B'].values.astype(np.int64)
        ts_g = point_df['G'].values.astype(np.int64)
        ts_r = point_df['R'].values.astype(np.int64)
        ts_n = point_df['NIR'].values.astype(np.int64)
        ts_s1 = point_df['SWIR1'].values.astype(np.int64)
        ts_s2 = point_df['SWIR2'].values.astype(np.int64)
        # THERMAL波段可能有空值，用中位数填充
        thermal_values = point_df['THERMAL'].fillna(point_df['THERMAL'].median())
        if thermal_values.isna().all():
            thermal_values = thermal_values.fillna(30000)  # 使用默认值
        ts_t = thermal_values.values.astype(np.int64)
        qas = point_df['cfmask'].values.astype(np.int64)
        
        return dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas
    

    
    def run_cold_for_point(self, point_id, point_df):
        """
        为单个点运行COLD算法
        
        Parameters:
        -----------
        point_id : int
            点ID
        point_df : pd.DataFrame
            点的时间序列数据
            
        Returns:
        --------
        dict: COLD结果统计
        """
        try:
            # 检查数据量 - 降至最低3个观测，适应极端云覆盖情况
            if len(point_df) < 3:  # 进一步降低至3个观测
                return {
                    'point_id': point_id,
                    'status': 'insufficient_data',
                    'num_observations': len(point_df),
                    'num_breaks': 0,
                    'error_message': f'观测数量不足: {len(point_df)} < 3'
                }
            
            # 准备数据
            dates, ts_b, ts_g, ts_r, ts_n, ts_s1, ts_s2, ts_t, qas = self.prepare_point_data(point_df)
            
            # 运行COLD算法
            if not PYXCCD_AVAILABLE:
                raise ImportError("pyxccd库不可用，无法运行COLD算法")
            
            try:
                cold_result = cold_detect(
                    dates=dates,
                    ts_b=ts_b,
                    ts_g=ts_g,
                    ts_r=ts_r,
                    ts_n=ts_n,
                    ts_s1=ts_s1,
                    ts_s2=ts_s2,
                    ts_t=ts_t,
                    qas=qas,
                    pos=int(point_id),
                    **self.cold_params
                )
            except Exception as e:
                logger.error(f"pyxccd COLD算法运行失败: {e}")
                raise
            
            # 统计结果 - cold_result是结构化数组
            num_breaks = len(cold_result) if cold_result is not None else 0
            
            # 提取断点信息
            break_dates = []
            if cold_result is not None and len(cold_result) > 0:
                for record in cold_result:
                    # pyxccd返回的是结构化数组，直接访问字段
                    if record['t_break'] > 0:  # t_break > 0 表示有效断点
                        break_dates.append(int(record['t_break']))
            
            return {
                'point_id': point_id,
                'status': 'success',
                'num_observations': len(point_df),
                'num_breaks': num_breaks,
                'break_dates': break_dates,
                'time_range': f"{point_df['date_str'].min()} - {point_df['date_str'].max()}",
                'error_message': None
            }
            
        except Exception as e:
            logger.error(f"点 {point_id} COLD算法运行失败: {str(e)}")
            return {
                'point_id': point_id,
                'status': 'error',
                'num_observations': len(point_df),
                'num_breaks': 0,
                'error_message': str(e)
            }
    
    def process_all_points(self, df, batch_size=100):
        """
        处理所有点的COLD算法
        
        Parameters:
        -----------
        df : pd.DataFrame
            预处理后的数据
        batch_size : int
            批处理大小
            
        Returns:
        --------
        list: 所有点的COLD结果
        """
        logger.info("开始运行COLD算法...")
        
        unique_points = df['id'].unique()
        total_points = len(unique_points)
        logger.info(f"总共需要处理 {total_points} 个点")
        
        results = []
        
        for i, point_id in enumerate(unique_points):
            # 获取点数据
            point_df = df[df['id'] == point_id].copy()
            
            # 运行COLD算法
            result = self.run_cold_for_point(point_id, point_df)
            results.append(result)
            
            # 进度报告
            if (i + 1) % batch_size == 0 or (i + 1) == total_points:
                logger.info(f"已处理 {i + 1}/{total_points} 个点 ({(i + 1)/total_points*100:.1f}%)")
        
        logger.info("COLD算法处理完成")
        return results
    
    def save_results(self, results):
        """
        保存COLD结果
        
        Parameters:
        -----------
        results : list
            COLD结果列表
        """
        logger.info("保存COLD结果...")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        output_file = self.output_dir / f"cold_results_{timestamp}.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"详细结果已保存到: {output_file}")
        
        # 生成统计报告
        self.generate_summary_report(results_df, timestamp)
        
        return output_file
    
    def generate_summary_report(self, results_df, timestamp):
        """
        生成统计报告
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            结果数据框
        timestamp : str
            时间戳
        """
        logger.info("生成统计报告...")
        
        # 基本统计
        total_points = len(results_df)
        successful_points = len(results_df[results_df['status'] == 'success'])
        error_points = len(results_df[results_df['status'] == 'error'])
        insufficient_data_points = len(results_df[results_df['status'] == 'insufficient_data'])
        
        # 断点统计
        success_df = results_df[results_df['status'] == 'success']
        if len(success_df) > 0:
            total_breaks = success_df['num_breaks'].sum()
            avg_breaks = success_df['num_breaks'].mean()
            max_breaks = success_df['num_breaks'].max()
            min_breaks = success_df['num_breaks'].min()
        else:
            total_breaks = avg_breaks = max_breaks = min_breaks = 0
        
        # 生成报告
        report = f"""
COLD算法处理统计报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== 处理概况 ===
总点数: {total_points}
成功处理: {successful_points} ({successful_points/total_points*100:.1f}%)
处理失败: {error_points} ({error_points/total_points*100:.1f}%)
数据不足: {insufficient_data_points} ({insufficient_data_points/total_points*100:.1f}%)

=== 断点统计 ===
总断点数: {total_breaks}
平均断点数: {avg_breaks:.2f}
最大断点数: {max_breaks}
最小断点数: {min_breaks}

=== 断点数量分布 ===
"""
        
        # 断点数量分布
        if len(success_df) > 0:
            break_counts = success_df['num_breaks'].value_counts().sort_index()
            for breaks, count in break_counts.items():
                report += f"{breaks}个断点: {count}个点 ({count/len(success_df)*100:.1f}%)\n"
        
        # 保存报告
        report_file = self.output_dir / f"cold_summary_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"统计报告已保存到: {report_file}")
        print(report)
    
    def run(self):
        """
        运行完整的COLD处理流程
        """
        try:
            logger.info("开始COLD处理流程")
            
            # 1. 加载数据
            df = self.load_data()
            
            # 2. 数据预处理
            processed_df = self.preprocess_data(df)
            
            # 3. 运行COLD算法
            results = self.process_all_points(processed_df)
            
            # 4. 保存结果
            output_file = self.save_results(results)
            
            logger.info("COLD处理流程完成")
            return output_file
            
        except Exception as e:
            logger.error(f"COLD处理流程失败: {e}")
            raise

def main():
    """
    主函数：运行COLD算法分析
    """
    # 配置路径
    data_dir = r"F:\BaiduSyncdisk\论文\东南亚10m人工林提取\代码\2.CCDC计算FD\数据\Singapore_PyCCD_RAW"  # 数据目录
    output_dir = "结果"  # 输出目录
    
    # 创建日志目录
    log_dir = Path("日志")
    log_dir.mkdir(exist_ok=True)
    
    try:
        # 创建处理器并运行
        processor = COLDProcessor(data_dir, output_dir)
        output_file = processor.run()
        
        print(f"\n=== COLD处理完成 ===")
        print(f"结果文件: {output_file}")
        print(f"详细日志请查看: 日志/cold_processing.log")
        
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()