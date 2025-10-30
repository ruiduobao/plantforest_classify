#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Markov Model Optimization Results Analysis Tool - Multi-Processing Accelerated Version
多进程加速版本，专门用于分析实际优化结果文件，避免内存和图像大小问题

Author: 锐多宝 (ruiduobao)
Created: 2024
Purpose: 通过多进程并行处理大幅提升马尔可夫模型优化结果分析的处理速度
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import psutil
warnings.filterwarnings('ignore')

class MultiProcessOptimizationAnalyzer:
    """多进程马尔可夫模型优化结果分析器"""
    
    def __init__(self, output_dir="optimization_analysis_results", chunk_size=700, n_processes=None):
        """
        初始化分析器
        
        Args:
            output_dir: 输出目录
            chunk_size: 数据块大小（行数）- 增大以提高效率
            n_processes: 进程数量，None表示自动检测CPU核心数
        """
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        # 自动检测CPU核心数，但保留一个核心给系统使用
        self.n_processes = max(22, mp.cpu_count() - 10)
        self.logger = self._setup_logger()
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置matplotlib参数避免图像过大
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['figure.dpi'] = 72
        plt.rcParams['savefig.dpi'] = 72
        
        # 记录系统信息
        self.logger.info(f"系统信息: CPU核心数={mp.cpu_count()}, 使用进程数={self.n_processes}")
        self.logger.info(f"可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('MultiProcessAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台输出
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # 文件输出
            log_file = os.path.join(self.output_dir, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            # 确保日志文件所在的目录存在，如果日志文件不存在则创建
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def analyze_optimization_results(self, original_files, optimized_files, years):
        """
        分析优化结果 - 多进程版本
        
        Args:
            original_files: 原始文件列表
            optimized_files: 优化后文件列表
            years: 年份列表
        """
        start_time = time.time()
        self.logger.info("开始马尔可夫模型优化结果分析（多进程加速版）...")
        self.logger.info(f"待处理文件对数量: {len(years)}")
        
        # 准备任务参数
        tasks = []
        for i, (orig_file, opt_file, year) in enumerate(zip(original_files, optimized_files, years)):
            tasks.append({
                'orig_file': orig_file,
                'opt_file': opt_file,
                'year': year,
                'chunk_size': self.chunk_size,
                'task_id': i + 1,
                'total_tasks': len(years)
            })
        
        # 使用多进程处理文件对
        change_stats = []
        detailed_stats = []
        
        self.logger.info(f"启动 {self.n_processes} 个进程进行并行处理...")
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(analyze_file_pair_worker, task): task for task in tasks}
            
            # 使用tqdm显示进度条
            with tqdm(total=len(tasks), desc="分析文件对", unit="对") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        file_stats, file_detailed = future.result()
                        
                        if file_stats:
                            change_stats.append(file_stats)
                            detailed_stats.extend(file_detailed)
                            self.logger.info(f"完成年份 {task['year']} 的分析")
                        else:
                            self.logger.warning(f"年份 {task['year']} 分析失败")
                            
                    except Exception as e:
                        self.logger.error(f"处理年份 {task['year']} 时出错: {str(e)}")
                    
                    pbar.update(1)
        
        if not change_stats:
            self.logger.error("没有文件被成功分析，退出程序")
            return
        
        # 按年份排序结果
        change_stats.sort(key=lambda x: x['year'])
        detailed_stats.sort(key=lambda x: x['year'])
        
        # 保存统计结果
        self._save_statistics(change_stats, detailed_stats)
        
        # 创建可视化图表
        self._create_enhanced_visualizations(change_stats)
        
        # 计算处理时间
        total_time = time.time() - start_time
        self.logger.info(f"优化结果分析完成！总耗时: {total_time:.2f} 秒")
        self.logger.info(f"平均每个文件对处理时间: {total_time/len(years):.2f} 秒")
    
    def _save_statistics(self, change_stats, detailed_stats):
        """保存统计结果到CSV文件"""
        # 保存汇总统计
        summary_df = pd.DataFrame(change_stats)
        summary_file = os.path.join(self.output_dir, "optimization_summary.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"汇总统计已保存到: {summary_file}")
        
        # 保存详细统计
        if detailed_stats:
            detailed_df = pd.DataFrame(detailed_stats)
            detailed_file = os.path.join(self.output_dir, "optimization_detailed.csv")
            detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"详细统计已保存到: {detailed_file}")
            
            # 保存变化类型汇总
            self._save_change_type_summary(detailed_df)
    
    def _save_change_type_summary(self, detailed_df):
        """保存变化类型汇总统计"""
        try:
            # 按变化类型汇总
            change_summary = detailed_df.groupby('change_type').agg({
                'pixel_count': 'sum',
                'percentage': 'mean'
            }).reset_index()
            
            change_summary = change_summary.sort_values('pixel_count', ascending=False)
            
            summary_file = os.path.join(self.output_dir, "change_type_summary.csv")
            change_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"变化类型汇总已保存到: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"保存变化类型汇总时出错: {str(e)}")
    
    def _create_enhanced_visualizations(self, change_stats):
        """创建增强版可视化图表"""
        try:
            df = pd.DataFrame(change_stats)
            
            # 创建多子图布局
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            # 1. 变化率趋势图
            ax1.plot(df['year'], df['change_ratio'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
            ax1.set_title('Annual Optimization Change Ratio Trend', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Year', fontsize=12)
            ax1.set_ylabel('Change Ratio (%)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. 变化像素数量柱状图
            bars = ax2.bar(df['year'], df['changed_pixels'], color='#A23B72', alpha=0.7)
            ax2.set_title('Annual Changed Pixel Count', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_ylabel('Changed Pixel Count', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=9)
            
            # 3. 总像素数与变化像素数对比
            width = 0.35
            x = np.arange(len(df['year']))
            ax3.bar(x - width/2, df['total_pixels'], width, label='Total Pixels', color='#F18F01', alpha=0.7)
            ax3.bar(x + width/2, df['changed_pixels'], width, label='Changed Pixels', color='#C73E1D', alpha=0.7)
            ax3.set_title('Total vs. Changed Pixel Count Comparison', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Year', fontsize=12)
            ax3.set_ylabel('Pixel Count', fontsize=12)
            ax3.set_xticks(x)
            ax3.set_xticklabels(df['year'], rotation=45)
            ax3.legend()
            
            # 4. 变化率统计信息
            ax4.axis('off')
            stats_text = f"""
            Summary Statistics:
            
            Average Change Ratio: {df['change_ratio'].mean():.2f}%
            Maximum Change Ratio: {df['change_ratio'].max():.2f}% (Year: {df.loc[df['change_ratio'].idxmax(), 'year']})
            Minimum Change Ratio: {df['change_ratio'].min():.2f}% (Year: {df.loc[df['change_ratio'].idxmin(), 'year']})
            
            Total Processed Pixels: {df['total_pixels'].sum():,}
            Total Changed Pixels: {df['changed_pixels'].sum():,}
            
            Processing Year Range: {df['year'].min()} - {df['year'].max()}
            Number of File Pairs: {len(df)}
            """
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = os.path.join(self.output_dir, "comprehensive_analysis_results.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comprehensive analysis chart saved to: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualization chart: {str(e)}")


def analyze_file_pair_worker(task):
    """
    工作进程函数：分析单个文件对
    
    Args:
        task: 包含文件路径和参数的字典
    
    Returns:
        tuple: (file_stats, file_detailed)
    """
    orig_file = task['orig_file']
    opt_file = task['opt_file']
    year = task['year']
    chunk_size = task['chunk_size']
    
    try:
        with rasterio.open(orig_file) as orig_src, rasterio.open(opt_file) as opt_src:
            # 检查文件尺寸是否匹配
            if orig_src.shape != opt_src.shape:
                return None, []
            
            height, width = orig_src.shape
            total_pixels = np.int64(height) * np.int64(width)
            changed_pixels = np.int64(0)
            change_details = defaultdict(lambda: np.int64(0))
            
            # 分块处理数据
            num_chunks = (height + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                row_start = chunk_idx * chunk_size
                row_end = min(row_start + chunk_size, height)
                
                # 读取数据块
                orig_chunk = orig_src.read(1, window=((row_start, row_end), (0, width)))
                opt_chunk = opt_src.read(1, window=((row_start, row_end), (0, width)))
                
                # 分析数据块变化
                chunk_changed, chunk_details = analyze_chunk_changes(orig_chunk, opt_chunk)
                
                changed_pixels += np.int64(chunk_changed)
                
                # 合并详细统计
                for change_type, count in chunk_details.items():
                    change_details[change_type] += np.int64(count)
            
            # 计算统计结果
            unchanged_pixels = total_pixels - changed_pixels
            change_ratio = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            file_stats = {
                'year': year,
                'total_pixels': int(total_pixels),
                'changed_pixels': int(changed_pixels),
                'unchanged_pixels': int(unchanged_pixels),
                'change_ratio': change_ratio
            }
            
            # 转换详细统计为列表格式
            file_detailed = []
            for change_type, pixel_count in change_details.items():
                if pixel_count > 0:  # 只记录有变化的类型
                    file_detailed.append({
                        'year': year,
                        'change_type': f"{change_type[0]}->{change_type[1]}",
                        'pixel_count': int(pixel_count),
                        'percentage': (pixel_count / total_pixels) * 100
                    })
            
            return file_stats, file_detailed
            
    except Exception as e:
        return None, []


def analyze_chunk_changes(orig_chunk, opt_chunk):
    """
    分析数据块变化 - 优化版本
    
    Args:
        orig_chunk: 原始数据块
        opt_chunk: 优化后数据块
    
    Returns:
        tuple: (changed_pixels, change_details)
    """
    # 计算变化掩膜
    changed_mask = orig_chunk != opt_chunk
    changed_pixels = np.sum(changed_mask, dtype=np.int64)
    
    # 计算详细变化统计
    change_details = {}
    if changed_pixels > 0:
        # 获取变化像素的值
        changed_orig = orig_chunk[changed_mask]
        changed_opt = opt_chunk[changed_mask]
        
        # 使用numpy的unique函数加速统计
        change_pairs = np.column_stack((changed_orig, changed_opt))
        unique_pairs, counts = np.unique(change_pairs, axis=0, return_counts=True)
        
        # 构建变化详情字典
        for (orig_val, opt_val), count in zip(unique_pairs, counts):
            key = (int(orig_val), int(opt_val))
            change_details[key] = np.int64(count)
    
    return changed_pixels, change_details


def main():
    """主函数"""
    # 文件路径配置
    base_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\2.GEE导出结果_结果合并\zone7"
    optimal_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\zone7首尾填充\\"
    
    # 原始文件列表（优化前）
    original_files = [
        os.path.join(base_dir, "zone7_2017.tif"),
        os.path.join(base_dir, "zone7_2018.tif"),
        os.path.join(base_dir, "zone7_2019.tif"),
        os.path.join(base_dir, "zone7_2020.tif"),
        os.path.join(base_dir, "zone7_2021.tif"),
        os.path.join(base_dir, "zone7_2022.tif"),
        os.path.join(base_dir, "zone7_2023.tif"),
        os.path.join(base_dir, "zone7_2024.tif")
    ]
    
    # 优化后文件列表（马尔可夫模型优化后）- 使用实际存在的带颜色映射表的文件
    # optimized_files = [
    #     os.path.join(optimal_dir, "optimized_zone7_2017_添加颜色映射表.tif"),
    #     os.path.join(optimal_dir, "optimized_zone7_2018_添加颜色映射表.tif"),
    #     os.path.join(optimal_dir, "optimized_zone7_2019_添加颜色映射表.tif"),
    #     os.path.join(optimal_dir, "optimized_zone7_2020_添加颜色映射表.tif"),
    #     os.path.join(optimal_dir, "optimized_zone7_2021_添加颜色映射表.tif"),
    #     os.path.join(optimal_dir, "optimized_zone7_2022_添加颜色映射表.tif"),
    #     os.path.join(optimal_dir, "optimized_zone7_2023_添加颜色映射表.tif"),
    #     os.path.join(optimal_dir, "optimized_zone7_2024_添加颜色映射表.tif")
    # ]
    optimized_files = [
        os.path.join(optimal_dir, "optimized_zone7_2017.tif"),
        os.path.join(optimal_dir, "optimized_zone7_2018.tif"),
        os.path.join(optimal_dir, "optimized_zone7_2019.tif"),
        os.path.join(optimal_dir, "optimized_zone7_2020.tif"),
        os.path.join(optimal_dir, "optimized_zone7_2021.tif"),
        os.path.join(optimal_dir, "optimized_zone7_2022.tif"),
        os.path.join(optimal_dir, "optimized_zone7_2023.tif"),
        os.path.join(optimal_dir, "optimized_zone7_2024.tif")
    ]
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    # 创建多进程分析器并运行分析
    analyzer = MultiProcessOptimizationAnalyzer(
        output_dir=optimal_dir + "optimization_analysis_results_multiprocess",
        chunk_size=700,  # 增大块大小以提高效率
        n_processes=22 # 自动检测CPU核心数
    )
    
    analyzer.analyze_optimization_results(original_files, optimized_files, years)


if __name__ == "__main__":
    main()