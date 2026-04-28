#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Markov Model Optimization Results Analysis Tool - Practical Version
Specifically designed for analyzing actual optimization result files, avoiding memory and image size issues

Author: 锐多宝 (ruiduobao)
Created: 2024
Purpose: 分析马尔可夫模型优化结果，避免内存和图像大小问题
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
import time
from tqdm import tqdm
import psutil
warnings.filterwarnings('ignore')

class OptimizationResultAnalyzer:
    """Markov Model Optimization Results Analyzer"""
    
    def __init__(self, output_dir="optimization_analysis_results", chunk_size=2000):
        """
        Initialize analyzer
        
        Args:
            output_dir: Output directory
            chunk_size: Chunk size (number of rows) - 增大以提高处理效率
        """
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.logger = self._setup_logger()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set matplotlib parameters to avoid oversized images
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['figure.dpi'] = 72
        plt.rcParams['savefig.dpi'] = 72
        
        # 记录系统信息
        self.logger.info(f"系统可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        self.logger.info(f"数据块大小设置为: {chunk_size} 行")
        
    def _setup_logger(self):
        """Setup logger with file output"""
        logger = logging.getLogger('OptimizationAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 控制台输出
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # 文件输出 - 添加日志文件
            log_file = os.path.join(self.output_dir, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def analyze_optimization_results(self, original_files, optimized_files, years):
        """
        Analyze optimization results with progress monitoring
        
        Args:
            original_files: List of original files
            optimized_files: List of optimized files
            years: List of years
        """
        start_time = time.time()
        self.logger.info("Starting Markov model optimization results analysis...")
        self.logger.info(f"待处理文件对数量: {len(years)}")
        
        change_stats = []
        detailed_stats = []
        
        # 使用tqdm显示进度条
        with tqdm(total=len(years), desc="分析文件对", unit="对") as pbar:
            for i, (orig_file, opt_file, year) in enumerate(zip(original_files, optimized_files, years)):
                self.logger.info(f"Analyzing file pair {i+1}/{len(years)}: Year {year}")
                pbar.set_postfix({"当前年份": year})
                
                try:
                    # Analyze single file pair
                    file_stats, file_detailed = self._analyze_file_pair(orig_file, opt_file, year)
                    
                    if file_stats:
                        change_stats.append(file_stats)
                        detailed_stats.extend(file_detailed)
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing year {year} files: {str(e)}")
                    continue
                
                pbar.update(1)
        
        if not change_stats:
            self.logger.error("No files were successfully analyzed, exiting program")
            return
        
        # Save statistics
        self._save_statistics(change_stats, detailed_stats)
        
        # Create visualizations (enhanced version)
        self._create_enhanced_visualizations(change_stats)
        
        # 计算处理时间
        total_time = time.time() - start_time
        self.logger.info(f"Optimization results analysis completed! 总耗时: {total_time:.2f} 秒")
        self.logger.info(f"平均每个文件对处理时间: {total_time/len(years):.2f} 秒")
    
    def _analyze_file_pair(self, orig_file, opt_file, year):
        """Analyze single file pair with enhanced progress monitoring"""
        self.logger.info(f"  Analyzing files for year {year}...")
        
        try:
            with rasterio.open(orig_file) as orig_src, rasterio.open(opt_file) as opt_src:
                # Check if file dimensions match
                if orig_src.shape != opt_src.shape:
                    self.logger.error(f"File dimensions do not match: {orig_src.shape} vs {opt_src.shape}")
                    return None, []
                
                height, width = orig_src.shape
                total_pixels = np.int64(height) * np.int64(width)
                changed_pixels = np.int64(0)
                change_details = defaultdict(lambda: np.int64(0))
                
                # 计算数据块数量
                num_chunks = (height + self.chunk_size - 1) // self.chunk_size
                self.logger.info(f"  文件尺寸: {height}x{width}, 总像素数: {total_pixels:,}, 数据块数: {num_chunks}")
                
                # Process by chunks with progress bar
                chunk_pbar = tqdm(total=num_chunks, desc=f"  处理年份{year}数据块", unit="块", leave=False)
                
                for chunk_idx in range(num_chunks):
                    row_start = chunk_idx * self.chunk_size
                    row_end = min(row_start + self.chunk_size, height)
                    
                    # Read data chunks
                    orig_chunk = orig_src.read(1, window=((row_start, row_end), (0, width)))
                    opt_chunk = opt_src.read(1, window=((row_start, row_end), (0, width)))
                    
                    # Calculate changes - 使用优化的分析函数
                    chunk_changed, chunk_details = self._analyze_chunk_optimized(
                        orig_chunk, opt_chunk
                    )
                    
                    changed_pixels += np.int64(chunk_changed)
                    
                    # Merge detailed statistics
                    for change_type, count in chunk_details.items():
                        change_details[change_type] += np.int64(count)
                    
                    chunk_pbar.update(1)
                
                chunk_pbar.close()
                
                # Calculate statistics
                unchanged_pixels = total_pixels - changed_pixels
                change_ratio = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                
                file_stats = {
                    'year': year,
                    'total_pixels': int(total_pixels),
                    'changed_pixels': int(changed_pixels),
                    'unchanged_pixels': int(unchanged_pixels),
                    'change_ratio': change_ratio
                }
                
                # Convert detailed statistics to list format
                file_detailed = []
                for change_type, pixel_count in change_details.items():
                    if pixel_count > 0:  # Only record types with changes
                        file_detailed.append({
                            'year': year,
                            'change_type': f"{change_type[0]}->{change_type[1]}",
                            'pixel_count': int(pixel_count),
                            'percentage': (pixel_count / total_pixels) * 100
                        })
                
                self.logger.info(f"  Year {year}: Total pixels={total_pixels:,}, Changed pixels={changed_pixels:,} ({change_ratio:.2f}%)")
                
                return file_stats, file_detailed
                
        except Exception as e:
            self.logger.error(f"Error processing file pair: {str(e)}")
            return None, []
    
    def _analyze_chunk_optimized(self, orig_chunk, opt_chunk):
        """Analyze data chunk - optimized version using numpy operations"""
        # Calculate change mask
        changed_mask = orig_chunk != opt_chunk
        changed_pixels = np.sum(changed_mask, dtype=np.int64)
        
        # Calculate detailed change statistics - 优化版本
        change_details = {}
        if changed_pixels > 0:
            # Get changed pixel positions
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
    
    def _save_statistics(self, change_stats, detailed_stats):
        """Save statistics to CSV files with enhanced output"""
        # Save summary statistics
        summary_df = pd.DataFrame(change_stats)
        summary_file = os.path.join(self.output_dir, "optimization_summary.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"Summary statistics saved to: {summary_file}")
        
        # Save detailed statistics
        if detailed_stats:
            detailed_df = pd.DataFrame(detailed_stats)
            detailed_file = os.path.join(self.output_dir, "optimization_detailed.csv")
            detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"Detailed statistics saved to: {detailed_file}")
            
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
        """Create enhanced visualizations with multiple charts"""
        try:
            df = pd.DataFrame(change_stats)
            
            # 创建多子图布局
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 变化率趋势图
            ax1.plot(df['year'], df['change_ratio'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
            ax1.set_title('年度优化变化率趋势', fontsize=14, fontweight='bold')
            ax1.set_xlabel('年份', fontsize=12)
            ax1.set_ylabel('变化率 (%)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. 变化像素数量柱状图
            bars = ax2.bar(df['year'], df['changed_pixels'], color='#A23B72', alpha=0.7)
            ax2.set_title('年度变化像素数量', fontsize=14, fontweight='bold')
            ax2.set_xlabel('年份', fontsize=12)
            ax2.set_ylabel('变化像素数', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=9)
            
            # 3. 总像素数与变化像素数对比
            width = 0.35
            x = np.arange(df['year'].size)
            ax3.bar(x - width/2, df['total_pixels'], width, label='总像素数', color='#F18F01', alpha=0.7)
            ax3.bar(x + width/2, df['changed_pixels'], width, label='变化像素数', color='#C73E1D', alpha=0.7)
            ax3.set_title('总像素数与变化像素数对比', fontsize=14, fontweight='bold')
            ax3.set_xlabel('年份', fontsize=12)
            ax3.set_ylabel('像素数', fontsize=12)
            ax3.set_xticks(x)
            ax3.set_xticklabels(df['year'], rotation=45)
            ax3.legend()
            
            # 4. 统计摘要
            ax4.axis('off')
            stats_text = f"""
            统计摘要:
            
            平均变化率: {df['change_ratio'].mean():.2f}%
            最大变化率: {df['change_ratio'].max():.2f}% (年份: {df.loc[df['change_ratio'].idxmax(), 'year']})
            最小变化率: {df['change_ratio'].min():.2f}% (年份: {df.loc[df['change_ratio'].idxmin(), 'year']})
            
            总处理像素数: {df['total_pixels'].sum():,}
            总变化像素数: {df['changed_pixels'].sum():,}
            
            处理年份范围: {df['year'].min()} - {df['year'].max()}
            处理文件数量: {len(df)} 对
            """
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            plt.tight_layout()
            
            # Save enhanced chart
            chart_file = os.path.join(self.output_dir, "enhanced_analysis_results.png")
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Enhanced visualization saved to: {chart_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")


def main():
    """Main function"""
    # File paths
    base_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\2.GEE导出结果_结果合并\zone7"
    optimal_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\zone7\\"
    
    # Original files (before optimization)
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
    
    # Optimized files (after Markov model optimization)
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
    
    # Create analyzer and run analysis - 使用更大的chunk_size提高效率
    analyzer = OptimizationResultAnalyzer(
        output_dir=optimal_dir+"optimization_analysis_results",
        chunk_size=2000  # 增大chunk_size以提高处理效率
    )
    
    analyzer.analyze_optimization_results(original_files, optimized_files, years)


if __name__ == "__main__":
    main()