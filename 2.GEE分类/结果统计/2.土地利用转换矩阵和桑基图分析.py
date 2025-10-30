#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
土地利用转换矩阵和桑基图分析脚本
功能：计算2017-2024年间人工林(1)、自然林(2)、其他(3)类别的转换矩阵，生成桑基图数据和可视化
作者：锐多宝 (ruiduobao)
创建时间：2025年1月
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import json
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def setup_logging(output_dir):
    """
    设置日志记录
    
    参数:
    - output_dir: 输出目录
    
    返回:
    - logger: 日志记录器
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"land_use_transition_analysis_{timestamp}.log")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存至: {log_file}")
    
    return logger

def find_classification_files(base_dir, years, file_pattern="*.tif"):
    """
    查找指定年份的分类结果文件
    
    参数:
    - base_dir: 基础目录路径
    - years: 年份列表
    - file_pattern: 文件匹配模式
    
    返回:
    - dict: {年份: 文件路径}
    """
    file_dict = {}
    
    # 直接在基础目录中查找所有匹配的文件
    pattern_path = os.path.join(base_dir, file_pattern)
    all_files = glob.glob(pattern_path)
    
    for year in years:
        # 查找包含年份的文件
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # 检查文件名中是否包含年份
            if str(year) in filename:
                file_dict[year] = file_path
                break
        
        if year not in file_dict:
            print(f"警告: 未找到 {year} 年的分类文件")
    
    return file_dict

def calculate_transition_matrix_parallel(raster1_path, raster2_path, categories=[1, 2, 3]):
    """
    计算两个栅格文件之间的转换矩阵（并行分块处理）
    
    参数:
    - raster1_path: 第一年栅格文件路径
    - raster2_path: 第二年栅格文件路径
    - categories: 类别列表
    
    返回:
    - dict: 转换矩阵和统计信息
    """
    try:
        with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
            # 检查两个栅格的空间参考是否一致
            if src1.shape != src2.shape:
                raise ValueError(f"栅格尺寸不匹配: {src1.shape} vs {src2.shape}")
            
            if src1.crs != src2.crs:
                raise ValueError(f"坐标系不匹配: {src1.crs} vs {src2.crs}")
            
            # 获取像素面积信息
            pixel_size_x, pixel_size_y = src1.res
            crs = src1.crs
            
            # 计算每个像素的面积
            if crs and crs.to_string() == 'EPSG:4326':
                # 地理坐标系，在东南亚地区的近似计算
                pixel_area = abs(pixel_size_x * pixel_size_y) * (111000 ** 2)  # 转换为平方米
            else:
                # 投影坐标系
                pixel_area = abs(src1.transform[0] * src1.transform[4])  # 像素面积（平方米）
            
            # 初始化转换矩阵
            n_categories = len(categories)
            transition_matrix = np.zeros((n_categories, n_categories), dtype=np.int64)
            
            # 分块处理以节省内存
            block_size = 1024  # 每次处理1024x1024像素块
            
            for row_start in range(0, src1.height, block_size):
                for col_start in range(0, src1.width, block_size):
                    # 计算当前块的窗口
                    row_end = min(row_start + block_size, src1.height)
                    col_end = min(col_start + block_size, src1.width)
                    
                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                    
                    # 读取当前块的数据
                    block1 = src1.read(1, window=window)
                    block2 = src2.read(1, window=window)
                    
                    # 创建有效数据掩码（排除NaN和无效值）
                    valid_mask = (~np.isnan(block1)) & (~np.isnan(block2))
                    
                    # 只处理有效像素
                    if np.any(valid_mask):
                        valid_block1 = block1[valid_mask]
                        valid_block2 = block2[valid_mask]
                        
                        # 计算当前块的转换矩阵
                        for i, cat1 in enumerate(categories):
                            for j, cat2 in enumerate(categories):
                                # 统计从cat1转换到cat2的像素数量
                                count = np.sum((valid_block1 == cat1) & (valid_block2 == cat2))
                                transition_matrix[i, j] += count
            
            # 构建结果字典
            result = {
                'transition_matrix': transition_matrix,
                'categories': categories,
                'pixel_area_sqm': pixel_area,
                'total_pixels': np.sum(transition_matrix),
                'metadata': {
                    'raster1_path': raster1_path,
                    'raster2_path': raster2_path,
                    'crs': str(src1.crs),
                    'shape': src1.shape,
                    'bounds': src1.bounds
                }
            }
            
            return result
            
    except Exception as e:
        raise Exception(f"计算转换矩阵时出错: {str(e)}")

def process_transition_pair(args):
    """
    处理单个年份对转换矩阵的包装函数（用于多进程）
    
    参数:
    - args: 包含文件路径对、年份对和类别的元组
    
    返回:
    - tuple: (年份对, 转换矩阵结果, 错误信息)
    """
    (year1, year2), (file1, file2), categories = args
    try:
        result = calculate_transition_matrix_parallel(file1, file2, categories)
        return (year1, year2), result, None
    except Exception as e:
        return (year1, year2), None, str(e)

def create_sankey_data(transition_results, categories=[1, 2, 3]):
    """
    创建适合桑基图的数据格式
    
    参数:
    - transition_results: 转换矩阵结果字典
    - categories: 类别列表
    
    返回:
    - list: 桑基图数据列表
    """
    category_names = {1: '人工林', 2: '自然林', 3: '其他'}
    sankey_data = []
    
    for (year1, year2), result in transition_results.items():
        if result is None:
            continue
            
        transition_matrix = result['transition_matrix']
        pixel_area = result['pixel_area_sqm']
        
        # 转换为面积（平方公里）
        area_matrix = transition_matrix * pixel_area / 1000000
        
        # 创建桑基图数据
        for i, source_cat in enumerate(categories):
            for j, target_cat in enumerate(categories):
                area_km2 = area_matrix[i, j]
                
                if area_km2 > 0:  # 只记录有转换的数据
                    sankey_data.append({
                        'period': f"{year1}-{year2}",
                        'source': f"{category_names[source_cat]}_{year1}",
                        'target': f"{category_names[target_cat]}_{year2}",
                        'source_category': category_names[source_cat],
                        'target_category': category_names[target_cat],
                        'area_km2': float(area_km2),
                        'pixel_count': int(transition_matrix[i, j]),
                        'year1': year1,
                        'year2': year2,
                        'is_change': source_cat != target_cat
                    })
    
    return sankey_data

def export_transition_matrices(transition_results, output_dir, categories=[1, 2, 3]):
    """
    导出转换矩阵到Excel文件
    
    参数:
    - transition_results: 转换矩阵结果字典
    - output_dir: 输出目录
    - categories: 类别列表
    """
    category_names = {1: '人工林', 2: '自然林', 3: '其他'}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_dir, f"land_use_transition_matrices_{timestamp}.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # 1. 汇总所有转换矩阵
        summary_data = []
        for (year1, year2), result in transition_results.items():
            if result is None:
                continue
                
            transition_matrix = result['transition_matrix']
            pixel_area = result['pixel_area_sqm']
            area_matrix = transition_matrix * pixel_area / 1000000  # 转换为平方公里
            
            for i, source_cat in enumerate(categories):
                for j, target_cat in enumerate(categories):
                    summary_data.append({
                        '时期': f"{year1}-{year2}",
                        '源类别': category_names[source_cat],
                        '目标类别': category_names[target_cat],
                        '面积(km²)': area_matrix[i, j],
                        '像素数量': transition_matrix[i, j],
                        '是否转换': '是' if source_cat != target_cat else '否'
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='转换矩阵汇总', index=False)
        
        # 2. 为每个时期创建单独的转换矩阵表
        for (year1, year2), result in transition_results.items():
            if result is None:
                continue
                
            transition_matrix = result['transition_matrix']
            pixel_area = result['pixel_area_sqm']
            area_matrix = transition_matrix * pixel_area / 1000000
            
            # 创建面积转换矩阵DataFrame
            area_df = pd.DataFrame(
                area_matrix,
                index=[f"{category_names[cat]}_{year1}" for cat in categories],
                columns=[f"{category_names[cat]}_{year2}" for cat in categories]
            )
            area_df.to_excel(writer, sheet_name=f'{year1}-{year2}年面积转换')
            
            # 创建像素数量转换矩阵DataFrame
            pixel_df = pd.DataFrame(
                transition_matrix,
                index=[f"{category_names[cat]}_{year1}" for cat in categories],
                columns=[f"{category_names[cat]}_{year2}" for cat in categories]
            )
            pixel_df.to_excel(writer, sheet_name=f'{year1}-{year2}年像素转换')
    
    return excel_path

def create_sankey_diagram(sankey_data, output_dir, period=None):
    """
    创建桑基图
    
    参数:
    - sankey_data: 桑基图数据
    - output_dir: 输出目录
    - period: 特定时期，如果为None则创建所有时期的图
    """
    if period:
        # 筛选特定时期的数据
        period_data = [d for d in sankey_data if d['period'] == period]
        if not period_data:
            print(f"未找到时期 {period} 的数据")
            return
        data_to_plot = period_data
        title_suffix = f"_{period}"
    else:
        data_to_plot = sankey_data
        title_suffix = "_all_periods"
    
    # 创建节点和链接
    nodes = set()
    for item in data_to_plot:
        nodes.add(item['source'])
        nodes.add(item['target'])
    
    node_list = sorted(list(nodes))
    node_indices = {node: i for i, node in enumerate(node_list)}
    
    # 定义颜色
    colors = {
        '人工林': 'rgba(34, 139, 34, 0.8)',    # 森林绿
        '自然林': 'rgba(0, 100, 0, 0.8)',      # 深绿
        '其他': 'rgba(139, 69, 19, 0.8)'       # 棕色
    }
    
    # 为节点分配颜色
    node_colors = []
    for node in node_list:
        for cat_name in colors.keys():
            if cat_name in node:
                node_colors.append(colors[cat_name])
                break
        else:
            node_colors.append('rgba(128, 128, 128, 0.8)')  # 默认灰色
    
    # 创建链接
    source_indices = []
    target_indices = []
    values = []
    link_colors = []
    
    for item in data_to_plot:
        source_idx = node_indices[item['source']]
        target_idx = node_indices[item['target']]
        
        source_indices.append(source_idx)
        target_indices.append(target_idx)
        values.append(item['area_km2'])
        
        # 链接颜色基于是否发生转换
        if item['is_change']:
            link_colors.append('rgba(255, 0, 0, 0.3)')  # 红色表示转换
        else:
            link_colors.append('rgba(0, 0, 255, 0.3)')  # 蓝色表示保持不变
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_list,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        )
    )])
    
    title = f"土地利用转换桑基图{title_suffix.replace('_', ' ')}"
    fig.update_layout(
        title_text=title,
        font_size=12,
        font_family="Microsoft YaHei",
        width=1200,
        height=800
    )
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"sankey_diagram{title_suffix}_{timestamp}.html")
    fig.write_html(html_path)
    
    # 尝试保存为PNG（需要kaleido库）
    try:
        png_path = os.path.join(output_dir, f"sankey_diagram{title_suffix}_{timestamp}.png")
        fig.write_image(png_path, width=1200, height=800, scale=2)
        print(f"桑基图已保存为: {png_path}")
    except Exception as e:
        print(f"保存PNG格式失败: {e}")
        print(f"桑基图HTML版本已保存为: {html_path}")
    
    return html_path

def create_transition_heatmap(transition_results, output_dir, categories=[1, 2, 3]):
    """
    创建转换矩阵热力图
    
    参数:
    - transition_results: 转换矩阵结果字典
    - output_dir: 输出目录
    - categories: 类别列表
    """
    category_names = {1: '人工林', 2: '自然林', 3: '其他'}
    
    # 计算每个时期的转换矩阵
    n_periods = len(transition_results)
    if n_periods == 0:
        return
    
    # 创建子图
    fig, axes = plt.subplots(1, n_periods, figsize=(6*n_periods, 5))
    if n_periods == 1:
        axes = [axes]
    
    for idx, ((year1, year2), result) in enumerate(transition_results.items()):
        if result is None:
            continue
            
        transition_matrix = result['transition_matrix']
        pixel_area = result['pixel_area_sqm']
        area_matrix = transition_matrix * pixel_area / 1000000  # 转换为平方公里
        
        # 创建标签
        labels = [category_names[cat] for cat in categories]
        
        # 创建热力图
        im = axes[idx].imshow(area_matrix, cmap='YlOrRd', aspect='auto')
        
        # 设置标签
        axes[idx].set_xticks(range(len(categories)))
        axes[idx].set_yticks(range(len(categories)))
        axes[idx].set_xticklabels([f"{label}\n{year2}" for label in labels])
        axes[idx].set_yticklabels([f"{label}\n{year1}" for label in labels])
        
        # 添加数值标注
        for i in range(len(categories)):
            for j in range(len(categories)):
                text = axes[idx].text(j, i, f'{area_matrix[i, j]:.1f}',
                                    ha="center", va="center", color="black", fontsize=10)
        
        axes[idx].set_title(f'{year1}-{year2}年土地利用转换\n(面积: km²)')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(output_dir, f"transition_heatmap_{timestamp}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"转换矩阵热力图已保存为: {png_path}")
    return png_path

def main():
    """
    主函数：执行土地利用转换矩阵分析和桑基图生成
    """
    # 配置参数
    BASE_DATA_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\第三次分类_样本筛选_每年逐年分类\GEE下载_逐年镶嵌_马尔科夫模型"  # 分类结果基础目录
    OUTPUT_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\第三次分类_样本筛选_每年逐年分类\GEE下载_逐年镶嵌_马尔科夫模型\2.土地利用转换矩阵和桑基图分析_变化检测分类"  # 输出目录
    YEARS = list(range(2017, 2025))  # 2017-2024年
    CATEGORIES = [1, 2, 3]  # 人工林(1), 自然林(2), 其他(3)
    FILE_PATTERN = "optimized_landcover_*.tif"  # 文件匹配模式，匹配合并后的分类结果
    
    # 设置日志
    logger = setup_logging(OUTPUT_DIR)
    logger.info("开始执行土地利用转换矩阵分析")
    logger.info(f"分析年份范围: {YEARS[0]}-{YEARS[-1]}")
    logger.info(f"统计类别: 人工林(1), 自然林(2), 其他(3)")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    
    try:
        # 查找分类文件
        logger.info("正在查找分类结果文件...")
        file_dict = find_classification_files(BASE_DATA_DIR, YEARS, FILE_PATTERN)
        
        if len(file_dict) < 2:
            logger.error("至少需要两年的分类文件才能计算转换矩阵")
            return
        
        logger.info(f"找到 {len(file_dict)} 个年份的分类文件")
        for year, file_path in file_dict.items():
            logger.info(f"  {year}年: {os.path.basename(file_path)}")
        
        # 准备年份对和文件对
        sorted_years = sorted(file_dict.keys())
        year_pairs = []
        file_pairs = []
        
        for i in range(len(sorted_years) - 1):
            year1, year2 = sorted_years[i], sorted_years[i + 1]
            file1, file2 = file_dict[year1], file_dict[year2]
            year_pairs.append((year1, year2))
            file_pairs.append((file1, file2))
        
        logger.info(f"将计算 {len(year_pairs)} 个年份对的转换矩阵")
        
        # 准备多进程处理的参数
        process_args = []
        for year_pair, file_pair in zip(year_pairs, file_pairs):
            process_args.append((year_pair, file_pair, CATEGORIES))
        
        # 多进程计算转换矩阵
        logger.info("开始多进程计算转换矩阵...")
        transition_results = {}
        failed_pairs = []
        
        # 使用进程池处理
        max_workers = min(mp.cpu_count(), len(process_args))
        logger.info(f"使用 {max_workers} 个进程进行并行处理")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(process_transition_pair, args): args for args in process_args}
            
            for future in as_completed(future_to_args):
                year_pair, result, error = future.result()
                
                if error:
                    logger.error(f"计算转换矩阵失败 {year_pair}: {error}")
                    failed_pairs.append((year_pair, error))
                else:
                    transition_results[year_pair] = result
                    logger.info(f"完成转换矩阵计算: {year_pair[0]}-{year_pair[1]}")
        
        if not transition_results:
            logger.error("未能成功计算任何转换矩阵")
            return
        
        # 创建桑基图数据
        logger.info("生成桑基图数据...")
        sankey_data = create_sankey_data(transition_results, CATEGORIES)
        
        # 导出结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 导出转换矩阵Excel文件
        logger.info("导出转换矩阵到Excel...")
        excel_path = export_transition_matrices(transition_results, OUTPUT_DIR, CATEGORIES)
        logger.info(f"转换矩阵Excel文件已保存至: {excel_path}")
        
        # 2. 导出桑基图数据CSV
        sankey_df = pd.DataFrame(sankey_data)
        csv_path = os.path.join(OUTPUT_DIR, f"sankey_data_{timestamp}.csv")
        sankey_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"桑基图数据CSV已保存至: {csv_path}")
        
        # 3. 创建桑基图可视化
        logger.info("创建桑基图可视化...")
        
        # 为每个时期创建单独的桑基图
        periods = sankey_df['period'].unique()
        for period in periods:
            try:
                html_path = create_sankey_diagram(sankey_data, OUTPUT_DIR, period)
                logger.info(f"{period}年桑基图已保存")
            except Exception as e:
                logger.error(f"创建{period}年桑基图失败: {e}")
        
        # 创建综合桑基图（如果数据不太多）
        if len(sankey_data) < 50:  # 避免图表过于复杂
            try:
                html_path = create_sankey_diagram(sankey_data, OUTPUT_DIR)
                logger.info("综合桑基图已保存")
            except Exception as e:
                logger.error(f"创建综合桑基图失败: {e}")
        
        # 4. 创建转换矩阵热力图
        logger.info("创建转换矩阵热力图...")
        try:
            heatmap_path = create_transition_heatmap(transition_results, OUTPUT_DIR, CATEGORIES)
            logger.info("转换矩阵热力图已保存")
        except Exception as e:
            logger.error(f"创建热力图失败: {e}")
        
        # 5. 保存详细JSON结果
        json_output = os.path.join(OUTPUT_DIR, f"transition_analysis_results_{timestamp}.json")
        results = {
            'transition_matrices': {f"{k[0]}-{k[1]}": {
                'transition_matrix': v['transition_matrix'].tolist() if v else None,
                'categories': v['categories'] if v else None,
                'pixel_area_sqm': v['pixel_area_sqm'] if v else None,
                'total_pixels': int(v['total_pixels']) if v else None
            } for k, v in transition_results.items()},
            'sankey_data': sankey_data,
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'years_analyzed': YEARS,
                'categories': {1: '人工林', 2: '自然林', 3: '其他'},
                'successful_transitions': len(transition_results),
                'failed_transitions': len(failed_pairs)
            }
        }
        
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"详细结果已保存至: {json_output}")
        
        # 输出摘要统计
        logger.info("\n=== 转换分析结果摘要 ===")
        for (year1, year2), result in transition_results.items():
            if result is None:
                continue
                
            transition_matrix = result['transition_matrix']
            pixel_area = result['pixel_area_sqm']
            area_matrix = transition_matrix * pixel_area / 1000000
            
            logger.info(f"\n{year1}-{year2}年转换统计:")
            category_names = {1: '人工林', 2: '自然林', 3: '其他'}
            
            # 计算各类别的净变化
            for i, cat in enumerate(CATEGORIES):
                inflow = np.sum(area_matrix[:, i]) - area_matrix[i, i]  # 流入
                outflow = np.sum(area_matrix[i, :]) - area_matrix[i, i]  # 流出
                net_change = inflow - outflow
                
                logger.info(f"  {category_names[cat]}: 流入{inflow:.2f}km², "
                          f"流出{outflow:.2f}km², 净变化{net_change:+.2f}km²")
        
        if failed_pairs:
            logger.warning(f"\n处理失败的年份对数量: {len(failed_pairs)}")
            for year_pair, error in failed_pairs:
                logger.warning(f"  {year_pair}: {error}")
        
        logger.info(f"\n转换分析完成！结果已保存至: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()