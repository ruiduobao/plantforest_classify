# 使用验证样本，计算精度

import os
import sys
import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr
import rasterio
from rasterio.mask import mask
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from pathlib import Path
from collections import defaultdict

# GEE结果数据，1=人工林；2=自然林；3=others；0是空值；文件名类似"zone1_2021.tif"，代表2021年zone1的分类结果
TIF_FILES= r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\2.GEE导出结果_结果合并"

# 验证点数据，名称类似Zone1_Merged_AllBand_Sample_filtered_test.shp，代表2017年-2024年zone1的验证样本
POINT_VALID=r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并_剔除异常样本_去掉不需要的属性\拆分训练集测试集的zone数据\测试集"
# 矢量的年属性。如果是2021年，则代表2021年的验证样本
YEAR_ATTR="year"
# 矢量点的验证属性 1=人工林；2=自然林；3=others
POINT_ATTR="landcover"
# 输出
Out_path=r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\统计发射矩阵"

def setup_logging(output_dir):
    """
    设置日志记录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_filename = f"accuracy_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"日志文件: {log_path}")
    return log_path

def parse_tif_filename(filename):
    """
    解析TIF文件名，提取zone和年份信息
    
    参数:
        filename: 文件名，如 "zone1_2021.tif"
    
    返回:
        tuple: (zone, year) 或 (None, None) 如果解析失败
    """
    try:
        # 移除文件扩展名
        basename = os.path.splitext(filename)[0]
        
        # 匹配模式: zone{数字}_{年份}
        pattern = r'zone(\d+)_(\d{4})'
        match = re.search(pattern, basename, re.IGNORECASE)
        
        if match:
            zone = int(match.group(1))
            year = int(match.group(2))
            return zone, year
        else:
            logging.warning(f"无法解析文件名: {filename}")
            return None, None
            
    except Exception as e:
        logging.error(f"解析文件名时出错 {filename}: {str(e)}")
        return None, None

def parse_shp_filename(filename):
    """
    解析SHP文件名，提取zone信息
    
    参数:
        filename: 文件名，如 "Zone1_Merged_AllBand_Sample_filtered_test.shp"
    
    返回:
        int: zone编号 或 None 如果解析失败
    """
    try:
        # 匹配模式: Zone{数字}
        pattern = r'Zone(\d+)'
        match = re.search(pattern, filename, re.IGNORECASE)
        
        if match:
            zone = int(match.group(1))
            return zone
        else:
            logging.warning(f"无法解析验证点文件名: {filename}")
            return None
            
    except Exception as e:
        logging.error(f"解析验证点文件名时出错 {filename}: {str(e)}")
        return None

def get_tif_files(tif_dir):
    """
    获取所有TIF文件并按zone和年份分组
    
    参数:
        tif_dir: TIF文件目录
    
    返回:
        dict: {(zone, year): filepath}
    """
    tif_files = {}
    
    if not os.path.exists(tif_dir):
        logging.error(f"TIF目录不存在: {tif_dir}")
        return tif_files
    
    for filename in os.listdir(tif_dir):
        if filename.lower().endswith('.tif'):
            zone, year = parse_tif_filename(filename)
            if zone is not None and year is not None:
                filepath = os.path.join(tif_dir, filename)
                tif_files[(zone, year)] = filepath
                logging.info(f"找到TIF文件: Zone{zone}, {year}年 - {filename}")
    
    logging.info(f"总共找到 {len(tif_files)} 个TIF文件")
    return tif_files

def get_validation_files(validation_dir):
    """
    获取所有验证点文件并按zone分组
    
    参数:
        validation_dir: 验证点文件目录
    
    返回:
        dict: {zone: filepath}
    """
    validation_files = {}
    
    if not os.path.exists(validation_dir):
        logging.error(f"验证点目录不存在: {validation_dir}")
        return validation_files
    
    for filename in os.listdir(validation_dir):
        if filename.lower().endswith('.shp'):
            zone = parse_shp_filename(filename)
            if zone is not None:
                filepath = os.path.join(validation_dir, filename)
                validation_files[zone] = filepath
                logging.info(f"找到验证点文件: Zone{zone} - {filename}")
    
    logging.info(f"总共找到 {len(validation_files)} 个验证点文件")
    return validation_files

def extract_raster_values_at_points(raster_path, points_gdf, year_filter=None):
    """
    提取栅格在验证点位置的值
    
    参数:
        raster_path: 栅格文件路径
        points_gdf: 验证点GeoDataFrame
        year_filter: 年份过滤器，只保留指定年份的点
    
    返回:
        pandas.DataFrame: 包含预测值和真实值的DataFrame
    """
    try:
        # 过滤指定年份的点
        if year_filter is not None and YEAR_ATTR in points_gdf.columns:
            filtered_points = points_gdf[points_gdf[YEAR_ATTR] == year_filter].copy()
            logging.info(f"过滤后的验证点数量 ({year_filter}年): {len(filtered_points)}")
        else:
            filtered_points = points_gdf.copy()
            logging.info(f"验证点总数量: {len(filtered_points)}")
        
        if len(filtered_points) == 0:
            logging.warning(f"没有找到 {year_filter} 年的验证点")
            return pd.DataFrame()
        
        # 打开栅格文件
        with rasterio.open(raster_path) as src:
            # 确保坐标系一致
            if filtered_points.crs != src.crs:
                logging.info(f"转换坐标系: {filtered_points.crs} -> {src.crs}")
                filtered_points = filtered_points.to_crs(src.crs)
            
            # 提取栅格值
            coords = [(x, y) for x, y in zip(filtered_points.geometry.x, filtered_points.geometry.y)]
            raster_values = list(src.sample(coords))
            
            # 创建结果DataFrame
            result_df = pd.DataFrame({
                'predicted': [val[0] if val[0] != src.nodata else 0 for val in raster_values],
                'actual': filtered_points[POINT_ATTR].values,
                'year': filtered_points[YEAR_ATTR].values if YEAR_ATTR in filtered_points.columns else [year_filter] * len(filtered_points)
            })
            
            # 过滤掉无效值
            valid_mask = (result_df['predicted'] != 0) & (result_df['actual'].notna())
            result_df = result_df[valid_mask]
            
            logging.info(f"有效验证点数量: {len(result_df)}")
            return result_df
            
    except Exception as e:
        logging.error(f"提取栅格值时出错: {str(e)}")
        return pd.DataFrame()

def calculate_accuracy_metrics(y_true, y_pred, class_names=None):
    """
    计算精度指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
    
    返回:
        dict: 包含各种精度指标的字典
    """
    if class_names is None:
        class_names = ['Planted Tree', 'Natural Tree', 'Other']
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
    
    # 计算总体精度
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    # 计算用户精度和生产者精度
    user_accuracy = []  # 用户精度 (行方向)
    producer_accuracy = []  # 生产者精度 (列方向)
    
    for i in range(len(class_names)):
        # 用户精度 = 对角线元素 / 行总和
        if cm[i, :].sum() > 0:
            ua = cm[i, i] / cm[i, :].sum()
        else:
            ua = 0.0
        user_accuracy.append(ua)
        
        # 生产者精度 = 对角线元素 / 列总和
        if cm[:, i].sum() > 0:
            pa = cm[i, i] / cm[:, i].sum()
        else:
            pa = 0.0
        producer_accuracy.append(pa)
    
    # 计算F1分数
    f1_scores = []
    for i in range(len(class_names)):
        if user_accuracy[i] + producer_accuracy[i] > 0:
            f1 = 2 * (user_accuracy[i] * producer_accuracy[i]) / (user_accuracy[i] + producer_accuracy[i])
        else:
            f1 = 0.0
        f1_scores.append(f1)
    
    # 计算Kappa系数
    n = cm.sum()
    po = overall_accuracy  # 观察一致性
    pe = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(len(class_names))) / (n * n)  # 期望一致性
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0
    
    return {
        'confusion_matrix': cm,
        'overall_accuracy': overall_accuracy,
        'user_accuracy': user_accuracy,
        'producer_accuracy': producer_accuracy,
        'f1_scores': f1_scores,
        'kappa': kappa,
        'class_names': class_names,
        'sample_count': len(y_true)
    }

def create_accuracy_table(metrics, zone, year):
    """
    创建精度表格
    
    参数:
        metrics: 精度指标字典
        zone: 区域编号
        year: 年份
    
    返回:
        pandas.DataFrame: 精度表格
    """
    class_names = metrics['class_names']
    
    # 创建详细精度表
    accuracy_data = []
    for i, class_name in enumerate(class_names):
        accuracy_data.append({
            'Zone': zone,
            'Year': year,
            'Class': class_name,
            'Class_Code': i + 1,
            'User_Accuracy': metrics['user_accuracy'][i],
            'Producer_Accuracy': metrics['producer_accuracy'][i],
            'F1_Score': metrics['f1_scores'][i]
        })
    
    # 添加总体指标
    accuracy_data.append({
        'Zone': zone,
        'Year': year,
        'Class': 'Overall',
        'Class_Code': 0,
        'User_Accuracy': metrics['overall_accuracy'],
        'Producer_Accuracy': metrics['overall_accuracy'],
        'F1_Score': np.mean(metrics['f1_scores'])
    })
    
    df = pd.DataFrame(accuracy_data)
    
    # 添加额外信息
    df['Kappa'] = metrics['kappa']
    df['Sample_Count'] = metrics['sample_count']
    
    return df

def save_confusion_matrix_plot(cm, class_names, zone, year, output_dir):
    """
    保存混淆矩阵图
    
    参数:
        cm: 混淆矩阵
        class_names: 类别名称
        zone: 区域编号
        year: 年份
        output_dir: 输出目录
    """
    try:
        plt.figure(figsize=(8, 6))
        
        # 创建热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - Zone{zone} {year}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # 保存图片
        plot_filename = f"confusion_matrix_zone{zone}_{year}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"混淆矩阵图已保存: {plot_path}")
        
    except Exception as e:
        logging.error(f"保存混淆矩阵图时出错: {str(e)}")

def calculate_emission_matrix(confusion_matrix):
    """
    计算发射矩阵（行归一化的混淆矩阵）
    
    参数:
        confusion_matrix: 混淆矩阵 (numpy array)
    
    返回:
        numpy.ndarray: 发射矩阵，每行表示真实类别对预测类别的概率分布
    """
    try:
        # 行归一化：每行除以该行的总和
        emission_matrix = confusion_matrix.astype(float)
        row_sums = emission_matrix.sum(axis=1, keepdims=True)
        
        # 避免除零错误
        row_sums[row_sums == 0] = 1
        
        emission_matrix = emission_matrix / row_sums
        
        return emission_matrix
        
    except Exception as e:
        logging.error(f"计算发射矩阵时出错: {str(e)}")
        return None

def collect_emission_matrices(all_results):
    """
    收集所有zone和年份的发射矩阵
    
    参数:
        all_results: 所有精度计算结果的列表
    
    返回:
        dict: {(zone, year): emission_matrix}
    """
    emission_matrices = {}
    
    for result_df in all_results:
        if not result_df.empty:
            # 获取该结果的zone和year信息
            zones = result_df['Zone'].unique()
            years = result_df['Year'].unique()
            
            for zone in zones:
                for year in years:
                    # 这里需要重新计算该zone和year的混淆矩阵
                    # 由于原始数据在process_single_zone_year中处理，我们需要修改该函数
                    pass
    
    return emission_matrices

def calculate_average_emission_matrix(emission_matrices_by_year):
    """
    计算所有年份的平均发射矩阵
    
    参数:
        emission_matrices_by_year: 按年份分组的发射矩阵字典
    
    返回:
        numpy.ndarray: 平均发射矩阵
    """
    try:
        all_matrices = []
        
        for year, matrices in emission_matrices_by_year.items():
            if matrices:
                # 计算该年份所有zone的平均矩阵
                year_matrices = list(matrices.values())
                if year_matrices:
                    year_avg = np.mean(year_matrices, axis=0)
                    all_matrices.append(year_avg)
        
        if all_matrices:
            # 计算所有年份的平均
            overall_avg = np.mean(all_matrices, axis=0)
            return overall_avg
        else:
            logging.warning("没有有效的发射矩阵用于计算平均值")
            return None
            
    except Exception as e:
        logging.error(f"计算平均发射矩阵时出错: {str(e)}")
        return None

def save_emission_matrices_text(emission_matrices_by_year, overall_avg_matrix, output_dir):
    """
    保存发射矩阵结果为文本格式
    
    参数:
        emission_matrices_by_year: 按年份分组的发射矩阵
        overall_avg_matrix: 总体平均发射矩阵
        output_dir: 输出目录
    """
    try:
        output_file = os.path.join(output_dir, "emission_matrices_2017_2024.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("发射矩阵计算结果 (2017-2024年)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("矩阵说明:\n")
            f.write("- 行: 真实类别 (1=人工林, 2=自然林, 3=其他)\n")
            f.write("- 列: 预测类别 (1=人工林, 2=自然林, 3=其他)\n")
            f.write("- 每行数值表示该真实类别被预测为各类别的概率\n\n")
            
            # 写入各年份的发射矩阵
            for year in sorted(emission_matrices_by_year.keys()):
                matrices = emission_matrices_by_year[year]
                if matrices:
                    f.write(f"{year}年发射矩阵:\n")
                    f.write("-" * 30 + "\n")
                    
                    # 计算该年份的平均矩阵
                    year_matrices = list(matrices.values())
                    if year_matrices:
                        year_avg = np.mean(year_matrices, axis=0)
                        
                        f.write("emission_matrix = np.array([\n")
                        for i, row in enumerate(year_avg):
                            class_names = ["人工林", "自然林", "其他"]
                            f.write(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],  # True = {i+1} ({class_names[i]})\n")
                        f.write("])\n\n")
                        
                        # 写入各zone的详细信息
                        f.write(f"  {year}年各Zone详细信息:\n")
                        for (zone, year_key), matrix in matrices.items():
                            if year_key == year:
                                f.write(f"    Zone{zone}: [{matrix[0,0]:.4f}, {matrix[0,1]:.4f}, {matrix[0,2]:.4f}; "
                                       f"{matrix[1,0]:.4f}, {matrix[1,1]:.4f}, {matrix[1,2]:.4f}; "
                                       f"{matrix[2,0]:.4f}, {matrix[2,1]:.4f}, {matrix[2,2]:.4f}]\n")
                        f.write("\n")
            
            # 写入总体平均矩阵
            if overall_avg_matrix is not None:
                f.write("总体平均发射矩阵 (2017-2024年):\n")
                f.write("=" * 40 + "\n")
                f.write("emission_matrix = np.array([\n")
                for i, row in enumerate(overall_avg_matrix):
                    class_names = ["人工林", "自然林", "其他"]
                    f.write(f"    [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}],  # True = {i+1} ({class_names[i]})\n")
                f.write("])\n\n")
                
                f.write("矩阵解释:\n")
                f.write(f"- 人工林被正确识别的概率: {overall_avg_matrix[0,0]:.4f}\n")
                f.write(f"- 人工林被误识为自然林的概率: {overall_avg_matrix[0,1]:.4f}\n")
                f.write(f"- 人工林被误识为其他的概率: {overall_avg_matrix[0,2]:.4f}\n")
                f.write(f"- 自然林被误识为人工林的概率: {overall_avg_matrix[1,0]:.4f}\n")
                f.write(f"- 自然林被正确识别的概率: {overall_avg_matrix[1,1]:.4f}\n")
                f.write(f"- 自然林被误识为其他的概率: {overall_avg_matrix[1,2]:.4f}\n")
                f.write(f"- 其他被误识为人工林的概率: {overall_avg_matrix[2,0]:.4f}\n")
                f.write(f"- 其他被误识为自然林的概率: {overall_avg_matrix[2,1]:.4f}\n")
                f.write(f"- 其他被正确识别的概率: {overall_avg_matrix[2,2]:.4f}\n")
        
        logging.info(f"发射矩阵文本文件已保存: {output_file}")
        
    except Exception as e:
        logging.error(f"保存发射矩阵文本文件时出错: {str(e)}")

def process_single_zone_year(tif_path, validation_path, zone, year, output_dir):
    """
    处理单个zone和年份的精度计算
    
    参数:
        tif_path: TIF文件路径
        validation_path: 验证点文件路径
        zone: 区域编号
        year: 年份
        output_dir: 输出目录
    
    返回:
        tuple: (精度结果表格, 发射矩阵)
    """
    try:
        logging.info(f"处理 Zone{zone} {year}年的精度计算...")
        
        # 读取验证点数据
        validation_gdf = gpd.read_file(validation_path)
        logging.info(f"验证点文件读取成功: {len(validation_gdf)} 个点")
        
        # 提取栅格值
        result_df = extract_raster_values_at_points(tif_path, validation_gdf, year)
        
        if len(result_df) == 0:
            logging.warning(f"Zone{zone} {year}年没有有效的验证点")
            return pd.DataFrame(), None
        
        # 计算精度指标
        metrics = calculate_accuracy_metrics(result_df['actual'], result_df['predicted'])
        
        # 计算发射矩阵
        emission_matrix = calculate_emission_matrix(metrics['confusion_matrix'])
        
        # 创建精度表格
        accuracy_table = create_accuracy_table(metrics, zone, year)
        
        # 保存混淆矩阵图
        save_confusion_matrix_plot(metrics['confusion_matrix'], metrics['class_names'], 
                                 zone, year, output_dir)
        
        # 保存详细结果
        detail_filename = f"accuracy_detail_zone{zone}_{year}.csv"
        detail_path = os.path.join(output_dir, detail_filename)
        result_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
        
        logging.info(f"Zone{zone} {year}年精度计算完成")
        logging.info(f"总体精度: {metrics['overall_accuracy']:.4f}")
        logging.info(f"Kappa系数: {metrics['kappa']:.4f}")
        
        return accuracy_table, emission_matrix
        
    except Exception as e:
        logging.error(f"处理Zone{zone} {year}年时出错: {str(e)}")
        return pd.DataFrame(), None

def merge_results_by_year(all_results):
    """
    按年份合并多个zone的统计结果
    
    参数:
        all_results: 所有结果的列表
    
    返回:
        dict: 按年份分组的合并结果
    """
    yearly_results = defaultdict(list)
    
    # 按年份分组
    for result_df in all_results:
        if not result_df.empty:
            for year in result_df['Year'].unique():
                year_data = result_df[result_df['Year'] == year]
                yearly_results[year].append(year_data)
    
    # 合并每年的结果
    merged_results = {}
    for year, year_data_list in yearly_results.items():
        if year_data_list:
            merged_df = pd.concat(year_data_list, ignore_index=True)
            merged_results[year] = merged_df
            
            # 计算年度统计
            overall_rows = merged_df[merged_df['Class'] == 'Overall']
            if not overall_rows.empty:
                avg_accuracy = overall_rows['User_Accuracy'].mean()
                avg_kappa = overall_rows['Kappa'].mean()
                total_samples = overall_rows['Sample_Count'].sum()
                
                logging.info(f"{year}年合并结果:")
                logging.info(f"  平均总体精度: {avg_accuracy:.4f}")
                logging.info(f"  平均Kappa系数: {avg_kappa:.4f}")
                logging.info(f"  总样本数: {total_samples}")
    
    return merged_results

def main():
    """
    主函数
    """
    # 创建输出目录
    os.makedirs(Out_path, exist_ok=True)
    
    # 设置日志
    log_path = setup_logging(Out_path)
    logging.info("开始精度计算...")
    
    try:
        # 获取TIF文件和验证点文件
        tif_files = get_tif_files(TIF_FILES)
        validation_files = get_validation_files(POINT_VALID)
        
        if not tif_files:
            logging.error("没有找到TIF文件")
            return
        
        if not validation_files:
            logging.error("没有找到验证点文件")
            return
        
        # 处理每个zone和年份的组合
        all_results = []
        emission_matrices = {}  # 存储发射矩阵 {(zone, year): emission_matrix}
        
        for (zone, year), tif_path in tif_files.items():
            if zone in validation_files:
                validation_path = validation_files[zone]
                
                # 处理单个zone和年份
                result_table, emission_matrix = process_single_zone_year(
                    tif_path, validation_path, zone, year, Out_path
                )
                
                if not result_table.empty and emission_matrix is not None:
                    all_results.append(result_table)
                    emission_matrices[(zone, year)] = emission_matrix
                    logging.info(f"Zone{zone} {year}年发射矩阵已收集")
            else:
                logging.warning(f"Zone{zone}没有对应的验证点文件")
        
        # 合并所有结果
        if all_results:
            # 保存所有结果
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_path = os.path.join(Out_path, "accuracy_results_all.csv")
            combined_results.to_csv(combined_path, index=False, encoding='utf-8-sig')
            logging.info(f"所有结果已保存: {combined_path}")
            
            # 按年份合并结果
            yearly_merged = merge_results_by_year(all_results)
            
            # 保存按年份合并的结果
            for year, merged_df in yearly_merged.items():
                yearly_path = os.path.join(Out_path, f"accuracy_results_{year}.csv")
                merged_df.to_csv(yearly_path, index=False, encoding='utf-8-sig')
                logging.info(f"{year}年合并结果已保存: {yearly_path}")
            
            # 创建汇总统计
            summary_data = []
            for year, merged_df in yearly_merged.items():
                overall_rows = merged_df[merged_df['Class'] == 'Overall']
                if not overall_rows.empty:
                    summary_data.append({
                        'Year': year,
                        'Zone_Count': len(overall_rows),
                        'Total_Samples': overall_rows['Sample_Count'].sum(),
                        'Avg_Overall_Accuracy': overall_rows['User_Accuracy'].mean(),
                        'Avg_Kappa': overall_rows['Kappa'].mean(),
                        'Min_Accuracy': overall_rows['User_Accuracy'].min(),
                        'Max_Accuracy': overall_rows['User_Accuracy'].max()
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(Out_path, "accuracy_summary.csv")
                summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
                logging.info(f"汇总统计已保存: {summary_path}")
        
        # 处理发射矩阵
        if emission_matrices:
            logging.info("开始处理发射矩阵...")
            
            # 按年份分组发射矩阵
            emission_matrices_by_year = defaultdict(dict)
            for (zone, year), matrix in emission_matrices.items():
                emission_matrices_by_year[year][(zone, year)] = matrix
            
            logging.info(f"收集到 {len(emission_matrices)} 个发射矩阵，涵盖 {len(emission_matrices_by_year)} 个年份")
            
            # 计算总体平均发射矩阵
            overall_avg_matrix = calculate_average_emission_matrix(emission_matrices_by_year)
            
            if overall_avg_matrix is not None:
                logging.info("总体平均发射矩阵计算完成:")
                logging.info(f"人工林识别准确率: {overall_avg_matrix[0,0]:.4f}")
                logging.info(f"自然林识别准确率: {overall_avg_matrix[1,1]:.4f}")
                logging.info(f"其他类别识别准确率: {overall_avg_matrix[2,2]:.4f}")
            
            # 保存发射矩阵文本文件
            save_emission_matrices_text(emission_matrices_by_year, overall_avg_matrix, Out_path)
            
            # 保存发射矩阵为CSV格式（可选）
            emission_summary = []
            for year in sorted(emission_matrices_by_year.keys()):
                matrices = emission_matrices_by_year[year]
                if matrices:
                    year_matrices = list(matrices.values())
                    if year_matrices:
                        year_avg = np.mean(year_matrices, axis=0)
                        emission_summary.append({
                            'Year': year,
                            'Zone_Count': len(year_matrices),
                            'P11': year_avg[0,0],  # 人工林->人工林
                            'P12': year_avg[0,1],  # 人工林->自然林
                            'P13': year_avg[0,2],  # 人工林->其他
                            'P21': year_avg[1,0],  # 自然林->人工林
                            'P22': year_avg[1,1],  # 自然林->自然林
                            'P23': year_avg[1,2],  # 自然林->其他
                            'P31': year_avg[2,0],  # 其他->人工林
                            'P32': year_avg[2,1],  # 其他->自然林
                            'P33': year_avg[2,2]   # 其他->其他
                        })
            
            if emission_summary:
                emission_df = pd.DataFrame(emission_summary)
                emission_csv_path = os.path.join(Out_path, "emission_matrices_summary.csv")
                emission_df.to_csv(emission_csv_path, index=False, encoding='utf-8-sig')
                logging.info(f"发射矩阵汇总CSV已保存: {emission_csv_path}")
        
        logging.info("精度计算和发射矩阵分析完成!")
        
    except Exception as e:
        logging.error(f"主程序执行时出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
