"""
SDPT数据筛选脚本 - 锐多宝(ruiduobao)
目的：筛选出SDPT数据中位于东南亚区划数据范围内的数据
作者：锐多宝(ruiduobao) - GIS和遥感科学专家
"""

import geopandas as gpd
import pandas as pd
import os
import sys
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count
import logging

# 忽略警告信息
warnings.filterwarnings('ignore')

def setup_logging(output_dir):
    """
    设置日志记录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件名
    log_filename = f"sdpt_filter_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(output_dir, log_filename)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_path

def analyze_geopackage_data(gdam_path):
    """
    分析GDAM区划数据的基础情况
    参数：
        gdam_path: GDAM数据文件路径
    返回：
        gdam_gdf: 读取的GeoDataFrame
    """
    logging.info("=" * 60)
    logging.info("开始分析GDAM区划数据")
    logging.info("=" * 60)
    
    try:
        # 读取GDAM区划数据
        logging.info(f"正在读取GDAM数据: {gdam_path}")
        gdam_gdf = gpd.read_file(gdam_path)
        
        # 基础信息统计
        logging.info(f"GDAM数据基础信息:")
        logging.info(f"  - 数据形状: {gdam_gdf.shape}")
        logging.info(f"  - 坐标系统: {gdam_gdf.crs}")
        logging.info(f"  - 几何类型: {gdam_gdf.geometry.geom_type.unique()}")
        
        # 字段信息
        logging.info(f"  - 字段列表: {list(gdam_gdf.columns)}")
        
        # 空间范围
        bounds = gdam_gdf.total_bounds
        logging.info(f"  - 空间范围: ")
        logging.info(f"    最小经度: {bounds[0]:.6f}")
        logging.info(f"    最小纬度: {bounds[1]:.6f}")
        logging.info(f"    最大经度: {bounds[2]:.6f}")
        logging.info(f"    最大纬度: {bounds[3]:.6f}")
        
        # 如果有国家字段，显示国家列表
        country_fields = ['NAME_0', 'COUNTRY', 'name', 'NAME', 'Country']
        for field in country_fields:
            if field in gdam_gdf.columns:
                countries = gdam_gdf[field].unique()
                logging.info(f"  - 包含的国家/地区 ({field}): {len(countries)}个")
                for country in sorted(countries):
                    logging.info(f"    {country}")
                break
        
        # 检查空值
        null_counts = gdam_gdf.isnull().sum()
        if null_counts.sum() > 0:
            logging.info(f"  - 空值统计:")
            for col, count in null_counts.items():
                if count > 0:
                    logging.info(f"    {col}: {count}")
        
        return gdam_gdf
        
    except Exception as e:
        logging.error(f"读取GDAM数据时出错: {str(e)}")
        return None

def analyze_sdpt_data(sdpt_path):
    """
    分析SDPT数据的基础情况
    参数：
        sdpt_path: SDPT数据文件路径
    返回：
        sdpt_gdf: 读取的GeoDataFrame
    """
    logging.info("=" * 60)
    logging.info("开始分析SDPT数据")
    logging.info("=" * 60)
    
    try:
        # 读取SDPT数据
        logging.info(f"正在读取SDPT数据: {sdpt_path}")
        sdpt_gdf = gpd.read_file(sdpt_path)
        
        # 基础信息统计
        logging.info(f"SDPT数据基础信息:")
        logging.info(f"  - 数据形状: {sdpt_gdf.shape}")
        logging.info(f"  - 坐标系统: {sdpt_gdf.crs}")
        logging.info(f"  - 几何类型: {sdpt_gdf.geometry.geom_type.unique()}")
        
        # 字段信息
        logging.info(f"  - 字段列表: {list(sdpt_gdf.columns)}")
        
        # 空间范围
        bounds = sdpt_gdf.total_bounds
        logging.info(f"  - 空间范围: ")
        logging.info(f"    最小经度: {bounds[0]:.6f}")
        logging.info(f"    最小纬度: {bounds[1]:.6f}")
        logging.info(f"    最大经度: {bounds[2]:.6f}")
        logging.info(f"    最大纬度: {bounds[3]:.6f}")
        
        # 显示前几行数据样例
        logging.info(f"  - 数据样例 (前5行):")
        for idx, row in sdpt_gdf.head().iterrows():
            logging.info(f"    行 {idx}: {dict(row.drop('geometry'))}")
        
        # 检查空值
        null_counts = sdpt_gdf.isnull().sum()
        if null_counts.sum() > 0:
            logging.info(f"  - 空值统计:")
            for col, count in null_counts.items():
                if count > 0:
                    logging.info(f"    {col}: {count}")
        
        # 如果有特定字段，进行统计
        if 'type' in sdpt_gdf.columns:
            type_counts = sdpt_gdf['type'].value_counts()
            logging.info(f"  - 类型统计:")
            for type_name, count in type_counts.items():
                logging.info(f"    {type_name}: {count}")
        
        return sdpt_gdf
        
    except Exception as e:
        logging.error(f"读取SDPT数据时出错: {str(e)}")
        return None

def fast_spatial_filter(gdam_gdf, sdpt_gdf, chunk_size=50000):
    """
    快速空间筛选，优化大数据处理
    策略：1. 边界框预筛选 2. 几何体简化 3. 分块处理
    参数：
        gdam_gdf: GDAM区划数据
        sdpt_gdf: SDPT数据
        chunk_size: 分块大小
    返回：
        filtered_sdpt: 筛选后的SDPT数据
    """
    logging.info("=" * 60)
    logging.info("开始执行快速空间筛选（优化版）")
    logging.info("=" * 60)
    
    try:
        # 确保两个数据集使用相同的坐标系
        if gdam_gdf.crs != sdpt_gdf.crs:
            logging.info(f"坐标系不一致，将SDPT数据从 {sdpt_gdf.crs} 转换为 {gdam_gdf.crs}")
            sdpt_gdf = sdpt_gdf.to_crs(gdam_gdf.crs)
        
        logging.info(f"筛选前SDPT数据量: {len(sdpt_gdf):,}")
        
        # 步骤1: 获取区划数据的边界框，进行快速预筛选
        logging.info("步骤1: 使用边界框进行快速预筛选...")
        gdam_bounds = gdam_gdf.total_bounds  # [minx, miny, maxx, maxy]
        
        # 扩展边界框，确保不遗漏边缘数据（扩展0.1度）
        buffer_degree = 0.1
        expanded_bounds = [
            gdam_bounds[0] - buffer_degree,  # minx
            gdam_bounds[1] - buffer_degree,  # miny  
            gdam_bounds[2] + buffer_degree,  # maxx
            gdam_bounds[3] + buffer_degree   # maxy
        ]
        
        logging.info(f"区划数据边界框: [{gdam_bounds[0]:.3f}, {gdam_bounds[1]:.3f}, {gdam_bounds[2]:.3f}, {gdam_bounds[3]:.3f}]")
        logging.info(f"扩展后边界框: [{expanded_bounds[0]:.3f}, {expanded_bounds[1]:.3f}, {expanded_bounds[2]:.3f}, {expanded_bounds[3]:.3f}]")
        
        # 使用边界框快速筛选
        bbox_mask = (
            (sdpt_gdf.geometry.bounds['minx'] <= expanded_bounds[2]) &  # 左边界小于右边界
            (sdpt_gdf.geometry.bounds['maxx'] >= expanded_bounds[0]) &  # 右边界大于左边界
            (sdpt_gdf.geometry.bounds['miny'] <= expanded_bounds[3]) &  # 下边界小于上边界
            (sdpt_gdf.geometry.bounds['maxy'] >= expanded_bounds[1])    # 上边界大于下边界
        )
        
        bbox_filtered = sdpt_gdf[bbox_mask].copy()
        logging.info(f"边界框筛选后数据量: {len(bbox_filtered):,} (筛选比例: {len(bbox_filtered)/len(sdpt_gdf)*100:.2f}%)")
        
        # 如果边界框筛选后数据量仍然很大，可以选择只返回边界框筛选结果
        if len(bbox_filtered) > 200000:  # 如果超过20万条记录
            logging.info("数据量仍然很大，建议只使用边界框筛选结果以提高速度")
            logging.info("如需更精确筛选，可以降低chunk_size或使用更强大的硬件")
            return bbox_filtered
        
        # 步骤2: 简化区划数据几何体以提高后续处理速度
        logging.info("步骤2: 简化区划数据几何体...")
        # 简化几何体，容差设为0.01度（约1km）
        gdam_simplified = gdam_gdf.copy()
        gdam_simplified['geometry'] = gdam_simplified.geometry.simplify(0.01, preserve_topology=True)
        
        # 创建简化后的联合几何体
        gdam_union = gdam_simplified.geometry.unary_union
        logging.info("区划数据几何体简化完成")
        
        # 步骤3: 对边界框筛选后的数据进行精确的几何相交筛选
        logging.info("步骤3: 对预筛选数据进行几何相交筛选...")
        
        if len(bbox_filtered) <= chunk_size:
            # 数据量不大，直接处理
            logging.info("数据量适中，直接进行几何相交筛选...")
            mask = bbox_filtered.geometry.intersects(gdam_union)
            filtered_sdpt = bbox_filtered[mask].copy()
        else:
            # 数据量大，分块处理
            logging.info(f"数据量较大，采用分块处理，块大小: {chunk_size:,}")
            filtered_chunks = []
            
            total_chunks = (len(bbox_filtered) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(bbox_filtered), chunk_size):
                chunk_num = i // chunk_size + 1
                logging.info(f"处理第 {chunk_num}/{total_chunks} 块...")
                
                chunk = bbox_filtered.iloc[i:i+chunk_size]
                mask = chunk.geometry.intersects(gdam_union)
                filtered_chunk = chunk[mask].copy()
                
                if len(filtered_chunk) > 0:
                    filtered_chunks.append(filtered_chunk)
                
                logging.info(f"第 {chunk_num} 块筛选结果: {len(filtered_chunk):,} 条记录")
            
            # 合并所有筛选结果
            if filtered_chunks:
                filtered_sdpt = pd.concat(filtered_chunks, ignore_index=True)
            else:
                filtered_sdpt = gpd.GeoDataFrame()
        
        logging.info(f"最终筛选后数据量: {len(filtered_sdpt):,}")
        logging.info(f"总体筛选比例: {len(filtered_sdpt)/len(sdpt_gdf)*100:.2f}%")
        
        # 统计筛选结果
        if len(filtered_sdpt) > 0:
            # 空间范围
            bounds = filtered_sdpt.total_bounds
            logging.info(f"筛选后数据空间范围:")
            logging.info(f"  最小经度: {bounds[0]:.6f}")
            logging.info(f"  最小纬度: {bounds[1]:.6f}")
            logging.info(f"  最大经度: {bounds[2]:.6f}")
            logging.info(f"  最大纬度: {bounds[3]:.6f}")
            
            # 如果有类型字段，统计各类型数量
            if 'type' in filtered_sdpt.columns:
                type_counts = filtered_sdpt['type'].value_counts()
                logging.info(f"筛选后各类型统计:")
                for type_name, count in type_counts.items():
                    logging.info(f"  {type_name}: {count:,}")
        
        return filtered_sdpt
        
    except Exception as e:
        logging.error(f"空间筛选时出错: {str(e)}")
        return None

def save_filtered_data(filtered_sdpt, output_path):
    """
    保存筛选后的数据
    参数：
        filtered_sdpt: 筛选后的SDPT数据
        output_path: 输出文件路径
    """
    logging.info("=" * 60)
    logging.info("保存筛选结果")
    logging.info("=" * 60)
    
    try:
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        logging.info(f"正在保存筛选结果到: {output_path}")
        filtered_sdpt.to_file(output_path, driver='GPKG')
        
        # 验证保存结果
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logging.info(f"文件保存成功，大小: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        logging.error(f"保存文件时出错: {str(e)}")
        return False

def main():
    """
    主函数
    """
    # 数据路径设置
    gdam_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\区划数据\东南亚国家\southeast_asia_combine.gpkg"
    sdpt_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\总体\sdpt2_shp.shp"
    
    # 输出路径设置
    output_dir = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\筛选结果"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"sdpt_filtered_{timestamp}.gpkg"
    output_path = os.path.join(output_dir, output_filename)
    
    # 设置日志
    log_path = setup_logging(output_dir)
    
    logging.info("SDPT数据筛选程序启动")
    logging.info(f"程序作者: 锐多宝(ruiduobao) - GIS和遥感科学专家")
    logging.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"日志文件: {log_path}")
    
    # 检查输入文件是否存在
    if not os.path.exists(gdam_path):
        logging.error(f"GDAM数据文件不存在: {gdam_path}")
        return
    
    if not os.path.exists(sdpt_path):
        logging.error(f"SDPT数据文件不存在: {sdpt_path}")
        return
    
    # 步骤1: 分析GDAM区划数据
    gdam_gdf = analyze_geopackage_data(gdam_path)
    if gdam_gdf is None:
        logging.error("无法读取GDAM数据，程序终止")
        return
    
    # 步骤2: 分析SDPT数据
    sdpt_gdf = analyze_sdpt_data(sdpt_path)
    if sdpt_gdf is None:
        logging.error("无法读取SDPT数据，程序终止")
        return
    
    # 步骤3: 执行快速空间筛选
    filtered_sdpt = fast_spatial_filter(gdam_gdf, sdpt_gdf)
    if filtered_sdpt is None or len(filtered_sdpt) == 0:
        logging.error("空间筛选失败或无匹配数据")
        return
    
    # 步骤4: 保存结果
    success = save_filtered_data(filtered_sdpt, output_path)
    if success:
        logging.info("=" * 60)
        logging.info("程序执行完成！")
        logging.info(f"筛选结果已保存到: {output_path}")
        logging.info(f"日志文件: {log_path}")
        logging.info("=" * 60)
    else:
        logging.error("保存结果失败")

if __name__ == "__main__":
    main()