# 拆分训练集和测试集
import os
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_output_directories(output_base_path):
    """创建输出目录结构"""
    train_dir = os.path.join(output_base_path, "训练集")
    test_dir = os.path.join(output_base_path, "测试集")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    return train_dir, test_dir

def extract_year_from_filename(filename):
    """从文件名中提取年份信息"""
    # 假设文件名格式为 ZoneX_Merged_AllBand_Sample_filtered.shp
    # 或者包含年份信息的其他格式
    import re
    
    # 尝试从文件名中提取4位数字年份
    year_match = re.search(r'(20\d{2})', filename)
    if year_match:
        return int(year_match.group(1))
    
    # 如果文件名中没有年份，可能需要从数据中的year字段获取
    return None

def get_year_from_data(gdf):
    """从数据的year字段获取年份信息"""
    if 'year' in gdf.columns:
        years = gdf['year'].unique()
        return years
    return []

def split_by_year(gdf, test_size=0.2, random_state=42):
    """按年份进行训练集和测试集拆分，然后合并所有年份的训练和测试数据"""
    all_train_data = []
    all_test_data = []
    year_stats = {}
    
    # 获取所有年份
    if 'year' in gdf.columns:
        years = gdf['year'].unique()
        
        for year in years:
            year_data = gdf[gdf['year'] == year].copy()
            
            if len(year_data) > 0:
                # 按80%和20%拆分
                if len(year_data) == 1:
                    # 如果只有一条记录，放入训练集
                    train_data = year_data
                    test_data = gpd.GeoDataFrame(columns=year_data.columns, crs=year_data.crs)
                else:
                    try:
                        # 尝试按landcover分层抽样
                        if 'landcover' in year_data.columns and len(year_data['landcover'].unique()) > 1:
                            train_data, test_data = train_test_split(
                                year_data, 
                                test_size=test_size, 
                                random_state=random_state,
                                stratify=year_data['landcover']
                            )
                        else:
                            # 如果无法分层，则随机拆分
                            train_data, test_data = train_test_split(
                                year_data, 
                                test_size=test_size, 
                                random_state=random_state
                            )
                    except ValueError:
                        # 如果分层抽样失败，使用随机拆分
                        train_data, test_data = train_test_split(
                            year_data, 
                            test_size=test_size, 
                            random_state=random_state
                        )
                
                # 收集训练和测试数据
                if len(train_data) > 0:
                    all_train_data.append(train_data)
                if len(test_data) > 0:
                    all_test_data.append(test_data)
                
                # 记录统计信息
                year_stats[year] = {
                    'total': len(year_data),
                    'train_count': len(train_data),
                    'test_count': len(test_data)
                }
    
    # 合并所有年份的数据
    if all_train_data:
        combined_train = gpd.pd.concat(all_train_data, ignore_index=True)
        combined_train = gpd.GeoDataFrame(combined_train, crs=gdf.crs)
    else:
        combined_train = gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
    
    if all_test_data:
        combined_test = gpd.pd.concat(all_test_data, ignore_index=True)
        combined_test = gpd.GeoDataFrame(combined_test, crs=gdf.crs)
    else:
        combined_test = gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
    
    return {
        'train': combined_train,
        'test': combined_test,
        'year_stats': year_stats,
        'total_train': len(combined_train),
        'total_test': len(combined_test)
    }

def process_shapefile(input_file, train_dir, test_dir):
    """处理单个shapefile文件"""
    try:
        print(f"正在处理: {os.path.basename(input_file)}")
        
        # 读取shapefile
        gdf = gpd.read_file(input_file)
        
        if len(gdf) == 0:
            print(f"  警告: {input_file} 为空文件")
            return None
        
        # 检查是否有year字段
        if 'year' not in gdf.columns:
            print(f"  警告: {input_file} 中没有找到'year'字段")
            return None
        
        print(f"  总样本数: {len(gdf)}")
        print(f"  包含年份: {sorted(gdf['year'].unique())}")
        
        # 按年份拆分数据
        split_result = split_by_year(gdf)
        
        if split_result['total_train'] == 0 and split_result['total_test'] == 0:
            print(f"  警告: {input_file} 拆分后没有数据")
            return None
        
        # 获取基础文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 训练集文件名
        train_filename = f"{base_name}_train.shp"
        train_path = os.path.join(train_dir, train_filename)
        
        # 测试集文件名
        test_filename = f"{base_name}_test.shp"
        test_path = os.path.join(test_dir, test_filename)
        
        # 保存训练集
        if len(split_result['train']) > 0:
            split_result['train'].to_file(train_path, encoding='utf-8')
            print(f"  保存训练集: {train_filename} ({len(split_result['train'])} 条记录)")
        
        # 保存测试集
        if len(split_result['test']) > 0:
            split_result['test'].to_file(test_path, encoding='utf-8')
            print(f"  保存测试集: {test_filename} ({len(split_result['test'])} 条记录)")
        
        # 统计信息
        file_stats = {
            'filename': base_name,
            'total_features': len(gdf),
            'train_total': split_result['total_train'],
            'test_total': split_result['total_test'],
            'year_stats': split_result['year_stats']
        }
        
        # 显示按年份的拆分统计
        print("  按年份拆分统计:")
        for year in sorted(split_result['year_stats'].keys()):
            stats = split_result['year_stats'][year]
            print(f"    {year}年: 总计 {stats['total']}, 训练集 {stats['train_count']}, 测试集 {stats['test_count']}")
        
        return file_stats
        
    except Exception as e:
        print(f"  错误: 处理 {input_file} 时出现错误: {str(e)}")
        return None

def main():
    # 输入和输出路径
    input_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并_剔除异常样本_去掉不需要的属性\合并后的zone数据"
    output_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_逐年合并_剔除异常样本_去掉不需要的属性\拆分训练集测试集的zone数据"
    
    print("=" * 60)
    print("按年份拆分训练集和测试集")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"拆分比例: 80% 训练集, 20% 测试集")
    print("=" * 60)
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录结构
    train_dir, test_dir = create_output_directories(output_dir)
    print(f"训练集输出目录: {train_dir}")
    print(f"测试集输出目录: {test_dir}")
    print()
    
    # 查找所有shapefile文件
    shp_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.shp'):
            shp_files.append(os.path.join(input_dir, file))
    
    if not shp_files:
        print("错误: 在输入目录中没有找到shapefile文件")
        return
    
    print(f"找到 {len(shp_files)} 个shapefile文件")
    print()
    
    # 处理统计信息
    all_stats = []
    success_count = 0
    error_count = 0
    
    # 处理每个shapefile
    for shp_file in shp_files:
        stats = process_shapefile(shp_file, train_dir, test_dir)
        if stats:
            all_stats.append(stats)
            success_count += 1
        else:
            error_count += 1
        print()
    
    # 生成统计报告
    print("=" * 60)
    print("处理完成统计")
    print("=" * 60)
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print()
    
    if all_stats:
        # 创建详细统计报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"训练测试集拆分统计报告_{timestamp}.csv")
        
        # 准备报告数据
        report_data = []
        total_train = 0
        total_test = 0
        total_features = 0
        
        for stats in all_stats:
            total_features += stats['total_features']
            total_train += stats['train_total']
            total_test += stats['test_total']
            
            # 为每个文件创建一行记录
            report_data.append({
                '文件名': stats['filename'],
                '总样本数': stats['total_features'],
                '训练集样本数': stats['train_total'],
                '测试集样本数': stats['test_total'],
                '训练集比例': f"{stats['train_total']/stats['total_features']*100:.1f}%" if stats['total_features'] > 0 else "0%",
                '测试集比例': f"{stats['test_total']/stats['total_features']*100:.1f}%" if stats['total_features'] > 0 else "0%"
            })
            
            # 为每个年份创建详细记录
            for year, year_stats in stats['year_stats'].items():
                report_data.append({
                    '文件名': f"{stats['filename']}_{year}年详情",
                    '总样本数': year_stats['total'],
                    '训练集样本数': year_stats['train_count'],
                    '测试集样本数': year_stats['test_count'],
                    '训练集比例': f"{year_stats['train_count']/year_stats['total']*100:.1f}%" if year_stats['total'] > 0 else "0%",
                    '测试集比例': f"{year_stats['test_count']/year_stats['total']*100:.1f}%" if year_stats['total'] > 0 else "0%"
                })
        
        # 保存报告
        df_report = pd.DataFrame(report_data)
        df_report.to_csv(report_file, index=False, encoding='utf-8-sig')
        
        print(f"详细统计报告已保存: {report_file}")
        print()
        print("总体统计:")
        print(f"  总样本数: {total_features:,}")
        print(f"  训练集样本数: {total_train:,} ({total_train/total_features*100:.1f}%)")
        print(f"  测试集样本数: {total_test:,} ({total_test/total_features*100:.1f}%)")
        
        # 按年份统计
        year_summary = {}
        for stats in all_stats:
            for year, year_stats in stats['year_stats'].items():
                if year not in year_summary:
                    year_summary[year] = {'total': 0, 'train': 0, 'test': 0}
                year_summary[year]['total'] += year_stats['total']
                year_summary[year]['train'] += year_stats['train_count']
                year_summary[year]['test'] += year_stats['test_count']
        
        print("\n按年份统计:")
        for year in sorted(year_summary.keys()):
            stats = year_summary[year]
            print(f"  {year}年: 总计 {stats['total']:,}, 训练集 {stats['train']:,}, 测试集 {stats['test']:,}")
    
    print("\n" + "=" * 60)
    print("拆分完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()