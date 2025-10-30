
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 分类结果
RF_class_result=r"D:\地理所\论文\东南亚10m人工林提取\数据\第二次分类\合并\merged_zone5_2024_20251024_160615.tif"
# 样本点
Points_SHP=r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果_本地分析\2024\3.筛选后样本点_20251023_230327.shp"
# 点的属性
Points_Attr="landcover"

Out_put_FILE=r"D:\地理所\论文\东南亚10m人工林提取\数据\第二次分类_马尔可夫模型_高性能\计算发射概率\emission_probabilities_2024.csv"

def extract_raster_values_at_points(raster_path, points_gdf):
    """
    提取样本点位置对应的栅格值
    """
    logger.info("开始提取样本点对应的栅格值...")
    
    # 确保点数据和栅格数据的坐标系一致
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        logger.info(f"栅格数据坐标系: {raster_crs}")
        
        # 如果坐标系不一致，转换点数据坐标系
        if points_gdf.crs != raster_crs:
            logger.info(f"点数据坐标系: {points_gdf.crs}, 转换为栅格坐标系")
            points_gdf = points_gdf.to_crs(raster_crs)
        
        # 提取栅格值
        raster_values = []
        coords = [(x, y) for x, y in zip(points_gdf.geometry.x, points_gdf.geometry.y)]
        
        # 使用rasterio的sample方法提取值
        for val in src.sample(coords):
            raster_values.append(val[0])
    
    logger.info(f"成功提取了 {len(raster_values)} 个点的栅格值")
    return np.array(raster_values)

def calculate_confusion_matrix():
    """
    计算混淆矩阵和精度评估指标
    """
    logger.info("开始计算混淆矩阵...")
    
    # 读取样本点数据
    logger.info(f"读取样本点数据: {Points_SHP}")
    try:
        points_gdf = gpd.read_file(Points_SHP)
        logger.info(f"样本点数量: {len(points_gdf)}")
        logger.info(f"样本点属性列: {list(points_gdf.columns)}")
        
        # 检查landcover属性是否存在
        if Points_Attr not in points_gdf.columns:
            logger.error(f"属性 '{Points_Attr}' 不存在于样本点数据中")
            logger.info(f"可用属性: {list(points_gdf.columns)}")
            return None
            
    except Exception as e:
        logger.error(f"读取样本点数据失败: {e}")
        return None
    
    # 检查栅格文件是否存在
    if not os.path.exists(RF_class_result):
        logger.error(f"栅格文件不存在: {RF_class_result}")
        return None
    
    # 提取栅格值
    try:
        raster_values = extract_raster_values_at_points(RF_class_result, points_gdf)
        reference_values = points_gdf[Points_Attr].values
        
        logger.info(f"参考值范围: {np.unique(reference_values)}")
        logger.info(f"预测值范围: {np.unique(raster_values)}")
        
    except Exception as e:
        logger.error(f"提取栅格值失败: {e}")
        return None
    
    # 过滤有效数据（排除NoData值）
    valid_mask = (raster_values != 0) & (~np.isnan(raster_values)) & (reference_values != 0) & (~np.isnan(reference_values))
    
    if np.sum(valid_mask) == 0:
        logger.error("没有有效的数据点用于计算混淆矩阵")
        return None
    
    reference_filtered = reference_values[valid_mask]
    predicted_filtered = raster_values[valid_mask]
    
    logger.info(f"有效数据点数量: {len(reference_filtered)}")
    logger.info(f"过滤后参考值范围: {np.unique(reference_filtered)}")
    logger.info(f"过滤后预测值范围: {np.unique(predicted_filtered)}")
    
    # 确保只包含地物类别1、2、3
    valid_classes = [1, 2, 3]
    class_mask = np.isin(reference_filtered, valid_classes) & np.isin(predicted_filtered, valid_classes)
    
    if np.sum(class_mask) == 0:
        logger.error("没有地物类别1、2、3的有效数据")
        return None
    
    reference_final = reference_filtered[class_mask]
    predicted_final = predicted_filtered[class_mask]
    
    logger.info(f"最终用于计算的数据点数量: {len(reference_final)}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(reference_final, predicted_final, labels=valid_classes)
    
    # 计算精度指标
    overall_accuracy = accuracy_score(reference_final, predicted_final)
    kappa = cohen_kappa_score(reference_final, predicted_final)
    
    # 创建混淆矩阵DataFrame
    cm_df = pd.DataFrame(cm, 
                        index=[f'参考_地物{i}' for i in valid_classes],
                        columns=[f'预测_地物{i}' for i in valid_classes])
    
    # 添加行和列的总计
    cm_df['行总计'] = cm_df.sum(axis=1)
    cm_df.loc['列总计'] = cm_df.sum(axis=0)
    
    # 计算用户精度和生产者精度
    user_accuracy = []
    producer_accuracy = []
    
    for i, class_id in enumerate(valid_classes):
        # 用户精度 = 对角线元素 / 行总计
        ua = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        user_accuracy.append(ua)
        
        # 生产者精度 = 对角线元素 / 列总计
        pa = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        producer_accuracy.append(pa)
    
    # 创建精度统计表
    accuracy_stats = pd.DataFrame({
        '地物类别': [f'地物{i}' for i in valid_classes],
        '用户精度': [f'{ua:.4f}' for ua in user_accuracy],
        '生产者精度': [f'{pa:.4f}' for pa in producer_accuracy],
        '样本数量': [np.sum(reference_final == class_id) for class_id in valid_classes]
    })
    
    # 输出结果
    logger.info("\n=== 混淆矩阵 ===")
    print(cm_df)
    
    logger.info(f"\n=== 精度评估 ===")
    logger.info(f"总体精度: {overall_accuracy:.4f}")
    logger.info(f"Kappa系数: {kappa:.4f}")
    
    logger.info("\n=== 各类别精度 ===")
    print(accuracy_stats)
    
    # 保存结果到CSV
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(Out_put_FILE), exist_ok=True)
        
        # 保存混淆矩阵
        confusion_matrix_file = Out_put_FILE.replace('.csv', '_confusion_matrix.csv')
        cm_df.to_csv(confusion_matrix_file, encoding='utf-8-sig')
        logger.info(f"混淆矩阵已保存到: {confusion_matrix_file}")
        
        # 保存精度统计
        accuracy_file = Out_put_FILE.replace('.csv', '_accuracy_stats.csv')
        
        # 添加总体精度信息
        summary_stats = pd.DataFrame({
            '指标': ['总体精度', 'Kappa系数', '总样本数'],
            '数值': [f'{overall_accuracy:.4f}', f'{kappa:.4f}', len(reference_final)]
        })
        
        # 合并所有统计信息
        with open(accuracy_file, 'w', encoding='utf-8-sig', newline='') as f:
            summary_stats.to_csv(f, index=False)
            f.write('\n')
            accuracy_stats.to_csv(f, index=False)
        
        logger.info(f"精度统计已保存到: {accuracy_file}")
        
        # 保存详细分类报告
        report = classification_report(reference_final, predicted_final, 
                                     target_names=[f'地物{i}' for i in valid_classes],
                                     output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_file = Out_put_FILE.replace('.csv', '_classification_report.csv')
        report_df.to_csv(report_file, encoding='utf-8-sig')
        logger.info(f"分类报告已保存到: {report_file}")
        
    except Exception as e:
        logger.error(f"保存结果文件失败: {e}")
    
    return {
        'confusion_matrix': cm_df,
        'accuracy_stats': accuracy_stats,
        'overall_accuracy': overall_accuracy,
        'kappa': kappa,
        'sample_count': len(reference_final)
    }

if __name__ == "__main__":
    logger.info("开始计算混淆矩阵和精度评估...")
    result = calculate_confusion_matrix()
    if result:
        logger.info("计算完成！")
    else:
        logger.error("计算失败！")