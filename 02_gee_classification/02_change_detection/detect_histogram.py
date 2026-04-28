"""
变化概率直方图分析和突变点检测脚本
目的：分析变化概率数据的分布特征，识别突变点，为变化监测提供阈值参考
作者：锐多宝 (ruiduobao)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from kneed import KneeLocator
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_output_directory(output_dir):
    """
    创建输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    return output_dir

def load_and_analyze_data(csv_file_path):
    """
    读取并分析CSV数据文件
    参数:
        csv_file_path: CSV文件路径
    返回:
        df: 数据框
        stats_info: 统计信息字典
    """
    print(f"正在读取数据文件: {csv_file_path}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 计算基本统计信息
    total_pixels = df['pixel_count'].sum()
    mean_bin = np.average(df['bin_max'], weights=df['pixel_count'])
    
    # 计算累积分布
    df['cumulative_pixels'] = df['pixel_count'].cumsum()
    df['cumulative_percentage'] = df['cumulative_pixels'] / total_pixels * 100
    
    # 计算概率密度
    df['probability_density'] = df['pixel_count'] / total_pixels
    
    stats_info = {
        'total_pixels': total_pixels,
        'mean_bin': mean_bin,
        'max_count_bin': df.loc[df['pixel_count'].idxmax(), 'bin_max'],
        'bins_count': len(df)
    }
    
    print(f"总像素数: {total_pixels:,.0f}")
    print(f"加权平均bin值: {mean_bin:.2f}")
    print(f"最大像素数对应的bin值: {stats_info['max_count_bin']}")
    
    return df, stats_info

def plot_histogram(df, output_dir):
    """
    绘制变化概率直方图
    """
    print("正在绘制直方图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('变化概率直方图分析', fontsize=16, fontweight='bold')
    
    # 1. 基本直方图
    ax1 = axes[0, 0]
    ax1.bar(df['bin_max'], df['pixel_count'], width=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('变化概率值')
    ax1.set_ylabel('像素数量')
    ax1.set_title('变化概率分布直方图')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 2. 对数尺度直方图
    ax2 = axes[0, 1]
    ax2.bar(df['bin_max'], df['pixel_count'], width=8, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('变化概率值')
    ax2.set_ylabel('像素数量 (对数尺度)')
    ax2.set_title('变化概率分布直方图 (对数尺度)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. 累积分布图
    ax3 = axes[1, 0]
    ax3.plot(df['bin_max'], df['cumulative_percentage'], 'o-', color='green', linewidth=2, markersize=4)
    ax3.set_xlabel('变化概率值')
    ax3.set_ylabel('累积百分比 (%)')
    ax3.set_title('累积分布函数')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='95%分位数')
    ax3.axhline(y=99, color='orange', linestyle='--', alpha=0.7, label='99%分位数')
    ax3.legend()
    
    # 4. 概率密度图
    ax4 = axes[1, 1]
    ax4.plot(df['bin_max'], df['probability_density'], 'o-', color='purple', linewidth=2, markersize=4)
    ax4.set_xlabel('变化概率值')
    ax4.set_ylabel('概率密度')
    ax4.set_title('概率密度分布')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    histogram_path = os.path.join(output_dir, 'histogram_analysis.png')
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    print(f"直方图已保存: {histogram_path}")
    
    return fig

def detect_change_points(df, output_dir):
    """
    使用多种方法检测突变点
    """
    print("正在检测突变点...")
    
    x = df['bin_max'].values
    y = df['pixel_count'].values
    
    # 对数据进行平滑处理
    y_smooth = gaussian_filter1d(y, sigma=1.5)
    
    # 方法1: 梯度变化检测
    gradient = np.gradient(y_smooth)
    gradient2 = np.gradient(gradient)  # 二阶导数
    
    # 方法2: 使用KneeLocator检测拐点
    try:
        kl = KneeLocator(x, y, curve='convex', direction='decreasing')
        knee_point = kl.knee if kl.knee is not None else None
    except:
        knee_point = None
    
    # 方法3: 基于梯度变化率的突变点检测
    # 寻找梯度变化最大的点
    gradient_change = np.abs(gradient2)
    gradient_peaks, _ = find_peaks(gradient_change, height=np.percentile(gradient_change, 90))
    
    # 方法4: 基于统计分布的阈值
    # 计算95%和99%分位数对应的bin值
    percentile_95 = df[df['cumulative_percentage'] >= 95]['bin_max'].iloc[0] if len(df[df['cumulative_percentage'] >= 95]) > 0 else None
    percentile_99 = df[df['cumulative_percentage'] >= 99]['bin_max'].iloc[0] if len(df[df['cumulative_percentage'] >= 99]) > 0 else None
    
    # 方法5: 基于像素数量急剧下降的点
    # 寻找像素数量下降超过50%的点
    pixel_ratio = y[1:] / y[:-1]
    sharp_drop_indices = np.where(pixel_ratio < 0.5)[0]
    sharp_drop_points = x[sharp_drop_indices] if len(sharp_drop_indices) > 0 else []
    
    # 整合所有检测结果
    change_points = {
        'knee_point': knee_point,
        'gradient_peaks': x[gradient_peaks] if len(gradient_peaks) > 0 else [],
        'percentile_95': percentile_95,
        'percentile_99': percentile_99,
        'sharp_drop_points': sharp_drop_points
    }
    
    # 绘制突变点检测结果
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('突变点检测分析', fontsize=16, fontweight='bold')
    
    # 1. 原始数据和平滑数据
    ax1 = axes[0, 0]
    ax1.plot(x, y, 'o-', alpha=0.5, label='原始数据', markersize=3)
    ax1.plot(x, y_smooth, '-', linewidth=2, label='平滑数据', color='red')
    if knee_point:
        ax1.axvline(x=knee_point, color='green', linestyle='--', linewidth=2, label=f'拐点: {knee_point}')
    ax1.set_xlabel('变化概率值')
    ax1.set_ylabel('像素数量')
    ax1.set_title('数据平滑和拐点检测')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. 梯度分析
    ax2 = axes[0, 1]
    ax2.plot(x, gradient, label='一阶梯度', color='blue')
    ax2.plot(x, gradient2, label='二阶梯度', color='orange')
    if len(gradient_peaks) > 0:
        ax2.scatter(x[gradient_peaks], gradient2[gradient_peaks], color='red', s=50, zorder=5, label='梯度峰值点')
    ax2.set_xlabel('变化概率值')
    ax2.set_ylabel('梯度值')
    ax2.set_title('梯度变化分析')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 累积分布和分位数
    ax3 = axes[1, 0]
    ax3.plot(df['bin_max'], df['cumulative_percentage'], 'o-', color='green', linewidth=2, markersize=4)
    if percentile_95:
        ax3.axvline(x=percentile_95, color='red', linestyle='--', linewidth=2, label=f'95%分位数: {percentile_95}')
    if percentile_99:
        ax3.axvline(x=percentile_99, color='orange', linestyle='--', linewidth=2, label=f'99%分位数: {percentile_99}')
    ax3.set_xlabel('变化概率值')
    ax3.set_ylabel('累积百分比 (%)')
    ax3.set_title('统计分位数阈值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 综合突变点分析
    ax4 = axes[1, 1]
    ax4.plot(x, y, 'o-', alpha=0.7, markersize=3, label='像素分布')
    
    # 标记所有检测到的突变点
    if knee_point:
        ax4.axvline(x=knee_point, color='green', linestyle='--', linewidth=2, label=f'拐点法: {knee_point}')
    if len(gradient_peaks) > 0:
        for peak in x[gradient_peaks]:
            ax4.axvline(x=peak, color='blue', linestyle=':', alpha=0.7)
        ax4.axvline(x=x[gradient_peaks[0]], color='blue', linestyle=':', alpha=0.7, label=f'梯度法: {x[gradient_peaks[0]]:.0f}')
    if percentile_95:
        ax4.axvline(x=percentile_95, color='red', linestyle='--', alpha=0.7, label=f'95%分位: {percentile_95}')
    if len(sharp_drop_points) > 0:
        ax4.axvline(x=sharp_drop_points[0], color='purple', linestyle='-.', alpha=0.7, label=f'急降点: {sharp_drop_points[0]:.0f}')
    
    ax4.set_xlabel('变化概率值')
    ax4.set_ylabel('像素数量')
    ax4.set_title('综合突变点检测结果')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    changepoint_path = os.path.join(output_dir, 'changepoint_detection.png')
    plt.savefig(changepoint_path, dpi=300, bbox_inches='tight')
    print(f"突变点检测图已保存: {changepoint_path}")
    
    return change_points, fig

def recommend_threshold(df, change_points, stats_info):
    """
    基于检测结果推荐阈值
    """
    print("正在分析推荐阈值...")
    
    recommendations = []
    
    # 收集所有有效的突变点
    valid_points = []
    
    if change_points['knee_point']:
        valid_points.append(('拐点法', change_points['knee_point']))
    
    if len(change_points['gradient_peaks']) > 0:
        valid_points.append(('梯度法', change_points['gradient_peaks'][0]))
    
    if change_points['percentile_95']:
        valid_points.append(('95%分位数', change_points['percentile_95']))
    
    if change_points['percentile_99']:
        valid_points.append(('99%分位数', change_points['percentile_99']))
    
    if len(change_points['sharp_drop_points']) > 0:
        valid_points.append(('急降点法', change_points['sharp_drop_points'][0]))
    
    # 分析每个阈值对应的像素比例
    for method, threshold in valid_points:
        # 计算该阈值以上的像素比例
        pixels_above = df[df['bin_max'] >= threshold]['pixel_count'].sum()
        percentage_above = pixels_above / stats_info['total_pixels'] * 100
        
        recommendations.append({
            'method': method,
            'threshold': threshold,
            'pixels_above': pixels_above,
            'percentage_above': percentage_above
        })
    
    # 根据像素比例推荐最佳阈值
    # 目标：选择能够筛选出合适数量像素的阈值（通常1-5%的像素用于变化监测）
    best_recommendation = None
    target_percentage = 3.0  # 目标3%的像素用于监测
    
    min_diff = float('inf')
    for rec in recommendations:
        diff = abs(rec['percentage_above'] - target_percentage)
        if diff < min_diff:
            min_diff = diff
            best_recommendation = rec
    
    return recommendations, best_recommendation

def generate_probability_cumulative_csv(df, output_dir):
    """
    生成概率值和累积百分比的CSV表格
    参数:
        df: 包含直方图数据的数据框
        output_dir: 输出目录
    返回:
        csv_file_path: 生成的CSV文件路径
    """
    print("正在生成概率值和累积百分比CSV表格...")
    
    # 动态获取数据的实际范围
    min_prob = int(df['bin_max'].min())
    max_prob = int(df['bin_max'].max())
    
    print(f"检测到的概率值范围: {min_prob} - {max_prob}")
    
    # 创建从最小值到最大值的概率值范围
    probability_values = list(range(min_prob, max_prob + 1))
    
    # 计算每个概率值对应的累积百分比
    cumulative_percentages = []
    total_pixels = df['pixel_count'].sum()
    
    for prob_val in probability_values:
        # 找到小于等于当前概率值的所有bin
        mask = df['bin_max'] <= prob_val
        if mask.any():
            # 计算累积像素数
            cumulative_pixels = df[mask]['pixel_count'].sum()
            # 计算累积百分比
            cumulative_percentage = (cumulative_pixels / total_pixels) * 100
        else:
            # 如果没有bin小于等于当前概率值，累积百分比为0
            cumulative_percentage = 0.0
        
        cumulative_percentages.append(cumulative_percentage)
    
    # 创建结果数据框
    result_df = pd.DataFrame({
        'probability_value': probability_values,
        'cumulative_percentage': cumulative_percentages
    })
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = os.path.join(output_dir, f'probability_cumulative_table_{timestamp}.csv')
    
    # 保存CSV文件
    result_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    
    print(f"概率值和累积百分比CSV表格已保存: {csv_file_path}")
    print(f"表格包含 {len(result_df)} 行数据，概率值范围: {min_prob}-{max_prob}")
    
    # 显示前几行和后几行数据作为预览
    print("\n数据预览:")
    print("前5行:")
    print(result_df.head().to_string(index=False))
    print("\n后5行:")
    print(result_df.tail().to_string(index=False))
    
    # 验证最终累积百分比是否接近100%
    final_percentage = cumulative_percentages[-1]
    print(f"\n最终累积百分比: {final_percentage:.2f}%")
    if final_percentage < 99.9:
        print("警告: 最终累积百分比未达到100%，请检查数据完整性")
    
    return csv_file_path

def save_results(df, change_points, recommendations, best_recommendation, stats_info, output_dir):
    """
    保存分析结果到文件
    """
    print("正在保存分析结果...")
    
    # 创建结果报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建日志文件
    log_file = os.path.join(output_dir, f'analysis_log_{timestamp}.txt')
    
    # 1. 保存详细分析结果
    results_file = os.path.join(output_dir, f'histogram_analysis_results_{timestamp}.txt')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("变化概率直方图分析结果报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析师: 锐多宝 (ruiduobao)\n\n")
        
        f.write("数据基本信息:\n")
        f.write("-" * 30 + "\n")
        f.write(f"总像素数: {stats_info['total_pixels']:,.0f}\n")
        f.write(f"加权平均bin值: {stats_info['mean_bin']:.2f}\n")
        f.write(f"最大像素数对应bin值: {stats_info['max_count_bin']}\n")
        f.write(f"bin数量: {stats_info['bins_count']}\n\n")
        
        f.write("突变点检测结果:\n")
        f.write("-" * 30 + "\n")
        if change_points['knee_point']:
            f.write(f"拐点法检测结果: {change_points['knee_point']}\n")
        if len(change_points['gradient_peaks']) > 0:
            f.write(f"梯度法检测结果: {change_points['gradient_peaks']}\n")
        if change_points['percentile_95']:
            f.write(f"95%分位数: {change_points['percentile_95']}\n")
        if change_points['percentile_99']:
            f.write(f"99%分位数: {change_points['percentile_99']}\n")
        if len(change_points['sharp_drop_points']) > 0:
            f.write(f"急降点法检测结果: {change_points['sharp_drop_points']}\n")
        f.write("\n")
        
        f.write("阈值推荐分析:\n")
        f.write("-" * 30 + "\n")
        for rec in recommendations:
            f.write(f"{rec['method']}: 阈值={rec['threshold']:.1f}, "
                   f"监测像素数={rec['pixels_above']:,.0f}, "
                   f"占比={rec['percentage_above']:.2f}%\n")
        
        f.write("\n最佳推荐阈值:\n")
        f.write("-" * 30 + "\n")
        if best_recommendation:
            f.write(f"推荐方法: {best_recommendation['method']}\n")
            f.write(f"推荐阈值: {best_recommendation['threshold']:.1f}\n")
            f.write(f"监测像素数: {best_recommendation['pixels_above']:,.0f}\n")
            f.write(f"监测像素占比: {best_recommendation['percentage_above']:.2f}%\n")
            f.write(f"\n解释: 该阈值能够筛选出约{best_recommendation['percentage_above']:.1f}%的像素进行变化监测，\n")
            f.write("这个比例既能保证监测的有效性，又不会产生过多的计算负担。\n")
    
    # 2. 保存CSV格式的推荐结果
    recommendations_df = pd.DataFrame(recommendations)
    csv_file = os.path.join(output_dir, f'threshold_recommendations_{timestamp}.csv')
    recommendations_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # 3. 保存处理后的原始数据
    processed_data_file = os.path.join(output_dir, f'processed_histogram_data_{timestamp}.csv')
    df.to_csv(processed_data_file, index=False, encoding='utf-8-sig')
    
    print(f"分析结果已保存:")
    print(f"  - 详细报告: {results_file}")
    print(f"  - 推荐阈值CSV: {csv_file}")
    print(f"  - 处理后数据: {processed_data_file}")
    
    return results_file, csv_file, processed_data_file

def main():
    """
    主函数：执行完整的分析流程
    """
    print("开始变化概率直方图分析...")
    print("=" * 60)
    
    # 输入输出路径
    csv_file_path = r"K:\地理所\论文\东南亚10m人工林提取\数据\GEE分类\变化概率\直方图\histogram_zone5_2018vs2019_bins10.csv"
    output_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\GEE分类\变化概率\直方图_分析结果\zone5_2018vs2019_bins"
    
    # 创建输出目录
    create_output_directory(output_dir)
    
    try:
        # 1. 读取和分析数据
        df, stats_info = load_and_analyze_data(csv_file_path)
        
        # 2. 绘制直方图
        hist_fig = plot_histogram(df, output_dir)
        
        # 3. 检测突变点
        change_points, change_fig = detect_change_points(df, output_dir)
        
        # 4. 推荐阈值
        recommendations, best_recommendation = recommend_threshold(df, change_points, stats_info)
        
        # 5. 生成概率值和累积百分比CSV表格
        probability_csv_path = generate_probability_cumulative_csv(df, output_dir)
        
        # 6. 保存结果
        results_files = save_results(df, change_points, recommendations, best_recommendation, stats_info, output_dir)
        
        # 6. 输出总结
        print("\n" + "=" * 60)
        print("分析完成！主要结果:")
        print("-" * 40)
        
        if best_recommendation:
            print(f"推荐阈值: {best_recommendation['threshold']:.1f}")
            print(f"推荐方法: {best_recommendation['method']}")
            print(f"监测像素占比: {best_recommendation['percentage_above']:.2f}%")
            print(f"监测像素数量: {best_recommendation['pixels_above']:,.0f}")
        
        print(f"\n所有结果文件已保存到: {output_dir}")
        print(f"概率值和累积百分比CSV表格: {probability_csv_path}")
        print("分析完成！")
        
        # 保存图表而不显示
        plt.close('all')  # 关闭所有图表，避免内存占用
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()