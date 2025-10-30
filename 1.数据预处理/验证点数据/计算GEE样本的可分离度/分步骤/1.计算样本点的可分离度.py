# spectral_separability_gdal.py
# 目的：计算GEE样本点的光谱可分离度，用于评估不同地物类别的可区分性
# 输出详细的可分离度结果、统计表和可视化图表

from osgeo import ogr
import numpy as np
import math
import sys
import pandas as pd  # 用于数据处理和CSV输出
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns  # 用于美化图表
from datetime import datetime  # 用于生成时间戳
import os  # 用于文件路径操作

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 参数：文件、图层、字段名
VECTOR_PATH = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\GEE结果\Zone5_2018_Merged_AllBand_Sample.shp"   # 替换为你的文件
output_dir = r"K:\地理所\论文\东南亚10m人工林提取\数据\验证点数据\上传到GEE的样本\经过了8年的土地覆盖数据filter_平衡样本_获得Ebed波段\计算样本点的可分离度"
LAYER_NAME = None             # None 则取第一个图层
class_field = "landcover"     # 类别字段名
# 生成从A01到A63的完整波段列表（63个波段）
bands = [f"A{i:02d}" for i in range(0, 64)]  

eps = 1e-6  # 协方差正则化项

def read_features(path, layer_name=None):
    """
    读取矢量文件中的要素属性
    参数：
        path: 矢量文件路径
        layer_name: 图层名称，None则取第一个图层
    返回：
        features: 要素属性列表
    """
    ds = ogr.Open(path, 0)
    if ds is None:
        raise RuntimeError("无法打开矢量文件: " + path)
    if layer_name is None:
        layer = ds.GetLayer(0)
    else:
        layer = ds.GetLayerByName(layer_name)
    features = []
    layer.ResetReading()
    for feat in layer:
        props = feat.items()
        features.append(props)
    return features

def group_by_class(features, class_field, bands):
    """
    按类别分组样本数据
    参数：
        features: 要素属性列表
        class_field: 类别字段名
        bands: 波段名称列表
    返回：
        groups: 按类别分组的光谱数据字典
    """
    groups = {}
    for f in features:
        cls = f[class_field]
        vec = np.array([float(f[b]) for b in bands], dtype=np.float64)
        groups.setdefault(cls, []).append(vec)
    for k in list(groups.keys()):
        groups[k] = np.vstack(groups[k])
    return groups

def mean_and_cov(X):
    """
    计算均值和协方差矩阵
    参数：
        X: 样本数据矩阵 (n x d)
    返回：
        mu: 均值向量
        cov: 协方差矩阵
    """
    # X: n x d
    mu = X.mean(axis=0)
    # 行向量中心化
    Xc = X - mu
    cov = np.dot(Xc.T, Xc) / (X.shape[0] - 1) if X.shape[0] > 1 else np.eye(X.shape[1]) * eps
    return mu, cov

def bhattacharyya_distance(mu1, cov1, mu2, cov2, reg=1e-6):
    """
    计算Bhattacharyya距离
    参数：
        mu1, mu2: 两个类别的均值向量
        cov1, cov2: 两个类别的协方差矩阵
        reg: 正则化参数
    返回：
        B: Bhattacharyya距离
    """
    d = mu1.shape[0]
    # 正则化协方差，防止奇异
    cov1_r = cov1 + np.eye(d) * reg
    cov2_r = cov2 + np.eye(d) * reg
    covm = 0.5 * (cov1_r + cov2_r)
    # 使用 slogdet 提高数值稳定性
    sign_m, logdet_m = np.linalg.slogdet(covm)
    sign1, logdet1 = np.linalg.slogdet(cov1_r)
    sign2, logdet2 = np.linalg.slogdet(cov2_r)
    if sign_m <= 0 or sign1 <= 0 or sign2 <= 0:
        # 极端情况下 fallback
        logdet_term = 0.0
    else:
        logdet_term = 0.5 * (logdet_m - 0.5*(logdet1 + logdet2))
    # Mahalanobis 样项
    try:
        inv_covm = np.linalg.inv(covm)
    except np.linalg.LinAlgError:
        inv_covm = np.linalg.pinv(covm)
    diff = (mu1 - mu2).reshape((d,1))
    term1 = 0.125 * float(diff.T.dot(inv_covm).dot(diff))
    term2 = logdet_term
    B = term1 + term2
    return float(B)

def compute_pairwise(groups):
    """
    计算所有类别对之间的可分离度
    参数：
        groups: 按类别分组的光谱数据
    返回：
        results: 可分离度结果列表
        stats: 各类别统计信息
    """
    classes = sorted(groups.keys())
    stats = {}
    for c in classes:
        X = groups[c]
        mu, cov = mean_and_cov(X)
        stats[c] = {"n": X.shape[0], "mu": mu, "cov": cov}
    results = []
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            c1 = classes[i]; c2 = classes[j]
            s1 = stats[c1]; s2 = stats[c2]
            B = bhattacharyya_distance(s1["mu"], s1["cov"], s2["mu"], s2["cov"], reg=eps)
            JM = 2.0 * (1.0 - math.exp(-B))  # 常见实现
            results.append((c1, c2, s1["n"], s2["n"], B, JM))
    return results, stats

def save_results_to_csv(results, stats, output_dir):
    """
    保存可分离度结果到CSV文件
    参数：
        results: 可分离度结果列表
        stats: 各类别统计信息
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存可分离度结果
    df_results = pd.DataFrame(results, columns=['类别1', '类别2', '类别1样本数', '类别2样本数', 'Bhattacharyya距离', 'JM距离'])
    results_file = os.path.join(output_dir, f"可分离度结果_{timestamp}.csv")
    df_results.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"可分离度结果已保存到: {results_file}")
    
    # 保存类别统计信息
    stats_data = []
    for cls, stat in stats.items():
        stats_data.append({
            '类别': cls,
            '样本数量': stat['n'],
            '均值_最小值': np.min(stat['mu']),
            '均值_最大值': np.max(stat['mu']),
            '均值_平均值': np.mean(stat['mu']),
            '协方差矩阵_迹': np.trace(stat['cov']),
            '协方差矩阵_行列式': np.linalg.det(stat['cov'])
        })
    
    df_stats = pd.DataFrame(stats_data)
    stats_file = os.path.join(output_dir, f"类别统计信息_{timestamp}.csv")
    df_stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"类别统计信息已保存到: {stats_file}")
    
    return df_results, df_stats, timestamp

def create_visualizations(df_results, df_stats, output_dir, timestamp):
    """
    创建可视化图表
    参数：
        df_results: 可分离度结果DataFrame
        df_stats: 类别统计信息DataFrame
        output_dir: 输出目录
        timestamp: 时间戳
    """
    # 设置图表样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 创建图表1：样本数量分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sample Separability Analysis Report', fontsize=16, fontweight='bold')
    
    # 子图1：各类别样本数量
    ax1 = axes[0, 0]
    bars = ax1.bar(df_stats['类别'], df_stats['样本数量'], color=sns.color_palette("husl", len(df_stats)))
    ax1.set_title('Sample Count Distribution by Class', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Land Cover Class')
    ax1.set_ylabel('Sample Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # 子图2：JM距离分布直方图
    ax2 = axes[0, 1]
    ax2.hist(df_results['JM距离'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('JM Distance Distribution Histogram', fontsize=12, fontweight='bold')
    ax2.set_xlabel('JM Distance')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df_results['JM距离'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df_results["JM距离"].mean():.3f}')
    ax2.legend()
    
    # 子图3：可分离度热力图
    ax3 = axes[1, 0]
    # 创建可分离度矩阵
    classes = sorted(set(df_results['类别1'].tolist() + df_results['类别2'].tolist()))
    separability_matrix = np.zeros((len(classes), len(classes)))
    
    for _, row in df_results.iterrows():
        i = classes.index(row['类别1'])
        j = classes.index(row['类别2'])
        separability_matrix[i, j] = row['JM距离']
        separability_matrix[j, i] = row['JM距离']
    
    im = ax3.imshow(separability_matrix, cmap='RdYlBu_r', aspect='auto')
    ax3.set_title('Inter-class JM Distance Heatmap', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(classes)))
    ax3.set_yticks(range(len(classes)))
    ax3.set_xticklabels(classes, rotation=45)
    ax3.set_yticklabels(classes)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('JM Distance')
    
    # 在热力图上添加数值
    for i in range(len(classes)):
        for j in range(len(classes)):
            if separability_matrix[i, j] > 0:
                text = ax3.text(j, i, f'{separability_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
    
    # 子图4：可分离度等级统计
    ax4 = axes[1, 1]
    # 定义可分离度等级
    def classify_separability(jm_distance):
        if jm_distance < 1.0:
            return 'Poor (<1.0)'
        elif jm_distance < 1.5:
            return 'Fair (1.0-1.5)'
        elif jm_distance < 1.8:
            return 'Good (1.5-1.8)'
        else:
            return 'Excellent (≥1.8)'
    
    df_results['可分离度等级'] = df_results['JM距离'].apply(classify_separability)
    separability_counts = df_results['可分离度等级'].value_counts()
    
    colors = ['red', 'orange', 'lightgreen', 'green']
    wedges, texts, autotexts = ax4.pie(separability_counts.values, labels=separability_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title('Separability Level Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, f"样本可分离度分析图_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {plot_file}")
    plt.show()
    
    # 创建详细的可分离度对比图
    fig2, ax = plt.subplots(figsize=(12, 8))
    
    # 按JM距离排序
    df_sorted = df_results.sort_values('JM距离', ascending=True)
    
    # 创建类别对标签
    pair_labels = [f"{row['类别1']} vs {row['类别2']}" for _, row in df_sorted.iterrows()]
    
    # 绘制水平条形图
    bars = ax.barh(range(len(df_sorted)), df_sorted['JM距离'], 
                   color=plt.cm.RdYlGn(df_sorted['JM距离']/2.0))
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(pair_labels, fontsize=10)
    ax.set_xlabel('JM Distance', fontsize=12)
    ax.set_title('Detailed Separability Comparison by Class Pairs', fontsize=14, fontweight='bold')
    
    # 添加参考线
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Poor/Fair Threshold (1.0)')
    ax.axvline(x=1.5, color='orange', linestyle='--', alpha=0.7, label='Fair/Good Threshold (1.5)')
    ax.axvline(x=1.8, color='green', linestyle='--', alpha=0.7, label='Good/Excellent Threshold (1.8)')
    ax.legend()
    
    # 在条形图上添加数值
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['JM距离'] + 0.02, i, f'{row["JM距离"]:.3f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存详细对比图
    detail_plot_file = os.path.join(output_dir, f"可分离度详细对比图_{timestamp}.png")
    plt.savefig(detail_plot_file, dpi=300, bbox_inches='tight')
    print(f"详细对比图已保存到: {detail_plot_file}")
    plt.show()

def print_summary_statistics(df_results, df_stats):
    """
    打印汇总统计信息
    参数：
        df_results: 可分离度结果DataFrame
        df_stats: 类别统计信息DataFrame
    """
    print("\n" + "="*60)
    print("样本可分离度分析汇总报告")
    print("="*60)
    
    print(f"\n1. 基本信息:")
    print(f"   - 总类别数: {len(df_stats)}")
    print(f"   - 总样本数: {df_stats['样本数量'].sum()}")
    print(f"   - 类别对数: {len(df_results)}")
    print(f"   - 波段数量: {len(bands)}")
    
    print(f"\n2. 样本分布:")
    for _, row in df_stats.iterrows():
        print(f"   - {row['类别']}: {row['样本数量']} 个样本")
    
    print(f"\n3. 可分离度统计:")
    print(f"   - JM距离范围: {df_results['JM距离'].min():.3f} - {df_results['JM距离'].max():.3f}")
    print(f"   - JM距离平均值: {df_results['JM距离'].mean():.3f}")
    print(f"   - JM距离标准差: {df_results['JM距离'].std():.3f}")
    
    # 可分离度等级统计
    def classify_separability(jm_distance):
        if jm_distance < 1.0:
            return '差'
        elif jm_distance < 1.5:
            return '一般'
        elif jm_distance < 1.8:
            return '良好'
        else:
            return '优秀'
    
    df_results['可分离度等级'] = df_results['JM距离'].apply(classify_separability)
    separability_counts = df_results['可分离度等级'].value_counts()
    
    print(f"\n4. 可分离度等级分布:")
    for level, count in separability_counts.items():
        percentage = (count / len(df_results)) * 100
        print(f"   - {level}: {count} 对 ({percentage:.1f}%)")
    
    # 找出可分离度最好和最差的类别对
    best_pair = df_results.loc[df_results['JM距离'].idxmax()]
    worst_pair = df_results.loc[df_results['JM距离'].idxmin()]
    
    print(f"\n5. 极值分析:")
    print(f"   - 最易区分的类别对: {best_pair['类别1']} vs {best_pair['类别2']} (JM距离: {best_pair['JM距离']:.3f})")
    print(f"   - 最难区分的类别对: {worst_pair['类别1']} vs {worst_pair['类别2']} (JM距离: {worst_pair['JM距离']:.3f})")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("开始计算样本可分离度...")
    print(f"输入文件: {VECTOR_PATH}")
    print(f"类别字段: {class_field}")
    print(f"波段数量: {len(bands)}")
    
    try:
        # 读取数据
        print("\n正在读取矢量数据...")
        feats = read_features(VECTOR_PATH, LAYER_NAME)
        print(f"成功读取 {len(feats)} 个要素")
        
        # 按类别分组
        print("正在按类别分组数据...")
        groups = group_by_class(feats, class_field, bands)
        print(f"发现 {len(groups)} 个类别")
        
        # 计算可分离度
        print("正在计算可分离度...")
        results, stats = compute_pairwise(groups)
        print(f"计算了 {len(results)} 个类别对的可分离度")
        

        
        # 保存结果到CSV
        print("\n正在保存结果...")
        df_results, df_stats, timestamp = save_results_to_csv(results, stats, output_dir)
        
        # 创建可视化图表
        print("正在生成可视化图表...")
        create_visualizations(df_results, df_stats, output_dir, timestamp)
        
        # 打印汇总统计
        print_summary_statistics(df_results, df_stats)
        
        print(f"\n分析完成！所有结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
