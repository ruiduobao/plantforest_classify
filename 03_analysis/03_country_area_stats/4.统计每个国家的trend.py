import pandas as pd
import matplotlib.pyplot as plt
import io
import os

# 1. 路径设置
# 使用 raw string (r"...") 以避免Windows路径中的反斜杠转义问题
output_dir = r"Z:\Mywork\论文\东南亚10m人工林提取\制图\3.逐年变化图\每个国家PFNF的趋势小图"

# 检查目录是否存在，不存在则创建（防止报错）
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"已自动创建目录: {output_dir}")
    except Exception as e:
        print(f"无法创建目录 (请检查Z盘连接或权限): {e}")
        exit()

# 2. 数据准备
data_str = """Country,Class,2017,2018,2019,2020,2021,2022,2023,2024
Vietnam,PF,9.79,9.43,9.58,9.67,9.81,9.68,9.75,9.90
Vietnam,NF,10.29,10.15,10.01,9.93,9.83,9.76,9.62,9.69
Timor-Leste,PF,0.08,0.09,0.10,0.07,0.08,0.08,0.12,0.13
Timor-Leste,NF,0.73,0.72,0.72,0.70,0.70,0.70,0.70,0.72
Thailand,PF,3.59,3.72,3.72,3.85,3.98,4.04,4.01,3.97
Thailand,NF,19.62,19.45,19.37,19.33,19.20,19.06,18.95,19.00
Singapore,PF,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
Singapore,NF,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01
Philippines,PF,7.01,6.84,6.94,7.05,7.19,7.18,7.26,7.31
Philippines,NF,12.85,12.76,12.62,12.56,12.51,12.47,12.36,12.29
Myanmar,PF,7.26,7.36,7.51,7.66,7.79,7.89,8.17,8.34
Myanmar,NF,35.60,35.40,35.07,34.77,34.47,34.24,34.01,34.12
Malaysia,PF,10.86,10.92,11.05,11.09,11.26,11.33,11.52,11.51
Malaysia,NF,18.49,18.39,18.22,18.10,17.94,17.80,17.62,17.53
Laos,PF,2.01,2.13,2.12,2.06,2.30,2.33,2.25,2.20
Laos,NF,16.24,16.05,15.77,15.54,15.33,15.06,14.69,14.46
Indonesia,PF,35.61,35.10,35.43,36.26,37.46,37.60,37.74,38.45
Indonesia,NF,115.20,114.74,113.89,113.49,113.04,112.43,111.30,110.77
Cambodia,PF,0.84,0.77,0.80,0.86,0.88,0.84,0.87,0.89
Cambodia,NF,6.45,6.19,6.01,5.81,5.57,5.44,5.37,5.37
Brunei,PF,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04
Brunei,NF,0.47,0.47,0.47,0.47,0.47,0.47,0.47,0.46"""

# 读取数据
df = pd.read_csv(io.StringIO(data_str))

# 数据清洗：填充合并单元格导致的空缺 Country 名
df['Country'] = df['Country'].ffill()

# 定义年份列
years = [str(y) for y in range(2017, 2025)]

# 3. 绘图循环
print(f"目标输出路径: {output_dir}")
print(f"开始生成 {len(df)} 张图片...\n")

for index, row in df.iterrows():
    country = row['Country']
    cls = row['Class']
    values = row[years].values
    
    # 创建画布：figsize=(3, 0.8) 适合迷你趋势图
    fig, ax = plt.subplots(figsize=(3, 0.8))
    
    # 绘制折线：黑色，线宽适中
    ax.plot(years, values, color='black', linewidth=1.5)
    
    # 移除所有坐标轴、刻度、边框
    ax.axis('off')
    
    # 构造文件名和完整路径
    filename = f"{country} - {cls}.png"
    filepath = os.path.join(output_dir, filename)
    
    # 保存图片
    # bbox_inches='tight', pad_inches=0 用于去除所有不必要的空白边距，只保留线条
    plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    
    print(f"已保存: {filename}")
    
    # 关闭当前图形，释放内存 (这是批量生成的关键)
    plt.close(fig)

print("\n处理完毕。")