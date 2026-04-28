import pandas as pd
import os

# --- 1. 文件路径设置 ---
# ！！重要！！
# 使用 'r' 字符串前缀来正确处理 Windows 路径中的反斜杠 '\'
file_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\3.GEE导出结果_结果合并_马尔可夫模型\统计面积\按类型统计2017-2024.csv"

# 定义要汇总的列
columns_to_sum = [
    'Plantation_Forest_Ha', 
    'Natural_Forest_Ha', 
    'Other_Ha'
]

# --- 2. 读取和处理数据 ---
try:
    # 使用 'utf-8-sig' 编码来读取可能由 GEE 或 Excel 导出的、
    # 带有 BOM (Byte Order Mark) 的 CSV 文件。
    # 如果此行出错，请尝试 encoding='gbk'
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    print(f"成功读取文件: {file_path}")
    print("-" * 30)

    # --- 3. 按年份分组并求和 ---
    # 按 'Year' 列分组，并对我们感兴趣的列应用 .sum()
    yearly_totals = df.groupby('Year')[columns_to_sum].sum()

    # --- 4. (可选) 计算 Km2 (平方公里) 以便阅读 ---
    # 1 平方公里 = 100 公顷
    yearly_totals['Plantation_Forest_Km2'] = yearly_totals['Plantation_Forest_Ha'] / 100
    yearly_totals['Natural_Forest_Km2'] = yearly_totals['Natural_Forest_Ha'] / 100
    yearly_totals['Other_Km2'] = yearly_totals['Other_Ha'] / 100

    # 调整列顺序，使 Ha 和 Km2 在一起
    output_columns = [
        'Plantation_Forest_Ha', 'Plantation_Forest_Km2',
        'Natural_Forest_Ha', 'Natural_Forest_Km2',
        'Other_Ha', 'Other_Km2'
    ]
    yearly_totals = yearly_totals[output_columns]

    # --- 5. 打印最终结果 ---
    print("2017-2024年 逐年总面积统计 (所有Zone合并)")
    
    # 使用 to_string 并设置浮点数格式，使其对齐
    print(yearly_totals.to_string(float_format="%.2f"))


    # --- 6. (可选) 保存结果到新的 CSV 文件 ---
    
    # 获取原始文件所在的目录
    output_directory = os.path.dirname(file_path)
    output_filename = os.path.join(output_directory, "年度总计_2017-2024_汇总.csv")
    
    try:
        yearly_totals.to_csv(output_filename, encoding='utf-8-sig', float_format='%.2f')
        print("-" * 30)
        print(f"\n统计结果已保存到: {output_filename}")
    except Exception as e:
        print(f"\n保存文件失败: {e}")


except FileNotFoundError:
    print(f"错误：文件未找到。")
    print(f"请确保CSV文件存在于以下路径: \n{file_path}")
except Exception as e:
    print(f"读取或处理文件时发生错误: {e}")
    print("如果提示编码错误(EncodingError)，请尝试将 'utf-8-sig' 更改为 'gbk'。")