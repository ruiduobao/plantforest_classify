#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查像素数据和面积计算
"""

import pandas as pd

# 读取测试结果
df = pd.read_excel('测试_等面积投影统计结果_20251103_183657.xlsx')

print("像素数据检查:")
print(df[['Zone', 'Year', 'Total_Pixels', 'Plantation_Pixels', 'Natural_Pixels', 'Other_Pixels']].head(3))

print("\n计算验证:")
for i in range(min(3, len(df))):
    row = df.iloc[i]
    calculated_total = row['Plantation_Pixels'] + row['Natural_Pixels'] + row['Other_Pixels']
    print(f"\n第{i+1}行 (Zone {row['Zone']}, Year {row['Year']}):")
    print(f"  人工林像素: {row['Plantation_Pixels']:,}")
    print(f"  自然林像素: {row['Natural_Pixels']:,}")
    print(f"  其他像素: {row['Other_Pixels']:,}")
    print(f"  各类像素之和: {calculated_total:,}")
    print(f"  记录的总像素: {row['Total_Pixels']:,}")
    print(f"  差异: {calculated_total - row['Total_Pixels']:,}")
    print(f"  总面积(公顷): {row['Total_Area_Ha']:.2f}")
    print(f"  按总像素计算面积: {row['Total_Pixels'] * 100 / 10000:.2f}")

print("\n负数面积检查:")
negative_rows = df[df['Total_Area_Ha'] < 0]
print(f"负数面积行数: {len(negative_rows)}")
if len(negative_rows) > 0:
    print(negative_rows[['Zone', 'Year', 'Total_Area_Ha', 'Total_Pixels']])