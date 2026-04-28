import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 1. 数据准备 (单位: 万平方公里 / Million Hectares)
# FAO Data (Source: FRA 2020 Country Reports, 2020 value)
# User Data (Source: Provided Table, 2020 value)
data = {
    'Country': ['Indonesia', 'Myanmar', 'Malaysia', 'Thailand', 'Laos', 'Philippines', 'Vietnam', 'Cambodia', 'Timor-Leste', 'Brunei', 'Singapore'],
    # User Data (2020)
    'User_NF': [113.49, 34.77, 18.10, 19.33, 15.54, 12.56, 9.93, 5.81, 0.70, 0.47, 0.01],
    'User_PF': [36.26, 7.66, 11.09, 3.85, 2.06, 7.05, 9.67, 0.86, 0.07, 0.04, 0.00],
    # FAO Data (2020) - Approximate based on FRA reports (in Million ha)
    'FAO_NF': [87.61, 28.12, 17.42, 16.34, 14.82, 6.81, 10.29, 8.38, 0.92, 0.37, 0.016],
    'FAO_PF': [4.53, 0.43, 1.70, 3.54, 1.77, 0.38, 4.35, 0.13, 0.00, 0.005, 0.00]
}

df = pd.DataFrame(data)

# 2. 绘图设置
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# --- 子图 A: 自然林 (Natural Forests) ---
ax1 = axes[0]
x_nf = df['FAO_NF']
y_nf = df['User_NF']

# 绘制散点
ax1.scatter(x_nf, y_nf, alpha=0.7, c='#2ca02c', s=80, edgecolors='k', label='Countries')

# 绘制 1:1 线
lims_nf = [0, max(max(x_nf), max(y_nf)) * 1.1]
ax1.plot(lims_nf, lims_nf, 'k--', alpha=0.5, label='1:1 Line')

# 计算统计量
r2_nf = r2_score(x_nf, y_nf)
rmse_nf = np.sqrt(mean_squared_error(x_nf, y_nf))

# 添加文本和标签
ax1.text(0.25, 0.95, f'$R^2 = {r2_nf:.2f}$\n$RMSE = {rmse_nf:.2f}$', transform=ax1.transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.set_title('(a) Natural Forests Area Comparison (Mha)', fontsize=14, fontweight='bold')
ax1.set_xlabel('FAO FRA 2020 (Million ha)', fontsize=12)
ax1.set_ylabel('This Dataset (Million ha)', fontsize=12)
ax1.set_xlim(lims_nf)
ax1.set_ylim(lims_nf)

# --- 子图 B: 人工林 (Plantation Forests) ---
ax2 = axes[1]
x_pf = df['FAO_PF']
y_pf = df['User_PF']

# 绘制散点
ax2.scatter(x_pf, y_pf, alpha=0.7, c='#d62728', s=80, edgecolors='k', label='Countries')

# 绘制 1:1 线
lims_pf = [0, max(max(x_pf), max(y_pf)) * 1.1]
ax2.plot(lims_pf, lims_pf, 'k--', alpha=0.5, label='1:1 Line')

# 计算统计量
r2_pf = r2_score(x_pf, y_pf)
rmse_pf = np.sqrt(mean_squared_error(x_pf, y_pf))

# 添加文本
ax2.text(0.25, 0.95, f'$R^2 = {r2_pf:.2f}$\n$RMSE = {rmse_pf:.2f}$', transform=ax2.transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.set_title('(b) Plantation Forests Area Comparison (Mha)', fontsize=14, fontweight='bold')
ax2.set_xlabel('FAO FRA 2020 (Million ha)', fontsize=12)
ax2.set_ylabel('This Dataset (Million ha)', fontsize=12)
ax2.set_xlim(lims_pf)
ax2.set_ylim(lims_pf)

# 标注显著偏离的国家 (例如 Indonesia, Malaysia, Vietnam)
offset = 1
for i, txt in enumerate(df['Country']):
    # 只标注面积较大的几个国家以避免拥挤
    if df['User_PF'][i] > 5 or df['User_NF'][i] > 20:
        ax1.annotate(txt, (df['FAO_NF'][i], df['User_NF'][i]), xytext=(5, -5), textcoords='offset points', fontsize=9)
        ax2.annotate(txt, (df['FAO_PF'][i], df['User_PF'][i]), xytext=(5, -5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.show()