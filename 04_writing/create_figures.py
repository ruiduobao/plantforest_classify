import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体和样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 模拟数据生成 ---
# 时间轴 (2014 - 2020)
t_model = np.linspace(2014, 2020, 300)
np.random.seed(42)
t_obs = np.sort(np.random.uniform(2014.1, 2019.9, 80)) # 随机观测时间点

# 谐波模型函数 (Harmonic Model): y = a + b*t + c*cos(2pi*t) + d*sin(2pi*t)
def harmonic(t, intercept, slope, cos_amp, sin_amp):
    return intercept + slope * (t - 2014) + cos_amp * np.cos(2 * np.pi * t) + sin_amp * np.sin(2 * np.pi * t)

# 图A数据：稳定自然林 (高NDVI, 稳定季节波动)
y_model_A = harmonic(t_model, 0.75, 0.01, 0.1, 0.05)
y_obs_A = harmonic(t_obs, 0.75, 0.01, 0.1, 0.05) + np.random.normal(0, 0.04, len(t_obs))

# 图B数据：突变人工林 (2017.5发生砍伐)
t_break = 2017.5
mask_before = t_model <= t_break
mask_after = t_model > t_break

y_model_B_before = harmonic(t_model[mask_before], 0.75, 0.01, 0.1, 0.05)
y_model_B_after = harmonic(t_model[mask_after], 0.35, 0.03, 0.05, 0.02) # 砍伐后NDVI骤降，随后有微弱的杂草恢复

obs_before = t_obs <= t_break
y_obs_B_before = harmonic(t_obs[obs_before], 0.75, 0.01, 0.1, 0.05) + np.random.normal(0, 0.04, sum(obs_before))
y_obs_B_after = harmonic(t_obs[~obs_before], 0.35, 0.03, 0.05, 0.02) + np.random.normal(0, 0.05, sum(~obs_before))

# --- 2. 绘制图表 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 图A: 稳定样本
ax1.scatter(t_obs, y_obs_A, color='#81C784', edgecolors='white', s=50, label='Observations (观测值)')
ax1.plot(t_model, y_model_A, color='#2E7D32', linewidth=2.5, label='Harmonic Model (拟合线)')
ax1.set_title('图A: 稳定样本 (Stable Sample) — Retained ✅', fontsize=14, fontweight='bold')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Spectral Index (e.g., NDVI / SWIR1)', fontsize=12)
ax1.set_xlim(2014, 2020)
ax1.set_ylim(0.1, 1.0)
ax1.legend(loc='lower left')

# 图B: 突变样本
# 前半段
ax2.scatter(t_obs[obs_before], y_obs_B_before, color='#81C784', edgecolors='white', s=50, label='Pre-break Obs')
ax2.plot(t_model[mask_before], y_model_B_before, color='#2E7D32', linewidth=2.5, label='Pre-break Model')
# 后半段
ax2.scatter(t_obs[~obs_before], y_obs_B_after, color='#9E9E9E', edgecolors='black', s=50, alpha=0.8, label='Post-break Obs (异常)')
ax2.plot(t_model[mask_after], y_model_B_after, color='#F57C00', linewidth=2.5, linestyle='-', label='Post-break Model')
# 断点 tBreak
ax2.axvline(x=t_break, color='#D32F2F', linestyle='--', linewidth=2, zorder=0)
ax2.text(t_break + 0.1, 0.85, 'tBreak\)', color='#D32F2F', fontsize=12, fontweight='bold')

ax2.set_title('图B: 突变样本 (Abrupt Change) — Masked ❌', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Spectral Index (e.g., NDVI / SWIR1)', fontsize=12)
ax2.set_xlim(2014, 2020)
ax2.set_ylim(0.1, 1.0)
ax2.legend(loc='lower left')

plt.tight_layout()
# plt.savefig('CCDC_Comparison.pdf', dpi=300) # 取消注释以导出高清矢量图
plt.show()