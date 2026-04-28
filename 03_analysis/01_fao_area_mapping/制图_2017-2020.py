import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from io import StringIO

# Data provided by the user
csv_data = """Country,Year,Natural_FAO,Natural_Raster,Planted_FAO,Planted_Raster,Total_FAO
Brunei,2017,374.80,472.57,5.20,40.98,380.00
Brunei,2018,374.80,472.75,5.20,39.80,380.00
Brunei,2019,374.80,471.81,5.20,40.48,380.00
Brunei,2020,374.80,471.11,5.20,40.48,380.00
Cambodia,2017,8585.00,6372.87,61.00,923.15,8646.00
Cambodia,2018,8516.00,6138.43,83.00,821.67,8599.00
Cambodia,2019,8447.00,5991.40,105.00,817.24,8552.00
Cambodia,2020,8378.00,5792.14,126.00,869.97,8504.00
Indonesia,2017,89289.50,114463.00,4660.30,36343.40,93949.80
Indonesia,2018,88728.80,114359.00,4615.43,35486.20,93344.30
Indonesia,2019,88168.20,113668.00,4570.57,35644.30,92738.70
Indonesia,2020,87607.50,113367.00,4525.70,36357.40,92133.20
Laos,2017,14980.40,16049.00,1718.60,2205.73,16699.00
Laos,2018,14928.40,15946.90,1736.15,2233.94,16664.50
Laos,2019,14876.30,15707.50,1753.70,2177.88,16630.00
Laos,2020,14824.20,15503.60,1771.25,2088.10,16595.50
Malaysia,2017,17543.60,18404.10,1720.88,10954.10,19264.50
Malaysia,2018,17501.40,18360.70,1712.96,10957.00,19214.30
Malaysia,2019,17459.20,18195.90,1705.04,11068.70,19164.20
Malaysia,2020,17416.90,18109.60,1697.12,11087.60,19114.00
Myanmar,2017,28986.30,35318.70,427.09,7535.23,29413.00
Myanmar,2018,28696.70,35274.60,427.09,7485.83,29123.30
Myanmar,2019,28407.20,34990.30,427.09,7588.04,28833.60
Myanmar,2020,28117.60,34745.70,427.09,7673.60,28543.90
Philippines,2017,6712.40,12716.10,371.53,7147.46,7083.93
Philippines,2018,6744.29,12686.50,374.53,6911.25,7118.82
Philippines,2019,6776.18,12578.30,377.52,6982.58,7153.70
Philippines,2020,6808.07,12544.10,380.52,7059.16,7188.59
Singapore,2017,16.11,13.45,0.00,0.98,16.11
Singapore,2018,15.93,13.33,0.00,0.83,15.93
Singapore,2019,15.75,13.00,0.00,0.82,15.75
Singapore,2020,15.57,13.12,0.00,0.84,15.57
Thailand,2017,16345.00,19333.90,3636.00,3873.02,19981.00
Thailand,2018,16342.00,19307.30,3603.00,3863.71,19945.00
Thailand,2019,16339.00,19254.80,3570.00,3829.06,19909.00
Thailand,2020,16336.00,19267.70,3537.00,3905.92,19873.00
Timor-Leste,2017,925.30,720.07,0.00,86.73,925.60
Timor-Leste,2018,923.90,721.07,0.00,95.14,923.90
Timor-Leste,2019,922.50,713.49,0.00,97.22,922.50
Timor-Leste,2020,921.10,695.94,0.00,74.28,921.10
Vietnam,2017,10236.40,10133.90,4178.97,9941.76,14415.40
Vietnam,2018,10255.50,10072.30,4235.77,9508.19,14491.30
Vietnam,2019,10274.60,9962.00,4292.57,9625.18,14567.20
Vietnam,2020,10293.70,9916.91,4349.37,9680.41,14643.10"""

# Create DataFrame
df = pd.read_csv(StringIO(csv_data))

# Convert units from 1000 ha to Million ha
df['Natural_FAO_Mha'] = df['Natural_FAO'] / 1000
df['Natural_Raster_Mha'] = df['Natural_Raster'] / 1000
df['Planted_FAO_Mha'] = df['Planted_FAO'] / 1000
df['Planted_Raster_Mha'] = df['Planted_Raster'] / 1000

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# --- Subplot A: Natural Forests ---
ax1 = axes[0]
x_nf = df['Natural_FAO_Mha']
y_nf = df['Natural_Raster_Mha']

# Scatter plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from io import StringIO

# Data provided by the user
csv_data = """Country,Year,Natural_FAO,Natural_Raster,Planted_FAO,Planted_Raster,Total_FAO
Brunei,2017,374.80,472.57,5.20,40.98,380.00
Brunei,2018,374.80,472.75,5.20,39.80,380.00
Brunei,2019,374.80,471.81,5.20,40.48,380.00
Brunei,2020,374.80,471.11,5.20,40.48,380.00
Cambodia,2017,8585.00,6372.87,61.00,923.15,8646.00
Cambodia,2018,8516.00,6138.43,83.00,821.67,8599.00
Cambodia,2019,8447.00,5991.40,105.00,817.24,8552.00
Cambodia,2020,8378.00,5792.14,126.00,869.97,8504.00
Indonesia,2017,89289.50,114463.00,4660.30,36343.40,93949.80
Indonesia,2018,88728.80,114359.00,4615.43,35486.20,93344.30
Indonesia,2019,88168.20,113668.00,4570.57,35644.30,92738.70
Indonesia,2020,87607.50,113367.00,4525.70,36357.40,92133.20
Laos,2017,14980.40,16049.00,1718.60,2205.73,16699.00
Laos,2018,14928.40,15946.90,1736.15,2233.94,16664.50
Laos,2019,14876.30,15707.50,1753.70,2177.88,16630.00
Laos,2020,14824.20,15503.60,1771.25,2088.10,16595.50
Malaysia,2017,17543.60,18404.10,1720.88,10954.10,19264.50
Malaysia,2018,17501.40,18360.70,1712.96,10957.00,19214.30
Malaysia,2019,17459.20,18195.90,1705.04,11068.70,19164.20
Malaysia,2020,17416.90,18109.60,1697.12,11087.60,19114.00
Myanmar,2017,28986.30,35318.70,427.09,7535.23,29413.00
Myanmar,2018,28696.70,35274.60,427.09,7485.83,29123.30
Myanmar,2019,28407.20,34990.30,427.09,7588.04,28833.60
Myanmar,2020,28117.60,34745.70,427.09,7673.60,28543.90
Philippines,2017,6712.40,12716.10,371.53,7147.46,7083.93
Philippines,2018,6744.29,12686.50,374.53,6911.25,7118.82
Philippines,2019,6776.18,12578.30,377.52,6982.58,7153.70
Philippines,2020,6808.07,12544.10,380.52,7059.16,7188.59
Singapore,2017,16.11,13.45,0.00,0.98,16.11
Singapore,2018,15.93,13.33,0.00,0.83,15.93
Singapore,2019,15.75,13.00,0.00,0.82,15.75
Singapore,2020,15.57,13.12,0.00,0.84,15.57
Thailand,2017,16345.00,19333.90,3636.00,3873.02,19981.00
Thailand,2018,16342.00,19307.30,3603.00,3863.71,19945.00
Thailand,2019,16339.00,19254.80,3570.00,3829.06,19909.00
Thailand,2020,16336.00,19267.70,3537.00,3905.92,19873.00
Timor-Leste,2017,925.30,720.07,0.00,86.73,925.60
Timor-Leste,2018,923.90,721.07,0.00,95.14,923.90
Timor-Leste,2019,922.50,713.49,0.00,97.22,922.50
Timor-Leste,2020,921.10,695.94,0.00,74.28,921.10
Vietnam,2017,10236.40,10133.90,4178.97,9941.76,14415.40
Vietnam,2018,10255.50,10072.30,4235.77,9508.19,14491.30
Vietnam,2019,10274.60,9962.00,4292.57,9625.18,14567.20
Vietnam,2020,10293.70,9916.91,4349.37,9680.41,14643.10"""

# Create DataFrame
df = pd.read_csv(StringIO(csv_data))

# Convert units from 1000 ha to Million ha
df['Natural_FAO_Mha'] = df['Natural_FAO'] / 1000
df['Natural_Raster_Mha'] = df['Natural_Raster'] / 1000
df['Planted_FAO_Mha'] = df['Planted_FAO'] / 1000
df['Planted_Raster_Mha'] = df['Planted_Raster'] / 1000

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# --- Subplot A: Natural Forests ---
ax1 = axes[0]
x_nf = df['Natural_FAO_Mha']
y_nf = df['Natural_Raster_Mha']

# Scatter plot
ax1.scatter(x_nf, y_nf, alpha=0.7, c='#2ca02c', s=60, edgecolors='k', label='Countries (2017-2020)')

# 1:1 Line
lims_nf = [0, max(max(x_nf), max(y_nf)) * 1.1]
ax1.plot(lims_nf, lims_nf, 'k--', alpha=0.5, label='1:1 Line')

# Metrics
n = len(df)
r2_nf = r2_score(x_nf, y_nf)
rmse_nf = np.sqrt(mean_squared_error(x_nf, y_nf))

# Text box
ax1.text(0.05, 0.95, f'$R^2 = {r2_nf:.2f}$\n$RMSE = {rmse_nf:.2f}$\n$n = {n}$', transform=ax1.transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.set_title('(a) Natural Forests Area Comparison (Mha)', fontsize=14, fontweight='bold')
ax1.set_xlabel('FAO FRA 2020 (Million ha)', fontsize=12)
ax1.set_ylabel('This Dataset (Million ha)', fontsize=12)
ax1.set_xlim(lims_nf)
ax1.set_ylim(lims_nf)

# --- Subplot B: Plantation Forests ---
ax2 = axes[1]
x_pf = df['Planted_FAO_Mha']
y_pf = df['Planted_Raster_Mha']

# Scatter plot
ax2.scatter(x_pf, y_pf, alpha=0.7, c='#d62728', s=60, edgecolors='k', label='Countries (2017-2020)')

# 1:1 Line
lims_pf = [0, max(max(x_pf), max(y_pf)) * 1.1]
ax2.plot(lims_pf, lims_pf, 'k--', alpha=0.5, label='1:1 Line')

# Metrics
r2_pf = r2_score(x_pf, y_pf)
rmse_pf = np.sqrt(mean_squared_error(x_pf, y_pf))

# Text box
ax2.text(0.05, 0.95, f'$R^2 = {r2_pf:.2f}$\n$RMSE = {rmse_pf:.2f}$\n$n = {n}$', transform=ax2.transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.set_title('(b) Plantation Forests Area Comparison (Mha)', fontsize=14, fontweight='bold')
ax2.set_xlabel('FAO FRA 2020 (Million ha)', fontsize=12)
ax2.set_ylabel('This Dataset (Million ha)', fontsize=12)
ax2.set_xlim(lims_pf)
ax2.set_ylim(lims_pf)

# Labeling Outliers (Country + Year)
# Define threshold for labeling to avoid clutter
# Only label major discrepancies for visual clarity
# For NF: Indonesia, Myanmar, Malaysia, Thailand, Laos, Philippines usually have larger values.
# For PF: Indonesia, Malaysia, Vietnam, Thailand, Philippines.
# We iterate and label points that are significant.

for i, row in df.iterrows():
    country = row['Country']
    year = int(row['Year'])
    label_text = country
    
    # Natural Forest Labels
    # Logic: Label if absolute area > 10 Mha OR absolute difference is very large
    # To avoid overlapping text for the same country across 4 years, we can alternate positions or just label the start/end years if they are close.
    # Here, simply labeling big players for demonstration.
    if row['Natural_Raster_Mha'] > 20 or row['Natural_FAO_Mha'] > 20:
         # Only label 2020 for clarity if points are clustered, OR label all if distinct.
         # Let's label 2020 and maybe 2017 for outliers to show trend?
         # The user asked for "every year corresponding value", so points are all there. 
         # Labeling ALL 44 points might be too much text. 
         # I will label points with large discrepancies (> 5 Mha diff) or large absolute value (> 15 Mha).
         # Adjusting text position for specific countries to minimize overlap.
         
         offset_x = 5
         offset_y = -5
         if country == 'Indonesia':
             offset_x = -60
             offset_y = 10
         elif country == 'Myanmar':
             offset_x = 5
             offset_y = 5
         
         # To reduce clutter, only label 2020 or if the point is very distinct.
         # Since the prompt asks for "outlier marking", I will mark the most significant ones.
         # Let's mark 2020 for big countries to keep it readable, or all if space permits.
         # I will mark all points for major outlier countries but use small font.
         if country in ['Indonesia', 'Myanmar'] and year == 2020:
             ax1.annotate(label_text, (row['Natural_FAO_Mha'], row['Natural_Raster_Mha']), 
                          xytext=(offset_x, offset_y), textcoords='offset points', fontsize=7, alpha=0.8)

    # Plantation Forest Labels
    # Logic: Label if Raster > 5 Mha (Major plantations)
    if row['Planted_Raster_Mha'] > 4:
        offset_x = 5
        offset_y = -5
        if country == 'Indonesia':
            offset_x = -60
            offset_y = 10
        elif country == 'Malaysia':
            offset_x = -40
            offset_y = 5
        elif country == 'Vietnam':
             offset_x = 5
             offset_y = 10
        elif country == 'Philippines':
             offset_x = 5
             offset_y = -15
        
        # Labeling specific countries with large PF discrepancies
        if country in ['Indonesia', 'Malaysia', 'Vietnam', 'Philippines'] and year == 2020:
            ax2.annotate(label_text, (row['Planted_FAO_Mha'], row['Planted_Raster_Mha']), 
                         xytext=(offset_x, offset_y), textcoords='offset points', fontsize=7, alpha=0.8)

plt.tight_layout()
plt.savefig('scatter_comparison_2017_2020.png', dpi=300)
plt.show()