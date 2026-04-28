import numpy as np
import pandas as pd

def calculate_area_ci(mapped_areas, confusion_matrix_counts):
    """
    Calculates bias-corrected area estimates, standard errors (SE),
    and 95% confidence intervals (CI) based on Olofsson et al. (2013, 2014).

    Args:
        mapped_areas (np.ndarray): 1D array of mapped areas for each class (strata).
                                   Order must match the confusion matrix.
                                   [PF, NF, Others]
        confusion_matrix_counts (np.ndarray): 2D array (NxN) of sample counts.
                                              Rows = Reference (True) Class
                                              Cols = Map (Predicted) Class
                                              [[n_PF_PF, n_PF_NF, n_PF_Oth],
                                               [n_NF_PF, n_NF_NF, n_NF_Oth],
                                               [n_Oth_PF, n_Oth_NF, n_Oth_Oth]]
    Returns:
        pd.DataFrame: A DataFrame with results for each class.
    """
    
    # Class labels, assuming order is PF (Class 1), NF (Class 2), Others (Class 3)
    labels = ['PF', 'NF', 'Others']
    num_classes = len(labels)
    
    # Total map area (A_total)
    A_total = np.sum(mapped_areas)
    
    # Strata weights (W_j) - Area proportion of each mapped class j (cols)
    W_j = mapped_areas / A_total
    
    # Column totals (n_j) - Total samples in each mapped class j
    n_j = np.sum(confusion_matrix_counts, axis=0) # axis=0 sums columns
    
    # Check for strata with zero samples
    if np.any(n_j == 0):
        print(f"Warning: A mapped stratum has 0 samples. Results may be invalid.")
        n_j[n_j == 0] = 1e-6 
        
    # --- Step 1 & 2: Calculate Bias-Corrected Area (A_hat_k) ---
    
    # p_ij = W_j * (n_ij / n_j)
    # Estimate proportion of the map for cell (i, j)
    # We use broadcasting: confusion_matrix_counts (3x3) / n_j (1x3)
    p_ij = W_j * (confusion_matrix_counts / n_j)
    
    # p_k_hat = Estimated area proportion for *reference* class k (summing rows)
    p_k_hat = np.sum(p_ij, axis=1) # axis=1 sums rows
    
    # A_k_hat = Bias-corrected area estimate for reference class k
    A_k_hat = p_k_hat * A_total
    
    # --- Step 3: Calculate Standard Error (SE) ---
    
    # Variance of p_k_hat (Olofsson 2014, Eq. 6)
    # V(p_k_hat) = sum_j [ W_j^2 * ( (n_kj/n_j)*(1 - n_kj/n_j) / (n_j - 1) ) ]
    
    # n_kj_over_n_j = n_kj / n_j
    # This is n_ij / n_j, but we name it n_kj_... to match the formula's perspective
    n_kj_over_n_j = confusion_matrix_counts / n_j
    
    # Handle (n_j - 1) == 0 case
    n_j_minus_1 = n_j - 1
    n_j_minus_1[n_j_minus_1 <= 0] = 1e-6 # Avoid division by zero if n_j was 1
    
    variance_terms = (W_j**2) * ( (n_kj_over_n_j * (1 - n_kj_over_n_j)) / (n_j_minus_1) )
    
    # Sum across columns (j) to get variance for each row (k)
    V_p_k_hat = np.sum(variance_terms, axis=1)
    
    # SE(p_k_hat)
    SE_p_k_hat = np.sqrt(V_p_k_hat)
    
    # SE(A_k_hat) - Standard Error of the Area
    SE_A_k_hat = SE_p_k_hat * A_total
    
    # --- Step 4: Calculate 95% Confidence Interval ---
    z_score = 1.96 # For 95% CI
    
    CI_half_width = z_score * SE_A_k_hat
    CI_lower = A_k_hat - CI_half_width
    CI_upper = A_k_hat + CI_half_width
    
    # --- Compile Results ---
    results = pd.DataFrame({
        'Class': labels,
        'Mapped_Area': mapped_areas,
        'Estimated_Area': A_k_hat,
        'SE_Area': SE_A_k_hat,
        'CI_95_Lower': CI_lower,
        'CI_95_Upper': CI_upper,
        'CI_Half_Width': CI_half_width
    })
    
    return results

# ==============================================================================
# 1. 你的“图上面积”数据 (MAPPED AREAS)
# 顺序: (PF, NF, OTHERS)
# ==============================================================================
mapped_areas_by_year = {
    2017: np.array([770983.1472,2359564.001,1330541.532]),
    2018: np.array([764152.5005,2343330.311,1353605.868]),
    2019: np.array([772920.2498,2321424.98,1366748.342]),
    2020: np.array([786113.5851,2307003.367,1367971.728]),
    2021: np.array([807968.3767,2290776.843,1362343.461]),
    2022: np.array([810245.2014,2274470.926,1376377.445]),
    2023: np.array([817405.1732,2250924.169,1392764.23]),
    2024: np.array([827414.733,2244267.396,1389411.444])
}

# ==============================================================================
# 2. 你的混淆矩阵 (CONFUSION MATRICES - SAMPLE COUNTS)
# 格式: np.array([[n_PF_PF, n_PF_NF, n_PF_Others],   <- (Reference = PF, actual_1)
#                  [n_NF_PF, n_NF_NF, n_NF_Others],   <- (Reference = NF, actual_2)
#                  [n_Oth_PF, n_Oth_NF, n_Oth_Others]]) <- (Reference = Others, actual_3)
#
# 列 (Columns) 对应 Pred_1 (PF), Pred_2 (NF), Pred_3 (Others)
# ==============================================================================
matrices_by_year = {
    2017: np.array([[26324,842,168],
                   [508,26756,71],
                   [260,71,26371]]),

    2018: np.array([[26276,712,144],
                   [500,26649,53],
                   [74,23,26574]]),

    2019: np.array([[26326,609,151],
                   [512,26602,26],
                   [74, 22, 26575]]),

    2020: np.array([[26365,558,135],
                   [622,26506,34],
                   [82,25,26515]]),

    2021: np.array([[26414,503,119],
                   [742,26473,33],
                   [82,38,26455]]),

    2022: np.array([[26523,510,139],
                   [775,26410,51],
                   [79,29,26548]]),

    2023: np.array([[26572,472,158],
                   [900,26252,42],
                   [81,28,26556]]),

    2024: np.array([[26474,474,157],
                   [988,26102,84],
                   [217, 85, 26272]])
}
# --- 运行计算 ---
all_results = []

print("--- 面积估算与不确定性计算 (Olofsson et al. 2014) ---")

# 确保按年份顺序处理
for year in sorted(mapped_areas_by_year.keys()):
    print(f"\nProcessing Year: {year}")
    
    mapped_areas = mapped_areas_by_year[year]
    conf_matrix = matrices_by_year[year]
    
    # 执行计算
    results_df = calculate_area_ci(mapped_areas, conf_matrix)
    results_df['Year'] = year
    
    # 打印当年的详细结果
    print(results_df.to_string(index=False, float_format="%.4f"))
    all_results.append(results_df)

# --- 汇总最终数据 ---
final_table = pd.concat(all_results)
final_table = final_table.set_index(['Year', 'Class'])

# --- 格式化输出最终绘图数据 ---
print("\n\n" + "="*40)
print(" 最终绘图数据: PF (人工林)")
print(" (Estimated_Area 作为中心值, 误差棒半宽为 CI_Half_Width)")
print("="*40)
pf_data = final_table.xs('PF', level='Class')[['Estimated_Area', 'CI_95_Lower', 'CI_95_Upper', 'SE_Area', 'CI_Half_Width']]
print(pf_data.to_string(float_format="%.4f"))

print("\n\n" + "="*40)
print(" 最终绘图数据: NF (自然林)")
print(" (Estimated_Area 作为中心值, 误差棒半宽为 CI_Half_Width)")
print("="*40)
nf_data = final_table.xs('NF', level='Class')[['Estimated_Area', 'CI_95_Lower', 'CI_95_Upper', 'SE_Area', 'CI_Half_Width']]
print(nf_data.to_string(float_format="%.4f"))