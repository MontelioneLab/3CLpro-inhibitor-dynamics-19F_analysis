#!/usr/bin/env python3
# ============================================================
# 07a_run_activation_grid_scan.py
# Dual-field DG‡ profile scan engine
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import least_squares

# ============================================================
# 1. SETUP & PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_600 = os.path.join(PROJECT_ROOT, "processed_ascii", "nirmat_VT_LB10_600")
DATA_800 = os.path.join(PROJECT_ROOT, "processed_ascii", "nirmat_VT_LB10_800")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target evaluation temperatures
temps_all = np.array([285, 288, 291, 293, 296, 298, 303, 308, 313, 315])

# ============================================================
# 2. FIXED THERMODYNAMIC & LINEARITY BASELINES
# ============================================================
DH = 1.55      
DS = 4.08      
R_kcal = 0.001987207

FREQ_600 = 564.5881492
FREQ_800 = 752.8281062
FIELD_RATIO = FREQ_800 / FREQ_600

dnu0_600 = 134.75
dnu_slope = 0.349

LB = 10.0
slopeA_600 = -1.521
slopeB_600 = -0.976
slopeA_800 = -3.526
slopeB_800 = -2.088

def pB_vanthoff(T):
    dG = DH - T * (DS / 1000.0)
    K = np.exp(-dG / (R_kcal * T))
    return K / (1.0 + K)

# ============================================================
# EYRING RATE CALCULATIONS
# ============================================================
kB = 1.380649e-23
h  = 6.62607015e-34

def kex_eyring(T, DG298, DHdd):
    DSdd = (DHdd - DG298) / 298.0
    DGdd_T = DHdd - T * DSdd
    return (kB * T / h) * np.exp(-DGdd_T / (R_kcal * T))

def linewidth_T(T, lw0, slope):
    return lw0 + slope * (T - 298.0)

def dnu600_linear(T):
    return dnu0_600 + dnu_slope * (T - 298.0)

def dnu800_linear(T):
    return (dnu0_600 * FIELD_RATIO) + dnu_slope * (T - 298.0)

# ============================================================
# 3. SAFE MULTI-FIELD DATA ACQUISITION
# ============================================================
def load_data(directory, temps):
    """
    FIXED: Targets 'spec_XK_LB10.dat' and securely averages in replicates.
    """
    dataset = []
    for T in temps:
        path_std = os.path.join(directory, f"spec_{T}K_LB10.dat")
        path_rep = os.path.join(directory, f"spec_{T}K_rep1_LB10.dat")
        
        valid_arrays = []
        if os.path.exists(path_std):
            valid_arrays.append(np.loadtxt(path_std))
        if os.path.exists(path_rep):
            valid_arrays.append(np.loadtxt(path_rep))
            
        if len(valid_arrays) == 0:
            continue
            
        x_axis = valid_arrays[0][:, 0]
        y_matrix = np.column_stack([arr[:, 1] for arr in valid_arrays])
        y_avg = np.mean(y_matrix, axis=1)
        
        mask = (x_axis > -900.0) & (x_axis < 900.0)
        dataset.append((T, x_axis[mask], y_avg[mask]))
    return dataset

data600 = load_data(DATA_600, temps_all)
data800 = load_data(DATA_800, temps_all)

# ============================================================
# 4. QUANTUM DNMR SIMULATOR
# ============================================================
def dnmr(f, kex, center, dnu, r2a, r2b, pB):
    pi = np.pi
    pA = 1.0 - pB
    wa = 2 * pi * center
    wb = 2 * pi * (center + dnu)
    w  = 2 * pi * f

    ka = kex * pB
    kb = kex * pA

    Ga = r2a * pi + 1j * (wa - w) + ka
    Gb = r2b * pi + 1j * (wb - w) + kb

    num = pA * (Gb + ka) + pB * (Ga + kb)
    den = Ga * Gb - ka * kb
    return np.real(num / den)

# ============================================================
# 5. SCAN TARGET CONFIGURATION RANGE
# ============================================================
DG_scan = np.arange(14.5, 15.8, 0.01)
results = []

for DG_FIXED in DG_scan:
    
    def objective(x):
        amp600, amp800 = x[0], x[1]
        LWA0_600, LWB0_600 = x[2], x[3]
        LWA0_800, LWB0_800 = x[4], x[5]
        DHdd = x[6]

        resid = []

        # --- Evaluate 600 MHz ---
        for T, f, y in data600:
            pB = pB_vanthoff(T)
            dnu = -dnu600_linear(T)
            kex = kex_eyring(T, DG_FIXED, DHdd)
            LWA = linewidth_T(T, LWA0_600, slopeA_600)
            LWB = linewidth_T(T, LWB0_600, slopeB_600)

            sim = dnmr(f, kex, 0.0, dnu, LWA + LB, LWB + LB, pB) * amp600
            resid.extend((sim - y) / np.max(y))

        # --- Evaluate 800 MHz ---
        for T, f, y in data800:
            pB = pB_vanthoff(T)
            dnu800 = -dnu800_linear(T)
            kex = kex_eyring(T, DG_FIXED, DHdd)
            LWA = linewidth_T(T, LWA0_800, slopeA_800)
            LWB = linewidth_T(T, LWB0_800, slopeB_800)

            sim = dnmr(f, kex, 0.0, dnu800, LWA + LB, LWB + LB, pB) * amp800
            resid.extend((sim - y) / np.max(y))

        # --- EXSY Restraints ---
        EXSY_T = np.array([288.0, 298.0, 308.0])
        EXSY_k = np.array([25.12, 48.07, 130.00])
        EXSY_sigma = np.array([(31.73 - 19.76)/4.0, (60.66 - 37.17)/4.0, (150.68 - 111.78)/4.0])

        for T_exsy, k_target, sigma in zip(EXSY_T, EXSY_k, EXSY_sigma):
            k_model = kex_eyring(T_exsy, DG_FIXED, DHdd)
            resid.append((k_model - k_target) / sigma)

        return np.array(resid)

    # Initial Guesses & Boundary Conditions
    x0 = np.array([2e11, 2e11, 64.0, 73.0, 119.0, 141.0, 15.0])
    lower = np.array([1e8, 1e8, 40.0, 40.0, 80.0, 80.0, 0.0])
    upper = np.array([1e13, 1e13, 120.0, 120.0, 180.0, 180.0, 35.0])

    result = least_squares(objective, x0, bounds=(lower, upper), max_nfev=2000)
    xfit = result.x
    RSS = np.sum(result.fun**2)

    LWA0_600, LWB0_600 = xfit[2], xfit[3]
    LWA0_800, LWB0_800 = xfit[4], xfit[5]
    DHdd = xfit[6]
    DSdd = (DHdd - DG_FIXED) / 298.0

    kex_values = [kex_eyring(T, DG_FIXED, DHdd) for T in temps_all]
    pB_values = [pB_vanthoff(T) for T in temps_all]
    dnu600_values = [dnu600_linear(T) for T in temps_all]
    dnu800_values = [dnu800_linear(T) for T in temps_all]
    
    k298 = kex_values[5]

    results.append({
        "DG298": DG_FIXED, "DHdd": DHdd, "DSdd_cal": DSdd * 1000.0, "RSS": RSS,
        "kex": kex_values, "k298": k298, "pB": pB_values,
        "dnu600": dnu600_values, "dnu800": dnu800_values,
        "LWA0_600": LWA0_600, "LWB0_600": LWB0_600,
        "LWA0_800": LWA0_800, "LWB0_800": LWB0_800
    })
    print(f"Locked Scan Point -> DG‡ = {DG_FIXED:.2f} | RSS = {RSS:.2f} | k298 = {k298:.1f} s^-1")

# ============================================================
# 6. EXPORT COMPILED TABULATIONS
# ============================================================
outfile_pkl = os.path.join(OUTPUT_DIR, "07a_DGscan_results.pkl")
with open(outfile_pkl, "wb") as f:
    pickle.dump(results, f)

rows = []
for r in results:
    row = {
        "DG298": r["DG298"], "DHdd": r["DHdd"], "DSdd_cal": r["DSdd_cal"],
        "RSS": r["RSS"], "k298": r["k298"],
        "LWA0_600": r["LWA0_600"], "LWB0_600": r["LWB0_600"],
        "LWA0_800": r["LWA0_800"], "LWB0_800": r["LWB0_800"]
    }
    for i, T in enumerate(temps_all):
        row[f"kex_{T}K"] = r["kex"][i]
        row[f"pB_{T}K"] = r["pB"][i]
    rows.append(row)

df_out = pd.DataFrame(rows)
csv_out = os.path.join(OUTPUT_DIR, "07a_DGscan_results.csv")
df_out.to_csv(csv_out, index=False)

# Comprehensive confidence boundary evaluations
arr_RSS, arr_DG, arr_DH, arr_DS, arr_k298 = df_out["RSS"].values, df_out["DG298"].values, df_out["DHdd"].values, df_out["DSdd_cal"].values, df_out["k298"].values
idx_min = np.argmin(arr_RSS)
RSS_min = arr_RSS[idx_min]

thresh_68, thresh_95 = 1.08, 1.25
mask_68, mask_95 = arr_RSS <= (RSS_min * thresh_68), arr_RSS <= (RSS_min * thresh_95)

txt_out = os.path.join(OUTPUT_DIR, "07a_DGscan_output_summary.txt")
summary_text = f"""============================================================
DUAL-FIELD ACTIVATION INFRASTRUCTURE GRID SCAN SUMMARY
============================================================
Minimum RSS (Global Optimum Value): {RSS_min:.4f}
Best-fit Coordinated Solution Vectors:
  - DG‡ (298 K) : {arr_DG[idx_min]:.2f} kcal/mol
  - DH‡        : {arr_DH[idx_min]:.2f} kcal/mol
  - DS‡        : {arr_DS[idx_min]:.2f} cal/mol/K
  - kex (298 K) : {arr_k298[idx_min]:.1f} s^-1

68% Confidence Interval Profile Limits (RSS <= {RSS_min * thresh_68:.3f}):
  - DG‡ (298 K) : [{arr_DG[mask_68].min():.2f} to {arr_DG[mask_68].max():.2f}] kcal/mol
  - kex (298 K) : [{arr_k298[mask_68].min():.1f} to {arr_k298[mask_68].max():.1f}] s^-1

95% Confidence Interval Profile Limits (RSS <= {RSS_min * thresh_95:.3f}):
  - DG‡ (298 K) : [{arr_DG[mask_95].min():.2f} to {arr_DG[mask_95].max():.2f}] kcal/mol
  - kex (298 K) : [{arr_k298[mask_95].min():.1f} to {arr_k298[mask_95].max():.1f}] s^-1
============================================================
"""
print("\n" + summary_text)
with open(txt_out, "w") as f:
    f.write(summary_text)
