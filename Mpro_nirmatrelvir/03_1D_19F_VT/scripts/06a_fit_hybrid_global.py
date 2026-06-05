#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# ============================================================
# 1. SETUP & PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_600 = os.path.join(PROJECT_ROOT, "processed_ascii", "nirmat_VT_LB10_600")
DATA_800 = os.path.join(PROJECT_ROOT, "processed_ascii", "nirmat_VT_LB10_800")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
OUTFILE_NPZ = os.path.join(OUTPUT_DIR, "06_vt_dual_fit_HYBRID.npz")

T6 = np.array([285, 288, 291, 293, 296, 298, 303, 308, 313, 315])

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
# EYRING ACTIVATION ENGINE
# ============================================================
kB = 1.380649e-23
h  = 6.62607015e-34

def kex_eyring(T, DG298, DHdd):
    DSdd = (DHdd - DG298) / 298.0
    DGdd_T = DHdd - T * DSdd
    return (kB * T / h) * np.exp(-DGdd_T / (R_kcal * T))

def linewidth_T(T, lw0, slope):
    return lw0 + slope * (T - 298.0)

# ============================================================
# 3. SAFE MULTI-FIELD DATA ACQUISITION
# ============================================================
def load_data(directory, temps):
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

data600 = load_data(DATA_600, T6)
data800 = load_data(DATA_800, T6)

if len(data600) == 0 or len(data800) == 0:
    print("❌ Critical Pipeline Error: Preprocessed arrays not found.")
    sys.exit(1)

# ============================================================
# 4. QUANTUM DNMR BLOCH-MCCONNELL SIMULATOR
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
# 5. UNIFIED MULTI-FIELD OBJECTIVE ENGINE
# ============================================================
def objective(x):
    amp600, amp800 = x[0], x[1]
    LWA0_600, LWB0_600 = x[2], x[3]
    LWA0_800, LWB0_800 = x[4], x[5]
    DG298, DHdd = x[6], x[7]

    resid = []

    for T, f, y in data600:
        pB = pB_vanthoff(T)
        dnu = -(dnu0_600 + dnu_slope * (T - 298.0))
        kex = kex_eyring(T, DG298, DHdd)
        
        LWA = linewidth_T(T, LWA0_600, slopeA_600)
        LWB = linewidth_T(T, LWB0_600, slopeB_600)

        sim = dnmr(f, kex, 0.0, dnu, LWA + LB, LWB + LB, pB) * amp600
        resid.extend((sim - y) / np.max(y))

    for T, f, y in data800:
        pB = pB_vanthoff(T)
        dnu800 = -(dnu0_600 * FIELD_RATIO + dnu_slope * (T - 298.0))
        kex = kex_eyring(T, DG298, DHdd)

        LWA = linewidth_T(T, LWA0_800, slopeA_800)
        LWB = linewidth_T(T, LWB0_800, slopeB_800)

        sim = dnmr(f, kex, 0.0, dnu800, LWA + LB, LWB + LB, pB) * amp800
        resid.extend((sim - y) / np.max(y))

    EXSY_T = np.array([288.0, 298.0, 308.0])
    EXSY_k = np.array([25.12, 48.07, 130.00])
    EXSY_sigma = np.array([(31.73 - 19.76)/4.0, (60.66 - 37.17)/4.0, (150.68 - 111.78)/4.0])

    for T_exsy, k_target, sigma in zip(EXSY_T, EXSY_k, EXSY_sigma):
        k_model = kex_eyring(T_exsy, DG298, DHdd)
        resid.append((k_model - k_target) / sigma)

    return np.array(resid)

# ============================================================
# INITIAL GUESS BOUNDARIES
# ============================================================
x0 = np.array([2e11, 2e11, 64.0, 67.0, 109.0, 131.0, 15.6, 15.0])
lower = np.array([1e8, 1e8, 40.0, 40.0, 80.0, 80.0, 15.0, 0.0])
upper = np.array([1e13, 1e13, 90.0, 100.0, 180.0, 180.0, 16.5, 35.0])

# ============================================================
# 6. EXECUTE OPTIMIZATION
# ============================================================
print("============================================================")
print("     LAUNCHING DUAL-FIELD HYBRID BLOCH-MCCONNELL FITTER     ")
print("============================================================\n")

result = least_squares(objective, x0, bounds=(lower, upper), verbose=2, max_nfev=2000)
xfit = result.x

amp600, amp800 = xfit[0], xfit[1]
LWA0_600, LWB0_600 = xfit[2], xfit[3]
LWA0_800, LWB0_800 = xfit[4], xfit[5]
DG298, DHdd = xfit[6], xfit[7]
DSdd = (DHdd - DG298) / 298.0

# ============================================================
# 7. EXPORT COMPATIBLE COMPLIANCE ARCHIVE
# ============================================================
np.savez(
    OUTFILE_NPZ,
    xfit=xfit,
    T6=T6,
    DH=DH,
    DS=DS,
    dnu_slope=dnu_slope,
    dnu_intercept=dnu0_600,  
    LWA0=LWA0_600,
    slopeA=slopeA_600,
    LWB0=LWB0_600,
    slopeB=slopeB_600,
    DG298=DG298,
    DHdd=DHdd,
    amp600=amp600,  # Explicitly saved to safeguard old scripts
    amp800=amp800,
    scaleA_800=LWA0_800 / LWA0_600,  
    scaleB_800=LWB0_800 / LWB0_600
)

print("\n" + "="*60)
print(f"🎉 SUCCESS: Unified fit results archived cleanly to:\n  👉 {OUTFILE_NPZ}")
print("="*60 + "\n")
