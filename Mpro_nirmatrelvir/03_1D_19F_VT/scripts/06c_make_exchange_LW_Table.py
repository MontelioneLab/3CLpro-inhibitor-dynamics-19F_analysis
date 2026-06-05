#!/usr/bin/env python3
# ============================================================
# 06c_make_exchange_linewidth_table.py
# Calculates exchange contribution parameters across fields
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

# ============================================================
# 1. SETUP & PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# FIXED: Synchronized to read from your step-prefixed master optimizer output
FITFILE = os.path.join(OUTPUT_DIR, "06_vt_dual_fit_HYBRID.npz")
OUTFILE = os.path.join(OUTPUT_DIR, "06c_exchange_linewidth_contribution_table.csv")

# ============================================================
# 2. LOAD COMPREHENSIVE FIT ARCHIVE
# ============================================================
if not os.path.exists(FITFILE):
    print(f"❌ Error: {FITFILE} not found. Please run your 06a master optimizer first.")
    sys.exit(1)

fit = np.load(FITFILE)
T6 = fit["T6"]

# Unpack global optimized variables from binary archive
DG298 = float(fit["DG298"])
DHdd  = float(fit["DHdd"])
DH    = float(fit["DH"])
DS    = float(fit["DS"])
dnu_slope = float(fit["dnu_slope"])

# FIXED: Re-anchored to read from your standardized dictionary key framework
LWA0_600 = float(fit["LWA0"])
slopeA_600 = float(fit["slopeA"])
LWB0_600 = float(fit["LWB0"])
slopeB_600 = float(fit["slopeB"])

dnu0_600 = float(fit["dnu_intercept"])

scaleA_800 = float(fit["scaleA_800"])
scaleB_800 = float(fit["scaleB_800"])

# Reconstruct 800 MHz intercepts using saved field scales
LWA0_800 = LWA0_600 * scaleA_800
LWB0_800 = LWB0_600 * scaleB_800

# Fixed 800 MHz slopes derived from Step 02c baseline calibrations
slopeA_800 = -3.680
slopeB_800 = -2.158

# ============================================================
# 3. CORE PHYSICAL CONSTANTS
# ============================================================
R_kcal = 0.001987207  # Gas constant in kcal/mol/K
kB = 1.380649e-23     # Boltzmann Constant (J/K)
h  = 6.62607015e-34    # Planck Constant (J*s)

# Spectrometer frequencies logged from Script 01
freq_600 = 564.5312351
freq_800 = 752.8281062
field_ratio = freq_800 / freq_600

# ============================================================
# 4. EQUILIBRIUM & KINETIC MATHEMATICS
# ============================================================
def dnu600_linear(T):
    return dnu0_600 + dnu_slope * (T - 298.0)

def dnu800_linear(T):
    return (dnu0_600 * field_ratio) + dnu_slope * (T - 298.0)

def pB_vanthoff(T):
    dG = DH - T * (DS / 1000.0)
    K = np.exp(-dG / (R_kcal * T))
    return K / (1.0 + K)

def kex_eyring(T, DG298, DHdd):
    DSdd = (DHdd - DG298) / 298.0
    DGdd_T = DHdd - T * DSdd
    return (kB * T / h) * np.exp(-DGdd_T / (R_kcal * T))

def linewidth_T(T, lw0, slope):
    return lw0 + slope * (T - 298.0)

# ============================================================
# 5. MATRIX COMPILATION LOOP
# ============================================================
rows = []

for T in T6:
    pB = pB_vanthoff(T)
    kex = kex_eyring(T, DG298, DHdd)

    # --- 600 MHz Field Extraction ---
    dnu600 = abs(dnu600_linear(T))
    LWA_600 = linewidth_T(T, LWA0_600, slopeA_600)
    LWB_600 = linewidth_T(T, LWB0_600, slopeB_600)

    kAB = pB * kex
    kBA = (1.0 - pB) * kex

    LWex_A = kAB / np.pi
    LWex_B = kBA / np.pi

    pctA = 100.0 * LWex_A / LWA_600
    pctB = 100.0 * LWex_B / LWB_600

    rows.append({
        "Field": 600, "T(K)": T, "kex": kex, "pB(%)": pB * 100.0,
        "dnu": dnu600, "kex/dw": kex / (2 * np.pi * dnu600),
        "LWA_BM": LWA_600, "LWB_BM": LWB_600,
        "LWex_A": LWex_A, "LWex_B": LWex_B,
        "%Ex_A": pctA, "%Ex_B": pctB
    })

    # --- 800 MHz Field Extraction ---
    dnu800 = abs(dnu800_linear(T))
    LWA_800 = linewidth_T(T, LWA0_800, slopeA_800)
    LWB_800 = linewidth_T(T, LWB0_800, slopeB_800)

    pctA = 100.0 * LWex_A / LWA_800
    pctB = 100.0 * LWex_B / LWB_800

    rows.append({
        "Field": 800, "T(K)": T, "kex": kex, "pB(%)": pB * 100.0,
        "dnu": dnu800, "kex/dw": kex / (2 * np.pi * dnu800),
        "LWA_BM": LWA_800, "LWB_BM": LWB_800,
        "LWex_A": LWex_A, "LWex_B": LWex_B,
        "%Ex_A": pctA, "%Ex_B": pctB
    })

# ============================================================
# 6. DATABASE FORMATTING & EXPORT
# ============================================================
df = pd.DataFrame(rows)
df = df.sort_values(by=["Field", "T(K)"]).reset_index(drop=True)

df["Field"] = df["Field"].astype(int)
df["T(K)"]  = df["T(K)"].astype(int)

for col in df.columns:
    if col in ["Field", "T(K)"]:
        continue
    df[col] = df[col].round(3) if col == "kex/dw" else df[col].round(1)

df.to_csv(OUTFILE, index=False)

print("\n" + "="*60)
print(f"🎉 SUCCESS: Exchange contribution table written safely to:\n  👉 {OUTFILE}")
print("="*60 + "\n")
print(df.to_string(index=False))
