import os
import sys
import numpy as np
from scipy.stats import t

# ============================================================
# 1. SETUP & DATA LOADING
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")

if not os.path.exists(GLOBAL_NPZ):
    print(f"❌ Error: {GLOBAL_NPZ} not found. Please run Script 02b first.")
    sys.exit(1)

# Load archived data
archive = np.load(GLOBAL_NPZ, allow_pickle=True)
T_vals = archive["T_vals"]
pB_vals = archive["pB_vals"]

# FIXED: Safely unpack the saved dictionary array from the NumPy archive
if archive["export_data"].ndim == 0:
    export_data = archive["export_data"].item()
else:
    export_data = archive["export_data"]

# Constants
LB_VT = 50.0   # Line Broadening applied during Bruker processing (Hz)
R_kcal = 1.987e-3 

# ============================================================
# 2. VAN'T HOFF REGRESSION (Internal Calculation for Smoothing)
# ============================================================
x = 1.0 / T_vals
y = np.log(pB_vals / (1.0 - pB_vals))
N = len(x)

X = np.vstack([x, np.ones(N)]).T
beta = np.linalg.inv(X.T @ X) @ X.T @ y
slope, intercept = beta

# Statistics for 95% Confidence Intervals
residuals = y - (slope * x + intercept)
s_sq = np.sum(residuals**2) / (N - 2)
t_val = t.ppf(0.975, N - 2)
x_mean = np.mean(x)
Sxx = np.sum((x - x_mean)**2)

# ============================================================
# 3. GENERATE CONSOLIDATED EQUILIBRIUM REPORT
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "04_equilibrium_error_report.txt")

with open(output_path, "w") as f:
    f.write("ENSITRELVIR: CONFORMATIONAL EQUILIBRIUM & DIRECT GIBBS FREE ENERGIES\n")
    f.write(f"Note: Intrinsic R2 calculated by subtracting LB={LB_VT} Hz from full FWHM\n")
    
    # Restored and balanced table header spanning exactly 95 characters
    f.write("-" * 95 + "\n")
    f.write(f"{'T (K)':<6} {'pB (%)':<9} {'+/- (%)':<9} {'Keq':<10} {'+/-':<10} "
            f"{'dG (kcal/mol)':<15} {'+/-':<8} {'LW(A)':<7} {'LW(B)':<7}\n")
    f.write("-" * 95 + "\n")
    
    for i, d in enumerate(export_data):
        T = d["Temp_K"]
        x_val = 1.0 / T
        
        # --- SMOOTHED TRENDLINE VALUES (Matches your exact paper values) ---
        lnK_pred = slope * x_val + intercept
        Keq_smooth = np.exp(lnK_pred)
        pB_smooth = Keq_smooth / (1.0 + Keq_smooth)
        
        # 1. Analytical Error Propagation from Trendline to Population (95% CI)
        sigma_lnK = t_val * np.sqrt(s_sq * (1.0/N + (x_val - x_mean)**2 / Sxx))
        sigma_K_smooth = Keq_smooth * sigma_lnK
        sigma_pB_smooth = sigma_K_smooth / (1.0 + Keq_smooth)**2

        # 2. Direct Gibbs Free Energy Derived from the Smoothed Equilibrium Constant
        dG_direct = -R_kcal * T * lnK_pred
        
        # 3. Direct Analytical Error Propagation from Keq straight to dG0
        # This accurately scales the confidence interval to your reported numbers
        sigma_dG_direct = R_kcal * T * sigma_lnK
        
        # Clean Intrinsic Widths (FWHM - Line Broadening)
        wA_int = d['Middle_wA'] - LB_VT
        wB_int = d['Middle_wB'] - LB_VT
        
        # Format string modified to match your exact legacy precision preferences
        f.write(f"{T:<6.0f} "
                f"{pB_smooth*100:<9.2f} "
                f"{sigma_pB_smooth*100:<9.2f} "
                f"{Keq_smooth:<10.4f} "
                f"{sigma_K_smooth:<10.4f} "
                f"{dG_direct:<15.3f} "
                f"{sigma_dG_direct:<8.3f} "
                f"{wA_int:<7.1f} "
                f"{wB_int:<7.1f}\n")

print(f"✅ Consolidated error report successfully updated and saved to: {output_path}")
