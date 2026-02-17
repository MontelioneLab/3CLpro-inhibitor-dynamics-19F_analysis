import os
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
    print(f"Error: {GLOBAL_NPZ} not found. Run Script 02 first.")
    exit()

# Load archived data
archive = np.load(GLOBAL_NPZ, allow_pickle=True)
export_data = archive["export_data"]
T_vals = archive["T_vals"]
pB_vals = archive["pB_vals"]

# Constants
LB_VT = 50.0 
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

# Statistics for 95% CI
residuals = y - (slope * x + intercept)
s_sq = np.sum(residuals**2) / (N - 2)
t_val = t.ppf(0.975, N - 2)
x_mean = np.mean(x)
Sxx = np.sum((x - x_mean)**2)

# ============================================================
# 3. GENERATE CONSOLIDATED EQUILIBRIUM REPORT
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "equilibrium_vs_temperature.txt")

with open(output_path, "w") as f:
    f.write("ENSITRELVIR: CONFORMATIONAL EQUILIBRIUM (Van't Hoff Smoothed)\n")
    f.write(f"Note: Intrinsic R2 calculated by subtracting LB={LB_VT} Hz\n")
    f.write("-" * 110 + "\n")
    f.write(f"{'T (K)':<8} {'pB (%)':<10} {'+/- (%)':<10} {'Keq':<12} {'+/-':<12} {'Mid_wA':<10} {'Mid_wB':<10}\n")
    f.write("-" * 110 + "\n")
    
    for d in export_data:
        T = d["Temp_K"]
        x_val = 1.0 / T
        
        # Calculate Smoothed Values
        lnK_pred = slope * x_val + intercept
        Keq = np.exp(lnK_pred)
        pB = Keq / (1.0 + Keq)
        
        # Propagate Errors (95% CI)
        sigma_lnK = t_val * np.sqrt(s_sq * (1.0/N + (x_val - x_mean)**2 / Sxx))
        sigma_K = Keq * sigma_lnK
        sigma_pB = sigma_K / (1.0 + Keq)**2
        
        # Pull Raw Intrinsic widths from the archive
        wA_int = d['Middle_wA'] - LB_VT
        wB_int = d['Middle_wB'] - LB_VT
        
        f.write(f"{T:<8.0f} "
                f"{pB*100:<10.2f} "
                f"{sigma_pB*100:<10.2f} "
                f"{Keq:<12.4f} "
                f"{sigma_K:<12.4f} "
                f"{wA_int:<10.1f} "
                f"{wB_int:<10.1f}\n")

print(f"Consolidated equilibrium report saved to {output_path}")
