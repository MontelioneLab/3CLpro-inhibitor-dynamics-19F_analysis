import os
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

# ============================================================
# 1. SETUP & DATA LOADING
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")

if not os.path.exists(GLOBAL_NPZ):
    print(f"Error: {GLOBAL_NPZ} not found. Run Script 02 first.")
    exit()

# Load archived data
archive = np.load(GLOBAL_NPZ, allow_pickle=True)
T_vals = archive["T_vals"]
pB_vals = archive["pB_vals"]

# Constants
R_kcal = 1.987e-3  
T0 = 298.15

# ============================================================
# 2. VAN'T HOFF REGRESSION & COVARIANCE
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
cov_beta = s_sq * np.linalg.inv(X.T @ X)
t_val = t.ppf(0.975, N - 2)
x_mean = np.mean(x)
Sxx = np.sum((x - x_mean)**2)

# Thermodynamics
dH = -slope * R_kcal
dS = intercept * R_kcal
dG_298 = dH - T0 * dS

# Covariance-Corrected Errors
sigma_dH = t_val * R_kcal * np.sqrt(cov_beta[0,0])
sigma_dS = t_val * R_kcal * np.sqrt(cov_beta[1,1])
cov_si = cov_beta[0,1]
cov_dH_dS = (-R_kcal) * R_kcal * cov_si
sigma_dG = np.sqrt(np.abs(sigma_dH**2 + (T0**2)*sigma_dS**2 - 2*T0*cov_dH_dS))

r2 = np.corrcoef(x, y)[0,1]**2

# ============================================================
# 3. CONSOLIDATED THERMO_RESULTS.TXT
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "thermo_results.txt")
with open(output_path, "w") as f:
    f.write("=== ENSITRELVIR THERMODYNAMIC & KINETIC SUMMARY ===\n\n")
    f.write("GROUND STATE THERMODYNAMICS (Van't Hoff)\n")
    f.write("-" * 45 + "\n")
    f.write(f"dH°          = {dH:.4f} ± {sigma_dH:.3f} kcal/mol\n")
    f.write(f"dS°          = {dS*1000:.3f} ± {sigma_dS*1000:.2f} cal/mol/K\n")
    f.write(f"dG° (298.15 K) = {dG_298:.4f} ± {sigma_dG:.3f} kcal/mol\n")
    f.write(f"R²           = {r2:.4f}\n\n")
    f.write("ACTIVATION KINETICS (Simulated from 1D Lineshapes)\n")
    f.write("-" * 45 + "\n")
    f.write("Note: Based on kex at 298 K derived from global fit analysis\n")
    f.write("dG‡ (298.15 K) = [Value] kcal/mol\n")

# ============================================================
# 4. VAN'T HOFF PLOT WITH 95% CI BAND
# ============================================================
T_line = np.linspace(min(T_vals) - 5, max(T_vals) + 5, 300)
x_line = 1.0 / T_line
y_line = slope * x_line + intercept

# CI band calculation
band = t_val * np.sqrt(s_sq * (1.0/N + (x_line - x_mean)**2 / Sxx))

# Use a slightly wider figure to accommodate the legend on the right
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

# Plot elements with blue dots and black outline
ax.scatter(1000.0/T_vals, y, color='blue', edgecolor='black', 
           linewidth=0.5, s=40, zorder=3, label='600 MHz Data')
ax.plot(1000.0/T_line, y_line, color='black', lw=1.5, label='Global Fit')
ax.fill_between(1000.0/T_line, y_line - band, y_line + band, 
                color='#dddddd', alpha=0.6, label='95% CI')

# Labels and Formatting
ax.set_xlabel("1000 / T (K$^{-1}$)", fontsize=10)
ax.set_ylabel("ln K", fontsize=10)

# Move legend outside to the right
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=9)

# Remove the inset text box as requested
# (Thermodynamic parameters are now only in thermo_results.txt)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "vanthoff_plot.png"), bbox_inches='tight')

print(f"Standardized Van't Hoff plot (Nirmatrelvir style) saved to: {FIGURE_DIR}")
# ============================================================
