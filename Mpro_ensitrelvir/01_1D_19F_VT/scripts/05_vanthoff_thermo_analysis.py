import os
import sys
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
    print(f"❌ Error: {GLOBAL_NPZ} not found. Please run Script 02b first.")
    sys.exit(1)

# Load archived data
archive = np.load(GLOBAL_NPZ, allow_pickle=True)
T_vals = archive["T_vals"]
pB_vals = archive["pB_vals"]

# Constants
R_kcal = 1.987e-3  
T0 = 298.0  # FIXED: Updated to exactly 298 K per reporting convention

# ============================================================
# 2. VAN'T HOFF REGRESSION & COVARIANCE
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
cov_beta = s_sq * np.linalg.inv(X.T @ X)
t_val = t.ppf(0.975, N - 2)
x_mean = np.mean(x)
Sxx = np.sum((x - x_mean)**2)

# Thermodynamics (Reference centered at 298 K)
dH = -slope * R_kcal
dS = intercept * R_kcal
dG_298 = dH - T0 * dS

# Covariance-Corrected Error Propagation
sigma_dH = t_val * R_kcal * np.sqrt(cov_beta[0,0])
sigma_dS = t_val * R_kcal * np.sqrt(cov_beta[1,1])
cov_si = cov_beta[0,1]
cov_dH_dS = (-R_kcal) * R_kcal * cov_si
sigma_dG = np.sqrt(np.abs(sigma_dH**2 + (T0**2)*sigma_dS**2 - 2*T0*cov_dH_dS))

r2 = np.corrcoef(x, y)[0,1]**2

# ============================================================
# 3. CONSOLIDATED THERMO_RESULTS.TXT
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "05_thermodyanamic_summary.txt")
with open(output_path, "w") as f:
    f.write("=== ENSITRELVIR THERMODYNAMIC & KINETIC SUMMARY ===\n\n")
    f.write("GROUND STATE THERMODYNAMICS (Van't Hoff Summary)\n")
    f.write("-" * 45 + "\n")
    # Swapped °, ±, and ² for safe ASCII alternatives: (deg), +/-, and R2
    f.write(f"dH (deg)     = {dH:.4f} +/- {sigma_dH:.3f} kcal/mol\n")
    f.write(f"dS (deg)     = {dS*1000:.3f} +/- {sigma_dS*1000:.2f} cal/mol/K\n")
    f.write(f"dG (298 K)   = {dG_298:.4f} +/- {sigma_dG:.3f} kcal/mol\n")
    f.write(f"R2           = {r2:.4f}\n\n")
    f.write("ACTIVATION KINETICS (Simulated from 1D Lineshapes)\n")
    f.write("-" * 45 + "\n")
    f.write("Note: Based on kex at 298 K derived from global fit analysis\n")
    f.write("dG# (298 K)  = [Value] kcal/mol\n") # Using '#' or 'double-dagger' text description

# ============================================================
# 4. VAN'T HOFF PLOT WITH 95% CI BAND
# ============================================================
T_line = np.linspace(min(T_vals) - 5, max(T_vals) + 5, 300)
x_line = 1.0 / T_line
y_line = slope * x_line + intercept

# FIXED: CI band calculations mapped correctly to unscaled 1/T line space
band = t_val * np.sqrt(s_sq * (1.0/N + (x_line - x_mean)**2 / Sxx))

# Figure Setup
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

# Plot elements with synchronized 1000/T mapping
ax.scatter(1000.0/T_vals, y, color='blue', edgecolor='black', 
           linewidth=0.5, s=40, zorder=3, label='600 MHz Data')
ax.plot(1000.0/T_line, y_line, color='black', lw=1.5, label='Global Fit')
ax.fill_between(1000.0/T_line, y_line - band, y_line + band, 
                color='#dddddd', alpha=0.6, label='95% CI')

# Labels and Formatting
ax.set_xlabel("1000 / T (K$^{-1}$)", fontsize=24)
ax.set_ylabel("ln K$_{eq}$", fontsize=24)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
ax.set_xticks([3.2, 3.4, 3.6])
ax.set_yticks([-0.4, -0.5, -0.6])

# Legend Layout
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10)

plt.tight_layout()
os.makedirs(FIGURE_DIR, exist_ok=True)

# Save both PNG raster and high-end journal PDF vectors
png_path = os.path.join(FIGURE_DIR, "05_vanthoff_plot.png")
pdf_path = os.path.join(FIGURE_DIR, "05_vanthoff_plot.pdf")
plt.savefig(png_path, bbox_inches='tight', dpi=300)
plt.savefig(pdf_path, bbox_inches='tight')

print(f"✅ Ensitrelvir Van't Hoff plot saved to:\n  👉 {png_path}\n  👉 {pdf_path}")
print(f"📝 Text summary saved to: {output_path}")
# ============================================================
