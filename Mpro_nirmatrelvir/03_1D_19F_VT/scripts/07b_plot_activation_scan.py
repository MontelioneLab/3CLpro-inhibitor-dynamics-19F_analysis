#!/usr/bin/env python3
# ============================================================
# 07b_plot_activation_scan.py
# Realignment-Safe Profile Plotter & Text Reporting Engine
# ============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# ============================================================
# STYLE & GLOBAL FORMATTING (MATCHES 09 MODEL)
# ============================================================
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 13
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["pdf.fonttype"] = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

SCAN_FILE = os.path.join(OUTPUT_DIR, "07a_DGscan_results.pkl")
EXSY_FILE = os.path.join(OUTPUT_DIR, "EXSY_bootstrap_results.csv")

if not os.path.exists(SCAN_FILE):
    print(f"❌ Error: {SCAN_FILE} not found. Run your profile scan engine first.")
    sys.exit(1)

with open(SCAN_FILE, "rb") as f:
    results = pickle.load(f)

print(f"Loaded {len(results)} scan results cleanly.")

# ============================================================
# EXTRACT DATA FIELDS
# ============================================================
DG  = np.array([r["DG298"] for r in results])
DH  = np.array([r["DHdd"] for r in results])
DS = np.array([r["DSdd_cal"] for r in results])
RSS = np.array([r["RSS"] for r in results])
k298 = np.array([r["k298"] for r in results])

LWA600 = np.array([r["LWA0_600"] for r in results])
LWB600 = np.array([r["LWB0_600"] for r in results])
LWA800 = np.array([r["LWA0_800"] for r in results])
LWB800 = np.array([r["LWB0_800"] for r in results])

RSS_min = np.min(RSS)
threshold = 1.25
acceptable = [r for r in results if r["RSS"] < RSS_min * threshold]

print(f"Minimum RSS = {RSS_min:.3f}")
print(f"Keeping {len(acceptable)} fits within {threshold:.2f} × RSS_min")

temps = np.array([285, 288, 291, 293, 296, 298, 303, 308, 313, 315])

# ============================================================
# GENERATE THE HIGH-FIDELITY SUMMARY TEXT LOG
# ============================================================
TXT_OUT = os.path.join(OUTPUT_DIR, "07b_DGscan_output_summary.txt")

idx_min   = np.argmin(RSS)
best_DG   = DG[idx_min]
best_DH   = DH[idx_min]
best_DS   = DS[idx_min]
best_k298 = k298[idx_min]

thresh_68 = 1.08  
thresh_95 = 1.25  

mask_68 = RSS <= (RSS_min * thresh_68)
mask_95 = RSS <= (RSS_min * thresh_95)

summary_text = f"""============================================================
DUAL-FIELD DNMR / EXSY PROFILE SCAN ANALYSIS SUMMARY
============================================================
Methodology Description:
This analysis performs a 1D structural grid scan across activation free energy 
bounds (DG‡ 298 K). At each step, a multi-variable nonlinear least-squares 
regression evaluates simultaneous lineshape fitting targets for 600 MHz and 
800 MHz spectra across 10 temperature profiles. Explicit internal EXSY kinetic 
restraints are applied using weighted bootstrap deviation penalties.

Statistical Confidence Interval Assignment:
Parameter distributions are defined using an expansion envelope tracking 
relative Residual Sum of Squares (RSS) tolerance limits above the global minimum.
By capturing the covariance profile paths directly, parameter compensation 
artifacts (such as enthalpy-entropy compensation error) are preserved.
  - 68% Confidence Bounds: RSS cutoff threshold = {thresh_68:.2f} × RSS_min
  - 95% Confidence Bounds: RSS cutoff threshold = {thresh_95:.2f} × RSS_min

------------------------------------------------------------
GLOBAL OPTIMIZATION RESULTS:
------------------------------------------------------------
Minimum Achieved RSS (Global Optimum) : {RSS_min:.4f}
Best-fit Activation Parameters:
  - DG‡ (298 K) : {best_DG:.2f} kcal/mol
  - DH‡        : {best_DH:.2f} kcal/mol
  - DS‡        : {best_DS:.2f} cal/mol/K
  - kex (298 K) : {best_k298:.1f} s^-1

------------------------------------------------------------
68% CONFIDENCE INTERVALS (1-Sigma Envelope):
------------------------------------------------------------
Keeping {np.sum(mask_68)} matching coordinates within RSS <= {RSS_min * thresh_68:.3f}
  - DG‡ (298 K) : [{DG[mask_68].min():.2f} to {DG[mask_68].max():.2f}] kcal/mol
  - DH‡        : [{DH[mask_68].min():.2f} to {DH[mask_68].max():.2f}] kcal/mol
  - DS‡        : [{DS[mask_68].min():.2f} to {DS[mask_68].max():.2f}] cal/mol/K
  - kex (298 K) : [{k298[mask_68].min():.1f} to {k298[mask_68].max():.1f}] s^-1

------------------------------------------------------------
95% CONFIDENCE INTERVALS (2-Sigma Envelope):
------------------------------------------------------------
Keeping {np.sum(mask_95)} matching coordinates within RSS <= {RSS_min * thresh_95:.3f}
  - DG‡ (298 K) : [{DG[mask_95].min():.2f} to {DG[mask_95].max():.2f}] kcal/mol
  - DH‡        : [{DH[mask_95].min():.2f} to {DH[mask_95].max():.2f}] kcal/mol
  - DS‡        : [{DS[mask_95].min():.2f} to {DS[mask_95].max():.2f}] cal/mol/K
  - kex (298 K) : [{k298[mask_95].min():.1f} to {k298[mask_95].max():.1f}] s^-1
============================================================
"""

with open(TXT_OUT, "w") as f:
    f.write(summary_text)
print(f"🎉 Saved Text Summary to disk: {TXT_OUT}")

# ============================================================
# FIGURE 1: RSS PROFILE (RESTORED FROM 09)
# ============================================================
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.plot(DG, RSS, 'ko-', lw=1.5, ms=4)
ax1.axhline(RSS_min * threshold, color='red', ls='--', lw=1.5, label=f'{threshold:.2f} × RSS_min')
ax1.set_xlabel(r"$\Delta G^\ddagger$(298 K) (kcal/mol)")
ax1.set_ylabel("RSS")
ax1.legend()
plt.tight_layout()

# ============================================================
# FIGURE 2: EYRING SPAGHETTI (RESTORED FROM 09)
# ============================================================
plt.rcParams.update({
    'font.size': 20, 'axes.labelsize': 20, 'xtick.labelsize': 20,
    'ytick.labelsize': 20, 'legend.fontsize': 20
})

fig2, ax2 = plt.subplots(figsize=(6.0, 5.2))
order = np.argsort(temps)
temps_sorted = temps[order]
x_eyr = 1000.0 / temps_sorted

for r in acceptable:
    kex_sorted = np.array(r["kex"])[order]
    y = np.log(kex_sorted / temps_sorted)
    ax2.plot(x_eyr, y, color='royalblue', lw=0.8, alpha=0.12, zorder=1)

best = min(results, key=lambda r: r["RSS"])
best_kex = np.array(best["kex"])[order]
best_y = np.log(best_kex / temps_sorted)
ax2.plot(x_eyr, best_y, color='red', lw=3, zorder=3, label='Best fit')

# --- EXSY Data Processing with Explicit Variable Scoping ---
if os.path.exists(EXSY_FILE):
    exsy = pd.read_csv(EXSY_FILE).sort_values("T")
    
    T_exsy = exsy["T"].values
    kex_med = exsy["median_kex"].values
    kex_low = exsy["CI2.5"].values
    kex_high = exsy["CI97.5"].values

    x_exsy = 1000.0 / T_exsy
    y_exsy = np.log(kex_med / T_exsy)

    y_low  = np.log(kex_low / T_exsy)
    y_high = np.log(kex_high / T_exsy)
    
    # FIXED: Declared as an independent local object to force precise asymmetric vector mapping
    yerr = [
        y_exsy - y_low,
        y_high - y_exsy
    ]
    
    ax2.errorbar(x_exsy, y_exsy, yerr=yerr,
                 fmt='o', color='darkred', ecolor='red', elinewidth=2.5, capsize=5,
                 markersize=8, markeredgecolor='black', markeredgewidth=1.0, zorder=4,
                 label='EXSY (95% Bootstrap CI)')

ax2.set_xlabel(r"1000 / $T$ (K$^{-1}$)", labelpad=8)
ax2.set_ylabel(r"$\ln(k_{\mathrm{ex}} / T)$", labelpad=8)
ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax2.set_yticks([-1.0, -2.0, -3.0, -4.0])
ax2.set_ylim(-4.5, -0.2)
ax2.set_xticks([3.2, 3.3, 3.4, 3.5])
ax2.set_xlim(3.18, 3.51)
ax2.legend(frameon=False, loc="lower left")
ax2.spines['top'].set_visible(False)   
ax2.spines['right'].set_visible(False)
plt.tight_layout()

# ============================================================
# FIGURE 4: DH VS DS SCATTER MATRIX (RESTORED FROM 09)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(6, 4))
sc = ax4.scatter(DH, DS, c=RSS, cmap='viridis_r', s=40)
ax4.set_xlabel(r"$\Delta H^\ddagger$ (kcal/mol)")
ax4.set_ylabel(r"$\Delta S^\ddagger$ (cal/mol·K)")
cbar = plt.colorbar(sc)
cbar.set_label("RSS")
plt.tight_layout()

# ============================================================
# FIGURE 5: DG VS K298 SCATTER MATRIX (RESTORED FROM 09)
# ============================================================
fig5, ax5 = plt.subplots(figsize=(6, 4))
sc5 = ax5.scatter(DG, k298, c=RSS, cmap='viridis_r', s=40)
ax5.set_xlabel(r"$\Delta G^\ddagger$(298 K) (kcal/mol)")
ax5.set_ylabel(r"$k_{\mathrm{ex}}$(298 K) (s$^{-1}$)")
ax5.axhspan(37, 61, color='red', alpha=0.15, label='EXSY-supported')
cbar5 = plt.colorbar(sc5)
cbar5.set_label("RSS")
plt.tight_layout()

# ============================================================
# EXPORT VISUAL PORTFOLIO HOOKS
# ============================================================
figures = {
    "07b_DGscan_RSS_profile": fig1,
    "07b_Eyring_profile_ensemble": fig2,
    "07b_DHdd_vs_DSdd_compensation": fig4,
    "07b_DGdd_vs_k298_velocity": fig5,
}

for name, fig in figures.items():
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURE_DIR, f"{name}.pdf"), bbox_inches='tight')
    print(f"Saved Image and Vector Assets -> {name}")

print("\n🎉 Success: 07b error bars and aesthetics are perfectly restored to matching standards.")
plt.close('all')
