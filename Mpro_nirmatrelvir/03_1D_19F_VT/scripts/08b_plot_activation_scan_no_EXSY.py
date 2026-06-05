#!/usr/bin/env python3
# ============================================================
# 08b_plot_control_activation_scan.py
# Plot dual-field unconstrained control landscape profiles (no EXSY)
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
# STYLE & GLOBAL FORMATTING (MATCHES 07B / 09 STYLES)
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

# Pulls cleanly from your unconstrained control data frames
SCAN_FILE = os.path.join(OUTPUT_DIR, "08a_DGscan_results_noEXSY.pkl")

if not os.path.exists(SCAN_FILE):
    print(f"❌ Error: {SCAN_FILE} not found. Run your control profile scan engine first.")
    sys.exit(1)

with open(SCAN_FILE, "rb") as f:
    results = pickle.load(f)

print(f"Loaded {len(results)} unconstrained control scan results.")

# ============================================================
# EXTRACT DATA FIELDS
# ============================================================
DG  = np.array([r["DG298"] for r in results])
DH  = np.array([r["DHdd"] for r in results])
DS  = np.array([r["DSdd_cal"] for r in results])
RSS = np.array([r["RSS"] for r in results])
k298 = np.array([r["k298"] for r in results])

LWA600 = np.array([r["LWA0_600"] for r in results])
LWB600 = np.array([r["LWB0_600"] for r in results])
LWA800 = np.array([r["LWA0_800"] for r in results])
LWB800 = np.array([r["LWB0_800"] for r in results])

RSS_min = np.min(RSS)
threshold = 1.25
acceptable = [r for r in results if r["RSS"] < RSS_min * threshold]

print(f"Minimum RSS (EXSY-Free Control) = {RSS_min:.3f}")
print(f"Keeping {len(acceptable)} fits within {threshold:.2f} × RSS_min")

temps = np.array([285, 288, 291, 293, 296, 298, 303, 308, 313, 315])

# ============================================================
# GENERATE THE UNCONSTRAINED SUMMARY TEXT LOG
# ============================================================
TXT_OUT = os.path.join(OUTPUT_DIR, "08b_DGscan_output_summary-noEXSY.txt")

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
DUAL-FIELD CONTROL DNMR PROFILE SCAN ANALYSIS SUMMARY (EXSY-FREE)
============================================================
Methodology Description:
This analysis performs a 1D structural grid scan across activation free energy 
bounds (DG‡ 298 K). At each step, a multi-variable nonlinear least-squares 
regression evaluates simultaneous lineshape fitting targets for 600 MHz and 
800 MHz spectra across 10 temperature profiles. CONTROL EXPERIMENT: No external 
EXSY kinetic restraints are applied during this scan sequence.

Statistical Confidence Interval Assignment:
Parameter distributions are defined using an expansion envelope tracking 
relative Residual Sum of Squares (RSS) tolerance limits above the global minimum.
By capturing the covariance profile paths directly, parameter compensation 
artifacts (such as enthalpy-entropy compensation error) are preserved.
  - 68% Confidence Bounds: RSS cutoff threshold = {thresh_68:.2f} × RSS_min
  - 95% Confidence Bounds: RSS cutoff threshold = {thresh_95:.2f} × RSS_min

------------------------------------------------------------
GLOBAL OPTIMIZATION RESULTS (UNCONSTRAINED LINESHAPE ONLY):
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
print(f"🎉 Saved EXSY-Free Control Text Summary to disk: {TXT_OUT}")

# ============================================================
# FIGURE 1: CONTROL RSS PROFILE
# ============================================================
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.plot(DG, RSS, 'ko-', lw=1.5, ms=4)
ax1.axhline(RSS_min * threshold, color='red', ls='--', lw=1.5, label=f'{threshold:.2f} × RSS_min')
ax1.set_xlabel(r"$\Delta G^\ddagger$(298 K) (kcal/mol)")
ax1.set_ylabel("RSS")
ax1.legend()
plt.tight_layout()

# ============================================================
# FIGURE 2: EYRING SPAGHETTI (CONTROL ENSEMBLE - NO EXSY OVERLAY)
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
ax2.plot(x_eyr, best_y, color='red', lw=3, zorder=3, label='Best VT-only fit')

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
# FIGURE 3: DH VS DS SCATTER MATRIX (CONTROL COMPENSATION PROFILE)
# ============================================================
fig3, ax3 = plt.subplots(figsize=(6, 4))
sc = ax3.scatter(DH, DS, c=RSS, cmap='viridis_r', s=40)
ax3.set_xlabel(r"$\Delta H^\ddagger$ (kcal/mol)")
ax3.set_ylabel(r"$\Delta S^\ddagger$ (cal/mol·K)")
cbar = plt.colorbar(sc)
cbar.set_label("RSS")
plt.tight_layout()

# ============================================================
# FIGURE 4: CONTROL DG VS K298 SCATTER MATRIX (NO HORIZONTAL EXSY SPAN)
# ============================================================
fig4, ax4 = plt.subplots(figsize=(6, 4))
sc4 = ax4.scatter(DG, k298, c=RSS, cmap='viridis_r', s=40)
ax4.set_xlabel(r"$\Delta G^\ddagger$(298 K) (kcal/mol)")
ax4.set_ylabel(r"$k_{\mathrm{ex}}$(298 K) (s$^{-1}$)")
# Note: axhspan is explicitly removed here to isolate the purely unconstrained landscape profile
cbar4 = plt.colorbar(sc4)
cbar4.set_label("RSS")
plt.tight_layout()

# ============================================================
# EXPORT VISUAL CONTROL PORTFOLIO ASSETS
# ============================================================
figures = {
    "08b_DGscan_RSS_profile": fig1,
    "08b_Eyring_profile_ensemble": fig2,
    "08b_DHdd_vs_DSdd": fig3,
    "08b_DGdd_vs_k298": fig4,
}

for name, fig in figures.items():
    png = os.path.join(FIGURE_DIR, f"{name}-noEXSY.png")
    pdf = os.path.join(FIGURE_DIR, f"{name}-noEXSY.pdf")
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f"Saved Image and Vector Control Assets -> {name}-noEXSY")

print("\n🎉 Success: 08b control plot assets and unconstrained matrices are completely updated.")
plt.close('all')
