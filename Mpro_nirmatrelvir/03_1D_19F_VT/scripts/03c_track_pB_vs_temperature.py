#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. SETUP & PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

# FIXED: Synchronized with the correct filename from your database step
CSV_FILE = os.path.join(OUTPUT_DIR, "02_lorentzian_summary.csv")
TXT_OUT = os.path.join(OUTPUT_DIR, "03c_pB_vantHoff_fit.txt")

# ============================================================
# 2. CONFIGURATIONS & SETTINGS
# ============================================================
# Trusted low-T points (Determined from Rex/LW < 10%)
FIT_TEMPS_600 = [285, 288, 291, 293, 296]
FIT_TEMPS_800 = [285, 288, 293, 298]

COLOR_600 = "#1f77b4"   # Classic Blue
COLOR_800 = "#c47a00"   # Dark Yellow-Orange
USE_SEM = False

# ============================================================
# PLOT STYLE SETTINGS
# ============================================================
plt.rcParams.update({
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.titlesize": 24,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ============================================================
# 3. DATA LOADING & REPLICATE HANDLING
# ============================================================
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: {CSV_FILE} not found. Run your Lorentzian fitter script first.")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)

# Aggregate replicate records cleanly
summary = df.groupby(["Field", "Temp"], as_index=False).agg({"pB": ["mean", "std", "count"]})
summary.columns = ["Field", "Temp", "pB_mean", "pB_std", "n"]

# Calculate error parameters
summary["pB_sem"] = summary["pB_std"] / np.sqrt(summary["n"])
summary["pB_err"] = summary["pB_sem"] if USE_SEM else summary["pB_std"]

# Map active inclusion masks
summary["used"] = False
summary.loc[((summary["Field"] == 600) & (summary["Temp"].isin(FIT_TEMPS_600))), "used"] = True
summary.loc[((summary["Field"] == 800) & (summary["Temp"].isin(FIT_TEMPS_800))), "used"] = True

used600 = summary[(summary["Field"] == 600) & (summary["used"])]
excl600 = summary[(summary["Field"] == 600) & (~summary["used"])]
used800 = summary[(summary["Field"] == 800) & (summary["used"])]
excl800 = summary[(summary["Field"] == 800) & (~summary["used"])]

# ============================================================
# 4. UNIFIED VAN 'T HOFF REGRESSION
# ============================================================
fit_df = pd.concat([used600, used800])
T_fit = fit_df["Temp"].values
pB_fit = fit_df["pB_mean"].values

# Calculate equilibrium constraints from combined fields
Keq = pB_fit / (1.0 - pB_fit)
lnK = np.log(Keq)
invT = 1.0 / T_fit

# Linear fit: lnK = -(dH/R)(1/T) + dS/R
coeffs = np.polyfit(invT, lnK, 1)
slope, intercept = coeffs

R = 0.0019872  # Gas constant in kcal/mol/K
dH = -slope * R
dS = intercept * R * 1000.0
dG298 = dH - 298.0 * (dS / 1000.0)

# ============================================================
# 5. WRITE SYSTEM LOGS (Clean ASCII strings)
# ============================================================
print("\n" + "="*60)
print("              COMBINED FIELD VAN'T HOFF SUMMARY")
print("="*60)
print(f"  dH0      = {dH:.3f} kcal/mol")
print(f"  dS0      = {dS:.3f} cal/mol/K")
print(f"  dG0(298) = {dG298:.3f} kcal/mol")
print("="*60)

with open(TXT_OUT, "w") as f:
    f.write("="*60 + "\n")
    f.write("VAN'T HOFF THERMODYNAMIC ANALYSIS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Error bar layout = {'SEM' if USE_SEM else 'SD'}\n\n")
    f.write(f"600 MHz included temperatures: {FIT_TEMPS_600}\n")
    f.write(f"800 MHz included temperatures: {FIT_TEMPS_800}\n\n")
    f.write(f"dH0      = {dH:.4f} kcal/mol\n")
    f.write(f"dS0      = {dS:.4f} cal/mol/K\n")
    f.write(f"dG0(298) = {dG298:.4f} kcal/mol\n")

# ============================================================
# 6. RENDER REPOSITORY VISUALIZATIONS
# ============================================================
T_plot = np.linspace(284, 316, 400)
invT_plot = 1.0 / T_plot
lnK_fit_plot = slope * invT_plot + intercept
Keq_plot = np.exp(lnK_fit_plot)
pB_plot = Keq_plot / (1.0 + Keq_plot)

fig, ax = plt.subplots(figsize=(8, 6.0))

# 600 MHz Markers (Included vs Excluded)
ax.errorbar(used600["Temp"], used600["pB_mean"], yerr=used600["pB_err"],
            fmt='o', color=COLOR_600, ms=10, lw=2, capsize=4, label="600 MHz")
ax.errorbar(excl600["Temp"], excl600["pB_mean"], yerr=excl600["pB_err"],
            fmt='o', markerfacecolor='none', markeredgecolor=COLOR_600,
            markeredgewidth=2, ecolor=COLOR_600, ms=9, lw=2, capsize=4)

# 800 MHz Markers (Included vs Excluded)
ax.errorbar(used800["Temp"], used800["pB_mean"], yerr=used800["pB_err"],
            fmt='o', color=COLOR_800, ms=10, lw=2, capsize=4, label="800 MHz")
ax.errorbar(excl800["Temp"], excl800["pB_mean"], yerr=excl800["pB_err"],
            fmt='o', markerfacecolor='none', markeredgecolor=COLOR_800,
            markeredgewidth=2, ecolor=COLOR_800, ms=10, lw=2, capsize=4)

# Projected Thermodynamic Fit Curve
ax.plot(T_plot, pB_plot, '--', color='black', lw=2, label="van't Hoff fit")

# Formatting Layout
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("p$_B$")  # Subscript format for publication
ax.set_xlim(284, 316)
ax.set_ylim(0.30, 0.50)
ax.set_yticks([0.30, 0.35, 0.40, 0.45, 0.50])

# Make tick labels bigger and clean up spacing
ax.tick_params(axis='both', labelsize=18)
ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))

# FIXED: Strip top/right frames for uniform look across the pipeline
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc="upper left")

plt.tight_layout()

# Save High-End Formats
PDF_OUT = os.path.join(FIGURE_DIR, "03c_pB_vs_temperature.pdf")
PNG_OUT = os.path.join(FIGURE_DIR, "03c_pB_vs_temperature.png")
plt.savefig(PDF_OUT, bbox_inches='tight')
plt.savefig(PNG_OUT, dpi=300, bbox_inches='tight')

print(f"✅ Population trend figures successfully exported:\n  👉 {PDF_OUT}\n  👉 {PNG_OUT}\n")
plt.close()
