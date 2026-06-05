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

# FIXED: Synchronized with the step-prefixed database file
CSV_FILE = os.path.join(OUTPUT_DIR, "02_lorentzian_summary.csv")
TXT_OUT = os.path.join(OUTPUT_DIR, "04_vanthoff_results.txt")

# ============================================================
# 2. TRUSTED TEMPERATURES & COLORS
# ============================================================
FIT_TEMPS_600 = [285, 288, 291, 293, 296]
FIT_TEMPS_800 = [285, 288, 293, 298]

COLOR_600 = "#1f77b4"  # Classic Blue
COLOR_800 = "#c47a00"  # Dark Yellow-Orange
FIT_COLOR = "black"
USE_SEM = False

# ============================================================
# PLOT STYLE SETTINGS
# ============================================================
plt.rcParams.update({
    "axes.labelsize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
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
# 3. DATA LOADING & FILTERING
# ============================================================
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: {CSV_FILE} not found. Run your Lorentzian fitter script first.")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)

# Average out replicates cleanly
summary = df.groupby(["Field", "Temp"], as_index=False).agg({"pB": ["mean", "std", "count"]})
summary.columns = ["Field", "Temp", "pB_mean", "pB_std", "n"]

summary["pB_sem"] = summary["pB_std"] / np.sqrt(summary["n"])
summary["pB_err"] = summary["pB_sem"] if USE_SEM else summary["pB_std"]

summary["used"] = False
summary.loc[((summary["Field"] == 600) & (summary["Temp"].isin(FIT_TEMPS_600))), "used"] = True
summary.loc[((summary["Field"] == 800) & (summary["Temp"].isin(FIT_TEMPS_800))), "used"] = True

used600 = summary[(summary["Field"] == 600) & (summary["used"])]
excl600 = summary[(summary["Field"] == 600) & (~summary["used"])]
used800 = summary[(summary["Field"] == 800) & (summary["used"])]
excl800 = summary[(summary["Field"] == 800) & (~summary["used"])]

fit_df = pd.concat([used600, used800]).sort_values("Temp")

# ============================================================
# 4. VAN'T HOFF REGRESSION MATHEMATICS
# ============================================================
T = fit_df["Temp"].values
pB = fit_df["pB_mean"].values

Keq = pB / (1.0 - pB)
lnK = np.log(Keq)
invT = 1000.0 / T  # Scale factor of 1000 applied to x-axis

coeffs = np.polyfit(invT, lnK, 1)
slope, intercept = coeffs

R = 0.0019872  # Gas constant in kcal/mol/K

# FIXED: Because the x-axis was multiplied by 1000, the calculated slope is 
# divided by 1000. To recover dH, we must DIVIDE by 1000, not multiply.
dH = -slope * R / 1000.0
dS = intercept * R

dG298 = dH - 298.0 * dS

# Convert to standard reporting units (kcal/mol and cal/mol/K)
dH_report = dH 
dS_report = dS * 1000.0

# ============================================================
# 5. GENERATE TERMINAL PRINTS & LOG REPORTS
# ============================================================
print("\n" + "="*60)
print("              TRUE VAN'T HOFF THERMODYNAMIC VALUES")
print("="*60)
print(f"  dH0      = {dH_report:.3f} kcal/mol")
print(f"  dS0      = {dS_report:.3f} cal/mol/K")
print(f"  dG0(298) = {dG298:.3f} kcal/mol")
print("="*60)

with open(TXT_OUT, "w") as f:
    f.write("="*60 + "\n")
    f.write("VAN'T HOFF MATRIX REGRESSION RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"600 MHz included temperatures: {FIT_TEMPS_600}\n")
    f.write(f"800 MHz included temperatures: {FIT_TEMPS_800}\n\n")
    f.write(f"dH0      = {dH_report:.4f} kcal/mol\n")
    f.write(f"dS0      = {dS_report:.4f} cal/mol/K\n")
    f.write(f"dG0(298) = {dG298:.4f} kcal/mol\n")

# ============================================================
# 6. RENDER VAN'T HOFF COORDINATE VISUALIZATIONS
# ============================================================
invT_plot = np.linspace(1000/315, 1000/284, 400)
lnK_fit = slope * invT_plot + intercept

fig, ax = plt.subplots(figsize=(8, 6.0))

# 600 MHz Traces (With propagated analytical fractional errors)
ax.errorbar(1000.0 / used600["Temp"], np.log(used600["pB_mean"] / (1.0 - used600["pB_mean"])),
            yerr=(used600["pB_err"] / (used600["pB_mean"] * (1.0 - used600["pB_mean"]))),
            fmt='o', color=COLOR_600, ms=11, lw=2, capsize=4, label="600 MHz")

ax.errorbar(1000.0 / excl600["Temp"], np.log(excl600["pB_mean"] / (1.0 - excl600["pB_mean"])),
            yerr=(excl600["pB_err"] / (excl600["pB_mean"] * (1.0 - excl600["pB_mean"]))),
            fmt='o', markerfacecolor='none', markeredgecolor=COLOR_600, markeredgewidth=2,
            ecolor=COLOR_600, ms=9, lw=2, capsize=4)

# 800 MHz Traces (With propagated analytical fractional errors)
ax.errorbar(1000.0 / used800["Temp"], np.log(used800["pB_mean"] / (1.0 - used800["pB_mean"])),
            yerr=(used800["pB_err"] / (used800["pB_mean"] * (1.0 - used800["pB_mean"]))),
            fmt='o', color=COLOR_800, ms=11, lw=2, capsize=4, label="800 MHz")

ax.errorbar(1000.0 / excl800["Temp"], np.log(excl800["pB_mean"] / (1.0 - excl800["pB_mean"])),
            yerr=(excl800["pB_err"] / (excl800["pB_mean"] * (1.0 - excl800["pB_mean"]))),
            fmt='o', markerfacecolor='none', markeredgecolor=COLOR_800, markeredgewidth=2,
            ecolor=COLOR_800, ms=9, lw=2, capsize=4)

# Projected Linear Regression Line
ax.plot(invT_plot, lnK_fit, '--', color=FIT_COLOR, lw=2, label="van't Hoff fit")

# Formatting Layout
ax.set_xlabel(r"1000 / T (K$^{-1}$)", labelpad=10)
ax.set_ylabel(r"ln($K_{eq}$)", labelpad=10)
ax.set_xlim(3.15, 3.55)
ax.set_xticks([3.2, 3.3, 3.4, 3.5])
ax.set_ylim(-0.9, -0.25)
ax.set_yticks([-0.8, -0.6, -0.4, -0.2])

# Custom tick sizing requested
ax.tick_params(axis='both', labelsize=18)

# FIXED: Removed background grid, stripped top/right bounding frames for crisp look
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc="upper right")

plt.tight_layout()

# Save High-End Formats
PDF_OUT = os.path.join(FIGURE_DIR, "04_vantHoff_plot.pdf")
PNG_OUT = os.path.join(FIGURE_DIR, "04_vantHoff_plot.png")
plt.savefig(PDF_OUT, bbox_inches='tight')
plt.savefig(PNG_OUT, dpi=300, bbox_inches='tight')

print(f"✅ Van't Hoff regression plots successfully exported:\n  👉 {PDF_OUT}\n  👉 {PNG_OUT}\n")
plt.close()
