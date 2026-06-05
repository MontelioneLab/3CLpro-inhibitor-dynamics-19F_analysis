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
TXT_OUT = os.path.join(OUTPUT_DIR, "03b_linewidth_fit_equations.txt")

# ============================================================
# 2. CONFIGURATIONS & SETTINGS
# ============================================================
LB = 10.0  # Line broadening subtracted to isolate true intrinsic linewidths
FIT_TEMPS_600 = [285, 288, 291, 293, 296]
FIT_TEMPS_800 = [285, 288, 293, 298]

COLOR_A = "black"
COLOR_B = "#4CAF50"  # Vibrant forest green
USE_SEM = False

# ============================================================
# PLOT STYLE SETTINGS
# ============================================================
plt.rcParams.update({
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ============================================================
# 3. DATA LOADING & EXTRACTION
# ============================================================
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: {CSV_FILE} not found. Run your Lorentzian fitter script first.")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)

# Aggregate replicate sets smoothly
summary = df.groupby(["Field", "Temp"], as_index=False).agg({
    "LW_A_Hz": ["mean", "std", "count"],
    "LW_B_Hz": ["mean", "std", "count"]
})
summary.columns = ["Field", "Temp", "LWA_mean", "LWA_std", "nA", "LWB_mean", "LWB_std", "nB"]

# Calculate error vectors
summary["LWA_sem"] = summary["LWA_std"] / np.sqrt(summary["nA"])
summary["LWB_sem"] = summary["LWB_std"] / np.sqrt(summary["nB"])
summary["LWA_err"] = summary["LWA_sem"] if USE_SEM else summary["LWA_std"]
summary["LWB_err"] = summary["LWB_sem"] if USE_SEM else summary["LWB_std"]

# Map active inclusion masks
summary["used"] = False
summary.loc[((summary["Field"] == 600) & (summary["Temp"].isin(FIT_TEMPS_600))), "used"] = True
summary.loc[((summary["Field"] == 800) & (summary["Temp"].isin(FIT_TEMPS_800))), "used"] = True

# ============================================================
# 4. POLYNOMIAL REGRESSION FUNCTION
# ============================================================
def fit_line(T, y):
    x = T - 298.0
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0], coeffs[1]  # returns slope, LW0

fit600 = summary[(summary["Field"] == 600) & (summary["used"])]
fit800 = summary[(summary["Field"] == 800) & (summary["used"])]

results = {}
for field, dset in zip([600, 800], [fit600, fit800]):
    T = dset["Temp"].values
    slopeA, LWA0 = fit_line(T, dset["LWA_mean"].values - LB)
    slopeB, LWB0 = fit_line(T, dset["LWB_mean"].values - LB)
    
    results[field] = {
        "slopeA": slopeA, "LWA0": LWA0,
        "slopeB": slopeB, "LWB0": LWB0
    }

# ============================================================
# 5. GENERATE TERMINAL PRINTS & LOGS
# ============================================================
print("\n" + "="*60)
print("              INTRINSIC LINEWIDTH(T) SLOPES")
print("="*60)
for field in [600, 800]:
    r = results[field]
    print(f"\n=== {field} MHz (Low-T Baseline Fit) ===")
    print(f"  LW_A(T) = {r['LWA0']:.2f} + ({r['slopeA']:.3f}) * (T-298)")
    print(f"  LW_B(T) = {r['LWB0']:.2f} + ({r['slopeB']:.3f}) * (T-298)")
print("="*60)

with open(TXT_OUT, "w") as f:
    f.write("="*60 + "\n")
    f.write("INTRINSIC LINEWIDTH(T) FITS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Error bar layout = {'SEM' if USE_SEM else 'SD'}\n\n")
    for field in [600, 800]:
        r = results[field]
        f.write(f"=== {field} MHz ===\n")
        f.write(f"dLW_A/dT = {r['slopeA']:.4f} Hz/K | Intercept at 298K = {r['LWA0']:.4f} Hz\n")
        f.write(f"dLW_B/dT = {r['slopeB']:.4f} Hz/K | Intercept at 298K = {r['LWB0']:.4f} Hz\n\n")
        f.write(f"=== {field} MHz (Low-T Baseline Fit) ===\n")
        f.write(f"  LW_A(T) = {r['LWA0']:.4f} + ({r['slopeA']:.4f}) * (T-298)\n")
        f.write(f"  LW_B(T) = {r['LWB0']:.4f} + ({r['slopeB']:.4f}) * (T-298)\n\n")
    f.write("="*60 + "\n")

# ============================================================
# 6. RENDER DUAL-PANEL PLOTS
# ============================================================
T_plot = np.linspace(284, 316, 400)
used600 = summary[(summary["Field"] == 600) & (summary["used"])]
excl600 = summary[(summary["Field"] == 600) & (~summary["used"])]
used800 = summary[(summary["Field"] == 800) & (summary["used"])]
excl800 = summary[(summary["Field"] == 800) & (~summary["used"])]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

for ax, field, used, excl in zip(axes, [600, 800], [used600, used800], [excl600, excl800]):
    r = results[field]
    fitA = r["LWA0"] + r["slopeA"] * (T_plot - 298.0)
    fitB = r["LWB0"] + r["slopeB"] * (T_plot - 298.0)

    # Plot Included Points (Correctly centered via LB subtraction)
    ax.errorbar(used["Temp"], used["LWA_mean"] - LB, yerr=used["LWA_err"],
                fmt='o', color=COLOR_A, ms=10, lw=2, capsize=4, label="A")
    ax.errorbar(used["Temp"], used["LWB_mean"] - LB, yerr=used["LWB_err"],
                fmt='o', color=COLOR_B, ms=10, lw=2, capsize=4, label="B")

    # FIXED: Applied the LB line-broadening subtraction here as well to fix the 10 Hz jumping artifact
    ax.errorbar(excl["Temp"], excl["LWA_mean"] - LB, yerr=excl["LWA_err"],
                fmt='o', markerfacecolor='none', markeredgecolor=COLOR_A,
                markeredgewidth=2, ecolor=COLOR_A, ms=10, lw=2, capsize=4)
    ax.errorbar(excl["Temp"], excl["LWB_mean"] - LB, yerr=excl["LWB_err"],
                fmt='o', markerfacecolor='none', markeredgecolor=COLOR_B,
                markeredgewidth=2, ecolor=COLOR_B, ms=10, lw=2, capsize=4)

    # Plot Linear Extrapolations
    ax.plot(T_plot, fitA, '--', color=COLOR_A, lw=3)
    ax.plot(T_plot, fitB, '--', color=COLOR_B, lw=3)

    # Panel Customizations
    ax.set_title(f"{field} MHz", fontsize=20, fontweight='bold', pad=10)
    ax.set_xlabel("Temperature (K)", labelpad=8)
    ax.set_xlim(284, 316)
    
    # FIXED: Replaced standard background grids with crisp journal spine frames
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel("Intrinsic Linewidth (Hz)", labelpad=10)
axes[1].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.05, 0.05))
plt.tight_layout()

# Save High-Resolution Pipeline Vectors
PDF_OUT = os.path.join(FIGURE_DIR, "03b_linewidth_vs_temperature.pdf")
PNG_OUT = os.path.join(FIGURE_DIR, "03b_linewidth_vs_temperature.png")
plt.savefig(PDF_OUT, bbox_inches='tight')
plt.savefig(PNG_OUT, dpi=300, bbox_inches='tight')

print(f"✅ Multi-field linewidth plots successfully exported:\n  👉 {PDF_OUT}\n  👉 {PNG_OUT}\n")
plt.close()
