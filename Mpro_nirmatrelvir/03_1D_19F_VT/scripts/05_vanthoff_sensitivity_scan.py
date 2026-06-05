#!/usr/bin/env python3

import os
import sys
import itertools
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
CSV_OUT = os.path.join(OUTPUT_DIR, "05_equilibrium_subset_scan.csv")
TXT_OUT = os.path.join(OUTPUT_DIR, "05_equilibrium_uncertainties_summary.txt")

# ============================================================
# 2. CONFIGURATIONS & SETTINGS
# ============================================================
R = 0.0019872      # Gas constant in kcal/mol/K
T_REF = 298.0
MIN_POINTS = 4
RSS_FACTOR = 1.15  # Acceptance threshold coefficient

# Trusted low-exchange temperatures
TRUSTED_600 = [285, 288, 291, 293, 296]
TRUSTED_800 = [285, 288, 293, 298]

# Optional higher-T points used for combinatoric sensitivity scan
OPTIONAL_600 = [298, 303, 308]
OPTIONAL_800 = [303]

# ============================================================
# PLOT STYLE SETTINGS
# ============================================================
plt.rcParams.update({
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "axes.titlesize": 18,
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

# Aggregate replicate arrays cleanly
summary = df.groupby(["Field", "Temp"], as_index=False).agg({"pB": ["mean", "std", "count"]})
summary.columns = ["Field", "Temp", "pB_mean", "pB_std", "n"]

df600 = summary[summary["Field"] == 600].copy()
df800 = summary[summary["Field"] == 800].copy()

# ============================================================
# 4. CORE THERMODYNAMIC FITTING ENGINE
# ============================================================
def vant_hoff_fit(T, pB):
    Keq = pB / (1.0 - pB)
    lnK = np.log(Keq)
    invT = 1.0 / T

    coeffs = np.polyfit(invT, lnK, 1)
    slope, intercept = coeffs

    dH = -slope * R
    dS = intercept * R * 1000.0
    dG298 = dH - T_REF * dS / 1000.0

    Keq298 = np.exp(-dG298 / (R * T_REF))
    pB298 = Keq298 / (1.0 + Keq298)

    lnK_fit = slope * invT + intercept
    rss = np.sum((lnK - lnK_fit)**2)

    return {"dH": dH, "dS": dS, "dG298": dG298, "pB298": pB298, "rss": rss}

# ============================================================
# 5. EXECUTE COMBINATORIC SUBSET SCAN
# ============================================================
optional600_sets = []
for n in range(len(OPTIONAL_600) + 1):
    optional600_sets.extend(itertools.combinations(OPTIONAL_600, n))

optional800_sets = []
for n in range(len(OPTIONAL_800) + 1):
    optional800_sets.extend(itertools.combinations(OPTIONAL_800, n))

results = []
for opt600 in optional600_sets:
    for opt800 in optional800_sets:
        temps600 = sorted(TRUSTED_600 + list(opt600))
        temps800 = sorted(TRUSTED_800 + list(opt800))

        sel600 = df600[df600["Temp"].isin(temps600)]
        sel800 = df800[df800["Temp"].isin(temps800)]

        combined = pd.concat([sel600, sel800])
        T = combined["Temp"].values
        pB = combined["pB_mean"].values

        if len(T) < MIN_POINTS:
            continue

        try:
            fit = vant_hoff_fit(T, pB)
            fit["temps600"] = temps600
            fit["temps800"] = temps800
            fit["npts"] = len(T)
            results.append(fit)
        except Exception:
            continue

results_df = pd.DataFrame(results)

# Apply RSS acceptance filtering
rss_min = results_df["rss"].min()
rss_cutoff = RSS_FACTOR * rss_min
results_df["accepted"] = results_df["rss"] <= rss_cutoff
accepted = results_df[results_df["accepted"]].copy()

# Export raw matrix log to disk
results_df.to_csv(CSV_OUT, index=False)

# ============================================================
# 6. CALCULATE COMPREHENSIVE STATISTICAL INTERVALS
# ============================================================
def summarize(param):
    vals = accepted[param].values
    return {
        "median": np.median(vals),
        "low68": np.percentile(vals, 16), "high68": np.percentile(vals, 84),
        "low95": np.percentile(vals, 2.5), "high95": np.percentile(vals, 97.5)
    }

stats_dH = summarize("dH")
stats_dS = summarize("dS")
stats_dG = summarize("dG298")
stats_pB = summarize("pB298")

# ============================================================
# 7. WRITE SYSTEM LOGS & REPORTS
# ============================================================
print("\n" + "="*60)
print("              NIRMATRELVIR SENSITIVITY SCAN RESULTS")
print("="*60)
print(f"  Total Fits Evaluated: {len(results_df)}")
print(f"  Accepted Fits Matrix: {len(accepted)}")
print(f"  RSS Global Minimum  : {rss_min:.4e}")
print(f"  RSS Cutoff Cap Bound: {rss_cutoff:.4e}")
print("="*60)

for label, stats in zip(["dH0", "dS0", "dG0(298)", "pB(298)"], [stats_dH, stats_dS, stats_dG, stats_pB]):
    print(f"  {label:<10} Median: {stats['median']:.3f} | 68% CI: {stats['low68']:.3f} to {stats['high68']:.3f}")

with open(TXT_OUT, "w") as f:
    f.write("="*60 + "\n")
    f.write("EQUILIBRIUM SUBSET SENSITIVITY SCAN SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"RSS minimum : {rss_min:.6e}\n")
    f.write(f"RSS cutoff  : {rss_cutoff:.6e}\n\n")

    for label, stats in zip(["dH0 (kcal/mol)", "dS0 (cal/mol/K)", "dG0 (298 K; kcal/mol)", "pB (298 K)"], 
                           [stats_dH, stats_dS, stats_dG, stats_pB]):
        f.write(f"=== {label} ===\n")
        f.write(f"  median : {stats['median']:.4f}\n")
        f.write(f"  68% CI : {stats['low68']:.4f} to {stats['high68']:.4f}\n")
        f.write(f"  95% CI : {stats['low95']:.4f} to {stats['high95']:.4f}\n\n")

# ============================================================
# 8. RENDER REPOSITORY MANIFOLD VISUALIZATIONS
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

# Render background combinations that were rejected due to high RSS variance
ax.scatter(results_df["dH"], results_df["dS"], color="lightgray", s=35, alpha=0.5, label="Rejected Subsets")

# Render thermodynamically accepted compensation models colored by free energy stability
sc = ax.scatter(accepted["dH"], accepted["dS"], c=accepted["dG298"], cmap="viridis",
                edgecolor="black", linewidth=0.5, s=70, zorder=3, label="Accepted Subsets")

# Configure Colorbar Metrics
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"$\Delta$G°(298) (kcal/mol)", fontsize=18)
cbar.ax.tick_params(labelsize=16)
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# FIXED: Replaced hardcoded ticks with automated max locator to handle actual data ranges dynamically
cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))

# Label Customizations
ax.set_xlabel(r"$\Delta$H° (kcal/mol)", fontsize=18, labelpad=8)
ax.set_ylabel(r"$\Delta$S° (cal/mol/K)", fontsize=18, labelpad=8)
ax.set_title("Equilibrium Compensation Manifold", fontsize=16, fontweight='bold', pad=12)

# FIXED: Removed old grid matrix, stripped top/right bounding frames for uniform look
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

# Export Vector Assets
PDF1 = os.path.join(FIGURE_DIR, "05_equilibrium_compensation_manifold.pdf")
PNG1 = os.path.join(FIGURE_DIR, "05_equilibrium_compensation_manifold.png")
plt.savefig(PDF1, bbox_inches='tight')
plt.savefig(PNG1, dpi=300, bbox_inches='tight')

print(f"\n✅ Sensitivity assets successfully written out:")
print(f"  👉 {PDF1}\n  👉 {PNG1}")
print(f"📝 Tabulated metrics logged to: {TXT_OUT}\n")
plt.close()
