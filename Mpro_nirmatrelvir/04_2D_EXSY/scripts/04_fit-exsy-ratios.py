import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.linalg import expm
from scipy.optimize import least_squares

# ============================================================
# SETTINGS & GLOBAL PLOT CONFIGURATION
# ============================================================

# Set large fonts globally for publication-quality scaling
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 18,     # Subplot title font size (e.g., "288 K")
    'axes.labelsize': 16,     # X and Y axis labels
    'xtick.labelsize': 14,    # X tick marks
    'ytick.labelsize': 14,    # Y tick marks
    'legend.fontsize': 12     # Legend font size
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../figures"))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../output"))

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(SCRIPT_DIR, "../processed_data/exsy_peak_intensities.csv")

INITIAL_KEX = 50.0

FIT_WINDOWS = {
    288: (30, 150),
    298: (10, 150),
    308: (5, 150)
}

#Ordered list of temperatures for left-to-right column mapping
TEMPERATURES = [288, 298, 308]

# ============================================================
# READ DATA
# ============================================================

raw = pd.read_csv(CSV_FILE)
raw.columns = raw.columns.str.strip()

# ============================================================
# DYNAMICALLY DETERMINE UNIFIED AXIS LIMITS
# ============================================================

all_ratios = []
for T, dfT in raw.groupby("T"):
    r_AB = dfT["h_AB"] / (dfT["h_AA"] + dfT["h_AB"])
    r_BA = dfT["h_BA"] / (dfT["h_BB"] + dfT["h_BA"])
    all_ratios.extend(r_AB.tolist())
    all_ratios.extend(r_BA.tolist())

max_ratio = max(all_ratios)
YLIM_RATIO = (-0.01, max_ratio * 1.12)

# ============================================================
# MATRIX MODEL & RESIDUALS
# ============================================================

def simulate_exsy_ratios(tmix_s, kex, pB, T1A, T1B):
    pA = 1.0 - pB
    kAB, kBA = pB * kex, pA * kex
    R1A, R1B = 1.0 / T1A, 1.0 / T1B

    R = np.array([
        [-(R1A + kAB),  kBA],
        [ kAB,         -(R1B + kBA)]
    ])

    E = expm(R * tmix_s)
    M_A = E @ np.array([pA, 0.0])
    M_B = E @ np.array([0.0, pB])

    return M_B[0] / (M_A[0] + M_B[0]), M_A[1] / (M_B[1] + M_A[1])

def residuals(params, df):
    kex, SAB, SBA = params[0], params[1], params[2]
    res = []
    for _, row in df.iterrows():
        rAB_model, rBA_model = simulate_exsy_ratios(row["tmix_ms"] / 1000.0, kex, row["pB"], row["T1A"], row["T1B"])
        res.extend([
            (SAB * rAB_model) - (row["h_AB"] / (row["h_AA"] + row["h_AB"])),
            (SBA * rBA_model) - (row["h_BA"] / (row["h_BB"] + row["h_BA"]))
        ])
    return np.array(res)

# ============================================================
# PASS 1: RUN FITS TO COLLECT GLOBAL RESIDUALS
# ============================================================
all_residuals = []
fit_results_cache = {}

for T, dfT in raw.groupby("T"):
    dfT = dfT.copy()
    tmin, tmax = FIT_WINDOWS[T]
    fit_mask = ((dfT["tmix_ms"] >= tmin) & (dfT["tmix_ms"] <= tmax))
    df_fit = dfT[fit_mask].copy()

    fit = least_squares(
        residuals, x0=[INITIAL_KEX, 1.0, 1.0],
        bounds=([0.01, 0.1, 0.1], [5000, 5.0, 5.0]), args=(df_fit,)
    )
    fit_results_cache[T] = (fit, df_fit, dfT[~fit_mask].copy())
    all_residuals.extend(residuals(fit.x, df_fit).tolist())

max_res_offset = max(abs(min(all_residuals)), abs(max(all_residuals))) * 1.65
YLIM_RESIDUAL = (-max_res_offset, max_res_offset)

# ============================================================
# PASS 2: GENERATE COMBINED 3-COLUMN MULTI-PANEL FIGURE
# ============================================================

# ============================================================
# PASS 2: GENERATE COMBINED 3-COLUMN MULTI-PANEL FIGURE
# ============================================================

# Adjusted figsize: 20% wider (15 -> 18) and 20% shorter (9 -> 7.2)
fig, axes = plt.subplots(2, 3, figsize=(18, 7.2), sharex='col', sharey='row')
results_list = []

# Map individual panels from grid: axes[row, col]
for col_idx, T in enumerate(TEMPERATURES):
    if T not in fit_results_cache:
        continue
        
    fit, df_fit, df_excluded = fit_results_cache[T]
    kex_fit, SAB_fit, SBA_fit = fit.x[0], fit.x[1], fit.x[2]
    
    results_list.append({"T": T, "kex": kex_fit, "SAB": SAB_fit, "SBA": SBA_fit})

    ax1 = axes[0, col_idx]  # Top Panel (Fits)
    ax2 = axes[1, col_idx]  # Bottom Panel (Residuals)

    # 1. Smooth fit curves
    tmix_plot = np.linspace(0, pd.concat([df_fit, df_excluded])["tmix_ms"].max(), 400)
    ratio_AB_fit, ratio_BA_fit = [], []
    pB, T1A, T1B = df_fit.iloc[0]["pB"], df_fit.iloc[0]["T1A"], df_fit.iloc[0]["T1B"]

    for tm in tmix_plot:
        rAB, rBA = simulate_exsy_ratios(tm / 1000.0, kex_fit, pB, T1A, T1B)
        ratio_AB_fit.append(SAB_fit * rAB)
        ratio_BA_fit.append(SBA_fit * rBA)

    # 2. Compute scatter points
    ratio_AB_fitpts = df_fit["h_AB"] / (df_fit["h_AA"] + df_fit["h_AB"])
    ratio_BA_fitpts = df_fit["h_BA"] / (df_fit["h_BB"] + df_fit["h_BA"])
    ratio_AB_excluded = df_excluded["h_AB"] / (df_excluded["h_AA"] + df_excluded["h_AB"])
    ratio_BA_excluded = df_excluded["h_BA"] / (df_excluded["h_BB"] + df_excluded["h_BA"])

    # --- TOP PANEL RENDERING ---
    # Restored descriptive ratio strings to the legend labels
    ax1.plot(tmix_plot, ratio_AB_fit, linewidth=2.5, color="C0", label="AB/(AA+AB) fit")
    ax1.plot(tmix_plot, ratio_BA_fit, linewidth=2.5, color="C1", label="BA/(BB+BA) fit")
    ax1.scatter(df_fit["tmix_ms"], ratio_AB_fitpts, s=70, color="C0", edgecolors='k', zorder=3)
    ax1.scatter(df_fit["tmix_ms"], ratio_BA_fitpts, s=70, color="C1", edgecolors='k', zorder=3)
    ax1.scatter(df_excluded["tmix_ms"], ratio_AB_excluded, s=80, facecolors="none", edgecolors="C0", linewidths=1.5)
    ax1.scatter(df_excluded["tmix_ms"], ratio_BA_excluded, s=80, facecolors="none", edgecolors="C1", linewidths=1.5)

    ax1.set_title(f"{T} K", pad=12)
    ax1.set_ylim(YLIM_RATIO)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    
    # Show y-label and legend only on the far left column panel
    if col_idx == 0:
        ax1.set_ylabel("Transfer ratio")
        ax1.legend(frameon=False, loc="upper left")

    # --- BOTTOM PANEL RENDERING ---
    res_vectors = residuals(fit.x, df_fit)
    residual_AB = res_vectors[0::2]
    residual_BA = res_vectors[1::2]

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.2)
    # Restored explicit residual tracking strings to the legend labels
    ax2.scatter(df_fit["tmix_ms"], -residual_AB, s=70, color="C0", edgecolors='k', zorder=3, label="AB residuals")
    ax2.scatter(df_fit["tmix_ms"], -residual_BA, s=70, color="C1", edgecolors='k', zorder=3, label="BA residuals")

    ax2.set_xlabel("Mixing time (ms)", labelpad=8)
    ax2.set_ylim(YLIM_RESIDUAL)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))

    if col_idx == 0:
        ax2.set_ylabel("Residual")
        ax2.legend(frameon=False, loc="upper left")

# Fine-tune spaces between subplots tightly to avoid giant blank gaps
plt.tight_layout(h_pad=1.0, w_pad=0.8)

# Save the unified composite graphic asset
fig_path = os.path.join(FIGURE_DIR, "exsy_combined_panels_fit.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"\nSaved combined panels figure to: {fig_path}")

plt.show()

# ============================================================
# FINAL DATA EXPORT
# ============================================================
results = pd.DataFrame(results_list)
CSV_OUTFILE = os.path.join(OUTPUT_DIR, "../output/04_EXSY_fit_results.csv")
results.to_csv(CSV_OUTFILE, index=False)
