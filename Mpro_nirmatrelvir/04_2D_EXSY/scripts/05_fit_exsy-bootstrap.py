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

# Significantly bumped font sizes for high-visibility downscaling
plt.rcParams.update({
    'font.size': 16,          # Base font size boosted
    'axes.titlesize': 22,     # Subplot titles
    'axes.labelsize': 18,     # X and Y labels
    'xtick.labelsize': 16,    # X tick markers
    'ytick.labelsize': 16,    # Y tick markers
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../figures"))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../output"))

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(SCRIPT_DIR, "../processed_data/exsy_peak_intensities.csv")

INITIAL_KEX = 50.0
N_BOOT = 500
RANDOM_SEED = 1234

FIT_WINDOWS = {
    288: (30, 150),
    298: (10, 150),
    308: (5, 150)
}

np.random.seed(RANDOM_SEED)

# ============================================================
# READ DATA
# ============================================================

raw = pd.read_csv(CSV_FILE)
raw.columns = raw.columns.str.strip()

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

def fit_dataset(df_fit):
    fit = least_squares(
        residuals, x0=[INITIAL_KEX, 1.0, 1.0],
        bounds=([0.01, 0.1, 0.1], [5000, 5.0, 5.0]), args=(df_fit,)
    )
    return fit.x

# ============================================================
# PASS 1: RUN BOOTSTRAP PIPELINES
# ============================================================
fit_cache = {}

for T, dfT in raw.groupby("T"):
    dfT = dfT.copy()
    tmin, tmax = FIT_WINDOWS[T]
    df_fit = dfT[((dfT["tmix_ms"] >= tmin) & (dfT["tmix_ms"] <= tmax))].copy()

    best_fit = fit_dataset(df_fit)
    best_kex, best_SAB, best_SBA = best_fit[0], best_fit[1], best_fit[2]

    orig_rAB_pred, orig_rBA_pred = [], []
    orig_rAB_exp, orig_rBA_exp = [], []

    for _, row in df_fit.iterrows():
        rAB_mod, rBA_mod = simulate_exsy_ratios(row["tmix_ms"] / 1000.0, best_kex, row["pB"], row["T1A"], row["T1B"])
        orig_rAB_pred.append(best_SAB * rAB_mod)
        orig_rBA_pred.append(best_SBA * rBA_mod)
        orig_rAB_exp.append(row["h_AB"] / (row["h_AA"] + row["h_AB"]))
        orig_rBA_exp.append(row["h_BA"] / (row["h_BB"] + row["h_BA"]))

    res_AB = np.array(orig_rAB_exp) - np.array(orig_rAB_pred)
    res_BA = np.array(orig_rBA_exp) - np.array(orig_rBA_pred)

    boot_kex = []
    for i in range(N_BOOT):
        boot_res_AB = np.random.choice(res_AB, size=len(res_AB), replace=True)
        boot_res_BA = np.random.choice(res_BA, size=len(res_BA), replace=True)

        syn_rAB = np.array(orig_rAB_pred) + boot_res_AB
        syn_rBA = np.array(orig_rBA_pred) + boot_res_BA

        df_boot = df_fit.copy()
        df_boot["h_AA"], df_boot["h_AB"] = 1.0 - syn_rAB, syn_rAB
        df_boot["h_BB"], df_boot["h_BA"] = 1.0 - syn_rBA, syn_rBA

        try:
            fit_boot = fit_dataset(df_boot)
            boot_kex.append(fit_boot[0])
        except:
            continue

    fit_cache[T] = (best_kex, best_SAB, best_SBA, np.array(boot_kex))

# ============================================================
# PASS 2: GENERATE INDIVIDUAL PLOTS (4x WIDER ASPECT RATIO)
# ============================================================
results = []
log_output = []

# Fixed figure dimensions to guarantee a wide, short layout
FIG_WIDTH = 6.8   # Stretched out to be about 4x wider than the skinny version
FIG_HEIGHT = 4.8  # Kept short to maximize the horizontal look

for T in sorted(fit_cache.keys()):
    best_kex, best_SAB, best_SBA, boot_kex = fit_cache[T]
    
    mean_kex = np.mean(boot_kex)
    median_kex = np.median(boot_kex)
    ci16, ci84 = np.percentile(boot_kex, 16), np.percentile(boot_kex, 84)
    ci025, ci975 = np.percentile(boot_kex, 2.5), np.percentile(boot_kex, 97.5)

    results.append({
        "T": T, "best_kex": best_kex, "mean_kex": mean_kex, "median_kex": median_kex,
        "CI16": ci16, "CI84": ci84, "CI2.5": ci025, "CI97.5": ci975,
        "best_SAB": best_SAB, "best_SBA": best_SBA
    })

    # Individual local padding limits 
    kex_range = boot_kex.max() - boot_kex.min()
    xlim_local = (boot_kex.min() - (kex_range * 0.20), boot_kex.max() + (kex_range * 0.20))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Histogram profile
    counts, bins, patches = ax.hist(boot_kex, bins=25, alpha=0.7, color="C0", edgecolor="k", zorder=2)
    max_count = max(counts)

    # Vertical Indicators
    ax.axvline(best_kex, color="red", linestyle="-", linewidth=3)
    ax.axvline(median_kex, color="black", linestyle="--", linewidth=3)
    ax.axvline(ci16, color="gray", linestyle=":", linewidth=1.5)
    ax.axvline(ci84, color="gray", linestyle=":", linewidth=1.5)

    # --- 95% CONFIDENCE INTERVAL RANGE VISUALIZATION ---
    ax.axvspan(ci025, ci975, ymin=0.0, ymax=0.06, color="gray", alpha=0.25, zorder=3)
    
    # Numeric 95% boundaries positioned at the base line
    ax.text(ci025, max_count * 0.08, f' {ci025:.1f}', color='dimgray', fontsize=12, ha='right', va='bottom')
    ax.text(ci975, max_count * 0.08, f'{ci975:.1f} ', color='dimgray', fontsize=12, ha='left', va='bottom')

    # Text callout annotation data block
    text_annotation = f"Best Fit: {best_kex:.1f}   Median: {median_kex:.1f}   95% CI: [{ci025:.1f}, {ci975:.1f}]"
    ax.text(0.03, 0.92, text_annotation, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='none'))

    # Display attributes
    ax.set_title(f"{T} K", pad=14)
    ax.set_xlabel("$k_{ex}$ (s$^{-1}$)", labelpad=8)
    ax.set_ylabel("Counts", labelpad=8)
    
    ax.set_xlim(xlim_local)
    ax.set_ylim(0, max_count * 1.25) # Leave a ceiling margin for the text header string
    
    # Precise control over tick density
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) # Clean, wide x-intervals
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2)) # Enforces exactly 3 labeled tick marks (Min, Mid, Max)

    plt.tight_layout()

    fig_path = os.path.join(FIGURE_DIR, f"05_bootstrap_{T}K.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close(fig)

    np.savetxt(f"{FIGURE_DIR}/bootstrap_values_{T}K.txt", boot_kex)

# ============================================================
# EXPORT METADATA LOGS
# ============================================================
results = pd.DataFrame(results)
CSV_OUTFILE = os.path.join(OUTPUT_DIR, "05_EXSY_bootstrap_results.csv")
results.to_csv(CSV_OUTFILE, index=False)
print(f"\nSaved master datasets to: {CSV_OUTFILE}")
