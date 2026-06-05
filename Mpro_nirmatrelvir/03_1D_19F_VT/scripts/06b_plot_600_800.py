#!/usr/bin/env python3

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# STYLE & GLOBAL FONTS
# ============================================================
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["pdf.fonttype"] = 42

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "processed_ascii")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

DIR_600 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_600")
DIR_800 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_800")
FIT_NPZ = os.path.join(OUTPUT_DIR, "06_vt_dual_fit_HYBRID.npz")

os.makedirs(FIGURE_DIR, exist_ok=True)

# ============================================================
# SETTINGS & CONSTANTS
# ============================================================
TEMPS_TO_PLOT = [288, 293, 298, 303, 308, 313]
STACK_OFFSET = 1.15
FIT_WINDOW = 2400
LB = 10.0
SIM_COLOR = "#D50032"  

# ============================================================
# FIXED FIELD VALUES & EXTRACTIONS
# ============================================================
slopeA_600 = -1.521
slopeB_600 = -0.976
slopeA_800 = -3.680
slopeB_800 = -2.158

if not os.path.exists(FIT_NPZ):
    print(f"❌ Error: {FIT_NPZ} not found. Run your global hybrid fit script first.")
    sys.exit(1)

fit = np.load(FIT_NPZ, allow_pickle=True)

DH = float(fit["DH"])
DS = float(fit["DS"]) 
dnu_slope = float(fit["dnu_slope"])
dnu_intercept = float(fit["dnu_intercept"])

# Recover structured parameters from core coordinates array scope safely
xfit_coor = fit["xfit"]
amp600   = float(xfit_coor[0])
amp800   = float(xfit_coor[1])
LWA0_600 = float(xfit_coor[2])
LWB0_600 = float(xfit_coor[3])
LWA0_800 = float(xfit_coor[4])
LWB0_800 = float(xfit_coor[5])

DG298 = float(fit["DG298"])
DHdd  = float(fit["DHdd"])
DSdd  = (DHdd - DG298) / 298.0

print("\n" + "="*50)
print("     ACTIVATION ARCHIVE RECOVERY METRICS")
print("="*50)
print(f"  DG#(298) = {DG298:.3f} kcal/mol")
print(f"  DH#      = {DHdd:.3f} kcal/mol")
print(f"  DS#      = {DSdd*1000:.3f} cal/mol/K\n")

# ============================================================
# METRIC LOOPS
# ============================================================
kB = 1.380649e-23
h  = 6.62607015e-34
R_kcal = 0.001987207

def get_kex(T):
    DGdd_T = DHdd - T * DSdd
    return (kB * T / h) * np.exp(-DGdd_T / (R_kcal * T))

def get_pB(T):
    dG = DH - T * (DS / 1000.0)
    K = np.exp(-dG / (R_kcal * T))
    return K / (1.0 + K)

def dnu600_linear(T):
    return dnu_intercept + dnu_slope * (T - 298.0)

def linewidth_T(T, lw0, slope):
    return lw0 + slope * (T - 298.0)

def dnmr(f, kex, center, dnu, r2a, r2b, pB):
    pi = np.pi
    pA = 1.0 - pB
    wa = 2 * pi * center
    wb = 2 * pi * (center + dnu)
    w  = 2 * pi * f

    ka = kex * pB
    kb = kex * pA

    Ga = r2a * pi + 1j * (wa - w) + ka
    Gb = r2b * pi + 1j * (wb - w) + kb

    num = pA * (Gb + ka) + pB * (Ga + kb)
    den = Ga * Gb - ka * kb
    return np.real(num / den)

# ============================================================
# LOAD SPECTRA
# ============================================================
def load_spectra(directory):
    spectra = {}
    files = sorted(glob.glob(os.path.join(directory, "spec_*K_LB10.dat")))

    for f in files:
        name = os.path.basename(f)
        try:
            temp_str = name.split("_")[1].replace("K", "")
            T = int(temp_str)
        except (IndexError, ValueError):
            continue

        if T in TEMPS_TO_PLOT:
            data = np.loadtxt(f)
            freq = data[:, 0]
            y = data[:, 1]

            mask = np.abs(freq) <= FIT_WINDOW / 2.0
            freq = freq[mask]
            y = y[mask]
            
            if np.max(y) > 0:
                y /= np.max(y)

            spectra[T] = (freq, y)

    return spectra

spec600 = load_spectra(DIR_600)
spec800 = load_spectra(DIR_800)

# ============================================================
# MULTI-PANEL PLOTTING ENGINE
# ============================================================
def plot_stack(ax, spectra, field="600"):
    temps = sorted(spectra.keys())
    
    freq_600_sys = 564.5312351
    freq_800_sys = 752.8281062
    field_ratio = freq_800_sys / freq_600_sys

    for i, T in enumerate(temps):
        freq, y_exp = spectra[T]

        lwa_600 = linewidth_T(T, LWA0_600, slopeA_600)
        lwb_600 = linewidth_T(T, LWB0_600, slopeB_600)
        lwa_800 = linewidth_T(T, LWA0_800, slopeA_800)
        lwb_800 = linewidth_T(T, LWB0_800, slopeB_800)

        kex = get_kex(T)
        pB = get_pB(T)

        if field == "600":
            dnu = -dnu600_linear(T)
            lwa_field = lwa_600
            lwb_field = lwb_600
            amp = amp600
        elif field == "800":
            dnu = -(dnu600_linear(T) * field_ratio)
            lwa_field = lwa_800
            lwb_field = lwb_800
            amp = amp800

        y_sim = dnmr(freq, kex, 0.0, dnu, lwa_field + LB, lwb_field + LB, pB) * amp
        
        if np.max(np.abs(y_sim)) > 0:
            y_sim /= np.max(np.abs(y_sim))

        y_shift = i * STACK_OFFSET

        ax.plot(freq, y_exp + y_shift, color="black", lw=1.7)
        ax.plot(freq, y_sim + y_shift, color=SIM_COLOR, lw=2.0, ls="--")

        ax.text(1100, y_shift + 0.15, f"{T} K", fontsize=14, ha="right")
        print(f"  [{field} MHz]  {T} K  -->  kex = {kex:7.1f} s^-1  |  pB = {pB:.3f}")

    ax.set_yticks([])
    ax.set_xlim(1200, -1200)  
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

# ============================================================
# GENERATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

print("="*50)
print("     GENERATING GRAPHICAL BLOCH-MCCONNELL OVERLAYS")
print("="*50)
plot_stack(axes[0], spec600, field="600")
plot_stack(axes[1], spec800, field="800")
print("="*50)

axes[0].set_title("600 MHz", fontsize=24, pad=10)
axes[1].set_title("800 MHz", fontsize=24, pad=10)

for ax in axes:
    ax.set_xlabel("Frequency (Hz)", labelpad=10)
    ax.spines[['top', 'right', 'left']].set_visible(False)

plt.tight_layout()

png_out = os.path.join(FIGURE_DIR, "06_dualfield_600_800_fit.png")
pdf_out = os.path.join(FIGURE_DIR, "06_dualfield_600_800_fit.pdf")

plt.savefig(png_out, dpi=300, bbox_inches="tight")
plt.savefig(pdf_out, bbox_inches="tight")

print(f"✅ Side-by-side lineshape figures successfully written:")
print(f"  👉 {png_out}\n  👉 {pdf_out}\n")

if "DISPLAY" in os.environ or os.name == 'nt' or sys.platform == 'darwin':
    plt.show()
else:
    plt.close()
