import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import least_squares

# ============================================================
# 1. SETUP & PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Data and Output Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_ascii", "ensitrelvir_298K_LB20_long_600MHz")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# File Paths
HIGH_SN_FILE = os.path.join(DATA_DIR, "spec_298_long_LB20.dat")
TXT_OUTPUT = os.path.join(OUTPUT_DIR, "high_res_validation_params.txt")
FIG_OUTPUT = os.path.join(FIGURE_DIR, "high_res_validation_plot.png")

# Constants for Targeted Zeroing & Sharpening
MIDDLE_PEAK_CENTER = -5000.0
WINDOW_HALF_WIDTH = 2000.0
LB_VT = 50.0   
LB_NEW = 20.0  
LB_DIFF = LB_VT - LB_NEW # 30 Hz sharpening correction

# ============================================================
# 2. DATA LOADING & DYNAMIC ARCHIVE RETRIEVAL
# ============================================================
def load_and_targeted_zero(filename, center_hz):
    try:
        raw = np.loadtxt(filename)
        if np.max(np.abs(raw[:,0])) < 500.0: raw[:,0] *= 600.0 * 1e3
        mask = (raw[:,0] > center_hz - WINDOW_HALF_WIDTH) & \
               (raw[:,0] < center_hz + WINDOW_HALF_WIDTH)
        freq = raw[mask, 0] - center_hz
        inten = raw[mask, 1]
        
        # Noise floor calculation using 1700-2000 Hz window
        mask_clean = (freq > 1700.0) & (freq < 2000.0)
        noise_floor = np.median(inten[mask_clean]) if np.sum(mask_clean) > 5 else np.min(inten)
        inten -= noise_floor
        if np.max(inten) > 0: inten /= np.max(inten)
        return freq, inten
    except Exception as e:
        print(f"Error loading {filename}: {e}"); return None, None

if not os.path.exists(GLOBAL_NPZ):
    print(f"Error: {GLOBAL_NPZ} not found. Run Script 02 first."); exit()

# Pull dynamic parameters from archive
archive = np.load(GLOBAL_NPZ, allow_pickle=True)
T_vals = archive["T_vals"]
pB_vals = archive["pB_vals"]
export_data = archive["export_data"]

# Find the 298 K entry for validation
idx = (np.abs(T_vals - 298.0)).argmin()
dyn_pB  = pB_vals[idx]
dyn_wA  = export_data[idx]['Middle_wA']
dyn_wB  = export_data[idx]['Middle_wB']
dyn_sep = 1189.0 # Spectral separation constant

# ============================================================
# 3. PHYSICS & ALIGNMENT FIT
# ============================================================
def two_lorentzians_slope(freq, wA, wB, center, separation, pB, height, baseline, slope):
    center_A = center + (separation * pB)
    peak_A = (1.0 - pB) * (wA**2 / ((freq - center_A)**2 + wA**2))
    center_B = center - (separation * (1.0 - pB))
    peak_B = pB * (wB**2 / ((freq - center_B)**2 + wB**2))
    return height * (peak_A + peak_B) + baseline + (slope * freq)

freq, y = load_and_targeted_zero(HIGH_SN_FILE, MIDDLE_PEAK_CENTER)

if freq is not None:
    # Sharpen the VT parameters for 20 Hz high-res data
    pred_wA = dyn_wA - LB_DIFF
    pred_wB = dyn_wB - LB_DIFF

    # Alignment fit using archived population
    mask_fit = freq > -1000.0
    freq_fit = freq[mask_fit]; y_fit = y[mask_fit]

    def res_align(p):
        return two_lorentzians_slope(freq_fit, pred_wA, pred_wB, p[0], dyn_sep + p[3], dyn_pB, p[1], p[2], p[4]) - y_fit
        
    p0 = [0.0, 1.0, 0.0, 0.0, 0.0]
    bounds = ([-300, 0.5, -0.2, -100, -0.005], [300, 1.5, 0.2, 100, 0.005])
    fit = least_squares(res_align, p0, bounds=bounds)
    shift_p, h_p, base_p, sep_corr_p, slope_p = fit.x

    # ============================================================
    # 4. PLOTTING & SUMMARY OUTPUT
    # ============================================================
    y_pred = two_lorentzians_slope(freq, pred_wA, pred_wB, shift_p, dyn_sep + sep_corr_p, dyn_pB, h_p, base_p, slope_p)
    
    # Save Text Summary
    with open(TXT_OUTPUT, "w") as f:
        f.write("ENSITRELVIR HIGH-RES VALIDATION (DYNAMIC ARCHIVE)\n")
        f.write(f"Source: {os.path.basename(GLOBAL_NPZ)} | Temp: {T_vals[idx]:.1f} K\n")
        f.write("-" * 55 + "\n")
        f.write(f"Population (pB): {dyn_pB*100:.2f}%\n")
        f.write(f"Archive wA/wB:  {dyn_wA:.1f} / {dyn_wB:.1f} Hz\n")
        f.write(f"Sharpened wA/wB: {pred_wA:.1f} / {pred_wB:.1f} Hz\n")
    
    # Generate Plot
    fig = plt.figure(figsize=(8, 6), dpi=150)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    ax = plt.subplot(gs[0])
    ax.plot(freq[mask_fit], y[mask_fit], 'o', color='#7f8c8d', ms=3, alpha=0.3, label="Data")
    ax.plot(freq, y_pred, '--', color='#e74c3c', lw=2, label=f"VT Prediction (pB={dyn_pB*100:.1f}%)")
    ax.invert_xaxis(); ax.set_xticks([]); ax.set_ylabel("Normalized Intensity"); ax.legend(frameon=False)
    
    ax_res = plt.subplot(gs[1])
    ax_res.plot(freq[mask_fit], y[mask_fit] - y_pred[mask_fit], '-', color='#27ae60', lw=1)
    ax_res.axhline(0, color='k', ls=':', alpha=0.5)
    ax_res.set_xlabel("Frequency (Hz)"); ax_res.set_ylabel("Resid."); ax_res.set_xlim(ax.get_xlim())
    
    plt.tight_layout()
    plt.savefig(FIG_OUTPUT, dpi=300)
    print(f"Validation complete. Saved plot and text to {OUTPUT_DIR} and {FIGURE_DIR}.")
