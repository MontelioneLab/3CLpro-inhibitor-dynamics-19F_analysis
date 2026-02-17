import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================
# 1. SETUP & PATHS (Restored to your original logic)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_ascii", "ensitrelvir_298K_LB20_long_600MHz")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

HIGH_SN_FILE = os.path.join(DATA_DIR, "spec_298_long_LB20.dat")
TXT_OUTPUT = os.path.join(OUTPUT_DIR, "ensitrelvir_validation_params.txt")
FIG_OUTPUT = os.path.join(FIGURE_DIR, "Ensitrelvir_Middle_Validation.png")

MIDDLE_PEAK_CENTER = -5000.0
WINDOW_HALF_WIDTH = 2000.0

# CONSTANTS (Your original hard-coded parameters for pB=40.5%)
LB_VT = 50.0   
LB_NEW = 20.0  
LB_DIFF = LB_VT - LB_NEW # 30 Hz sharpening correction

VT_pB_298 = 0.405
VT_wA     = 162.4
VT_wB     = 150.0 
VT_sep    = 1189.0

# ============================================================
# 2. PHYSICS ENGINE
# ============================================================
def two_lorentzians_slope(freq, wA, wB, center, separation, pB, height, baseline, slope):
    center_A = center + (separation * pB)
    peak_A = (1.0 - pB) * (wA**2 / ((freq - center_A)**2 + wA**2))
    center_B = center - (separation * (1.0 - pB))
    peak_B = pB * (wB**2 / ((freq - center_B)**2 + wB**2))
    return height * (peak_A + peak_B) + baseline + (slope * freq)

# ============================================================
# 3. DATA LOADING (Your original targeted zeroing)
# ============================================================
def load_and_targeted_zero(filename, center_hz):
    try:
        raw = np.loadtxt(filename)
        if np.max(np.abs(raw[:,0])) < 500.0: raw[:,0] *= 600.0 * 1e3
        mask = (raw[:,0] > center_hz - WINDOW_HALF_WIDTH) & \
               (raw[:,0] < center_hz + WINDOW_HALF_WIDTH)
        freq = raw[mask, 0] - center_hz
        inten = raw[mask, 1]
        
        # Noise floor calculation (Targeted zero)
        mask_clean = (freq > 1700.0) & (freq < 2000.0)
        noise_floor = np.median(inten[mask_clean]) if np.sum(mask_clean) > 5 else np.min(inten)
        inten -= noise_floor
        if np.max(inten) > 0: inten /= np.max(inten)
        return freq, inten
    except Exception as e:
        print(f"Error loading {filename}: {e}"); return None, None

freq, y = load_and_targeted_zero(HIGH_SN_FILE, MIDDLE_PEAK_CENTER)

if freq is not None:
    # SHARPEN THE VT PARAMETERS
    pred_wA = VT_wA - LB_DIFF
    pred_wB = VT_wB - LB_DIFF

    # ============================================================
    # 4. ALIGNMENT FIT (Original Mask)
    # ============================================================
    mask_fit = freq > -1000.0
    freq_fit = freq[mask_fit]; y_fit = y[mask_fit]

    def res_align(p):
        current_sep = VT_sep + p[3]
        model = two_lorentzians_slope(freq_fit, pred_wA, pred_wB, p[0], current_sep, VT_pB_298, p[1], p[2], p[4])
        return model - y_fit
        
    p0 = [0.0, 1.0, 0.0, 0.0, 0.0]
    bounds = ([-300, 0.5, -0.2, -100, -0.005], [300, 1.5, 0.2, 100, 0.005])
    
    fit = least_squares(res_align, p0, bounds=bounds)
    shift_p, h_p, base_p, sep_corr_p, slope_p = fit.x
    final_sep_pred = VT_sep + sep_corr_p

    # ============================================================
    # 6. PLOTTING
    # ============================================================
    y_pred = two_lorentzians_slope(freq, pred_wA, pred_wB, shift_p, final_sep_pred, VT_pB_298, h_p, base_p, slope_p)
    fig = plt.figure(figsize=(8, 6), dpi=150)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    ax = plt.subplot(gs[0])
    ax.plot(freq[mask_fit], y[mask_fit], 'o', color='#7f8c8d', ms=3, alpha=0.3, label="Data (Included)")
    ax.plot(freq[~mask_fit], y[~mask_fit], 'x', color='#ecf0f1', ms=3, alpha=0.6, label="Data (Excluded)")
    ax.plot(freq, y_pred, '--', color='#e74c3c', lw=2.5, label=f"VT Prediction (pB={VT_pB_298*100:.1f}%)")
    ax.invert_xaxis(); ax.set_xticks([]); ax.set_ylabel("Normalized Intensity")
    ax.legend(frameon=False)
    
    ax_res = plt.subplot(gs[1])
    ax_res.plot(freq[mask_fit], y[mask_fit] - y_pred[mask_fit], '-', color='#27ae60', lw=1.2)
    ax_res.axhline(0, color='k', ls=':', alpha=0.5)
    ax_res.set_xlabel("Frequency (Hz)"); ax_res.set_ylabel("Resid."); ax_res.set_xlim(ax.get_xlim()) 
    
    plt.tight_layout(); plt.show()
