import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import least_squares

# ============================================================
# 1. SETUP & PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_ascii", "ensitrelvir_298K_LB20_long_600MHz")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

HIGH_SN_FILE = os.path.join(DATA_DIR, "spec_298_long_LB20.dat")
TXT_OUTPUT = os.path.join(OUTPUT_DIR, "06b_high_res_validation_params.txt")
FIG_OUTPUT_PNG = os.path.join(FIGURE_DIR, "06b_high_res_validation_plot.png")
FIG_OUTPUT_PDF = os.path.join(FIGURE_DIR, "06b_high_res_validation_plot.pdf")

MIDDLE_PEAK_CENTER = -5000.0
WINDOW_HALF_WIDTH = 2000.0
LB_VT = 50.0   
LB_NEW = 20.0  
LB_DIFF = LB_VT - LB_NEW  # 30 Hz sharpening correction

# ============================================================
# 2. DYNAMIC ARCHIVE RETRIEVAL
# ============================================================
if not os.path.exists(GLOBAL_NPZ):
    print(f"❌ Error: {GLOBAL_NPZ} not found. Run Script 02b first.")
    sys.exit(1)

archive = np.load(GLOBAL_NPZ, allow_pickle=True)
T_vals = archive["T_vals"]

# Safely unpack saved dictionary array from NumPy container
if archive["export_data"].ndim == 0:
    export_data = archive["export_data"].item()
else:
    export_data = archive["export_data"]

idx = (np.abs(T_vals - 298.0)).argmin()
dyn_pB  = export_data[idx]['Population_B']
dyn_wA  = export_data[idx]['Middle_wA']  # Master FWHM
dyn_wB  = export_data[idx]['Middle_wB']  # Master FWHM
dyn_sep = export_data[idx]['Middle_sep']

# ============================================================
# 3. UNIFIED SYMMETRIC PHYSICS ENGINE (Normalized Baseline Slope)
# ============================================================
def two_lorentzians_symmetric_validation_slope(freq, wA, wB, center, separation, pB, height, baseline, slope):
    center_A = center + separation / 2.0
    center_B = center - separation / 2.0

    peak_A = (1.0 - pB) * (wA**2 / ((freq - center_A)**2 + wA**2))
    peak_B = pB * (wB**2 / ((freq - center_B)**2 + wB**2))
    
    return height * (peak_A + peak_B) + baseline + (slope * (freq / 1000.0))

# ============================================================
# 4. SAFE DATA LOADING
# ============================================================
def load_and_targeted_zero(filename, center_hz):
    try:
        raw = np.loadtxt(filename)
        if np.max(np.abs(raw[:,0])) < 500.0: 
            raw[:,0] *= 1e3  
            
        mask = (raw[:,0] > center_hz - WINDOW_HALF_WIDTH) & \
               (raw[:,0] < center_hz + WINDOW_HALF_WIDTH)
        
        freq = raw[mask, 0] - center_hz
        inten = raw[mask, 1]
        
        if len(inten) == 0:
            raise RuntimeError("Mask window returned empty array.")
        
        mask_clean = (freq > 1700.0) & (freq < 2000.0)
        noise_floor = np.median(inten[mask_clean]) if np.sum(mask_clean) > 5 else np.min(inten)
        inten -= noise_floor
        if np.max(inten) > 0: 
            inten /= np.max(inten)
            
        return freq, inten
    except Exception as e:
        print(f"\nCRITICAL ERROR loading {os.path.basename(filename)}: {e}")
        return None, None

freq, y = load_and_targeted_zero(HIGH_SN_FILE, MIDDLE_PEAK_CENTER)

# ============================================================
# 5. 6-PARAMETER ALIGNMENT FIT (Optimizing Peak B Width)
# ============================================================
if freq is not None:
    # Convert master FWHM data to HWHM before calculating sharpening shifts
    pred_wA = (dyn_wA / 2.0) - LB_DIFF
    pred_wB = (dyn_wB / 2.0) - LB_DIFF

    mask_fit = (freq > -1000.0) & (freq < 1200.0)
    freq_fit = freq[mask_fit]
    y_fit = y[mask_fit]

    def res_align(p):
        center_offset, height, baseline, sep_correction, slope, fitted_wB = p
        current_sep = dyn_sep + sep_correction
        return two_lorentzians_symmetric_validation_slope(
            freq_fit, pred_wA, fitted_wB, center_offset, current_sep, dyn_pB, height, baseline, slope
        ) - y_fit
        
    p0 = [0.0, 1.0, 0.0, 0.0, 0.0, pred_wB]
    
    fit_bounds = ([-300, 0.5, -0.05, -100, -0.05, pred_wB - 5.0], 
                  [ 300, 1.5,  0.05,  100,  0.05, pred_wB + 20.0])
    
    fit = least_squares(res_align, p0, bounds=fit_bounds, loss='soft_l1')
    shift_p, h_p, base_p, sep_corr_p, slope_p, wB_p = fit.x
    final_sep = dyn_sep + sep_corr_p

    # Multiply by 2.0 for true FWHM representation in outputs
    fwhm_pred_wB = pred_wB * 2.0
    fwhm_opt_wB = wB_p * 2.0

    print("\n" + "="*50)
    print("      HIGH-RES WIDTH RECOVERY ANALYSIS (FWHM)")
    print("="*50)
    print(f"  Archive Predicted wB:  {fwhm_pred_wB:.1f} Hz")
    print(f"  High-Res Optimized wB: {fwhm_opt_wB:.1f} Hz")
    print(f"  Linewidth Deficit:     {fwhm_opt_wB - fwhm_pred_wB:+.1f} Hz")
    print(f"  Fitted Sep Correction: {sep_corr_p:+.1f} Hz")
    print("="*50)

    # ============================================================
    # 6. PLOTTING & LOG REPORTING
    # ============================================================
    y_pred = two_lorentzians_symmetric_validation_slope(
        freq, pred_wA, wB_p, shift_p, final_sep, dyn_pB, h_p, base_p, slope_p
    )
    
    with open(TXT_OUTPUT, "w") as f:
        f.write("ENSITRELVIR HIGH-RES OPTIMIZED VALIDATION LOG\n")
        f.write(f"Source: {os.path.basename(GLOBAL_NPZ)} | Temp: {T_vals[idx]:.1f} K\n")
        f.write("-" * 60 + "\n")
        f.write(f"Optimized Population (pB):    {dyn_pB*100:.2f}%\n")
        f.write(f"Master Archive Baseline Sep:  {dyn_sep:.2f} Hz\n")
        f.write(f"High-Res Recovered Sep:       {final_sep:.2f} Hz\n")
        f.write(f"Archive Predicted wB (FWHM):  {fwhm_pred_wB:.1f} Hz\n")
        f.write(f"Recovered True High-Res FWHM: {fwhm_opt_wB:.1f} Hz\n")
    
    fig = plt.figure(figsize=(9, 7), dpi=150)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.15)
    
    ax = plt.subplot(gs[0])
    ax.plot(freq[mask_fit], y[mask_fit], 'o', color='#7f8c8d', ms=4, alpha=0.25, label="High S/N 1D Exp Data")
    
    # REVERTED: Restored the exact original projection label format from the previous version
    ax.plot(freq, y_pred, '--', color='#d62728', lw=2.5, label=f"VT Projected Model (pB={dyn_pB*100:.1f}%)")
    
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_ylabel("Normalized Intensity", fontsize=14)
    
    # REVERTED: Restored the exact original title format from the previous version
    ax.set_title("298 K Middle Fluorine High-Res Validation", fontsize=16, fontweight='bold', pad=10)
    
    ax.legend(frameon=False, fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    
    ax_res = plt.subplot(gs[1])
    ax_res.plot(freq[mask_fit], y[mask_fit] - y_pred[mask_fit], '-', color='#27ae60', lw=1.2)
    ax_res.axhline(0, color='k', ls=':', alpha=0.5)
    ax_res.set_xlabel("Frequency (Hz)", fontsize=14)
    ax_res.set_ylabel("Residuals", fontsize=14)
    ax_res.set_xlim(ax.get_xlim())
    ax_res.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIG_OUTPUT_PNG, dpi=300)
    plt.savefig(FIG_OUTPUT_PDF, bbox_inches='tight')
    print(f"\n✅ Validation complete. Plots with standard labels saved cleanly to:\n  👉 {FIG_OUTPUT_PNG}\n  👉 {FIG_OUTPUT_PDF}")
else:
    print("❌ ERROR: Data matrix empty. Validation terminated.")
