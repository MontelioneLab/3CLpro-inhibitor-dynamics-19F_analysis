import os
import glob
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks

# ============================================================
# 1. SETUP & AUTOMATED PARAMETER LOADING
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_ascii", "ensitrelvir_VT_LB50_600")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PEAK_CENTERS = [6000.0, -5000.0, -7000.0] 
WINDOW_HALF_WIDTH = 1000.0  
R_kcal = 1.987e-3 

# Dynamic loading of dnu linearity params from Script 02a output
csv_params_path = os.path.join(OUTPUT_DIR, "raw_peak_maxima.csv")
if os.path.exists(csv_params_path):
    print("📈 Found Script 02a output. Computing optimal global regression...")
    df_params = pd.read_csv(csv_params_path)
    X_val = df_params["Temp_K"].values
    Y_val = df_params["Delta_Nu_Hz"].values
    slope_dnu, inter_all = np.polyfit(X_val, Y_val, 1)
    intercept_298_dnu = inter_all + slope_dnu * 298
    print(f"   Loaded: dnu = {intercept_298_dnu:.1f} + {slope_dnu:.2f} * (T - 298)")
else:
    print("⚠️ Warning: raw_peak_maxima.csv not found. Falling back to default literature parameters.")
    intercept_298_dnu = 1186.0
    slope_dnu = 1.33

# ============================================================
# 2. STANDARDIZED PHYSICS ENGINE
# ============================================================
def two_lorentzians_symmetric(freq, wA, wB, center, separation, pB, height):
    center_A = center + separation / 2.0
    center_B = center - separation / 2.0

    peak_A = (1.0 - pB) * (wA**2 / ((freq - center_A)**2 + wA**2))
    peak_B = pB * (wB**2 / ((freq - center_B)**2 + wB**2))
    return height * (peak_A + peak_B)

# ============================================================
# 3. LOAD DATA
# ============================================================
def load_and_chop_spectra(directory):
    pattern = os.path.join(directory, "spec_*.dat")
    file_list = sorted(glob.glob(pattern))
    data_storage = {} 
    
    for fname in file_list:
        try:
            # FIXED: Handles 'spec_283K_LB50.dat' naming flawlessly
            base_name = os.path.basename(fname)
            temp_str = base_name.split("_")[1].replace("K", "").replace(".dat", "")
            T = int(temp_str)
            
            raw = np.loadtxt(fname)
            if np.max(np.abs(raw[:,0])) < 500.0: 
                raw[:,0] *= 1e3 
                
            regions_list = []
            for center in PEAK_CENTERS:
                mask = (raw[:,0] > center - WINDOW_HALF_WIDTH) & (raw[:,0] < center + WINDOW_HALF_WIDTH)
                freq_chunk, int_chunk = raw[mask, 0], raw[mask, 1]
                if len(int_chunk) > 0:
                    int_chunk -= np.min(int_chunk) 
                    if np.max(int_chunk) > 0: 
                        int_chunk /= np.max(int_chunk)
                regions_list.append((freq_chunk - center, int_chunk))
            data_storage[T] = regions_list
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse file {os.path.basename(fname)}. Error: {e}")
    return data_storage

spectra_data = load_and_chop_spectra(DATA_DIR)
temps = sorted(spectra_data.keys())

if not temps:
    print("❌ ERROR: No valid spectrum files parsed. Check data names.")
    sys.exit(1)

# ============================================================
# 4. FITTING LOOP
# ============================================================
hero_pB_values, hero_temps, export_data = [], [], []

print(f"\nProcessing Global Fits across {len(temps)} temperatures...")

for T in temps:
    regions = spectra_data[T]
    
    # --- STEP 1: MIDDLE PEAK ---
    freq_m, y_m = regions[1] 
    peaks, _ = find_peaks(y_m, prominence=0.05)
    f_peaks = freq_m[peaks]
    y_peaks = y_m[peaks]

    mask = (f_peaks > -1000) & (f_peaks < 1200)
    f_peaks, y_peaks = f_peaks[mask], y_peaks[mask]
    
    idx = np.argsort(y_peaks)[-2:]
    f_left, f_right = np.sort(f_peaks[idx])
    middle_midpoint = 0.5 * (f_left + f_right)
    freq_m_shifted = freq_m - middle_midpoint

    def res_middle(p):
        pB, wA, wB, height = p
        # Dynamically calculated separation based on Script 02a parameters
        sep = intercept_298_dnu + slope_dnu * (T - 298)
        return two_lorentzians_symmetric(freq_m_shifted, wA, wB, 0.0, sep, pB, height) - y_m

    fit_m = least_squares(res_middle, [0.40, 140., 120., 1.0], bounds=([0.01, 50., 50., 0.1], [0.60, 500., 200., 2.0]), loss='soft_l1')
    best_pB, m_wA, m_wB, m_h = fit_m.x
    m_sep = intercept_298_dnu + slope_dnu * (T - 298)
    
    hero_pB_values.append(best_pB)
    hero_temps.append(T)
    print(f"T={T} K | pB={best_pB:.3f} | dnu={m_sep:.1f} Hz | lwA={m_wA * 2.0:.1f} | lwB={m_wB * 2.0:.1f}")

    # --- STEP 2: LEFT PEAK ---
    freq_l, y_l = regions[0]
    def res_left(p): return two_lorentzians_symmetric(freq_l, *p[:4], best_pB, p[4]) - y_l
    fit_l = least_squares(res_left, [120., 120., 0.0, -200.0, 1.0], bounds=([50., 50., -500., -1000.0, 0.1], [250., 250., 500., -150.0, 2.0]), loss='soft_l1')
    l_wA, l_wB, l_cen, l_sep, l_h = fit_l.x

    # --- STEP 3: RIGHT PEAK ---
    freq_r, y_r = regions[2]
    def res_right(p): return two_lorentzians_symmetric(freq_r, *p[:4], best_pB, p[4]) - y_r
    fit_r = least_squares(res_right, [120., 120., 0.0, 200.0, 1.0], bounds=([50., 50., -500., 100.0, 0.1], [250., 250., 500., 1000., 2.0]), loss='soft_l1')
    r_wA, r_wB, r_cen, r_sep, r_h = fit_r.x

    export_data.append({
        "Temp_K": T, "Population_B": best_pB,
        "Middle_wA": m_wA * 2.0, "Middle_wB": m_wB *2.0, "Middle_cen": 0.0, "Middle_sep": m_sep, "Middle_h": m_h,
        "Middle_midpoint": middle_midpoint,
        "Left_wA": l_wA * 2.0, "Left_wB": l_wB * 2.0, "Left_cen": l_cen, "Left_sep": l_sep, "Left_h": l_h,
        "Right_wA": r_wA * 2.0, "Right_wB": r_wB * 2.0, "Right_cen": r_cen, "Right_sep": r_sep, "Right_h": r_h
    })

# ============================================================
# 5. THERMO CALCULATION & SAVING
# ============================================================
T_arr, pB_arr = np.array(hero_temps), np.array(hero_pB_values)
slope, intercept = np.polyfit(1.0/T_arr, np.log(pB_arr / (1.0 - pB_arr)), 1)
dH_calc, dS_calc = -slope * R_kcal, intercept * R_kcal * 1000.0
dG298_calc = dH_calc - (2985 * dS_calc / 1000.0)

# Output detailed results to screen for transparency
print("\n" + "="*60)
print(" GLOBAL THERMODYNAMIC SIMULATION SUMMARY")
print("="*60)
print(f"  ΔH° (Enthalpy):          {dH_calc:.2f} kcal/mol")
print(f"  ΔS° (Entropy):           {dS_calc:.2f} cal/(mol·K)")
print(f"  ΔGo @ 298 K (Free E):  {dG298_calc:.2f} kcal/mol")

print("="*60)

GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")
np.savez(
    GLOBAL_NPZ, 
    export_data=export_data, 
    T_vals=T_arr, 
    pB_vals=pB_arr, 
    dH=dH_calc, 
    dS=dS_calc, 
    dG298=dG298_calc
)
print(f"Archive updated successfully: {GLOBAL_NPZ}")
