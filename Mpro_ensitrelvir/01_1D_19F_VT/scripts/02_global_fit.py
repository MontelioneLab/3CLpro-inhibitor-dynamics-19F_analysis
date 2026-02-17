import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import os
import pandas as pd

# ============================================================
# 1. SETUP
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "processed_ascii",
    "ensitrelvir_VT_LB50_600"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ORDER: Left (Downfield/6k), Middle (-5k), Right (Upfield/-7k)
PEAK_CENTERS = [ 6000.0,  -5000.0,  -7000.0 ] 
WINDOW_HALF_WIDTH = 2000.0  
EXCLUDE_TEMPS = [] 
R_kcal = 1.987e-3 

# ============================================================
# 2. PHYSICS ENGINE
# ============================================================
def two_lorentzians(freq, wA, wB, center, separation, pB, height):
    center_A = center + (separation * pB) 
    peak_A = (1.0 - pB) * (wA**2 / ((freq - center_A)**2 + wA**2))
    center_B = center - (separation * (1.0 - pB)) 
    peak_B = pB * (wB**2 / ((freq - center_B)**2 + wB**2))
    return height * (peak_A + peak_B)

# ============================================================
# 3. LOAD DATA
# ============================================================
def load_and_chop_spectra(directory):
    pattern = os.path.join(directory, "spec_*.dat")
    file_list = sorted(glob.glob(pattern))
    print(f"Loading {directory}...")
    data_storage = {} 
    
    for fname in file_list:
        try:
            T_str = os.path.basename(fname).split("_")[1].replace(".dat","")
            T = int(T_str)
            if T in EXCLUDE_TEMPS: continue
            raw = np.loadtxt(fname)
            if np.max(np.abs(raw[:,0])) < 500.0: raw[:,0] *= 1e3 
            regions_list = []
            for center in PEAK_CENTERS:
                mask = (raw[:,0] > center - WINDOW_HALF_WIDTH) & \
                       (raw[:,0] < center + WINDOW_HALF_WIDTH)
                freq_chunk = raw[mask, 0]
                int_chunk = raw[mask, 1]
                if len(int_chunk) > 0:
                    int_chunk -= np.min(int_chunk) 
                    if np.max(int_chunk) > 0: 
                        int_chunk /= np.max(int_chunk)
                freq_chunk = freq_chunk - center 
                regions_list.append( (freq_chunk, int_chunk) )
            data_storage[T] = regions_list
        except: pass
    return data_storage

spectra_data = load_and_chop_spectra(DATA_DIR)
temps = sorted(spectra_data.keys())

# ============================================================
# 4. FITTING LOOP (Updated with Expanded Bounds)
# ============================================================
hero_pB_values = []
hero_temps = []
fit_curves = {}
export_data = []

print("\nProcessing Fits with Floating Linewidths...")

for T in temps:
    regions = spectra_data[T]
    fit_curves[T] = [None, None, None]
    
    # --- STEP 1: MIDDLE PEAK (Index 1) ---
    freq_m, y_m = regions[1] 

    def res_middle(p): return two_lorentzians(freq_m, *p[1:5], p[0], p[5]) - y_m

# Original bounds that gave pB ~ 40.5%
    p0 = [0.25, 150., 100., 250., 1100., 1.0]
    lb = [0.01, 50., 50., -500., 800., 0.1]
    ub = [0.49, 250., 150., 1000., 2000., 2.0] # Width B capped at 150 Hz

    fit_m = least_squares(res_middle, p0, bounds=(lb, ub), loss='soft_l1')
    best_pB = fit_m.x[0]
    
    hero_pB_values.append(best_pB)
    hero_temps.append(T)
    fit_curves[T][1] = two_lorentzians(freq_m, *fit_m.x[1:5], best_pB, fit_m.x[5])
    
    m_wA, m_wB, m_cen, m_sep, m_h = fit_m.x[1], fit_m.x[2], fit_m.x[3], fit_m.x[4], fit_m.x[5]

    # --- STEP 2: LEFT PEAK (Index 0) ---
    freq_l, y_l = regions[0]
    def res_left(p): return two_lorentzians(freq_l, *p[:4], best_pB, p[4]) - y_l
    
    p0_l = [120., 120., 0.0, -200.0, 1.0]
    lb_l = [50., 50., -500., -1000.0, 0.1] 
    ub_l = [250., 250., 500., -150.0, 2.0] 

    fit_l = least_squares(res_left, p0_l, bounds=(lb_l, ub_l), loss='soft_l1')
    fit_curves[T][0] = two_lorentzians(freq_l, *fit_l.x[:4], best_pB, fit_l.x[4])
    l_wA, l_wB, l_cen, l_sep, l_h = fit_l.x[0], fit_l.x[1], fit_l.x[2], fit_l.x[3], fit_l.x[4]
    
    # --- STEP 3: RIGHT PEAK (Index 2) ---
    freq_r, y_r = regions[2]
    def res_right(p): return two_lorentzians(freq_r, *p[:4], best_pB, p[4]) - y_r
    
    p0_r = [120., 120., 0.0, 200.0, 1.0]
    lb_r = [50., 50., -500., 100.0, 0.1]
    ub_r = [250., 250., 500., 1000., 2.0] 

    fit_r = least_squares(res_right, p0_r, bounds=(lb_r, ub_r), loss='soft_l1')
    fit_curves[T][2] = two_lorentzians(freq_r, *fit_r.x[:4], best_pB, fit_r.x[4])
    r_wA, r_wB, r_cen, r_sep, r_h = fit_r.x[0], fit_r.x[1], fit_r.x[2], fit_r.x[3], fit_r.x[4]

    export_data.append({
        "Temp_K": T, "Population_B": best_pB,
        "Middle_wA": m_wA, "Middle_wB": m_wB, "Middle_cen": m_cen, "Middle_sep": m_sep, "Middle_h": m_h,
        "Left_wA": l_wA, "Left_wB": l_wB, "Left_cen": l_cen, "Left_sep": l_sep, "Left_h": l_h,
        "Right_wA": r_wA, "Right_wB": r_wB, "Right_cen": r_cen, "Right_sep": r_sep, "Right_h": r_h
    })

# ============================================================
# 5. THERMO CALCULATION & REPORTING
# ============================================================
T_arr = np.array(hero_temps)
pB_arr = np.array(hero_pB_values)
ln_K = np.log(pB_arr / (1.0 - pB_arr))
mask = np.isfinite(ln_K)

slope, intercept = np.polyfit(1.0/T_arr[mask], ln_K[mask], 1)
dH_calc = -slope * R_kcal
dS_calc = intercept * R_kcal * 1000.0
dG_298 = dH_calc - (298.15 * dS_calc / 1000.0)
r2_val = np.corrcoef(1.0/T_arr[mask], ln_K[mask])[0,1]**2

# ============================================================
# 7. ARCHIVE FOR DOWNSTREAM SCRIPTS
# ============================================================
GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")
np.savez(GLOBAL_NPZ, 
         export_data=export_data, 
         T_vals=T_arr, 
         pB_vals=pB_arr,
         dH=dH_calc, dS=dS_calc, dG298=dG_298)

print(f"Archive updated: {GLOBAL_NPZ}")
