import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ============================================================
# PATHS & CONFIGURATION
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_ascii", "ensitrelvir_VT_LB50_600")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIDDLE_CENTER = -5000.0
WINDOW_HALF_WIDTH = 1000.0

# ============================================================
# LOAD DATA
# ============================================================
def load_middle_region(directory):
    pattern = os.path.join(directory, "spec_*.dat")
    file_list = sorted(glob.glob(pattern))
    data_storage = {}
    
    for fname in file_list:
        try:
            base_name = os.path.basename(fname)
            temp_str = base_name.split("_")[1].replace("K", "").replace(".dat", "")
            T = int(temp_str)
            
            raw = np.loadtxt(fname)
            if np.max(np.abs(raw[:,0])) < 500.0: 
                raw[:,0] *= 1e3  # Convert to Hz
                
            mask = (raw[:,0] > MIDDLE_CENTER - WINDOW_HALF_WIDTH) & \
                   (raw[:,0] < MIDDLE_CENTER + WINDOW_HALF_WIDTH)
            
            freq_chunk = raw[mask, 0]
            int_chunk = raw[mask, 1]
            
            if len(int_chunk) > 0:
                int_chunk -= np.min(int_chunk)
                if np.max(int_chunk) > 0:
                    int_chunk /= np.max(int_chunk)
                    
            data_storage[T] = (freq_chunk, int_chunk)
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse file {os.path.basename(fname)}. Error: {e}")
    return data_storage

spectra_data = load_middle_region(DATA_DIR)
temps = sorted(spectra_data.keys())

# ============================================================
# TRUE COORDINATE EXTRACTION ENGINE
# ============================================================
raw_maxima_results = []
tracking_log_lines = []  # List to accumulate individual peak extraction prints

header_str = f"Extracting Raw Maxima from {len(temps)} spectra..."
print(f"\n{header_str}")
tracking_log_lines.append(header_str)

for T in temps:
    freq_raw, y_raw = spectra_data[T]
    
    peaks, _ = find_peaks(y_raw, prominence=0.05)
    f_peaks_raw = freq_raw[peaks]
    y_peaks = y_raw[peaks]
    
    mask = (f_peaks_raw > MIDDLE_CENTER - 1000) & (f_peaks_raw < MIDDLE_CENTER + 1200)
    f_peaks_raw = f_peaks_raw[mask]
    y_peaks = y_peaks[mask]
    
    if len(f_peaks_raw) >= 2:
        idx = np.argsort(y_peaks)[-2:]
        f_two_peaks = f_peaks_raw[idx]
        
        peak_A_candidates = f_two_peaks[f_two_peaks > MIDDLE_CENTER]
        peak_B_candidates = f_two_peaks[f_two_peaks <= MIDDLE_CENTER]
        
        if len(peak_A_candidates) > 0 and len(peak_B_candidates) > 0:
            abs_max_A = peak_A_candidates[0]
            abs_max_B = peak_B_candidates[0]
            
            local_A = abs_max_A - MIDDLE_CENTER
            local_B = abs_max_B - MIDDLE_CENTER
            
            dnu_raw = np.abs(abs_max_A - abs_max_B)
            
            raw_maxima_results.append({
                "Temp_K": T,
                "Peak_A_Hz": local_A,
                "Peak_B_Hz": local_B,
                "Delta_Nu_Hz": dnu_raw
            })
            log_line = f"T={T} K | Peak A (+Hz): {local_A:>.1f} | Peak B (-Hz): {local_B:>.1f} | dnu: {dnu_raw:.1f} Hz"
            print(log_line)
            tracking_log_lines.append(log_line)
        else:
            msg = f"T={T} K | Warning: Failed to find split peak components on both sides of center."
            print(msg)
            tracking_log_lines.append(msg)
    else:
        msg = f"T={T} K | Warning: Less than 2 peaks detected with prominence >= 0.05"
        print(msg)
        tracking_log_lines.append(msg)

# ============================================================
# LINEAR REGRESSION ANALYSIS & FILE WRITING
# ============================================================
if len(raw_maxima_results) >= 2:
    df = pd.DataFrame(raw_maxima_results).sort_values("Temp_K")
    
    X = df["Temp_K"].values
    Y = df["Delta_Nu_Hz"].values
    
    # Fit 1: All temperatures
    slope_all, inter_all = np.polyfit(X, Y, 1)
    r2_all = np.corrcoef(X, Y)[0, 1]**2
    inter_298_all = inter_all + slope_all * 298
    
    # Fit 2: Low temperatures (<= 298 K)
    low_mask = X <= 298
    if np.sum(low_mask) >= 2:
        slope_low, inter_low = np.polyfit(X[low_mask], Y[low_mask], 1)
        r2_low = np.corrcoef(X[low_mask], Y[low_mask])[0, 1]**2
        inter_298_low = inter_low + slope_low * 298
    else:
        slope_low, inter_298_low, r2_low = np.nan, np.nan, np.nan

    # Build the report string dynamically, integrating the tracking logs
    report_lines = []
    report_lines.extend(tracking_log_lines) # Inserts the T-by-T data block at the top
    report_lines.extend([
        "\n" + "="*60,
        " INDEPENDENT LINEARITY CONFIRMATION REPORT",
        "="*60,
        "FIT TYPE 1: USING ALL TEMPERATURES",
        f"  Slope:          {slope_all:.3f} Hz/K",
        f"  298K Intercept: {inter_298_all:.1f} Hz",
        f"  R-squared (R²): {r2_all:.4f}",
        f"  Script 02 Ready:  dnu = {inter_298_all:.1f} + {slope_all:.1f}*(T - 298)",
        "-" * 60
    ])
    
    if not np.isnan(slope_low):
        report_lines.extend([
            "FIT TYPE 2: USING ONLY LOW TEMPERATURES (<= 298 K)",
            f"  Slope:          {slope_low:.3f} Hz/K",
            f"  298K Intercept: {inter_298_low:.1f} Hz",
            f"  R-squared (R²): {r2_low:.4f}",
            f"  Script 02 Ready:  dnu = {inter_298_low:.1f} + {slope_low:.1f}*(T - 298)",
            "="*60
        ])
    else:
        report_lines.extend([
            "FIT TYPE 2: USING ONLY LOW TEMPERATURES (<= 298 K)",
            "  Insufficient data points (T <= 298 K) to calculate a separate fit.",
            "="*60
        ])

    # Join lines to print to screen
    report_text = "\n".join(report_lines)
    print("\n" + "="*60 + "\n  SAVING LOG SUMMARY TO FILE\n" + "="*60)
    
    # Write everything to output/02a_dnu_slope_fits.txt
    txt_out_path = os.path.join(OUTPUT_DIR, "02a_dnu_slope_fits.txt")
    with open(txt_out_path, "w") as txt_file:
        txt_file.write(report_text + "\n")
    print(f"📝 Saved text summary to: {txt_out_path}")
    
    # Save the raw numbers CSV
    csv_out_path = os.path.join(OUTPUT_DIR, "02a_tracked_peak_maxima.csv")
    df.to_csv(csv_out_path, index=False, float_format="%.6f") 
    print(f"📊 Saved raw coordinates to: {csv_out_path}")

else:
    print("\n❌ Error: Insufficient data points found to calculate linear regressions.")
