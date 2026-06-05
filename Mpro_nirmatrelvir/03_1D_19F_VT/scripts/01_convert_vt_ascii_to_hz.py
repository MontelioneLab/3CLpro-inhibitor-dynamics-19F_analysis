import os
import sys
import glob
import numpy as np

# ============================================================
# 1. PATHS & SPECTROMETER CONFIGURATIONS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "raw_bruker")
PROCESSED_BASE = os.path.join(PROJECT_ROOT, "processed_ascii")

# Unified spectral sweep parameters
LEFT_PPM = 2.0
RIGHT_PPM = -2.0

DATASETS = [
    {
        "raw_subdir": "19F_VT_600MHz_Nirmat400-Mpro400_2mM_EDTA_2026-05-07/temp_ascii_from_Bruker",
        "output_subdir": "nirmat_VT_LB10_600",
        "sfo1_mhz": 564.588149193046
    },
    {
        "raw_subdir": "19F_VT_800MHz_Nirmat400-Mpro400_2mM_EDTA_2026-05-09/temp_ascii_from_Bruker",
        "output_subdir": "nirmat_VT_LB10_800",
        "sfo1_mhz": 752.8281062
    }
]

# ============================================================
# 2. CORE PROCESSING PIPELINE
# ============================================================
print("============================================================")
print("     NIRMATRELVIR MULTI-FIELD DATA PREPROCESSING PIPELINE    ")
print("============================================================\n")

for ds in DATASETS:
    target_raw_path = os.path.join(RAW_DIR, ds["raw_subdir"])
    target_out_path = os.path.join(PROCESSED_BASE, ds["output_subdir"])
    
    if not os.path.exists(target_raw_path):
        print(f"⚠️ Warning: Missing raw directory: {target_raw_path}. Skipping.")
        continue
        
    os.makedirs(target_out_path, exist_ok=True)
    print(f"📂 Processing Subdirectory: {ds['output_subdir']}")
    print(f"   Using SFO1 Frequency:  {ds['sfo1_mhz']:.4f} MHz")
    print(f"   Target Destination:    {target_out_path}\n")
    
    search_pattern = os.path.join(target_raw_path, "asci_*.txt")
    file_list = sorted(glob.glob(search_pattern))
    
    for fname in file_list:
        base_name = os.path.basename(fname)
        
        if "_trim" in base_name:
            continue
            
        try:
            # Parse Temperature and identify 1000-series replicates
            temp_str = base_name.split("_")[1].replace(".txt", "")
            raw_temp = int(temp_str)
            
            if raw_temp >= 1000:
                T = raw_temp - 1000
                out_name = f"spec_{T}K_rep1_LB10.dat"
            else:
                T = raw_temp
                out_name = f"spec_{T}K_LB10.dat"
            
            with open(fname) as f:
                lines = [l for l in f if not l.startswith("#")]
            intensity = np.array([float(l.strip()) for l in lines])
            
            raw_size = len(intensity)
            if raw_size in [16579, 16282]:
                intensity = intensity[:-1]
                current_size = raw_size - 1
            else:
                current_size = raw_size
                
            ppm = np.linspace(LEFT_PPM, RIGHT_PPM, current_size)
            hz = ppm * ds["sfo1_mhz"]
            hz_centered = hz - hz.mean()
            
            full_out_path = os.path.join(target_out_path, out_name)
            matrix_out = np.column_stack([hz_centered, intensity])
            np.savetxt(full_out_path, matrix_out, fmt="%12.6f %12.6e")
            print(f"  👉 Converted {base_name} -> {out_name}")
            
        except Exception as e:
            print(f"  ⚠️ Failed to process file {base_name}: {e}")
 
    print("-" * 60)

print("\n🎉 Multi-field preprocessing complete. All 600 MHz and 800 MHz spectra are synchronized!")
