import numpy as np
import glob
import os
import sys

# ============================================================
# 1. SETUP PATHS (Aligned with Script 02 logic)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Path relative to PROJECT_ROOT (01_1D_19F_VT)
DATA_DIR = os.path.normpath(os.path.join(
    PROJECT_ROOT, 
    "raw_bruker", 
    "Ensitrelvir-Mpro_400uM_VT_2026-01-29_600MHz", 
    "temp-ascii-from-bruker"
))

# Output to processed_ascii
PROCESSED_DIR = os.path.normpath(os.path.join(
    PROJECT_ROOT, 
    "processed_ascii", 
    "ensitrelvir_VT_LB50_600"
))

# ------------------ USER-DEFINED CONSTANTS ------------------
# Verify this size against your Ensitrelvir ASCII files!
TARGET_SIZE = 172882 
LEFT_PPM  = -110.0
RIGHT_PPM = -150.0
SFO1_MHZ = 564.5565299  # 19F on 600 MHz system

# ============================================================
# 2. MAIN LOOP
# ============================================================
print(f"Searching in: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    print(f"❌ ERROR: Source directory not found!")
else:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    search_path = os.path.join(DATA_DIR, "asci_*.txt")
    files = sorted(glob.glob(search_path))
    print(f"Found {len(files)} raw VT ASCII files.")

    for fname in files:
        with open(fname) as f:
            lines = [l for l in f if not l.startswith("#")]
        
        data = np.array([float(l.strip()) for l in lines])
        actual_size = len(data)

        # Trimming/Validation logic
        if actual_size == TARGET_SIZE + 1:
            intensity = data[:TARGET_SIZE]
        elif actual_size == TARGET_SIZE:
            intensity = data
        else:
            print(f"  ⚠️ Skipping {os.path.basename(fname)}: size {actual_size} != {TARGET_SIZE}")
            continue

        # Frequency Calculation
        ppm = np.linspace(LEFT_PPM, RIGHT_PPM, TARGET_SIZE)
        hz_vals = ppm * SFO1_MHZ 
        hz_centered = hz_vals - hz_vals.mean()

        # Extract temperature (e.g., 'asci_283.txt' -> '283')
        base_no_ext = os.path.splitext(os.path.basename(fname))[0]
        temp_val = base_no_ext.split("_")[1]

        # Save to processed_ascii as spec_{T}K_LB50.dat
        base_name = f"spec_{temp_val}K_LB50.dat"
        outname = os.path.join(PROCESSED_DIR, base_name)
    
        np.savetxt(outname, np.column_stack([hz_centered, intensity]), fmt="%12.6f %12.6e")
        print(f"  ✅ Saved: {base_name}")

print("\nVT Batch conversion complete.")
