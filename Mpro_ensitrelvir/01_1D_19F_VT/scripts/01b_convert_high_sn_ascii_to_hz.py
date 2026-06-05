import sys
import glob
import os
import numpy as np

# ============================================================
# 1. SETUP PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Source file path
RAW_FILE = os.path.normpath(os.path.join(
    PROJECT_ROOT, 
    "raw_bruker", 
    "Ensitrelvir-Mpro_400uM_long_1D_T1s_2026-01-29_600MHz", 
    "temp_ascii_from_Bruker",
    "asci-long-LB20_298.txt"
))

# Destination directory and filename
PROCESSED_DIR = os.path.normpath(os.path.join(
    PROJECT_ROOT, 
    "processed_ascii", 
    "ensitrelvir_298K_LB20_long_600MHz"
))
OUT_NAME = "spec_298_long_LB20.dat"

# ------------------ USER-DEFINED CONSTANTS ------------------
# We will use the standard Ensitrelvir size and window settings
TARGET_SIZE = 172881 
LEFT_PPM  = -110.0
RIGHT_PPM = -150.0
SFO1_MHZ = 564.5565299  # 19F on 600 MHz system

# ============================================================
# 2. EXECUTION
# ============================================================
if not os.path.exists(RAW_FILE):
    print(f"❌ ERROR: Source file not found at: {RAW_FILE}")
else:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Read and skip Bruker headers
    with open(RAW_FILE) as f:
        lines = [l for l in f if not l.startswith("#")]
    
    data = np.array([float(l.strip()) for l in lines])
    actual_size = len(data)

    # Validate size
    if actual_size == TARGET_SIZE + 1:
        intensity = data[:TARGET_SIZE]
    elif actual_size == TARGET_SIZE:
        intensity = data
    else:
        print(f"❌ ERROR: Unexpected file size {actual_size}. Check the Ensitrelvir export.")
        exit()

    # Frequency Calculation (ppm -> Hz centered)
    ppm = np.linspace(LEFT_PPM, RIGHT_PPM, TARGET_SIZE)
    hz_vals = ppm * SFO1_MHZ 
    hz_centered = hz_vals - hz_vals.mean()

    # Save to the specific 'long' subfolder
    out_path = os.path.join(PROCESSED_DIR, OUT_NAME)
    np.savetxt(out_path, np.column_stack([hz_centered, intensity]), fmt="%12.6f %12.6e")
    
    print(f"✅ Success! Processed file saved to: {out_path}")
