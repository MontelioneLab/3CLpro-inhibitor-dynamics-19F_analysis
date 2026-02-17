import numpy as np
import os

# ============================================================
# 1. SETUP PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Ensure this script is in the 'Mpro_nirmatrelvir/04_C145A_1D_19F_298K/scripts' folder
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Source file path
RAW_FILE = os.path.normpath(os.path.join(
    PROJECT_ROOT, 
    "raw_data", 
    "temp_ascii_from_Bruker",
    "asci_C145A_nirmat_298.txt"
))

# Destination folder and standardized name
PROCESSED_DIR = os.path.normpath(os.path.join(
    PROJECT_ROOT, 
    "processed_ascii", 
    "C145A_nirmat_298K_600MHz"
))
OUT_NAME = "spec_298.dat"

# ------------------ USER-DEFINED CONSTANTS ------------------
# TARGET_SIZE updated to 20840 as confirmed by your file header
TARGET_SIZE = 20840 
LEFT_PPM  = -70.0
RIGHT_PPM = -74.0
SFO1_MHZ = 564.5881492  # 600 MHz system (Pharaoh)

# ============================================================
# 2. EXECUTION
# ============================================================
if not os.path.exists(RAW_FILE):
    print(f"❌ ERROR: Source file not found at: {RAW_FILE}")
else:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    with open(RAW_FILE) as f:
        # Load data, skipping comment lines
        lines = [l for l in f if not l.startswith("#")]
    
    data = np.array([float(l.strip()) for l in lines])
    actual_size = len(data)

    # Logic to handle exact size or size + 1
    if actual_size == TARGET_SIZE + 1:
        intensity = data[:TARGET_SIZE]
        print(f"  ✅ Trimmed 1 point to match {TARGET_SIZE}")
    elif actual_size == TARGET_SIZE:
        intensity = data
        print(f"  ✅ Size matches {TARGET_SIZE} OK")
    else:
        print(f"❌ ERROR: Unexpected size {actual_size}. Wanted {TARGET_SIZE}")
        exit()

    # Create the frequency axis based on the 20840 points
    ppm = np.linspace(LEFT_PPM, RIGHT_PPM, TARGET_SIZE)
    hz_vals = ppm * SFO1_MHZ 
    hz_centered = hz_vals - hz_vals.mean()

    # Save output to standardized .dat format
    out_path = os.path.join(PROCESSED_DIR, OUT_NAME)
    np.savetxt(out_path, np.column_stack([hz_centered, intensity]), fmt="%12.6f %12.6e")
    
    print(f"✅ Success! C145A standardized file saved to: {out_path}")
