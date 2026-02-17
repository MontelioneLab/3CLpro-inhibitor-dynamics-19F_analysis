import numpy as np
import glob
import os

# ============================================================
# 1. SETUP PATHS (EXACT MATCH TO YOUR SCRIPT 02)
# ============================================================
# Script is in 'scripts/', so PROJECT_ROOT is '02_1D_19F_VT'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# TARGET_SIZE confirmed by your grep
TARGET_SIZE = 16283

DATASETS = {
    "600MHz": {
        # Path relative to PROJECT_ROOT (02_1D_19F_VT)
        "rel_raw": "raw_bruker/Nirmat-Mpro_400uM_VT_2026-01-24_600MHz/temp-ascii-from-Bruker",
        "rel_processed": "processed_ascii/nirmat_VT_LB10_600",
        "sfo1": 564.5881492 # From your 2.add_x_in_Hz_centered.py
    },
    "800MHz": {
        "rel_raw": "raw_bruker/Nirmat-Mpro_400uM_VT_2026-01-28_800MHz/temp-ascii-from-Bruker",
        "rel_processed": "processed_ascii/nirmat_VT_LB10_800",
        "sfo1": 752.8430000 # Standard 800 MHz field
    }
}

LEFT_PPM, RIGHT_PPM = -70.0, -74.0 # Confirmed by your file header

# ============================================================
# 2. BATCH PROCESSING
# ============================================================
for label, config in DATASETS.items():
    # Construct final absolute paths
    DATA_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, config["rel_raw"]))
    PROCESSED_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, config["rel_processed"]))
    
    print(f"\n--- Processing {label} ---")
    print(f"Looking in: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Skipping: Directory not found!")
        continue

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Match 'asci' with one 'i'
    files = sorted(glob.glob(os.path.join(DATA_DIR, "asci_*.txt")))
    print(f"Found {len(files)} files.")

    for fname in files:
        with open(fname) as f:
            # Skip Bruker header lines
            lines = [l for l in f if not l.startswith("#")]
        
        data = np.array([float(l.strip()) for l in lines])
        
        # Trimming logic to match TARGET_SIZE
        if len(data) == TARGET_SIZE + 1:
            intensity = data[:TARGET_SIZE]
        elif len(data) == TARGET_SIZE:
            intensity = data
        else:
            print(f"  ⚠️ Skipping {os.path.basename(fname)}: size {len(data)}")
            continue

        # Frequency logic
        ppm = np.linspace(LEFT_PPM, RIGHT_PPM, TARGET_SIZE)
        hz_vals = ppm * config["sfo1"]
        hz_centered = hz_vals - hz_vals.mean()

        # Save to 'processed_ascii' as spec_{T}.dat
        base_name = os.path.basename(fname).replace("asci_", "spec_").replace(".txt", ".dat")
        out_path = os.path.join(PROCESSED_DIR, base_name)
        
        np.savetxt(out_path, np.column_stack([hz_centered, intensity]), fmt="%12.6f %12.6e")
        print(f"  ✅ Saved: {base_name}")

print("\nBatch processing finished.")
