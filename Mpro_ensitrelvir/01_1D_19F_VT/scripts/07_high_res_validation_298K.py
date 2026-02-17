import os
import numpy as np

# ============================================================
# 1. SETUP & ARCHIVE LOADING
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")

if not os.path.exists(GLOBAL_NPZ):
    print(f"Error: {GLOBAL_NPZ} not found. Run Script 02 first.")
    exit()

# Load the dynamic results from Script 02
archive = np.load(GLOBAL_NPZ, allow_pickle=True)
T_vals = archive["T_vals"]
pB_vals = archive["pB_vals"]

# Find the 298 K entry specifically for validation
target_T = 298.0
idx = (np.abs(T_vals - target_T)).argmin()
pB_298 = pB_vals[idx]

# ============================================================
# 2. WRITE VALIDATION PARAMETERS FILE
# ============================================================
validation_path = os.path.join(OUTPUT_DIR, "validation_params_298K.txt")

with open(validation_path, "w") as f:
    f.write("ENSITRELVIR HIGH-RES VALIDATION PARAMETERS (298 K)\n")
    f.write("-" * 50 + "\n")
    f.write(f"Source Archive: {os.path.basename(GLOBAL_NPZ)}\n")
    f.write(f"Validated Temp: {T_vals[idx]:.1f} K\n")
    f.write(f"Population pB:  {pB_298:.4f} ({pB_298*100:.1f}%)\n")
    f.write(f"Equilibrium K:  {pB_298/(1-pB_298):.4f}\n")
    
    # Extracting fitted widths for the 298K record
    export_data = archive["export_data"]
    d_298 = export_data[idx]
    f.write(f"Fitted wA:      {d_298['Middle_wA']:.2f} Hz\n")
    f.write(f"Fitted wB:      {d_298['Middle_wB']:.2f} Hz\n")
    f.write("-" * 50 + "\n")

print(f"Validation parameters written to: {validation_path}")
