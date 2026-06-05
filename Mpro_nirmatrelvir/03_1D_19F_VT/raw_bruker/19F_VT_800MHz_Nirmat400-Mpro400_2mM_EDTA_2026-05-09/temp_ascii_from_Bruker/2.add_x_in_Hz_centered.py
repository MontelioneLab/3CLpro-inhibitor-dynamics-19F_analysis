import numpy as np
import glob

# ------------------ USER-DEFINED CONSTANTS ------------------

LEFT_PPM  = 2.0
RIGHT_PPM = -2.0
TARGET_SIZE = 16578

# NOTE: This value is in MHz (Millions of Hz). 
SFO1_MHZ = 752.8281062 

# ------------------ MAIN LOOP ------------------

for fname in sorted(glob.glob("asci_*_trim.txt")):

    print(f"\nProcessing {fname}")

    with open(fname) as f:
        intensity = np.array([float(l.strip()) for l in f])

    if len(intensity) != TARGET_SIZE:
        raise ValueError(f"{fname}: unexpected size {len(intensity)}")

    ppm = np.linspace(LEFT_PPM, RIGHT_PPM, TARGET_SIZE)

    # CORRECTION: ppm * MHz = Hz (The factors cancel out)
    hz = ppm * SFO1_MHZ 

    # center frequency axis
    hz_centered = hz - hz.mean()

    outname = fname.replace("asci_", "spec_").replace("_trim.txt", ".dat")
    out = np.column_stack([hz_centered, intensity])
    np.savetxt(outname, out, fmt="%12.6f %12.6e")

    print(f"  wrote {outname}")

print("\nAll spectra converted successfully.")
