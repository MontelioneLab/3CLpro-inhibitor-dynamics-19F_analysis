import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. SETUP & DIRECTORY STRUCTURE
# ============================================================
# Path logic for GitHub structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_ascii", "C145A_nirmat_298K_LB5_600MHz")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output" )
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# File names
FILE_NAME = os.path.join(DATA_DIR, 'spec_298.dat')
TXT_OUTPUT = os.path.join(OUTPUT_DIR, 'C145A_fit_params.txt')
FIG_OUTPUT = os.path.join(FIGURE_DIR, 'C145A_line-shape_fit.png')

# CONSTANTS FROM WT FIT
# Intrinsic LW without LB. At 600 MHz and 298K, LWA = 33.3 Hz. LWB = 77.3 Hz.
# The C145A-NVM data was processed with LB=5.
#WT_WIDTH_A = 49.7
#WT_WIDTH_B = 88.9
WT_WIDTH_A = 38.3
WT_WIDTH_B = 82.3

# ============================================================
# 2. PHYSICS MODEL (2-Site Exchange)
# ============================================================
def dnmr_2site_robust(freq_hz, kex, center_offset, r2a, r2b, delta_nu, pB):
    pA = 1.0 - pB
    pi = np.pi
    v = freq_hz - center_offset
    vA, vB = +0.5*delta_nu, -0.5*delta_nu
    w, wA, wB = 2*pi*v, 2*pi*vA, 2*pi*vB
    R2A, R2B = r2a*pi, r2b*pi 
    aA = R2A - 1j*(wA - w)
    aB = R2B - 1j*(wB - w)
    numerator = (pA*aB + pB*aA + kex)
    denominator = (aA*aB + kex*(aA + aB))
    return np.abs( (numerator/denominator).real )

# ============================================================
# 3. LOAD DATA
# ============================================================
try:
    data = np.loadtxt(FILE_NAME)
    max_val = np.max(np.abs(data[:,0]))
    if max_val < 0.5: data[:,0] *= 1e6
    elif max_val < 500.0 and max_val > 0.5: data[:,0] *= 1e3
    
    max_idx = np.argmax(data[:,1])
    center_hz = data[max_idx, 0]
    data[:,0] -= center_hz
    
    mask = np.abs(data[:,0]) <= 600.0
    freq = data[mask, 0]
    y_exp = data[mask, 1]
    y_exp /= np.max(y_exp) 
    print(f"Loaded {FILE_NAME}")
except Exception as e:
    print(f"Error loading {FILE_NAME}: {e}")
    exit()

# ============================================================
# 4. FIT LOGIC
# ============================================================
def residuals(params):
    kex, pB, dNu, shift_offset, base = params
    sim = dnmr_2site_robust(freq, kex, shift_offset, WT_WIDTH_A, WT_WIDTH_B, dNu, pB)
    if np.max(sim) > 0: sim /= np.max(sim)
    return (sim + base) - y_exp

p0 = [100.0, 0.2, 100.0, 0.0, 0.0]
lower = [0.0, 0.01, 20.0, -100.0, -0.2]
upper = [5000, 0.50, 300.0, 100.0, 0.2]

result = least_squares(residuals, p0, bounds=(lower, upper))
x = result.x

# ============================================================
# 5. REPORT, SAVE, & PLOT
# ============================================================
# Writing to TXT file
with open(TXT_OUTPUT, "w") as f:
    f.write("=== MPro C145A MUTANT FIT RESULTS ===\n")
    f.write(f"Source Data     : {FILE_NAME}\n")
    f.write(f"Assumed Width A : {WT_WIDTH_A:.1f} Hz\n")
    f.write(f"Assumed Width B : {WT_WIDTH_B:.1f} Hz\n")
    f.write("-" * 35 + "\n")
    f.write(f"k_ex            : {x[0]:.2f} s^-1\n")
    f.write(f"Population B    : {x[1]*100:.2f} %\n")
    f.write(f"Separation (dNu): {x[2]:.2f} Hz\n")

print(f"Results written to {TXT_OUTPUT}")

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(freq, y_exp, 'ko', alpha=0.3, label='Data (C145A)')
sim_best = dnmr_2site_robust(freq, x[0], x[3], WT_WIDTH_A, WT_WIDTH_B, x[2], x[1])
sim_best /= np.max(sim_best)
plt.plot(freq, sim_best + x[4], 'r-', lw=2, label='Global Fit')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Intensity")
plt.title(f"MPro C145A Fit: pB={x[1]:.2f}, k={x[0]:.0f}")
plt.legend()
plt.gca().invert_xaxis()

# Saving the figure
plt.savefig(FIG_OUTPUT, dpi=300)
print(f"Figure saved to {FIG_OUTPUT}")
plt.show()
