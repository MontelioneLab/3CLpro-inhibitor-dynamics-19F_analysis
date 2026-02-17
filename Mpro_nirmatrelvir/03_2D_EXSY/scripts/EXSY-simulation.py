import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import matplotlib.gridspec as gridspec
import os

# ============================================================
# 1. SETUP & DIRECTORY STRUCTURE
# ============================================================
# Path logic for GitHub structure (same as your MPro scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# File names
TXT_OUTPUT = os.path.join(OUTPUT_DIR, 'EXSY_simulation_summary.txt')
FIG_OUTPUT = os.path.join(FIGURE_DIR, 'Simulated_EXSY.png')

# ============================================================
# 2. FINAL PARAMETERS
# ============================================================
k_ex     = 106.0   # s^-1
pB       = 0.44    # Minor state
pA       = 1.0 - pB
T1       = 1.60    # s 

# Widths (Intrinsic + 20 Hz LB)
width_A  = 33.0 + 20.0  # 53.0 Hz
width_B  = 77.0 + 20.0  # 97.0 Hz

# Frequencies (Hz)
freq_A   = +67.0
freq_B   = -67.0

# Calculated Rates
k_AB = k_ex * pB
k_BA = k_ex * pA

# ============================================================
# 3. CALCULATION ENGINE
# ============================================================
def get_intensities(tm):
    R1 = 1.0 / T1
    K = [[-(R1 + k_AB),   +k_BA    ],
         [  +k_AB,      -(R1 + k_BA)]]
    E = scipy.linalg.expm(np.array(K) * tm)
    
    # Magnetization (Volume)
    V_AA = E[0,0] * pA
    V_BA = E[0,1] * pB
    V_AB = E[1,0] * pA
    V_BB = E[1,1] * pB
    return V_AA, V_BB, V_AB, V_BA

def gaussian_2d_height(volume, sig_x, sig_y):
    return volume / (2 * np.pi * sig_x * sig_y)

def make_gaussian(X, Y, height, x0, y0, sig_x, sig_y):
    return height * np.exp(-((X-x0)**2/(2*sig_x**2) + (Y-y0)**2/(2*sig_y**2)))

# ============================================================
# 4. WRITE SUMMARY TO TXT
# ============================================================
with open(TXT_OUTPUT, "w") as f:
    f.write("=== EXSY SIMULATION PARAMETERS ===\n")
    f.write(f"Exchange Rate (k_ex) : {k_ex:.1f} s^-1\n")
    f.write(f"Population B (pB)    : {pB*100:.1f} %\n")
    f.write(f"Longitudinal T1      : {T1:.2f} s\n")
    f.write(f"k_AB / k_BA          : {k_AB:.2f} / {k_BA:.2f} s^-1\n")
    f.write("-" * 35 + "\n")
    f.write(f"Width A (effective)  : {width_A:.1f} Hz\n")
    f.write(f"Width B (effective)  : {width_B:.1f} Hz\n")
    f.write(f"Frequency Separation : {abs(freq_A - freq_B):.1f} Hz\n")
    f.write("-" * 35 + "\n")
    f.write("Mixing Time (ms) | Peak Ratios (Height-based)\n")
    for tm in [0.010, 0.030, 0.100, 0.200]:
        V_AA, V_BB, V_AB, V_BA = get_intensities(tm)
        sig_A = width_A / 2.355
        sig_B = width_B / 2.355
        h_AA = gaussian_2d_height(V_AA, sig_A, sig_A)
        h_BB = gaussian_2d_height(V_BB, sig_B, sig_B)
        h_AB = gaussian_2d_height(V_AB, sig_B, sig_A)
        r_diag = h_BB / h_AA
        r_cross = h_AB / h_AA
        f.write(f"{tm*1000:16.0f} | Diag B/A: {r_diag:.3f}, Cross/A: {r_cross:.3f}\n")

print(f"Summary written to {TXT_OUTPUT}")

# ============================================================
# 5. PLOTTING
# ============================================================
mix_times = [0.010, 0.030, 0.100, 0.200]
x = np.linspace(-200, 200, 300) 
y = np.linspace(-200, 200, 300)
X, Y = np.meshgrid(x, y)
sig_A = width_A / 2.355
sig_B = width_B / 2.355

fig = plt.figure(figsize=(12, 3.5), dpi=300) 
gs = gridspec.GridSpec(1, 4, wspace=0.05) 

plt.rcParams.update({
    'font.sans-serif': 'Arial', 
    'font.size': 10,
    'lines.linewidth': 1.0
})

for i, tm in enumerate(mix_times):
    ax = plt.subplot(gs[i])
    V_AA, V_BB, V_AB, V_BA = get_intensities(tm)
    h_AA = gaussian_2d_height(V_AA, sig_A, sig_A)
    h_BB = gaussian_2d_height(V_BB, sig_B, sig_B)
    h_AB = gaussian_2d_height(V_AB, sig_B, sig_A)
    h_BA = gaussian_2d_height(V_BA, sig_A, sig_B)
    
    Z = np.zeros_like(X)
    Z += make_gaussian(X, Y, h_AA, freq_A, freq_A, sig_A, sig_A)
    Z += make_gaussian(X, Y, h_BB, freq_B, freq_B, sig_B, sig_B)
    Z += make_gaussian(X, Y, h_AB, freq_B, freq_A, sig_B, sig_A)
    Z += make_gaussian(X, Y, h_BA, freq_A, freq_B, sig_A, sig_B)
   
# Dynamic Levels: Start at 15% of max to clear the base 
    max_z = np.max(Z)
    levels = np.linspace(0.15 * max_z, 1.05 * max_z, 15) 
# Linewidths
    ax.contour(X, Y, Z, levels=levels, colors=['#003366'], linewidths=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(150, -150) 
    ax.set_ylim(150, -150)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{tm*1000:.0f} ms", fontsize=11, fontweight='bold')
    
    ratio_BA = h_BB / h_AA
    ratio_CrossA = h_AB / h_AA 
    label_text = f"Diag B/A: {ratio_BA:.2f}\nCross/A: {ratio_CrossA:.2f}"
    ax.text(0.5, -0.05, label_text, ha='center', va='top', transform=ax.transAxes, fontsize=9, color='black', linespacing=1.3)

plt.tight_layout()
plt.savefig(FIG_OUTPUT, dpi=300, bbox_inches='tight')
print(f"Saved figure to {FIG_OUTPUT}")
plt.show()
