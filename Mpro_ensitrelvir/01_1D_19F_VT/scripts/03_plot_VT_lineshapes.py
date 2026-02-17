import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import glob, os

# ============================================================
# 1. STYLE & PATH SETUP
# ============================================================
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["pdf.fonttype"] = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_ascii", "ensitrelvir_VT_LB50_600")
GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")

PEAK_CENTERS = [ 6000.0,  -5000.0,  -7000.0 ] 
WINDOW_HALF_WIDTH = 2000.0  

# ============================================================
# 2. PHYSICS & LOADING ENGINE (Matches Script 02 Exactly)
# ============================================================
def two_lorentzians(freq, wA, wB, center, separation, pB, height):
    center_A = center + (separation * pB) 
    peak_A = (1.0 - pB) * (wA**2 / ((freq - center_A)**2 + wA**2))
    center_B = center - (separation * (1.0 - pB)) 
    peak_B = pB * (wB**2 / ((freq - center_B)**2 + wB**2))
    return height * (peak_A + peak_B)

def load_and_chop_spectra(directory):
    pattern = os.path.join(directory, "spec_*.dat")
    file_list = sorted(glob.glob(pattern))
    data_storage = {} 
    for fname in file_list:
        try:
            T = int(os.path.basename(fname).split("_")[1].replace(".dat",""))
            raw = np.loadtxt(fname)
            if np.max(np.abs(raw[:,0])) < 500.0: raw[:,0] *= 1e3 
            regions_list = []
            for center in PEAK_CENTERS:
                mask = (raw[:,0] > center - WINDOW_HALF_WIDTH) & \
                       (raw[:,0] < center + WINDOW_HALF_WIDTH)
                freq_chunk, int_chunk = raw[mask, 0], raw[mask, 1]
                if len(int_chunk) > 0:
                    int_chunk -= np.min(int_chunk) 
                    if np.max(int_chunk) > 0: int_chunk /= np.max(int_chunk)
                regions_list.append((freq_chunk - center, int_chunk))
            data_storage[T] = regions_list
        except: pass
    return data_storage

# ============================================================
# 3. EXECUTION
# ============================================================
# Load raw data and the fit archive
spectra_data = load_and_chop_spectra(DATA_DIR)
data_archive = np.load(GLOBAL_NPZ, allow_pickle=True)
export_data = data_archive["export_data"]
T_vals = data_archive["T_vals"]

fig = plt.figure(figsize=(14, 8), dpi=150)
gs = gridspec.GridSpec(1, 3)
titles = ["Left Peak (6 kHz)", "Middle Peak", "Right Peak (-7 kHz)"]
fit_color = "#D50032" 

for p_idx in range(3):
    ax = plt.subplot(gs[p_idx])
    y_spacing = 1.3
    for i, T in enumerate(T_vals):
        d = export_data[i]
        freq, y = spectra_data[T][p_idx]
        pB = d["Population_B"]
        
        # Map back to the specific parameters that gave you the "Good Fit"
        if p_idx == 0:   # Left
            sim = two_lorentzians(freq, d["Left_wA"], d["Left_wB"], d["Left_cen"], d["Left_sep"], pB, d["Left_h"])
        elif p_idx == 1: # Middle
            sim = two_lorentzians(freq, d["Middle_wA"], d["Middle_wB"], d["Middle_cen"], d["Middle_sep"], pB, d["Middle_h"])
        else:            # Right
            sim = two_lorentzians(freq, d["Right_wA"], d["Right_wB"], d["Right_cen"], d["Right_sep"], pB, d["Right_h"])

        offset = i * y_spacing
        ax.plot(freq, y + offset, 'o', color='#2c3e50', ms=2, alpha=0.3, markeredgewidth=0)
        ax.plot(freq, sim + offset, '-', color=fit_color, lw=1.5)
        if p_idx == 0:
            ax.text(np.min(freq)-100, offset + 0.35, f"{T} K", fontsize=11)

    ax.set_title(titles[p_idx], fontsize=16, fontweight='bold', pad=15)
    ax.invert_xaxis()
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.spines[['top', 'right', 'left']].set_visible(False)

plt.tight_layout()
os.makedirs(FIGURE_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURE_DIR, "VT_lineshapes_fits.png"), dpi=300)
plt.show()
