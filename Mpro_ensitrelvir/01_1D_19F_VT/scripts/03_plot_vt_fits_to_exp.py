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
# 2. UNIFIED PHYSICS ENGINE (Standardized Symmetric Definitions)
# ============================================================
def two_lorentzians_symmetric(freq, wA, wB, center, separation, pB, height):
    # Core physics engine uses HWHM. Inputs from NPZ are FWHM, so they get halved below.
    center_A = center + separation / 2.0
    center_B = center - separation / 2.0

    peak_A = (1.0 - pB) * (wA**2 / ((freq - center_A)**2 + wA**2))
    peak_B = pB * (wB**2 / ((freq - center_B)**2 + wB**2))
    return height * (peak_A + peak_B)

def load_and_chop_spectra(directory):
    pattern = os.path.join(directory, "spec_*.dat")
    file_list = sorted(glob.glob(pattern))
    data_storage = {} 
    for fname in file_list:
        try:
            # FIXED: Handles 'spec_283K_LB50.dat' naming seamlessly
            base_name = os.path.basename(fname)
            temp_str = base_name.split("_")[1].replace("K", "").replace(".dat", "")
            T = int(temp_str)
            
            raw = np.loadtxt(fname)
            if np.max(np.abs(raw[:,0])) < 500.0: 
                raw[:,0] *= 1e3 
            regions_list = []
            for center in PEAK_CENTERS:
                mask = (raw[:,0] > center - WINDOW_HALF_WIDTH) & \
                       (raw[:,0] < center + WINDOW_HALF_WIDTH)
                freq_chunk, int_chunk = raw[mask, 0], raw[mask, 1]
                if len(int_chunk) > 0:
                    int_chunk -= np.min(int_chunk) 
                    if np.max(int_chunk) > 0: 
                        int_chunk /= np.max(int_chunk)
                regions_list.append((freq_chunk - center, int_chunk))
            data_storage[T] = regions_list
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse file {os.path.basename(fname)}: {e}")
    return data_storage

# ============================================================
# 3. EXECUTION & RENDERING
# ============================================================
spectra_data = load_and_chop_spectra(DATA_DIR)

if not os.path.exists(GLOBAL_NPZ):
    print(f"❌ ERROR: Global fit archive not found at: {GLOBAL_NPZ}. Run Script 02b first.")
    exit()

data_archive = np.load(GLOBAL_NPZ, allow_pickle=True)

# FIXED: Safely unpack the saved dictionary array from NumPy archive
if data_archive["export_data"].ndim == 0:
    export_data = data_archive["export_data"].item()
else:
    export_data = data_archive["export_data"]

T_vals = data_archive["T_vals"]

fig = plt.figure(figsize=(15, 7), dpi=150)
gs = gridspec.GridSpec(1, 3)
titles = ["Left Peak", "Middle Peak", "Right Peak"]
fit_color = "#D62728" 

for p_idx in range(3):
    ax = plt.subplot(gs[p_idx])
    y_spacing = 1.3
    
    for i, T in enumerate(T_vals):
        d = export_data[i]
        freq, y = spectra_data[T][p_idx]
        pB = d["Population_B"]
        
        # FIXED: Convert true FWHM values back to HWHM (divide by 2.0) for the physics equation
        if p_idx == 0:     # Left
            sim = two_lorentzians_symmetric(freq, d["Left_wA"]/2.0, d["Left_wB"]/2.0, d["Left_cen"], d["Left_sep"], pB, d["Left_h"])
        elif p_idx == 1:   # Middle
            freq_shifted = freq - d["Middle_midpoint"]
            sim = two_lorentzians_symmetric(freq_shifted, d["Middle_wA"]/2.0, d["Middle_wB"]/2.0, 0.0, d["Middle_sep"], pB, d["Middle_h"])
        else:              # Right
            sim = two_lorentzians_symmetric(freq, d["Right_wA"]/2.0, d["Right_wB"]/2.0, d["Right_cen"], d["Right_sep"], pB, d["Right_h"])

        offset = i * y_spacing
        ax.plot(freq, y + offset, color='#2c3e50', linewidth=1.5, alpha=0.8)
        ax.plot(freq, y + offset, 'o', color='#2c3e50', ms=2, alpha=0.2, markeredgewidth=0)
        ax.plot(freq, sim + offset, linestyle='--', color=fit_color, lw=2.5, alpha=0.9)
        
        if p_idx == 0:
            ax.text(np.min(freq) - 150, offset + 0.35, f"{T} K", fontsize=20, fontweight="regular")

    ax.set_title(titles[p_idx], fontsize=28, fontweight='bold', pad=15)
    ax.invert_xaxis()
    ax.set_yticks([])
    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.spines[['top', 'right', 'left']].set_visible(False)

plt.tight_layout()
os.makedirs(FIGURE_DIR, exist_ok=True)

# 1. Save the outputs first
plt.savefig(os.path.join(FIGURE_DIR, "03_VT_fit_600MHz_data.png"), dpi=300)
plt.savefig(os.path.join(FIGURE_DIR, "03_VT_fit_600MHz_data.pdf"), bbox_inches='tight')
print("🎉 Success! Plots saved cleanly to the figures directory.")

# 2. Safely attempt to pop up the window if a display environment exists
if "DISPLAY" in os.environ or os.name == 'nt' or sys.platform == 'darwin':
    print("📊 Opening interactive figure window...")
    plt.show()
else:
    print("🖥️ Headless environment detected. Skipping interactive figure popup.")
