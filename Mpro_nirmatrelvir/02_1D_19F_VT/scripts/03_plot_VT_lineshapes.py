import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob, os

# ============================================================
# 1. STYLE & PATH SETUP
# ============================================================
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]
mpl.rcParams["axes.labelsize"] = 18    # Larger axis labels
mpl.rcParams["xtick.labelsize"] = 14   # Larger tick labels
mpl.rcParams["pdf.fonttype"] = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figures")
DATA_ROOT = os.path.join(PROJECT_ROOT, "processed_ascii")

DIR_600 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_600")
DIR_800 = os.path.join(DATA_ROOT, "nirmat_VT_LB10_800")
GLOBAL_NPZ = os.path.join(OUTPUT_DIR, "vt_global_fit.npz")

STACK_OFFSET = 1.3
LABEL_OFFSET = 0.35
FIT_WINDOW_WIDTH = 1200.0

# ============================================================
# 2. LOAD GLOBAL FIT RESULTS
# ============================================================
fit_data = np.load(GLOBAL_NPZ)
xg = fit_data["xg"]
# Params: [dG#, dH#, dG0, dH0, r2a6, r2b6, r2a8, r2b8, b6, b8, dnu6, g6, dnu8, g8, shifts...]
dG, dH, dG_eq, dH_eq = xg[:4]
r2a6, r2b6, r2a8, r2b8 = xg[4:8]
b6, b8 = xg[8:10]
dnu6, g6, dnu8, g8 = xg[10:14]

kB = 1.3806e-23; h = 6.626e-34; R = 8.314; T0 = 298.15

# ============================================================
# 3. MODELS
# ============================================================
def get_visc_scale(T): return 10**(247.8/(T-140) - 247.8/(T0-140))

def kex_eyring(T):
    dG_T = dH*4184*(1 - T/T0) + dG*4184*(T/T0)
    return (kB * T / h) * np.exp(-dG_T / (8.314 * T))

def pB_vanthoff(T):
    dG_T = dH_eq*4184*(1 - T/T0) + dG_eq*4184*(T/T0)
    K = np.exp(-dG_T / (8.314 * T))
    return K / (1 + K)

def dnmr_2site(freq, kex, shift, r2a, r2b, dnu, pB):
    pi = np.pi; pA = 1.0 - pB; v = freq - shift
    aA = r2a*pi - 1j*(2*pi*(0.5*dnu - v))
    aB = r2b*pi - 1j*(2*pi*(-0.5*dnu - v))
    num = pA*aB + pB*aA + kex; den = aA*aB + kex*(aA+aB)
    return np.abs((num/den).real)

# ============================================================
# 4. LOAD AND NORMALIZE
# ============================================================
def load_spectra(directory):
    out = {}
    files = sorted(glob.glob(os.path.join(directory, "spec_*.dat")))
    for f in files:
        T = int(os.path.basename(f).split("_")[1].replace(".dat", ""))
        d = np.loadtxt(f)
        d[:, 0] -= d[np.argmax(d[:, 1]), 0]
        mask = np.abs(d[:, 0]) <= FIT_WINDOW_WIDTH / 2
        out[T] = (d[mask, 0], d[mask, 1] / np.max(d[mask, 1]))
    return out

spec600, spec800 = load_spectra(DIR_600), load_spectra(DIR_800)
temps600, temps800 = sorted(spec600), sorted(spec800)

# ============================================================
# 5. PLOTTING ENGINE 
# ============================================================
def plot_stack(ax, spectra, field="600"):
    temps = sorted(spectra)
    incl_600 = [t for t in temps600 if t != 283]
    incl_800 = [t for t in temps800 if t != 283]
    
    for i, T in enumerate(temps):
        f, y = spectra[T]
        sc, k, pB = get_visc_scale(T), kex_eyring(T), pB_vanthoff(T)

        if field == "600":
            r2a, r2b, dnu, base = r2a6*sc, r2b6*sc, dnu6 + g6*(T-T0), b6
            fit_color = "#D50032"
            if T in incl_600:
                shift = xg[14 + incl_600.index(T)]; style = "-"
            else:
                sim_tmp = dnmr_2site(f, k, 0.0, r2a, r2b, dnu, pB)
                shift = f[np.argmax(y)] - f[np.argmax(sim_tmp)]; style = "--"
        else:
            r2a, r2b, dnu, base = r2a8*sc, r2b8*sc, dnu8 + g8*(T-T0), b8
            fit_color = "navy"
            if T in incl_800:
                shift = xg[14 + len(incl_600) + incl_800.index(T)]; style = "-"
            else:
                sim_tmp = dnmr_2site(f, k, 0.0, r2a, r2b, dnu, pB)
                shift = f[np.argmax(y)] - f[np.argmax(sim_tmp)]; style = "--"

        sim = dnmr_2site(f, k, shift, r2a, r2b, dnu, pB)
        sim /= np.max(sim)
        offset = i * STACK_OFFSET

        ax.plot(f, y + offset, color="black", lw=1.5)
        ax.plot(f, sim + base + offset, color=fit_color, lw=1.5, ls=style)
        ax.text(f[0]-50, offset+LABEL_OFFSET, f"{T} K", fontsize=11)

    ax.invert_xaxis()
    ax.set_yticks([])
    # Reduce tick density for a cleaner look
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))

# ============================================================
# 6. GENERATE FIGURE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
plot_stack(axes[0], spec600, field="600")
plot_stack(axes[1], spec800, field="800")

axes[0].set_title("600 MHz", fontsize=14, fontweight="bold")
axes[1].set_title("800 MHz", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Normalized Intensity (offset)")
for ax in axes: ax.set_xlabel("Frequency (Hz)")

plt.tight_layout()
os.makedirs(FIGURE_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURE_DIR, "VT_lineshape_fits.png"), dpi=300)
plt.show()
