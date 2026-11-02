import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical order (reference palette, light mode)
C = {"mae": "#2a78d6", "eps": "#eb6834", "boltz": "#1baf7a", "noisy": "#eda100",
     "mae_noreheat": "#6250d6", "mae_fixed": "#e87ba4", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}
LS = {"mae": "-", "eps": "--", "boltz": "-.", "noisy": ":", "mae_noreheat": (0, (5, 1, 1, 1)), "mae_fixed": (0, (1, 1))}
LABEL = {"mae": "MAE (proposed)", "eps": r"$\epsilon$-greedy", "boltz": "Boltzmann", "noisy": "NoisyNet",
         "mae_noreheat": "MAE w/o reheating", "mae_fixed": "MAE w/o rate control"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.5, "lines.linewidth": 1.4, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL1 = 3.5   # IEEE single column (in)
COL2 = 7.16  # IEEE double column (in)
