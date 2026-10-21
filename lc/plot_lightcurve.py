import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Typography for paper-ready plots
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

data_file = "lc/MeerKAT_GRS1915+105_all_NEW.txt"

mjds = []
fluxes_mJy = []
errs_mJy = []

with open(data_file, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                mjd = float(parts[1])
                flux_jy = float(parts[2])
                err_jy = float(parts[5])  # Err_Total

                mjds.append(mjd)
                fluxes_mJy.append(flux_jy * 1000.0)
                errs_mJy.append(err_jy * 1000.0)
            except ValueError:
                continue

mjds = np.array(mjds)
fluxes_mJy = np.array(fluxes_mJy)
errs_mJy = np.array(errs_mJy)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.errorbar(
    mjds,
    fluxes_mJy,
    yerr=errs_mJy,
    fmt="o",
    color="black",
    markersize=6,
    capsize=3,
    elinewidth=1.2,
    markeredgecolor="black",
    markerfacecolor="dodgerblue",
    zorder=4,
    label="MeerKAT (1.28 GHz)",
)

import datetime

target_dates = [
    "2025-01-11 06:03:36",
    "2025-02-02 06:13:52",
    "2025-03-02 02:44:41",
    "2025-03-24 01:35:33",
    "2025-04-26 23:09:36",
    "2025-06-10 20:15:37",
    "2025-06-28 19:00:35",
    "2025-07-26 17:12:41",
    "2025-08-29 14:59:34",
    "2025-10-04 12:32:38",
    "2025-10-25 11:33:34.255",
    "2025-12-02 08:45:24.052",
]

all_epochs = []
for d in target_dates:
    try:
        dt = datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        dt = datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f")
    ts = dt.timestamp()
    all_epochs.append(ts / 86400.0 + 40587.0)

from matplotlib.colors import Normalize

norm = Normalize(vmin=min(all_epochs), vmax=max(all_epochs))
cmap = plt.cm.viridis

for i, mjd in enumerate(all_epochs):
    label = "MeerKAT S-band Epochs" if i == 0 else None
    ax.axvline(
        x=mjd, color=cmap(norm(mjd)), linestyle="--", alpha=0.8, zorder=1, label=label
    )

# Aesthetics
ax.set_xlabel("Time (MJD)", fontsize=18)
ax.set_ylabel("Flux Density (mJy)", fontsize=18)
ax.set_yscale("log")
ax.set_xlim(left=60000)

ax.tick_params(axis="both", which="major", labelsize=14, length=7, width=1.2)
ax.tick_params(axis="both", which="minor", labelsize=14, length=4, width=1)
ax.minorticks_on()

ax.legend(loc="best", fontsize=16, frameon=True, edgecolor="black")

plt.tight_layout()
output_path = "lc/MeerKAT_GRS1915_Lightcurve.png"
plt.savefig(output_path, bbox_inches="tight")
plt.close(fig)

print(f"Plot saved to {output_path}")
