"""
Appendix figure: apparent position angle of each lobe, measured from the fitted core, in every resolved epoch.

The SE lobe is plotted at PA + 180 deg so that both lobes sit on the same (NW-pointing) jet axis, as in the fits.
The fitted PA_n and PA_s + 180 of the linear model (posterior median and 68 per cent band) and the scatter angle
theta_scatter about their mean are overplotted. PA errors combine the lobe and fitted-core position errors to first
order, sigma_PA = sqrt((dy sigma_x)^2 + (dx sigma_y)^2) / r^2.

Input: data/absolute_kinematics/absolute_kinematics_plot_data_model1_without_limits.npz (positions and linear fit).
Also prints the per-epoch PAs and the chi-square of each lobe about its fitted PA.

  python src/plot_pa_evolution.py [--out figures/paper/pa_evolution.png]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out", default="figures/paper/pa_evolution.png")
args = p.parse_args()

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

LINEAR = "#E69F00"  # linear-model colour of Fig. 3

z = np.load("data/absolute_kinematics/absolute_kinematics_plot_data_model1_without_limits.npz")
lin = z["flat_samples"]  # T_ej, v_n, v_s, PA_n, PA_s, theta_scatter, x_c, y_c
ex_c, ey_c = np.std(lin[:, 6]), np.std(lin[:, 7])
scatter = np.median(lin[:, 5])
# Fit convention: both PAs point NW, so PA_s here is already the SE lobe's PA + 180
fit_pa = {"n": lin[:, 3] % 360, "s": lin[:, 4] % 360}
axis_pa = np.median(0.5 * (fit_pa["n"] + fit_pa["s"]))

fig, ax = plt.subplots(figsize=(3.6, 2.8))
t_lo, t_hi = 60770, 61030  # Epoch 12 (NW only) at MJD 61011
for side, ls in (("n", "-"), ("s", "--")):
    q16, q50, q84 = np.percentile(fit_pa[side], [16, 50, 84])
    ax.axhspan(q16, q84, color=LINEAR, alpha=0.2, lw=0)
    ax.axhline(q50, color=LINEAR, lw=1.1, ls=ls)
for sgn in (-1, 1):
    ax.axhline(axis_pa + sgn * scatter, color="0.5", lw=0.7, ls=":")

print(f"Fitted core error ({ex_c:.4f}, {ey_c:.4f}) arcsec; theta_scatter {scatter:.2f} deg")
for side, marker, mfc, dt in (("n", "^", "black", -1.5), ("s", "v", "white", 1.5)):
    t = z[f"{side}_mjd"]
    dx, dy = z[f"{side}_da"], z[f"{side}_dd"]  # already offsets from the fitted core (pipeline n_da = x - x_c)
    ex, ey = np.hypot(z[f"{side}_da_err"], ex_c), np.hypot(z[f"{side}_dd_err"], ey_c)
    pa = np.degrees(np.arctan2(dx, dy)) % 360
    if side == "s":
        pa = (pa + 180) % 360
    err = np.degrees(np.hypot(dy * ex, dx * ey) / (dx**2 + dy**2))
    ax.errorbar(t + dt, pa, yerr=err, fmt=marker, ms=4.5, color="black", mfc=mfc, mew=0.7, elinewidth=0.6,
                label="NW (receding)" if side == "n" else r"SE (approaching), PA $+\,180^\circ$")
    ref = np.median(fit_pa[side])
    chi2 = np.sum(((pa - ref) / err) ** 2)
    print(f"{side.upper()}: " + ", ".join(f"{m:.1f}: {a:.1f}+/-{e:.1f}" for m, a, e in zip(t, pa, err)))
    print(f"   about fitted PA {ref:.2f}: chi2 = {chi2:.1f} for {len(t)} points; "
          f"rms {np.sqrt(np.mean((pa - ref) ** 2)):.1f} deg, max |dPA| {np.max(np.abs(pa - ref)):.1f} deg")

ax.set_xlim(t_lo, t_hi)
ax.set_xticks([60800, 60850, 60900, 60950])
ax.set_ylim(axis_pa - 30, axis_pa + 32)
ax.set_xlabel("MJD", fontsize=8)
ax.set_ylabel(r"PA ($^\circ$ E of N)", fontsize=8)
# The same angle read for the SE lobe, which points the opposite way
sec = ax.secondary_yaxis("right", functions=(lambda a: a - 180, lambda a: a + 180))
sec.set_ylabel(r"SE lobe PA ($^\circ$ E of N)", fontsize=8)
sec.tick_params(labelsize=7, direction="in")
ax.tick_params(labelsize=7, direction="in", top=True)
ax.legend(loc="upper left", fontsize=6.5, frameon=False, handletextpad=0.3)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=300, bbox_inches="tight")
fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
print(f"Saved {args.out}")
