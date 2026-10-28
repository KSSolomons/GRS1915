"""
Kinematics figure for the paper: sky positions of the lobes (left) and their separation along the jet axis against time
with the two models of Sect. 3 (right).

(a) Lobe offsets from the fitted core (Epochs 5-11, frame-corrected), coloured by date, with the fitted jet axis and the
    scatter-angle envelope of the linear sky-motion fit.
(b) Separation along the jet axis with the linear sky-motion fit (posterior median and 68 per cent band).
(c) The same with the decelerating model at fixed launch speeds beta_0 = 0.5, 0.7 and 0.99 (D = 9.4 kpc),
    in blue, green and vermillion (--beta0-family).
Grey points are the perpendicular offsets. The shaded band marks the onset of the large flare (MJD 60673-60686).

Inputs: data/absolute_kinematics/absolute_kinematics_plot_data_model1_without_limits.npz (linear fit and the positions),
data/physical_kinematics_samples_decel_sym_nolimits_b<beta_0>.npz (decelerating fits at fixed beta_0).

  python src/plot_kinematics_figure.py [--out figures/paper/kinematics.png]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ltt_model import separation

p = argparse.ArgumentParser()
p.add_argument("--out", default="figures/paper/kinematics.png")
p.add_argument("--n-draws", type=int, default=2000, help="Posterior draws for the model bands")
p.add_argument("--dist", type=float, default=9.4, help="Distance of the decelerating fit (kpc)")
p.add_argument("--beta0-family", type=float, nargs="+", default=[0.5, 0.7, 0.99],
               help="Panel (c): one decelerating fit per fixed beta_0 (needs its samples file); one value gives the single-model style")
args = p.parse_args()
rng = np.random.default_rng(1915)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

LINEAR = dict(color="#E69F00", label="Linear model")
DECEL = dict(color="#009E73", label=r"Decelerating model ($\beta_0 = 0.99$)")
FLARE = (60673.3, 60686.3)  # onset of the large flare, between the last quiet L-band point and Epoch 1

z = np.load("data/absolute_kinematics/absolute_kinematics_plot_data_model1_without_limits.npz")
lin = z["flat_samples"]  # T_ej, v_n, v_s, PA_n, PA_s, theta_scatter, x_c, y_c
pa_n, pa_s = np.median(lin[:, 3]), np.median(lin[:, 4])
scatter = np.median(lin[:, 5])

lobes = {}
for side, pa in (("n", pa_n), ("s", pa_s)):
    t = z[f"{side}_mjd"]
    dx, dy = z[f"{side}_da"], z[f"{side}_dd"]  # already offsets from the fitted core (pipeline n_da = x - x_c)
    ex, ey = z[f"{side}_da_err"], z[f"{side}_dd_err"]
    s, c = np.sin(np.radians(pa)), np.cos(np.radians(pa))
    # Same projection as the fits: r_par along PA (positive towards the NW for both, PA_s points NW too)
    lobes[side] = dict(t=t, dx=dx, dy=dy, ex=ex, ey=ey,
                       par=dx * s + dy * c, perp=dx * c - dy * s,
                       epar=np.hypot(ex * s, ey * c), eperp=np.hypot(ex * c, ey * s))

t_grid = np.linspace(60250, 61010, 400)


def band(curves):
    return np.percentile(curves, [16, 50, 84], axis=0)


draws = lin[rng.choice(len(lin), args.n_draws, replace=False)]
dt = np.clip(t_grid[None, :] - draws[:, [0]], 0.0, None)
lin_n = band(np.abs(draws[:, [1]]) * dt)
lin_s = band(-np.abs(draws[:, [2]]) * dt)

def decel_bands(beta0_fixed):
    """Posterior median and 68 per cent band of r_par for the decelerating fit with beta_0 held at beta0_fixed."""
    zd = np.load(f"data/physical_kinematics_samples_decel_sym_nolimits_b{beta0_fixed:g}.npz")
    names = list(zd["names"])
    dec = zd["samples"][rng.choice(len(zd["samples"]), args.n_draws, replace=False)]
    col = {n: dec[:, [names.index(n)]] for n in names}
    gb = 10 ** col["lg_gb0"]
    kin = (col["t_ej"], gb / np.sqrt(1 + gb**2), col["theta"], col["tau"], args.dist)
    return band(separation(t_grid[None, :], *kin, +1)), band(-separation(t_grid[None, :], *kin, -1))


dec_n, dec_s = decel_bands(0.99)
family = []
if args.beta0_family:
    # Distinct Okabe-Ito hues (no orange, which is the linear model), so the overlapping bands stay separable;
    # every pair keeps OKLab dE >= 8.6 under protan, deutan and tritan simulation
    shades = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"][: len(args.beta0_family)]
    family = [(b0, shade, *decel_bands(b0)) for b0, shade in zip(sorted(args.beta0_family), shades)]

fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.45], wspace=0.38, hspace=0.18)

# (a) sky positions
ax = fig.add_subplot(gs[:, 0])
norm = Normalize(60780, 60985)
cmap = plt.get_cmap("viridis")
r_max = 6.0
for pa in (pa_n, pa_s + 180.0):
    a1, a2 = np.radians(pa - scatter), np.radians(pa + scatter)
    ax.fill([0, r_max * np.sin(a1), r_max * np.sin(a2)], [0, r_max * np.cos(a1), r_max * np.cos(a2)],
            color="0.92", lw=0, zorder=0)
axis = np.radians(0.5 * (pa_n + pa_s))  # both PAs point NW in the fit convention
ax.plot([-r_max * np.sin(axis), r_max * np.sin(axis)], [-r_max * np.cos(axis), r_max * np.cos(axis)],
        ls="--", color="0.55", lw=0.7, zorder=1)
for side, marker in (("n", "^"), ("s", "v")):
    L = lobes[side]
    order = np.argsort(L["t"])
    # Path of the lobe in time order, so the drift between epochs can be followed
    ax.plot(L["dx"][order], L["dy"][order], color="0.35", lw=0.6, zorder=2)
    ax.errorbar(L["dx"], L["dy"], xerr=L["ex"], yerr=L["ey"], fmt="none", ecolor="0.6", elinewidth=0.5,
                alpha=0.7, zorder=2)
    sc = ax.scatter(L["dx"], L["dy"], c=L["t"], cmap=cmap, norm=norm, marker=marker, s=42, edgecolor="black",
                    linewidth=0.5, zorder=3)
ax.plot(0, 0, "*", color="black", ms=10, mec="white", mew=0.6, zorder=4)
ax.text(-0.25, -0.15, "core", fontsize=6.5, ha="left", va="top")
# Jet axis quoted by its SE-pointing value, East of North, as in the text
ax.text(-2.2, 5.6, f"PA = {(0.5 * (pa_n + pa_s) + 180) % 360:.0f}$^\\circ$ E of N", fontsize=6.5, color="0.4", ha="right")
ax.set_xlim(3.4, -2.4)
ax.set_ylim(-5.0, 4.8)
# Equal scales on both axes; the limits (not the box) adjust, so the panel fills its full height
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel(r"$\Delta\alpha\cos\delta$ offset (arcsec)", fontsize=8)
ax.set_ylabel(r"$\Delta\delta$ offset (arcsec)", fontsize=8)
ax.tick_params(labelsize=7, direction="in", top=True, right=True)
ax.set_title("(a)", fontsize=8, loc="left")
# Compass (East to the left) in the empty south-west corner
cx, cy, arm = -1.4, -3.9, 0.8
ax.annotate("", xy=(cx, cy + arm), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.2"))
ax.annotate("", xy=(cx + arm, cy), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.2"))
ax.text(cx, cy + arm + 0.1, "N", fontsize=6.5, ha="center", va="bottom")
ax.text(cx + arm + 0.1, cy, "E", fontsize=6.5, ha="right", va="center")
# Date colour bar along the right side of the panel, in MJD
cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.03, fraction=0.07, aspect=40)
cb.set_ticks([60800, 60850, 60900, 60950])
cb.set_label("MJD", fontsize=9, labelpad=3)
cb.ax.tick_params(labelsize=8, length=2.5)
plt.setp(cb.ax.get_yticklabels(), rotation=90, va="center")

# (b, c) separation along the jet axis
for k, (model, nb, sb, title) in enumerate(((LINEAR, lin_n, lin_s, "(b)"), (DECEL, dec_n, dec_s, "(c)"))):
    ax = fig.add_subplot(gs[k, 1])
    ax.axvspan(*FLARE, color="0.85", lw=0, zorder=0)
    curves = [(model["color"], model["label"], nb, sb)]
    if k == 1 and family:
        curves = [(shade, rf"$\beta_0 = {b0:g}$", fn, fs) for b0, shade, fn, fs in family]
    for color, label, cn, cs in curves:
        for b in (cn, cs):
            # Nothing before launch: the band starts where any draw has launched, the median where the median has
            moving = (b[0] != 0) | (b[2] != 0)
            ax.fill_between(t_grid, b[0], b[2], where=moving, color=color, alpha=0.3 if len(curves) == 1 else 0.2,
                            lw=0)
            ax.plot(np.where(b[1] != 0, t_grid, np.nan), b[1], color=color, lw=1.4 if len(curves) == 1 else 1.1,
                    label=label if b is cn else None)
    for side, marker in (("n", "^"), ("s", "v")):
        L = lobes[side]
        ax.errorbar(L["t"], L["par"], yerr=L["epar"], fmt=marker, ms=4, color="black", mfc="black", elinewidth=0.6)
        ax.errorbar(L["t"], L["perp"], yerr=L["eperp"], fmt=marker, ms=3, color="0.6", mfc="none", elinewidth=0.5)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(t_grid[0], t_grid[-1])
    ax.set_ylim(-5.2, 5.2)
    ax.set_ylabel(r"$r_\parallel$ (arcsec)", fontsize=8)
    ax.tick_params(labelsize=7, direction="in")
    if len(curves) > 1:
        ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2, columnspacing=1.0, handlelength=1.5,
                  title="Decelerating model", title_fontsize=6.5, alignment="left")
    else:
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_title(title, fontsize=8, loc="left")
    if k == 0:
        ax.tick_params(labelbottom=False)
        top = ax.secondary_xaxis("top", functions=(lambda m: m, lambda m: m))
        dates = ["2024-04-01", "2024-10-01", "2025-04-01", "2025-10-01"]
        top.set_xticks([Time(d).mjd for d in dates])
        top.set_xticklabels([Time(d).to_datetime().strftime("%Y %b") for d in dates], fontsize=6.5)
        ax.text(np.mean(FLARE) - 8, -4.9, "flare\nonset", fontsize=6, ha="right", va="bottom", color="0.35")
    else:
        ax.set_xlabel("MJD", fontsize=8)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=300, bbox_inches="tight")
fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
print(f"Saved {args.out}")
