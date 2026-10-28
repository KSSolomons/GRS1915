"""
Epoch 3-4 figure for the paper: post-peel residuals with the predicted knot positions, and the detection threshold.

(a, b) Post-peel Epoch 3 and 4 images around the core, with the median predicted NW and SE knot positions (16-84 per cent
       range along the jet as a bar) of the linear sky-motion fit (Model 1) and the near-c
       decelerating model (decel_sym, beta_0 = 0.99, 9.4 kpc), all without limits, the jet axis and the r_lim circle.
(c, d) 5 sigma_emp(r) against radius for each epoch, with the expected knot flux S_exp: a knot of S_exp is detectable where
       S_exp lies above the curve (beyond r_lim). The predicted separations sit at S_exp.

Inputs: FITS/peel/ep{3,4}/ (postpeel_tools), data/peel_regions/limits_posterior_check.npz (check_limits_posterior.py).

  python src/plot_peel_limits_figure.py [--out figures/paper/peel_limits_ep34.png]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.patches import Circle, Ellipse, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epoch_limits import JET_PA_DEG as JET_PA
from postpeel_tools import PostPeelImage

p = argparse.ArgumentParser()
p.add_argument("--checks", default="data/peel_regions/limits_posterior_check.npz")
p.add_argument("--out", default="figures/paper/peel_limits_ep34.png")
p.add_argument("--half-width", type=float, default=9.0, help="Half-width of the image panels (arcsec)")
args = p.parse_args()

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Okabe-Ito colours (colour-blind safe), with a different marker per model as a second cue
# Sizes decrease so that coinciding predictions stay visible as nested markers
MODELS = {
    "ballistic": dict(label="Linear model", color="#E69F00", marker="o", ms=8),
    "decel_sym": dict(label=r"Decelerating model ($\beta_0 = 0.99$)", color="#009E73", marker="^", ms=5),
}
EPOCH_STYLE = {3: dict(ls="-", color="0.15"), 4: dict(ls="--", color="0.45")}

z = np.load(args.checks)
ul_r = z["ul_r"]
imgs = {3: PostPeelImage(3), 4: PostPeelImage(4)}


def predicted(model, j):
    """Median (E, N) of each knot at limit epoch j, and its 16-84 per cent separation range, per side."""
    out = {}
    for side, sign, pa_key in (("NW", 1.0, "pa_n"), ("SE", -1.0, "pa_s")):
        r = z[f"{model}_r_{side}"][:, j]
        pa = np.radians(np.median(z[f"{model}_{pa_key}"]))
        unit = sign * np.array([np.sin(pa), np.cos(pa)])
        q16, q50, q84 = np.percentile(r, [16, 50, 84])
        out[side] = (q50 * unit, q16 * unit, q84 * unit, q50)
    return out


fig = plt.figure(figsize=(7.0, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.32)
hw = args.half_width
radii = np.round(np.arange(1.0, 4.501, 0.05), 3)  # same grid as simple_limits.py
jet = np.radians(JET_PA)

for k, (ep, im) in enumerate(imgs.items()):
    j = k  # limit epochs are in time order: Epoch 3, then 4
    date = Time(im.mjd, format="mjd").to_datetime().strftime("%Y %b %d")
    pred = {model: predicted(model, j) for model in MODELS}

    # Top: post-peel image (signed residual, diverging map centred on zero)
    ax = fig.add_subplot(gs[0, k])
    pix = im.pix
    extent = [im.E[0, 0] + pix / 2, im.E[0, -1] - pix / 2, im.N[0, 0] - pix / 2, im.N[-1, 0] + pix / 2]
    vmax = np.percentile(np.abs(im.img[(im.r > 1.5) & (im.r < 6.0)]), 99.5)
    pcm = ax.imshow(im.img, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"mJy beam$^{-1}$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    # Jet axis through the core, r_lim circle, core, beam
    ax.plot([0.65 * hw * np.sin(jet), -0.65 * hw * np.sin(jet)], [0.65 * hw * np.cos(jet), -0.65 * hw * np.cos(jet)], color="0.35", lw=0.6, ls=(0, (6, 3)), zorder=1)
    ax.add_patch(Circle((0, 0), ul_r[j], fill=False, ec="black", ls=":", lw=1.1))
    ax.plot(0, 0, "+", color="black", ms=6, mew=0.9)
    # Beam in the lower-right (SW) corner on a white key box, so it is not read as part of the image
    a, b, phi = im.bmaj / 2, im.bmin / 2, np.radians(im.bpa_deg)
    ex, ey = np.hypot(a * np.sin(phi), b * np.cos(phi)), np.hypot(a * np.cos(phi), b * np.sin(phi))
    pad = 0.35
    ax.add_patch(Rectangle((-hw, -hw), 2 * (ex + pad), 2 * (ey + pad), fc="white", ec="0.4", lw=0.6, zorder=5))
    ax.add_patch(Ellipse((-hw + ex + pad, -hw + ey + pad), im.bmin, im.bmaj, angle=-im.bpa_deg,
                         fc="0.6", ec="black", lw=0.8, zorder=6))
    for model, sty in MODELS.items():
        for side, (mid, lo, hi, _) in pred[model].items():
            ax.plot([lo[0], hi[0]], [lo[1], hi[1]], color=sty["color"], lw=1.5, solid_capstyle="round", zorder=3)
            ax.plot(*mid, sty["marker"], ms=sty["ms"], mec="black", mew=0.6, zorder=4,
                    color=sty["color"] if side == "SE" else "white")
    for lab, s in (("SE", 1), ("NW", -1)):
        ax.text(s * 0.73 * hw * np.sin(jet), s * 0.73 * hw * np.cos(jet), lab, fontsize=6.5, color="0.2",
                ha="center", va="center", bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.8))
    ax.set_title(f"({'ab'[k]}) Epoch {ep}, {date}", fontsize=8, loc="left")
    ax.set_xlim(hw, -hw)
    ax.set_ylim(-hw, hw)
    ax.set_aspect("equal")
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_yticks([-8, -4, 0, 4, 8])
    ax.set_xlabel(r"$\Delta\alpha$ (arcsec)", fontsize=7.5)
    if k == 0:
        ax.set_ylabel(r"$\Delta\delta$ (arcsec)", fontsize=7.5)
    ax.tick_params(labelsize=6.5, direction="in")

    # Bottom: detection threshold against radius for this epoch
    ax = fig.add_subplot(gs[1, k])
    thr = np.array([5 * im.sigma_emp(r)[0] for r in radii])
    ax.axvspan(1.0, ul_r[j], color="0.9", lw=0, zorder=0)
    ax.plot(radii, thr, color="black", lw=1.4, label=r"$5\sigma_{\rm emp}(r)$")
    ax.plot(radii, 0.6 * thr, color="0.55", lw=1.0, ls="-.", label=r"$3\sigma_{\rm emp}(r)$")
    ax.axhline(im.s_expected, color="0.4", ls="--", lw=0.9, label=r"$S_{\rm exp}$")
    ax.axvline(ul_r[j], color="black", ls=":", lw=0.9)
    ax.text(ul_r[j] + 0.05, 9.7, rf"$r_{{\rm lim}} = {ul_r[j]:.2f}^{{\prime\prime}}$", fontsize=6.5, ha="left", va="top")
    for model, sty in MODELS.items():
        for side, (_, _, _, r50) in pred[model].items():
            ax.plot(r50, im.s_expected, sty["marker"], ms=sty["ms"] * 0.8, mec="black", mew=0.6, zorder=3,
                    color=sty["color"] if side == "SE" else "white")
    ax.set_xlim(1.0, 4.5)
    ax.set_ylim(0, 10)
    ax.set_title(f"({'cd'[k]}) Epoch {ep}: detection threshold", fontsize=8, loc="left")
    ax.set_xlabel("Radius from core (arcsec)", fontsize=7.5)
    if k == 0:
        ax.set_ylabel("Flux density (mJy)", fontsize=7.5)
    ax.tick_params(labelsize=6.5, direction="in")

# One legend for the whole figure: models by colour and shape, sides by fill, curves by line style
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Line2D([], [], ls="none", marker=s["marker"], ms=s["ms"] * 0.9, color=s["color"], mec="black", mew=0.6, label=s["label"])
           for s in MODELS.values()]
handles += [Line2D([], [], ls="none", marker="o", ms=6, color="0.5", mec="black", mew=0.6, label="SE knot (filled)"),
            Line2D([], [], ls="none", marker="o", ms=6, color="white", mec="black", mew=0.6, label="NW knot (open)"),
            Line2D([], [], color="black", lw=1.4, label=r"$5\sigma_{\rm emp}(r)$"),
            Line2D([], [], color="0.55", lw=1.0, ls="-.", label=r"$3\sigma_{\rm emp}(r)$"),
            Line2D([], [], color="0.4", ls="--", lw=0.9, label=r"$S_{\rm exp}$"),
            Line2D([], [], color="black", ls=":", lw=0.9, label=r"$r_{\rm lim}$"),
            Patch(fc="0.9", ec="none", label=r"knot of $S_{\rm exp}$ not detectable")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 0.04),
           handletextpad=0.3, columnspacing=1.2)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=300, bbox_inches="tight")
fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
print(f"Saved {args.out}")
