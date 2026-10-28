#!/usr/bin/env python3
"""
Light curve and spectral-index evolution of GRS 1915+105 from the final in-band fits.

Reads alpha/inband_alpha_final.txt (written by alpha/inband_systematics.py) and the
MeerKAT L-band light curve, and writes two figures to figures/paper/:
  inband_flux_evolution.png   flux density of the core at L band and S band and of the resolved lobes (main text);
  inband_alpha_evolution.png  in-band spectral index of the core, and of the lobes over the epochs used for their
                              mean (6-8), with that mean (Appendix B).
The two-panel version (inband_spectral_evolution.png) is no longer written: python src/plot_inband_spectra.py --both

Run from the repository root: python src/plot_inband_spectra.py
"""

import os
import sys

import matplotlib
import numpy as np
from astropy.time import Time

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FINAL = os.path.join(ROOT, "alpha", "inband_alpha_final.txt")
L_BAND = os.path.join(ROOT, "lc", "MeerKAT_GRS1915+105_all_NEW.txt")
OUT_EVOLUTION = os.path.join(ROOT, "figures", "paper", "inband_spectral_evolution.png")
OUT_FLUX = os.path.join(ROOT, "figures", "paper", "inband_flux_evolution.png")
OUT_ALPHA = os.path.join(ROOT, "figures", "paper", "inband_alpha_evolution.png")

EPOCH_LABEL = {"1736575328": 1, "1738471273": 2, "1740883517": 3, "1742780174": 4, "1745709012": 5,
               "1749586570": 6, "1751137270": 7, "1753549996": 8, "1756479610": 9, "1759581193": 10,
               "1761391938": 11, "1764664923": 12}
XLIM = (60655, 61060)  # the S-band monitoring ends at MJD ~61011; later L-band points are not shown

# Reference palette, light mode: categorical slots 1-3 (validated all-pairs), ink and rules
# Okabe-Ito (colour-vision-deficiency safe), as in the kinematics and peel figures: blue, orange, bluish green
L_COLOR, S_COLOR, LOBE_COLOR = "#0072B2", "#E69F00", "#009E73"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3de", "#ffffff"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK_2,
                     "ytick.color": INK_2, "axes.linewidth": 0.8})


def read_final(path):
    core, lobes, means = [], [], {}
    section = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("# CORE"):
                section = "core"
            elif line.startswith("# LOBES"):
                section = "lobes"
            elif line.startswith("# Mean lobe alpha"):
                label = line.split("(")[1].split(",")[0]
                value = line.split(":")[1].split("(")[0].split("+/-")
                means[label] = (float(value[0]), float(value[1]))
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if section == "core":
                core.append(dict(epoch=p[0][:10], mjd=float(p[1]), s0=float(p[4]), s0e=float(p[5]),
                                 alpha=float(p[6]), alpha_e=float(p[7])))
            elif section == "lobes":
                lobes.append(dict(epoch=p[0][:10], lobe=p[1], mjd=float(p[2]), s0=float(p[5]), s0e=float(p[6]),
                                  alpha=float(p[7]), alpha_e=float(p[8]), in_mean=p[-1] == "*"))
    return core, lobes, means


def read_l_band(path):
    mjd, flux, err = [], [], []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if line.startswith("#") or len(p) < 6:
                continue
            mjd.append(float(p[1]))
            flux.append(float(p[2]) * 1e3)
            err.append(float(p[5]) * 1e3)
    order = np.argsort(mjd)
    return np.array(mjd)[order], np.array(flux)[order], np.array(err)[order]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.spines["top"].set_visible(False)  # drawn by the twin date axis


def evolution_figure(core, lobes, means, lband, panels="ab", out=OUT_EVOLUTION):
    """panels: "a" (flux density), "b" (spectral index) or "ab" (both, shared time axis)."""
    l_mjd, l_flux, l_err = lband
    xlim = XLIM
    ratios = {"a": 1.15, "b": 1.0}
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 6.2 if len(panels) == 2 else 3.6), sharex=True, dpi=300,
                             gridspec_kw={"height_ratios": [ratios[k] for k in panels], "hspace": 0.08})
    axes = dict(zip(panels, np.atleast_1d(axes)))
    c_mjd = np.array([c["mjd"] for c in core])
    ax1, ax2 = axes.get("a"), axes.get("b")
    for ax in axes.values():
        style(ax)

    # (a) flux density
    if ax1 is not None:
        m = (l_mjd > xlim[0]) & (l_mjd < xlim[1])
        ax1.errorbar(l_mjd[m], l_flux[m], yerr=l_err[m], fmt="o", ms=4.5, color=L_COLOR, mec="white", mew=0.8,
                     elinewidth=1.0, capsize=0, zorder=3, label="Core, L band (1.28 GHz)")
        ax1.errorbar(c_mjd, [c["s0"] for c in core], yerr=[c["s0e"] for c in core], fmt="s", ms=5, color=S_COLOR,
                     mec="white", mew=0.8, elinewidth=1.0, capsize=0, zorder=4, label="Core, S band (3.05 GHz)")
        for lobe, marker, name in (("N", "^", "NW lobe (receding)"), ("S", "v", "SE lobe (approaching)")):
            rows = [r for r in lobes if r["lobe"] == lobe]
            ax1.errorbar([r["mjd"] for r in rows], [r["s0"] for r in rows], yerr=[r["s0e"] for r in rows],
                         fmt=marker, ms=5, color=LOBE_COLOR, mec="white", mew=0.8, elinewidth=1.0, capsize=0,
                         zorder=4, label=f"{name}, S band")
        ax1.set_yscale("log")
        ax1.set_ylim(0.05, 3000)
        ax1.set_ylabel("Flux density (mJy)")
        handles = [plt.Line2D([], [], ls="", marker=m, ms=5, color=c, mec="white", mew=0.8, label=l) for m, c, l in (
            ("o", L_COLOR, "Core, L band (1.28 GHz)"), ("s", S_COLOR, "Core, S band (3.05 GHz)"),
            ("^", LOBE_COLOR, "NW lobe (receding), S band"), ("v", LOBE_COLOR, "SE lobe (approaching), S band"))]
        ax1.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.5, handletextpad=0.3)

    # (b) spectral index
    if ax2 is not None:
        ax2.axhline(0.0, color=MUTED, lw=0.8, zorder=1)
        ax2.errorbar(c_mjd, [c["alpha"] for c in core], yerr=[c["alpha_e"] for c in core], fmt="s-", ms=5,
                     color=S_COLOR, lw=1.2, mec="white", mew=0.8, elinewidth=1.0, capsize=0, zorder=4,
                     label="Core, in band (2.79–3.34 GHz)")
        for e in core:
            n = EPOCH_LABEL.get(e["epoch"])
            if n is not None and n <= 5:
                ax2.annotate(str(n), (e["mjd"], e["alpha"]), xytext=(-7, -10), textcoords="offset points",
                             ha="center", fontsize=7, color=INK_2)

        for lobe, marker in (("N", "^"), ("S", "v")):
            # Only the epochs in the mean (6-8): Epoch 5 sits on the core's wing, Epochs 9-11 have sigma_alpha > 1
            rows = [r for r in lobes if r["lobe"] == lobe and r["in_mean"]]
            ax2.errorbar([r["mjd"] for r in rows], [r["alpha"] for r in rows], yerr=[r["alpha_e"] for r in rows],
                         fmt=marker, ms=6, color=LOBE_COLOR, mec="white", mew=0.6, elinewidth=1.0, capsize=0, zorder=4)
        if "N+S" in means:
            mean, err = means["N+S"]
            used = [r["mjd"] for r in lobes if r["in_mean"]]
            x0, x1 = min(used) - 8, max(used) + 8
            ax2.fill_between([x0, x1], mean - err, mean + err, color=LOBE_COLOR, alpha=0.18, lw=0, zorder=2)
            ax2.plot([x0, x1], [mean, mean], color=LOBE_COLOR, lw=1.5, zorder=2)
            ax2.text(x1 + 4, mean, rf"lobes, Epochs 6–8: $\alpha = {mean:.2f} \pm {err:.2f}$", fontsize=7.5,
                     color=INK, va="center", zorder=5, bbox=dict(fc="white", ec="none", pad=1.5, alpha=0.9))

        ax2.set_ylim(-3.0, 1.15)
        ax2.set_xlim(xlim)
        ax2.set_xlabel("MJD")
        ax2.set_ylabel(r"Spectral index $\alpha$ ($S_\nu \propto \nu^{\alpha}$)")
        handles = [
            plt.Line2D([], [], marker="s", ms=5, color=S_COLOR, lw=1.2, mec="white", mew=0.8, label="Core, in band (2.79–3.34 GHz)"),
            plt.Line2D([], [], ls="", marker="^", ms=5, color=LOBE_COLOR, mec="white", mew=0.8, label="NW lobe, in band"),
            plt.Line2D([], [], ls="", marker="v", ms=5, color=LOBE_COLOR, mec="white", mew=0.8, label="SE lobe, in band"),
        ]
        ax2.legend(handles=handles, loc="upper right", ncol=1, frameon=False, fontsize=7, handletextpad=0.3,
                   columnspacing=1.0)

    first = next(iter(axes.values()))
    top = first.twiny()
    top.set_xlim(xlim)
    months = [(2025, 1), (2025, 3), (2025, 5), (2025, 7), (2025, 9), (2025, 11)]
    ticks = [Time(f"{y}-{m:02d}-01").mjd for y, m in months]
    top.set_xticks(ticks)
    top.set_xticklabels([Time(t, format="mjd").strftime("%b %Y") for t in ticks], fontsize=8, color=INK_2)
    top.tick_params(direction="out", length=3, width=0.8)
    for side in ("right", "left", "bottom"):
        top.spines[side].set_visible(False)
    if len(panels) == 2:
        for k, ax in axes.items():
            ax.text(0.008, 0.97, f"({k})", transform=ax.transAxes, fontsize=9, color=INK, va="top", fontweight="bold")
    axes[panels[-1]].set_xlabel("MJD")
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    core, lobes, means = read_final(FINAL)
    lband = read_l_band(L_BAND)
    os.makedirs(os.path.dirname(OUT_EVOLUTION), exist_ok=True)
    evolution_figure(core, lobes, means, lband, "a", OUT_FLUX)
    evolution_figure(core, lobes, means, lband, "b", OUT_ALPHA)
    if "--both" in sys.argv:
        evolution_figure(core, lobes, means, lband)
