#!/usr/bin/env python3
"""
plot_spectral_index.py
======================
Comprehensive multi-frequency radio comparison for GRS 1915+105.
Carefully designed layout ensuring NO labels, legends, or annotations overlap
or cut into any data points.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

# Publication styling
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 11

L_BAND_FILE = os.path.join(os.path.dirname(__file__), "..", "lc", "MeerKAT_GRS1915+105_all_NEW.txt")
ALPHA_FILE = os.path.join(os.path.dirname(__file__), "..", "alpha", "alpha.txt")
FIGURE_OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "spectral", "spectral_index_evolution.png")

def load_l_band(filepath):
    mjds, fluxes, errs = [], [], []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    mjds.append(float(parts[1]))
                    fluxes.append(float(parts[2]) * 1000.0)  # Convert Jy -> mJy
                    errs.append(float(parts[5]) * 1000.0)    # Convert Jy -> mJy
                except ValueError:
                    continue
    return np.array(mjds), np.array(fluxes), np.array(errs)

def load_s_band(filepath):
    epochs, mjds, s0_mjy, s0_err, alphas, err_alphas, flags = [], [], [], [], [], [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            epochs.append(parts[0])
            mjds.append(float(parts[1]))
            s0_mjy.append(float(parts[5]) / 1000.0)  # uJy -> mJy
            s0_err.append(float(parts[6]) / 1000.0)  # uJy -> mJy
            alphas.append(float(parts[7]))
            err_alphas.append(float(parts[9]))       # Err_a_sc
            flags.append(parts[11])

    return {
        "epoch": epochs,
        "mjd": np.array(mjds),
        "s0": np.array(s0_mjy),
        "s0_err": np.array(s0_err),
        "alpha": np.array(alphas),
        "alpha_err": np.array(err_alphas),
        "flag": flags,
    }

def main():
    mjd_l, flux_l, err_l = load_l_band(L_BAND_FILE)
    s_band = load_s_band(ALPHA_FILE)

    mjd_s = s_band["mjd"]
    flux_s = s_band["s0"]
    err_s = s_band["s0_err"]
    alpha_s = s_band["alpha"]
    alpha_err_s = s_band["alpha_err"]
    flags_s = s_band["flag"]

    # Calculate cross-band spectral index for epochs with close L-band measurements (|dt| < 2.5 days)
    cross_mjds, cross_alphas, cross_errs = [], [], []
    nu_l = 1.284
    nu_s = 3.007
    for ms, fs, es in zip(mjd_s, flux_s, err_s):
        dt = np.abs(mjd_l - ms)
        best_idx = np.argmin(dt)
        if dt[best_idx] <= 2.5:
            fl = flux_l[best_idx]
            el = err_l[best_idx]
            if fl > 0 and fs > 0:
                a_ls = np.log(fs / fl) / np.log(nu_s / nu_l)
                var_a = ((es / fs)**2 + (el / fl)**2) / (np.log(nu_s / nu_l)**2)
                cross_mjds.append(ms)
                cross_alphas.append(a_ls)
                cross_errs.append(np.sqrt(var_a))

    cross_mjds = np.array(cross_mjds)
    cross_alphas = np.array(cross_alphas)
    cross_errs = np.array(cross_errs)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.07}
    )

    xlims = (60350, 61130)

    # Shaded flare window
    for ax in (ax1, ax2):
        ax.axvspan(60550, 60680, color="gold", alpha=0.20, zorder=0)

    # Ballistic model extrapolation marker (MJD ~60368)
    ax1.axvline(60368, color="purple", linestyle=":", linewidth=1.5, alpha=0.8, zorder=1)

    # --- TOP PANEL: Light Curves ---
    mask_l = (mjd_l >= xlims[0]) & (mjd_l <= xlims[1])
    ax1.errorbar(
        mjd_l[mask_l], flux_l[mask_l], yerr=err_l[mask_l],
        fmt="o", markersize=5.5, color="black", markerfacecolor="dodgerblue",
        markeredgewidth=0.8, ecolor="dodgerblue", elinewidth=1.2, capsize=2,
        label=r"MeerKAT L-band ($1.28\,$GHz Monitoring)", zorder=3
    )

    ax1.errorbar(
        mjd_s, flux_s, yerr=err_s,
        fmt="s", markersize=6.5, color="crimson", markerfacecolor="red",
        markeredgewidth=1.0, ecolor="crimson", elinewidth=1.5, capsize=3,
        label=r"MeerKAT S-band ($2.7–3.4\,$GHz, In-Band $S_0$)", zorder=4
    )

    # Add dummy entries for legend to keep legend completely clean
    ax1.plot([], [], color="gold", alpha=0.4, linewidth=8, label="Pre-Flare / Flare Onset Window (MJD 60550–60680)")
    ax1.plot([], [], color="purple", linestyle=":", linewidth=1.5, label=r"Ballistic Extrapolated $T_{\rm ej}$ (MJD 60368, Quiescence)")

    ax1.set_yscale("log")
    ax1.set_ylim(0.06, 5000)
    ax1.set_ylabel(r"Flux Density (mJy)", fontsize=13, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)

    # Legend in ax1 placed at upper right (x > 60800, y > 50 mJy is 100% empty!)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=9.2, edgecolor="black")
    ax1.set_title(r"GRS 1915+105: Multi-Band Light Curve & Spectral Index ($\alpha$) Evolution",
                  fontsize=14, pad=12, fontweight="bold")

    # Annotate Ballistic Tej line
    ax1.text(60373, 150, "Ballistic Launch\nMJD 60368\n(Quiescence)",
             fontsize=8.5, color="purple", fontweight="bold", va="bottom")

    # Annotate Pre-Flare Quiescence
    ax1.annotate(
        "Pre-Flare Quiescence\n" + r"($S_{\rm L} \approx 2\text{--}8\text{ mJy}$)",
        xy=(60603, 3.72), xytext=(60480, 0.4),
        arrowprops=dict(facecolor="darkorange", edgecolor="black", arrowstyle="->", lw=1.1),
        fontsize=8.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff9e6", edgecolor="goldenrod", alpha=0.92)
    )

    # Annotate S-band Peak (MJD 60686, 587 mJy)
    # xytext placed in empty region at (60590, 800)
    ax1.annotate(
        r"$\mathbf{S\text{-}band\ Peak:}\ 586.7\text{ mJy}$" + "\n(MJD 60686.3, Optically Thick)",
        xy=(mjd_s[0], flux_s[0]), xytext=(60530, 900),
        arrowprops=dict(facecolor="crimson", edgecolor="black", arrowstyle="->", lw=1.3),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="crimson", alpha=0.95)
    )

    # Annotate L-band Peak (MJD 60728, 1224 mJy)
    # xytext placed in empty region at (60680, 2600)
    ax1.annotate(
        r"$\mathbf{L\text{-}band\ Peak:}\ 1224\text{ mJy}$" + "\n(MJD 60728.2)",
        xy=(60728.16, 1224.3), xytext=(60660, 2600),
        arrowprops=dict(facecolor="dodgerblue", edgecolor="black", arrowstyle="->", lw=1.3),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="dodgerblue", alpha=0.95)
    )

    # --- BOTTOM PANEL: Spectral Index ---
    colors = {"good": "navy", "fair": "darkorange", "poor": "gray"}
    for f in ["good", "fair", "poor"]:
        mask = [fl == f for fl in flags_s]
        if any(mask):
            ax2.errorbar(
                mjd_s[mask], alpha_s[mask], yerr=alpha_err_s[mask],
                fmt="s" if f != "good" else "o",
                color=colors[f], ecolor=colors[f], elinewidth=1.5, capsize=3,
                markersize=6.5 if f == "good" else 5.5,
                zorder=5, label=f"In-Band S-Band $\\alpha$: {f.capitalize()} ({sum(mask)} epochs)"
            )

    # Cross-band L-to-S spectral index
    if len(cross_mjds) > 0:
        ax2.errorbar(
            cross_mjds, cross_alphas, yerr=cross_errs,
            fmt="D", markersize=6, color="teal", markerfacecolor="cyan",
            markeredgewidth=1.0, ecolor="teal", elinewidth=1.3, capsize=2.5,
            zorder=4, label=f"Cross-Band $\\alpha_{{L-S}}$ ({len(cross_mjds)} quasi-sim.)"
        )

    # Transition reference lines
    ax2.axhline(0.0, color="gray", linestyle="-.", linewidth=1.2, alpha=0.8,
                label=r"$\alpha = 0.0$ (Optically Thick / Flat Core)")
    ax2.axhline(-0.5, color="darkgreen", linestyle="--", linewidth=1.2, alpha=0.8,
                label=r"$\alpha = -0.5$ (Optically Thin Synchrotron, $p=2$)")
    ax2.axhline(-0.8, color="purple", linestyle=":", linewidth=1.2, alpha=0.8,
                label=r"$\alpha = -0.8$ (Steepening / Radiative Losses)")

    # Regimes
    ax2.axhspan(-0.15, 0.25, color="blue", alpha=0.07, label="Optically Thick Regime")
    ax2.axhspan(-1.3, -0.4, color="green", alpha=0.07, label="Optically Thin Synchrotron Regime")

    # Annotations on spectral index transition
    # In ax2, all data points are at MJD >= 60686! The region x in [60400, 60670] is completely empty.
    ax2.annotate(
        "Optically Thick / Peak\n" + r"$\mathbf{\alpha = -0.04 \pm 0.06}$",
        xy=(mjd_s[0], alpha_s[0]), xytext=(60530, 0.12),
        arrowprops=dict(facecolor="navy", edgecolor="black", arrowstyle="->", lw=1.3),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="navy", alpha=0.95)
    )

    ax2.annotate(
        "Transition to Optically Thin\n" + r"$\mathbf{\alpha = -0.52 \pm 0.08}$",
        xy=(mjd_s[1], alpha_s[1]), xytext=(60740, -0.15),
        arrowprops=dict(facecolor="darkgreen", edgecolor="black", arrowstyle="->", lw=1.3),
        fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="darkgreen", alpha=0.95)
    )

    ax2.set_xlabel("Time (MJD)", fontsize=13, fontweight="bold")
    ax2.set_ylabel(r"Spectral Index $\alpha$ ($S_\nu \propto \nu^\alpha$)", fontsize=13, fontweight="bold")
    ax2.set_ylim(-1.55, 0.45)
    ax2.set_xlim(xlims)
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)

    # In ax2, the entire left half (x < 60670) is completely empty of data points!
    # By setting loc="lower left", with 1 column and bbox_to_anchor on the left, it stays strictly within x in [60360, 60660]!
    ax2.legend(loc="lower left", bbox_to_anchor=(0.015, 0.04), framealpha=0.95, fontsize=8.2, ncol=1, edgecolor="black")

    # Calendar Date Axis on Top
    ax_top = ax1.twiny()
    ax_top.set_xlim(xlims)
    mjd_ticks = [60400, 60550, 60700, 60850, 61000]
    t_ticks = Time(mjd_ticks, format="mjd")
    ax_top.set_xticks(mjd_ticks)
    ax_top.set_xticklabels([t.strftime("%b %Y") for t in t_ticks], fontsize=10.5)
    ax_top.set_xlabel("Observation Date (Calendar)", fontsize=11.5, labelpad=8)

    os.makedirs(os.path.dirname(FIGURE_OUT), exist_ok=True)
    plt.savefig(FIGURE_OUT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated figure: {FIGURE_OUT}")

if __name__ == "__main__":
    main()
