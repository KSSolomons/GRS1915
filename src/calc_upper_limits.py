#!/usr/bin/env python3
"""
calculate_upper_limits.py
=========================
Calculates the spatial (position) separation upper limit and flux density upper limit
for an unresolved radio jet component in an interferometric observation (GRS 1915+105, Epoch 4).

Methodology:
1. Synthesized Beam Projection:
   The synthesized beam is an elliptical 2D Gaussian defined by BMAJ, BMIN, and BPA.
   The effective projected FWHM resolution along an arbitrary position angle (PA_jet) is:
       theta_jet = sqrt( (BMAJ * cos(Delta_phi))^2 + (BMIN * sin(Delta_phi))^2 )
   where Delta_phi = |PA_jet - BPA|.

2. Spatial Blending / Resolution Criterion:
   For a faint jet component next to a much brighter core, the components will remain
   blended at separations up to roughly the full synthesized beam FWHM.
   Therefore, the conservative separation upper limit is:
       Delta_r_upper = theta_jet.

3. Positional Accuracy Upper Limit:
   The positional accuracy of an interferometric detection / upper limit is:
       sigma_pos = theta_beam / (2 * SNR)
   where the equivalent circular synthesized beam is:
       theta_beam = sqrt(theta_maj * theta_min)
   and SNR = S_jet / RMS_peeled (using the local residual RMS of the core-peeled image).
   At a 3-sigma detection threshold, sigma_pos = theta_beam / (2 * 3) = theta_beam / 6.

4. Flux Density Constraint:
   The 3-sigma point-source detection upper limit is 3 * RMS.
"""

import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epoch_limits import JET_PA_DEG, UPPER_LIMIT_BLOCKS, beam_fwhm_along_pa

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication styling
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 11


def plot_beam_and_limits(
    bmaj, bmin, bpa_deg, jet_pa_deg, theta_jet, r_upper, output_path
):
    """
    Generates an illustrative diagnostic plot showing the synthesized beam,
    the jet axis, and the unresolved core boundary (separation upper limit).
    Uses astronomical sky coordinates: East is Left, North is Up.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=300)

    bpa_rad = np.radians(bpa_deg)
    t = np.linspace(0, 2 * np.pi, 300)

    # 1. Synthesized beam ellipse boundary (FWHM)
    # In beam intrinsic frame: major axis along y', minor axis along x'
    x_beam = (bmin / 2.0) * np.cos(t)
    y_beam = (bmaj / 2.0) * np.sin(t)
    # Rotate by BPA (measured East of North):
    ra_beam = x_beam * np.cos(bpa_rad) + y_beam * np.sin(bpa_rad)
    dec_beam = -x_beam * np.sin(bpa_rad) + y_beam * np.cos(bpa_rad)

    ax.fill(
        ra_beam,
        dec_beam,
        color="lightsteelblue",
        alpha=0.35,
        label=f'Synthesized Beam ({bmaj:.2f}" $\\times$ {bmin:.2f}")',
    )
    ax.plot(ra_beam, dec_beam, color="royalblue", linewidth=1.8)

    # 2. Unresolved core region (Full FWHM radius)
    ra_unres = ra_beam * 2.0
    dec_unres = dec_beam * 2.0
    ax.fill(
        ra_unres,
        dec_unres,
        color="orange",
        alpha=0.3,
        label=r"Unresolved Blending Limit ($\theta_{\rm beam}$)",
    )
    ax.plot(ra_unres, dec_unres, color="darkorange", linestyle="--", linewidth=1.5)

    # 3. Draw Jet trajectory line
    rad_s = np.radians(jet_pa_deg)
    line_len = bmaj * 0.72
    dx_s = line_len * np.sin(rad_s)
    dy_s = line_len * np.cos(rad_s)
    dx_n = -dx_s
    dy_n = -dy_s

    ax.plot(
        [dx_n, dx_s],
        [dy_n, dy_s],
        color="firebrick",
        linestyle="-",
        linewidth=1.8,
        label=f"Jet Trajectory (PA = {jet_pa_deg:.1f}$^\\circ$)",
    )

    # 4. Upper limit points along the jet axis
    x_upper_s = r_upper * np.sin(rad_s)
    y_upper_s = r_upper * np.cos(rad_s)
    x_upper_n = -r_upper * np.sin(rad_s)
    y_upper_n = -r_upper * np.cos(rad_s)

    ax.scatter(
        [x_upper_n, x_upper_s],
        [y_upper_n, y_upper_s],
        color="maroon",
        marker="X",
        s=100,
        zorder=7,
        label="Separation Upper Limits",
    )

    # Draw core position
    ax.scatter(
        [0],
        [0],
        color="black",
        marker="+",
        s=120,
        linewidth=2,
        zorder=6,
        label="Stationary Core",
    )

    # Annotate beam FWHM along jet
    arrow_x1 = (theta_jet / 2.0) * np.sin(rad_s)
    arrow_y1 = (theta_jet / 2.0) * np.cos(rad_s)
    ax.annotate(
        "",
        xy=(arrow_x1, arrow_y1),
        xytext=(-arrow_x1, -arrow_y1),
        arrowprops=dict(arrowstyle="<->", color="maroon", lw=1.8),
    )

    # Place text label beside arrow
    ax.text(
        arrow_x1 + 0.25,
        arrow_y1 - 0.25,
        f"$\\theta_{{\\rm jet}} = {theta_jet:.2f}''$",
        color="maroon",
        fontweight="bold",
        fontsize=11,
        ha="left",
        va="top",
    )

    # Aesthetics: standard astronomical orientation (East to the Left)
    limit_axis = bmaj * 1.2
    ax.set_xlim(limit_axis, -limit_axis)  # Invert X for standard RA
    ax.set_ylim(-limit_axis, limit_axis)
    ax.set_aspect("equal")

    ax.set_xlabel(
        r"$\Delta \alpha \cos(\delta)$ Offset [arcsec] (East $\leftarrow$)", fontsize=11
    )
    ax.set_ylabel(r"$\Delta \delta$ Offset [arcsec] (North $\uparrow$)", fontsize=11)
    ax.set_title(
        "Synthesized Beam & Jet Position Upper Limit (Epoch 4)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=8.8,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved diagnostic figure to: {output_path}")


def calc_positional_accuracy(bmaj_arcsec, bmin_arcsec, snr):
    """
    Computes the radio interferometric positional accuracy:
        sigma_pos = theta_beam / (2 * SNR)
    where theta_beam = sqrt(theta_maj * theta_min).
    """
    theta_beam = np.sqrt(bmaj_arcsec * bmin_arcsec)
    sigma_pos = theta_beam / (2.0 * snr) if snr > 0 else np.nan
    return theta_beam, sigma_pos


def process_block(
    name,
    bmaj_deg,
    bmin_deg,
    bpa_deg,
    mjd_obs,
    rms_jy,
    jet_pa_deg,
    t_ej,
    output_path,
    rms_peeled_jy=None,
    s_jet_jy=None,
):
    bmaj_arcsec = bmaj_deg * 3600.0
    bmin_arcsec = bmin_deg * 3600.0
    rms_ujy = rms_jy * 1e6
    three_sigma_ujy = 3.0 * rms_ujy

    # 1. Equivalent circular synthesized beam
    theta_beam = np.sqrt(bmaj_arcsec * bmin_arcsec)

    # 2. Projected beam FWHM along jet axis
    theta_jet, delta_phi = beam_fwhm_along_pa(
        bmaj_arcsec, bmin_arcsec, bpa_deg, jet_pa_deg
    )
    r_upper = theta_jet

    # 3. Positional Accuracy (sigma_pos = theta_beam / (2 * SNR))
    effective_rms = rms_peeled_jy if rms_peeled_jy is not None else rms_jy
    if s_jet_jy is not None and effective_rms is not None and effective_rms > 0:
        snr_empirical = s_jet_jy / effective_rms
        _, sigma_pos_model = calc_positional_accuracy(bmaj_arcsec, bmin_arcsec, snr_empirical)
    else:
        snr_empirical = np.nan
        sigma_pos_model = np.nan

    # Positional accuracy at 3-sigma detection threshold: SNR = 3 -> sigma_pos = theta_beam / 6
    _, sigma_pos_3sigma = calc_positional_accuracy(bmaj_arcsec, bmin_arcsec, 3.0)

    dt_days = mjd_obs - t_ej
    mu_upper_mas_per_day = (r_upper * 1000.0) / dt_days if dt_days > 0 else np.nan

    print("\n" + "=" * 65)
    print(f"      RADIO SYNTHESIZED BEAM & UPPER LIMIT ANALYSIS: {name}")
    print("=" * 65)
    print(f"Synthesized Beam:")
    print(f"  BMAJ              : {bmaj_arcsec:.3f} arcsec ({bmaj_deg:.6e} deg)")
    print(f"  BMIN              : {bmin_arcsec:.3f} arcsec ({bmin_deg:.6e} deg)")
    print(f"  BPA               : {bpa_deg:.2f} deg (East of North)")
    print(f"  theta_beam (geom) : {theta_beam:.3f} arcsec (equivalent circular FWHM)")
    print(f"\nJet Trajectory:")
    print(f"  Jet Position Angle: {jet_pa_deg:.2f} deg")
    print(
        f"  Angle to Beam Major: {delta_phi:.2f} deg (near-perpendicular to major axis)"
    )
    print(f"\nEffective Spatial Resolution & Positional Accuracy:")
    print(f"  Beam FWHM along jet axis (theta_jet) : {theta_jet:.3f} arcsec")
    print(f"  --> Position Upper Limit (Full FWHM) : {r_upper:.3f} arcsec")
    print(f"  Positional Accuracy (3-sigma thresh) : {sigma_pos_3sigma:.3f} arcsec (theta_beam / 6)")
    if rms_peeled_jy is not None:
        print(f"\nPeeled Image Noise & SNR Constraints:")
        print(f"  Peeled RMS Noise  : {rms_peeled_jy * 1e6:.1f} uJy/beam ({rms_peeled_jy:.4e} Jy/beam)")
        if s_jet_jy is not None:
            print(f"  Predicted Jet Flux: {s_jet_jy * 1e3:.2f} mJy")
            print(f"  Empirical SNR     : {snr_empirical:.1f}")
            print(f"  --> Positional Accuracy (Model SNR) : {sigma_pos_model:.3f} arcsec")
    else:
        print(f"\nFlux Density Constraints (Thermal / Unpeeled):")
        print(f"  Thermal RMS Noise : {rms_ujy:.3f} uJy/beam ({rms_jy:.4e} Jy/beam)")
        print(f"  3-sigma Upper Limit: {three_sigma_ujy:.2f} uJy/beam")
        print(f"  (Note: Pending 3GC peeled RMS noise for local SNR calculation)")

    if dt_days > 0:
        print(f"\nKinematic Constraints (relative to T_ej = {t_ej:.2f}):")
        print(f"  Time baseline (dt): {dt_days:.2f} days")
        print(f"  Proper Motion Limit: < {mu_upper_mas_per_day:.2f} mas/day")
    print("=" * 65)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_beam_and_limits(
        bmaj_arcsec, bmin_arcsec, bpa_deg, jet_pa_deg, theta_jet, r_upper, output_path
    )

    return {
        "name": name,
        "mjd": mjd_obs,
        "bmaj_arcsec": bmaj_arcsec,
        "bmin_arcsec": bmin_arcsec,
        "theta_beam": theta_beam,
        "theta_jet": theta_jet,
        "r_upper": r_upper,
        "sigma_pos_3sigma": sigma_pos_3sigma,
        "sigma_pos_model": sigma_pos_model,
        "rms_peeled_jy": rms_peeled_jy,
    }


T_EJ_REFERENCE = 60471.77  # reference launch epoch for the printed proper-motion limits


def main():
    results = []
    for i, block in enumerate(UPPER_LIMIT_BLOCKS, start=1):
        res = process_block(
            block["name"],
            block["bmaj_deg"],
            block["bmin_deg"],
            block["bpa_deg"],
            block["mjd_obs"],
            block["rms_jy"],
            JET_PA_DEG,
            T_EJ_REFERENCE,
            f"figures/peel/block{i}_upper_limit_beam.png",
            rms_peeled_jy=block.get("rms_peeled_jy"),
            s_jet_jy=block.get("s_jet_jy"),
        )
        results.append(res)


if __name__ == "__main__":
    main()
