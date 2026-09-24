#!/usr/bin/env python3
"""
calculate_upper_limits.py
=========================
Calculates the spatial (position) separation upper limit and flux density upper limit
for an unresolved radio jet component in an interferometric observation (GRS 1915+105, Epoch 4).

Methodology:
1. Synthesized Beam Projection:
   The synthesized beam is an elliptical 2D Gaussian defined by BMAJ, BMIN, and BPA.
   The effective FWHM resolution along an arbitrary position angle (PA_jet) is:
       theta_jet = 1 / sqrt( (cos(Delta_phi) / BMAJ)^2 + (sin(Delta_phi) / BMIN)^2 )
   where Delta_phi = |PA_jet - BPA|.

2. Spatial Blending / Resolution Criterion:
   Two Gaussian point sources separated by less than half the beam FWHM (HWHM = theta_jet / 2)
   blend into a single unimodal peak (Sparrow / Rayleigh resolution criterion) and cannot
   be deconvolved or resolved from the core.
   Therefore, the separation upper limit is:
       Delta_r_upper = theta_jet / 2.

3. Flux Density Constraint:
   The 3-sigma point-source detection upper limit is 3 * RMS.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Publication styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 11

def beam_fwhm_along_pa(bmaj, bmin, bpa_deg, target_pa_deg):
    """
    Computes the effective beam FWHM along a specific Position Angle (target_pa_deg).
    
    Parameters:
    -----------
    bmaj : float
        Major axis FWHM (arcsec).
    bmin : float
        Minor axis FWHM (arcsec).
    bpa_deg : float
        Beam Position Angle (degrees, East of North).
    target_pa_deg : float
        Target Position Angle along which to project (degrees, East of North).
        
    Returns:
    --------
    theta_proj : float
        Projected beam FWHM along target_pa (arcsec).
    delta_phi_deg : float
        Absolute angle between target PA and beam major axis (degrees).
    """
    delta_phi_rad = np.radians(target_pa_deg - bpa_deg)
    cos_phi = np.cos(delta_phi_rad)
    sin_phi = np.sin(delta_phi_rad)
    
    # Polar equation for an ellipse
    theta_proj = 1.0 / np.sqrt((cos_phi / bmaj)**2 + (sin_phi / bmin)**2)
    delta_phi_deg = np.degrees(np.abs(delta_phi_rad)) % 180.0
    if delta_phi_deg > 90.0:
        delta_phi_deg = 180.0 - delta_phi_deg
        
    return theta_proj, delta_phi_deg

def plot_beam_and_limits(bmaj, bmin, bpa_deg, jet_pa_deg, theta_jet, r_upper, output_path):
    """
    Generates an illustrative diagnostic plot showing the synthesized beam,
    the jet axis, and the unresolved core boundary (separation upper limit).
    Uses astronomical sky coordinates: East is Left, North is Up.
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=300)
    
    bpa_rad = np.radians(bpa_deg)
    t = np.linspace(0, 2*np.pi, 300)
    
    # 1. Synthesized beam ellipse boundary (FWHM)
    # In beam intrinsic frame: major axis along y', minor axis along x'
    x_beam = (bmin / 2.0) * np.cos(t)
    y_beam = (bmaj / 2.0) * np.sin(t)
    # Rotate by BPA (measured East of North):
    ra_beam = x_beam * np.cos(bpa_rad) + y_beam * np.sin(bpa_rad)
    dec_beam = -x_beam * np.sin(bpa_rad) + y_beam * np.cos(bpa_rad)
    
    ax.fill(ra_beam, dec_beam, color='lightsteelblue', alpha=0.35,
            label=f'Synthesized Beam ({bmaj:.2f}\" $\\times$ {bmin:.2f}\")')
    ax.plot(ra_beam, dec_beam, color='royalblue', linewidth=1.8)
    
    # 2. Unresolved core region (HWHM = FWHM / 2)
    ra_unres = ra_beam / 2.0
    dec_unres = dec_beam / 2.0
    ax.fill(ra_unres, dec_unres, color='orange', alpha=0.3,
            label=r'Unresolved Blending Limit ($\theta_{\rm beam}/2$)')
    ax.plot(ra_unres, dec_unres, color='darkorange', linestyle='--', linewidth=1.5)
    
    # 3. Draw Jet trajectory line
    rad_s = np.radians(jet_pa_deg)
    line_len = bmaj * 0.72
    dx_s = line_len * np.sin(rad_s)
    dy_s = line_len * np.cos(rad_s)
    dx_n = -dx_s
    dy_n = -dy_s
    
    ax.plot([dx_n, dx_s], [dy_n, dy_s], color='firebrick', linestyle='-', linewidth=1.8,
            label=f'Jet Trajectory (PA = {jet_pa_deg:.1f}$^\\circ$)')
    
    # Mark the upper limit point on the Southern jet axis
    x_lim = r_upper * np.sin(rad_s)
    y_lim = r_upper * np.cos(rad_s)
    ax.scatter([x_lim], [y_lim], color='crimson', s=70, zorder=5,
               label=f'Position Upper Limit ($r_{{\\rm max}} = {r_upper:.2f}\'\'$)')
    
    # Draw core position
    ax.scatter([0], [0], color='black', marker='+', s=120, linewidth=2, zorder=6, label='Stationary Core')
    
    # Annotate beam FWHM along jet
    arrow_x1 = (theta_jet / 2.0) * np.sin(rad_s)
    arrow_y1 = (theta_jet / 2.0) * np.cos(rad_s)
    ax.annotate('', xy=(arrow_x1, arrow_y1), xytext=(-arrow_x1, -arrow_y1),
                arrowprops=dict(arrowstyle='<->', color='maroon', lw=1.8))
    
    # Place text label beside arrow
    ax.text(arrow_x1 + 0.25, arrow_y1 - 0.25, f'$\\theta_{{\\rm jet}} = {theta_jet:.2f}\'\'$',
            color='maroon', fontweight='bold', fontsize=11, ha='left', va='top')
    
    # Aesthetics: standard astronomical orientation (East to the Left)
    limit_axis = bmaj * 0.70
    ax.set_xlim(limit_axis, -limit_axis)  # Invert X for standard RA
    ax.set_ylim(-limit_axis, limit_axis)
    ax.set_aspect('equal')
    
    ax.set_xlabel(r'$\Delta \alpha \cos(\delta)$ Offset [arcsec] (East $\leftarrow$)', fontsize=11)
    ax.set_ylabel(r'$\Delta \delta$ Offset [arcsec] (North $\uparrow$)', fontsize=11)
    ax.set_title('Synthesized Beam & Jet Position Upper Limit (Epoch 4)', fontsize=12, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.92, fontsize=8.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved diagnostic figure to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute position and flux upper limits for an unresolved radio component."
    )
    # Default parameters from GRS 1915+105 Observation 4
    parser.add_argument('--bmaj_deg', type=float, default=1.118431641616e-03,
                        help="Beam Major axis in degrees (default: 1.1184e-3 deg = 4.026 arcsec)")
    parser.add_argument('--bmin_deg', type=float, default=6.765129150517e-04,
                        help="Beam Minor axis in degrees (default: 6.7651e-4 deg = 2.435 arcsec)")
    parser.add_argument('--bpa', type=float, default=41.41138047977,
                        help="Beam Position Angle in degrees (default: 41.41 deg)")
    parser.add_argument('--jet_pa', type=float, default=137.6,
                        help="Jet Position Angle in degrees (default: 137.6 deg)")
    parser.add_argument('--rms_jy', type=float, default=4.831714277465e-06,
                        help="Local RMS background noise in Jy/beam (default: 4.832e-6 Jy/beam)")
    parser.add_argument('--mjd_obs', type=float, default=60836.845,
                        help="Observation MJD of Epoch 4 (default: 60836.845)")
    parser.add_argument('--t_ej', type=float, default=60471.77,
                        help="Estimated ejection time MJD (default: ~Epoch 1 MJD)")
    parser.add_argument('--output_fig', type=str, default='figures/epoch4_upper_limit_beam.png',
                        help="Path to save diagnostic plot")

    args = parser.parse_args()

    # Convert beam to arcsec
    bmaj_arcsec = args.bmaj_deg * 3600.0
    bmin_arcsec = args.bmin_deg * 3600.0
    bpa_deg = args.bpa
    jet_pa_deg = args.jet_pa
    rms_ujy = args.rms_jy * 1e6
    three_sigma_ujy = 3.0 * rms_ujy

    # Calculate effective FWHM along jet direction
    theta_jet, delta_phi = beam_fwhm_along_pa(bmaj_arcsec, bmin_arcsec, bpa_deg, jet_pa_deg)
    
    # Position upper limit (HWHM = FWHM / 2)
    r_upper = theta_jet / 2.0

    # Proper motion upper limit
    dt_days = args.mjd_obs - args.t_ej
    mu_upper_mas_per_day = (r_upper * 1000.0) / dt_days if dt_days > 0 else np.nan

    print("\n" + "="*65)
    print("      RADIO SYNTHESIZED BEAM & UPPER LIMIT ANALYSIS")
    print("="*65)
    print(f"Synthesized Beam:")
    print(f"  BMAJ              : {bmaj_arcsec:.3f} arcsec ({args.bmaj_deg:.6e} deg)")
    print(f"  BMIN              : {bmin_arcsec:.3f} arcsec ({args.bmin_deg:.6e} deg)")
    print(f"  BPA               : {bpa_deg:.2f} deg (East of North)")
    print(f"\nJet Trajectory:")
    print(f"  Jet Position Angle: {jet_pa_deg:.2f} deg")
    print(f"  Angle to Beam Major: {delta_phi:.2f} deg (near-perpendicular to major axis)")
    print(f"\nEffective Spatial Resolution:")
    print(f"  Beam FWHM along jet axis (theta_jet) : {theta_jet:.3f} arcsec")
    print(f"  --> Position Upper Limit (FWHM / 2)  : {r_upper:.3f} arcsec")
    print(f"\nFlux Density Constraints:")
    print(f"  Local RMS Noise   : {rms_ujy:.3f} uJy/beam ({args.rms_jy:.4e} Jy/beam)")
    print(f"  3-sigma Upper Limit: {three_sigma_ujy:.2f} uJy/beam")
    if dt_days > 0:
        print(f"\nKinematic Constraints (relative to T_ej = {args.t_ej:.2f}):")
        print(f"  Time baseline (dt): {dt_days:.2f} days")
        print(f"  Proper Motion Limit: < {mu_upper_mas_per_day:.2f} mas/day")
    print("="*65)

    print("\n[READY-TO-USE PAPER / MANUSCRIPT TEXT]")
    print("-" * 65)
    print(
        f"\"In our fourth observation (Epoch 4), no discrete knot could be distinguished "
        f"from the core component. To constrain the ejecta kinematics, we calculate an "
        f"upper limit on the component separation based on the synthesized beam geometry. "
        f"Given a synthesized beam of {bmaj_arcsec:.2f}'' x {bmin_arcsec:.2f}'' at a position "
        f"angle of {bpa_deg:.1f} deg, the effective beam FWHM projected along the jet trajectory "
        f"(PA ~ {jet_pa_deg:.1f} deg) is theta_jet = {theta_jet:.2f}''. Under the Sparrow "
        f"resolution criterion (FWHM/2), any component separated by r > {r_upper:.2f}'' would "
        f"have been resolved from the core or caused measurable elongation in the 2D Gaussian fit. "
        f"We therefore adopt a spatial upper limit of r <= {r_upper:.2f}'' for Epoch 4, alongside "
        f"a 3-sigma peak flux density upper limit of S_nu < {three_sigma_ujy:.1f} uJy/beam "
        f"(1-sigma RMS = {rms_ujy:.2f} uJy/beam).\""
    )
    print("-" * 65 + "\n")

    # Generate plot
    os.makedirs(os.path.dirname(args.output_fig), exist_ok=True)
    plot_beam_and_limits(bmaj_arcsec, bmin_arcsec, bpa_deg, jet_pa_deg, theta_jet, r_upper, args.output_fig)

if __name__ == '__main__':
    main()
