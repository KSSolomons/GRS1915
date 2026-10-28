"""
Shared constants for the unresolved Epochs 1-4 and the 2025 ejection campaign.

Every script that needs the blending upper limits or the campaign cut imports them
from here, so a new or re-measured limit only has to be entered once:

    calc_upper_limits.py                       (prints and plots the limits)
    pipeline_absolute_kinematics_with_limits.py
    pipeline_physical_kinematics.py
    plot_absolute_separation.py, plot_combined_kinematics_figure.py
    scratch/fit_flux.py

To add a limit for Epoch 1 or 2, fill in its rms_peeled_jy and set "in_fit": True.
The measured Epoch 3-4 limits (r_lim, sigma_pos) are in the "measured" entries; blending_limits() still returns the beam-FWHM ones.
This module only depends on numpy, so importing it has no side effects.
"""

import numpy as np

# Jet components before this MJD belong to older ejecta, not the 2025 campaign
EJECTION_CAMPAIGN_START = 60600.0

JET_PA_DEG = 137.6  # jet axis from a preliminary fit to the resolved ejecta

UPPER_LIMIT_BLOCKS = [
    {
        "name": "Block 1 (S1)",
        "bmaj_deg": 1.261684512901e-03,
        "bmin_deg": 6.778116366529e-04,
        "bpa_deg": 45.14841404913,
        "mjd_obs": 60686.2525,
        "rms_jy": 4.831714277465e-06,
        "rms_peeled_jy": None,  # Fill with 3GC peeled RMS
        "s_jet_jy": 0.00984,    # ~9.8 mJy: mean of NW/SE exponential fade fits to Epochs 5-11
        "in_fit": False,        # core too bright for a limit until the peeled rms is measured
    },
    {
        "name": "Block 2 (S2)",
        "bmaj_deg": 1.105131853119e-03,
        "bmin_deg": 7.246734168800e-04,
        "bpa_deg": 30.86987498950,
        "mjd_obs": 60708.2596296,
        "rms_jy": 4.831714277465e-06,
        "rms_peeled_jy": 1.061756531718e-04,  # Measured 3GC peeled RMS: 106.2 uJy/beam
        "s_jet_jy": 0.00671,    # ~6.7 mJy: mean of NW/SE exponential fade fits to Epochs 5-11
        "in_fit": False,        # provisional peeled rms; enable once re-measured
    },
    {
        "name": "Block 3",
        "bmaj_deg": 1.766882585384e-03,
        "bmin_deg": 6.449474413115e-04,
        "bpa_deg": 40.88001696354,
        "mjd_obs": 60736.11478,
        "rms_jy": 4.831714277465e-06,
        "rms_peeled_jy": 8.11e-04,  # sigma_emp at r_lim (S_exp, 5 sigma; 0.05" grid, was 779 uJy on the 0.25" grid); was 661 uJy (injection, SE side), before that 310.1 uJy
        "s_jet_jy": 0.00413,    # ~4.1 mJy: mean of NW/SE exponential fade fits to Epochs 5-11
        "r_lim_arcsec": 2.20,   # limit radius, S_exp, 5 sigma, from S >= 5 sigma_emp(r) (simple_limits.py, 0.05" grid; 2.25 on the 0.25" grid); injection gave 2.25/2.50
        "sigma_pos_arcsec": 0.369,  # b / (2 SNR) at r_lim
        "measured": {  # {threshold (sigma): {S / S_exp: (r_lim, sigma_pos, sigma_emp in Jy) or None}}; data/peel_regions/limits_ep34_simple.txt
            5: {1.0: (2.20, 0.369, 8.11e-4), 0.5: None, 0.25: None},
            3: {1.0: (1.80, 0.593, 1.302e-3), 0.5: (2.45, 0.621, 6.82e-4), 0.25: None},
        },
        "in_fit": True,
    },
    {
        "name": "Block 4",
        "bmaj_deg": 1.118431641616e-03,
        "bmin_deg": 6.765129150517e-04,
        "bpa_deg": 41.41138047977,
        "mjd_obs": 60758.0668287,
        "rms_jy": 4.831714277465e-06,
        "rms_peeled_jy": 5.41e-04,  # sigma_emp at r_lim (S_exp, 5 sigma; 0.05" grid, was 359 uJy on the 0.25" grid); was 306 uJy (injection, SE side)
        "s_jet_jy": 0.00283,    # ~2.8 mJy: mean of NW/SE exponential fade fits to Epochs 5-11
        "r_lim_arcsec": 1.40,   # limit radius, S_exp, 5 sigma, from S >= 5 sigma_emp(r) (simple_limits.py, 0.05" grid; 1.50 on the 0.25" grid); injection gave 1.50/1.75
        "sigma_pos_arcsec": 0.299,  # b / (2 SNR) at r_lim
        "measured": {
            5: {1.0: (1.40, 0.299, 5.41e-4), 0.5: (1.85, 0.313, 2.83e-4), 0.25: None},
            3: {1.0: (1.35, 0.367, 6.64e-4), 0.5: (1.45, 0.470, 4.25e-4), 0.25: (2.25, 0.522, 2.36e-4)},
        },
        "in_fit": True,
    },
]


def beam_fwhm_along_pa(bmaj, bmin, bpa_deg, target_pa_deg):
    """
    Effective beam FWHM along a position angle.

    Returns (theta_proj, delta_phi_deg): the projected FWHM in the units of bmaj/bmin,
    and the angle between target_pa_deg and the beam major axis folded into [0, 90] deg.
    """
    delta_phi_rad = np.radians(target_pa_deg - bpa_deg)
    theta_proj = np.sqrt((bmaj * np.cos(delta_phi_rad)) ** 2 + (bmin * np.sin(delta_phi_rad)) ** 2)
    delta_phi_deg = np.degrees(np.abs(delta_phi_rad)) % 180.0
    if delta_phi_deg > 90.0:
        delta_phi_deg = 180.0 - delta_phi_deg
    return theta_proj, delta_phi_deg


def blending_limits(blocks=None, jet_pa_deg=JET_PA_DEG):
    """
    MJDs, projected-beam blending limits (arcsec) and 3-sigma positional accuracies
    (theta_beam / 6, arcsec) of the blocks flagged "in_fit", in time order.
    """
    blocks = [b for b in (blocks or UPPER_LIMIT_BLOCKS) if b["in_fit"]]
    blocks = sorted(blocks, key=lambda b: b["mjd_obs"])
    mjd = np.array([b["mjd_obs"] for b in blocks])
    r_upper = np.array(
        [beam_fwhm_along_pa(b["bmaj_deg"] * 3600.0, b["bmin_deg"] * 3600.0, b["bpa_deg"], jet_pa_deg)[0] for b in blocks]
    )
    sigma_pos_3sigma = np.array([np.sqrt(b["bmaj_deg"] * b["bmin_deg"]) * 3600.0 / 6.0 for b in blocks])
    return mjd, r_upper, sigma_pos_3sigma


def measured_limits(threshold=5, s_factor=1.0, blocks=None):
    """
    MJDs, limit radii and positional accuracies (arcsec) of the peeled Epoch 3-4 images (src/simple_limits.py:
    S >= threshold x sigma_emp(r)), for knots of S_exp x s_factor detected at `threshold` sigma. Blocks flagged "in_fit"
    without a limit at that flux (None: not detectable within 4 arcsec) are left out. sigma_emp does not depend on the
    side, so one value serves NW and SE.
    """
    rows = []
    for b in sorted((b for b in (blocks or UPPER_LIMIT_BLOCKS) if b["in_fit"] and "measured" in b), key=lambda b: b["mjd_obs"]):
        v = b["measured"][threshold][s_factor]
        if v is not None:
            rows.append((b["mjd_obs"], v[0], v[1]))
    mjd, r_lim, sigma_pos = (np.array(c) for c in zip(*rows))
    return mjd, r_lim, sigma_pos
