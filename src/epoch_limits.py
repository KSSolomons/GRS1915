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
        "rms_peeled_jy": 3.100985734235e-04,  # Measured 3GC peeled RMS: 310.1 uJy/beam
        "s_jet_jy": 0.00413,    # ~4.1 mJy: mean of NW/SE exponential fade fits to Epochs 5-11
        "in_fit": True,
    },
    {
        "name": "Block 4",
        "bmaj_deg": 1.118431641616e-03,
        "bmin_deg": 6.765129150517e-04,
        "bpa_deg": 41.41138047977,
        "mjd_obs": 60758.0668287,
        "rms_jy": 4.831714277465e-06,
        "rms_peeled_jy": None,  # Fill with 3GC peeled RMS
        "s_jet_jy": 0.00283,    # ~2.8 mJy: mean of NW/SE exponential fade fits to Epochs 5-11
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
