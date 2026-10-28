"""
Light-travel-time separation of a decelerating ejecton (exponential slow-down, optional coasting phase), vectorised.
Same model as src/pipeline_physical_kinematics.py (that script runs its full fit on import, so it cannot be imported).
"""

import numpy as np
from scipy.special import lambertw

AU_PER_LD = 173.144633  # astronomical units per light-day


def intrinsic_time(dt_obs, beta0, theta_deg, tau, side, t_coast=0.0):
    """Coordinate time since launch t' at observer time dt_obs after launch (days); side = +1 receding, -1 approaching."""
    dt_obs, beta0, theta_deg, tau, t_coast = np.broadcast_arrays(*map(np.asarray, (dt_obs, beta0, theta_deg, tau, t_coast)))
    cth = np.cos(np.radians(theta_deg))
    doppler = 1.0 + side * beta0 * cth
    dt_coast = t_coast * doppler
    dt_dec = np.maximum(dt_obs - dt_coast, 0.0)
    a = side * beta0 * tau * cth
    with np.errstate(over="ignore", invalid="ignore"):
        s = dt_dec - a + tau * np.real(lambertw((a / tau) * np.exp(-(dt_dec - a) / tau)))
    return np.where(dt_obs <= dt_coast, dt_obs / doppler, t_coast + s)


def separation(t_obs, t_ej, beta0, theta_deg, tau, dist_kpc, side, t_coast=0.0):
    """Projected separation from the core (arcsec, positive) at observer time t_obs (MJD)."""
    dt = np.asarray(t_obs) - np.asarray(t_ej)
    tp = intrinsic_time(np.maximum(dt, 0.0), beta0, theta_deg, tau, side, t_coast)
    rad = beta0 * np.minimum(tp, t_coast) + beta0 * tau * (-np.expm1(-np.maximum(tp - t_coast, 0.0) / tau))
    return np.where(dt > 0, rad * np.sin(np.radians(theta_deg)) * AU_PER_LD / (dist_kpc * 1000.0), 0.0)
