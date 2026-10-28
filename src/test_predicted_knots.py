"""
Forward test of each kinematic model's predicted knot positions in the post-peel Epoch 3 and 4 images.

For every model of the no-limits maximum-likelihood fit at each distance (data/physical_kinematics_report.npz, so the
prediction is not shaped by the limit being tested), the predicted (E, N) offsets of both knots are computed with the
pipeline's convention (receding NW along +parallel, approaching SE along -parallel, offsets from the fitted core). A
beam-shaped component is fixed at each, plus a free core residual, and only the amplitudes are fitted. The noise is
sigma_emp, the scatter of the same fixed-position fit at 36 PAs at that radius away from the jet axes (includes the ripple).

  python src/test_predicted_knots.py [--fits data/physical_kinematics_report.npz] [--out data/peel_regions/predicted_knot_test.txt]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postpeel_tools import EPOCHS, PostPeelImage, en

p = argparse.ArgumentParser()
p.add_argument("--fits", default="data/physical_kinematics_report.npz")
p.add_argument("--out", default="data/peel_regions/predicted_knot_test.txt")
p.add_argument("--config", type=int, nargs="+", default=None,
               help="Indices of the report configurations to test (default: every no-limits configuration)")
args = p.parse_args()

# apparent_separation lives in the pipeline script, which runs the whole grid on import; copy the function instead
from scipy.special import lambertw

C = 1.0
AU_PER_LD = 173.144633  # AU per light-day, as in the pipeline


def intrinsic_time(dt_obs, beta0, theta_deg, tau, side, t_coast=0.0):
    cth = np.cos(np.radians(theta_deg))
    doppler = 1.0 + side * beta0 * cth
    if not np.isfinite(tau):
        return dt_obs / doppler
    dt_coast = t_coast * doppler
    dt_dec = max(dt_obs - dt_coast, 0.0)
    A = side * beta0 * tau * cth / C
    s = dt_dec - A + tau * np.real(lambertw((A / tau) * np.exp(-(dt_dec - A) / tau)))
    return dt_obs / doppler if dt_obs <= dt_coast else t_coast + s


def separation(t_obs, t_ej, beta0, theta_deg, tau, dist_kpc, side, t_coast=0.0):
    dt = t_obs - t_ej
    if dt <= 0:
        return 0.0
    tp = intrinsic_time(dt, beta0, theta_deg, tau, side, t_coast)
    if np.isfinite(tau):
        rad = beta0 * min(tp, t_coast) + beta0 * tau * (-np.expm1(-max(tp - t_coast, 0.0) / tau))
    else:
        rad = beta0 * tp
    return rad * np.sin(np.radians(theta_deg)) * AU_PER_LD / (dist_kpc * 1000.0)


z = np.load(args.fits)
imgs = {ep: PostPeelImage(ep) for ep in EPOCHS}
rows = []
for i in range(len(z["D"])):
    if (i not in args.config) if args.config is not None else z["limits"][i]:
        continue
    for model in z["models"]:
        d = dict(zip(z[f"c{i}_{model}_names"], z[f"c{i}_{model}_p"]))
        gb = 10 ** d["lg_gb0"]
        beta0 = gb / np.sqrt(1 + gb**2)
        tau_a = tau_r = d.get("tau", np.inf)
        tau_a, tau_r = d.get("tau_a", tau_a), d.get("tau_r", tau_r)
        kin = (d["t_ej"], beta0, d["theta"])
        for ep, im in imgs.items():
            sn = separation(im.mjd, *kin, tau_r, z["D"][i], +1, d.get("t_coast", 0.0))
            ss = -separation(im.mjd, *kin, tau_a, z["D"][i], -1, d.get("t_coast", 0.0))
            pos = [(sn * np.sin(np.radians(d["pa_n"])), sn * np.cos(np.radians(d["pa_n"]))),
                   (ss * np.sin(np.radians(d["pa_s"])), ss * np.cos(np.radians(d["pa_s"])))]
            amps, _, _ = im.fit_fixed(pos)
            for side, (e, n), amp, s in zip(("NW", "SE"), pos, amps, (sn, -ss)):
                r, pa = np.hypot(e, n), np.degrees(np.arctan2(e, n)) % 360
                sig = im.sigma_emp(r)[0]
                rows.append((model, z["D"][i], ep, side, r, pa, amp, sig, im.s_expected, (im.s_expected - amp) / sig))

hdr = ("Forward test: flux measured at each model's predicted knot position in the post-peel image (no-limits fits).\n"
       "S_meas = fitted amplitude (mJy/beam) with a free core residual; sigma_emp = scatter of fixed-position amplitudes at the\n"
       "same radius, 36 PAs, +/-30 deg around both jet directions excluded; S_expected = exponential fade extrapolated back (mJy).\n"
       "pred_r < ~1.2\" is inside the core residual and not a clean test.\n"
       f"{'model':<12}{'D_kpc':>6}{'epoch':>6}{'side':>5}{'pred_r':>8}{'pred_PA':>8}{'S_meas':>8}{'sig_emp':>8}{'S_exp':>7}{'(Sexp-Smeas)/sig':>18}")
with open(args.out, "w") as fp:
    fp.write(hdr + "\n")
    for m, D, ep, side, r, pa, a, sg, se, dev in rows:
        fp.write(f"{m:<12}{D:6.1f}{ep:6d}{side:>5}{r:8.2f}{pa:8.1f}{a:8.2f}{sg:8.2f}{se:7.2f}{dev:18.1f}\n")
print(open(args.out).read())
