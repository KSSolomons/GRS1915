"""
Posterior check of the Epoch 3-4 non-detections: for every posterior sample, the predicted NW and SE knot separations at
Epochs 3 and 4, compared with the limit radius r_lim for a knot of flux S_exp (src/epoch_limits.py, measured_limits()).
A sample "passes" an epoch and side if its predicted separation is below r_lim, i.e. the knot would have been too close to
the core to detect. The limits are not used in any of the fits.

Models:
  ballistic    apparent-motion Model 1, no limits (data/absolute_kinematics/absolute_kinematics_plot_data_model1_without_limits.npz)
  app_decel    apparent-motion Model 5 (smooth deceleration, free T_ej), no limits (..._model5_without_limits.npz)
  decel_sym    decelerating model, no limits, beta_0 fixed (--decel-samples, default beta_0 = 0.99)

  python src/check_limits_posterior.py [--dist 9.4] [--n-sub 5000] [--out data/peel_regions/limits_posterior_check.txt]
"""

import argparse
import os
import sys

import numpy as np
from scipy.special import lambertw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epoch_limits import measured_limits

p = argparse.ArgumentParser()
p.add_argument("--dist", type=float, default=9.4, help="Distance (kpc) the physical-model samples were drawn at")
p.add_argument("--n-sub", type=int, default=5000, help="Random subsample of each chain")
p.add_argument("--decel-samples", default="data/physical_kinematics_samples_decel_sym_nolimits_b0.99.npz")
p.add_argument("--seed", type=int, default=1915)
p.add_argument("--out", default="data/peel_regions/limits_posterior_check.txt")
args = p.parse_args()
rng = np.random.default_rng(args.seed)

C = 1.0
AU_PER_LD = 173.144633  # AU per light-day, as in the pipeline


# apparent_separation lives in the pipeline script, which runs the whole grid on import; copy the function instead
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


def subsample(chain):
    return chain[rng.choice(len(chain), size=min(args.n_sub, len(chain)), replace=False)]


UL_MJD, UL_R, _ = measured_limits(threshold=5, s_factor=1.0)
EPOCH_LABEL = {0: 3, 1: 4}

# Predicted separations, shape (n_samples, n_epochs) per side, and the position angles of the two lobes per sample
pred = {}
pa = {}

z = np.load("data/absolute_kinematics/absolute_kinematics_plot_data_model1_without_limits.npz")
ch = subsample(z["flat_samples"])  # T_ej, v_n, v_s, PA_n, PA_s, theta_scatter, x_c, y_c (arcsec/day)
dt = np.clip(UL_MJD[None, :] - ch[:, [0]], 0.0, None)
pred["ballistic"] = {"NW": np.abs(ch[:, [1]]) * dt, "SE": np.abs(ch[:, [2]]) * dt}
pa["ballistic"] = (ch[:, 3], ch[:, 4])

z = np.load("data/absolute_kinematics/absolute_kinematics_plot_data_model5_without_limits.npz")
ch = subsample(z["flat_samples"])  # T_ej, v0_n, tau_n, v0_s, tau_s, PA_n, PA_s, theta_scatter, x_c, y_c
dt = np.clip(UL_MJD[None, :] - ch[:, [0]], 0.0, None)
pred["app_decel"] = {
    "NW": np.abs(ch[:, [1]]) * ch[:, [2]] * -np.expm1(-dt / ch[:, [2]]),
    "SE": np.abs(ch[:, [3]]) * ch[:, [4]] * -np.expm1(-dt / ch[:, [4]]),
}
pa["app_decel"] = (ch[:, 5], ch[:, 6])

for model, path in (("decel_sym", args.decel_samples),):
    z = np.load(path)
    names = list(z["names"])
    ch = subsample(z["samples"])
    sides = {"NW": [], "SE": []}
    for q in ch:
        d = dict(zip(names, q))
        gb = 10 ** d["lg_gb0"]
        kin = (d["t_ej"], gb / np.sqrt(1 + gb**2), d["theta"], d["tau"])
        tc = d.get("t_coast", 0.0)
        sides["NW"].append([separation(t, *kin, args.dist, +1, tc) for t in UL_MJD])
        sides["SE"].append([separation(t, *kin, args.dist, -1, tc) for t in UL_MJD])
    pred[model] = {k: np.array(v) for k, v in sides.items()}
    pa[model] = (ch[:, names.index("pa_n")], ch[:, names.index("pa_s")])

lines = [
    "Posterior check of the Epoch 3-4 non-detections (limits not used in the fits).",
    "r_lim = radius beyond which a knot of flux S_exp would be detected at 5 sigma_emp (src/simple_limits.py).",
    "pass = fraction of posterior samples whose predicted knot lies inside r_lim (consistent with the non-detection).",
    f"Physical-model samples at D = {args.dist} kpc; ballistic and app_decel = apparent-motion Models 1 and 5 (distance-free).",
    f"{'model':<12}{'epoch':>6}{'side':>5}{'r_lim':>7}{'pred_r [16, 50, 84]':>24}{'pass':>7}",
]
for model, sides in pred.items():
    for j, (mjd, r_lim) in enumerate(zip(UL_MJD, UL_R)):
        for side, r in sides.items():
            q16, q50, q84 = np.percentile(r[:, j], [16, 50, 84])
            frac = np.mean(r[:, j] < r_lim)
            lines.append(f"{model:<12}{EPOCH_LABEL[j]:6d}{side:>5}{r_lim:7.2f}   [{q16:5.2f}, {q50:5.2f}, {q84:5.2f}]{frac:9.3f}")
    both = np.all(np.hstack([sides["NW"] < UL_R, sides["SE"] < UL_R]), axis=1)
    lines.append(f"{model:<12}{'all four inside r_lim':>40}{np.mean(both):9.3f}")
    lines.append("")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as fp:
    fp.write("\n".join(lines) + "\n")
print("\n".join(lines))

# Samples for the figure (plot_peel_limits_figure.py). PA convention of the pipelines: the NW knot lies at
# r (sin pa_n, cos pa_n) and the SE knot at -r (sin pa_s, cos pa_s), in (E, N) arcsec from the core.
store = {"ul_mjd": UL_MJD, "ul_r": UL_R}
for model in pred:
    store[f"{model}_r_NW"], store[f"{model}_r_SE"] = pred[model]["NW"], pred[model]["SE"]
    store[f"{model}_pa_n"], store[f"{model}_pa_s"] = pa[model]
np.savez(os.path.splitext(args.out)[0] + ".npz", **store)
