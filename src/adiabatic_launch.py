"""
Launch epoch from the lobe fade, assuming adiabatic expansion (van der Laan 1966): an optically thin blob expanding at
constant speed fades as S ~ (t - T_ej)^(-2p), p = 1 - 2 alpha. Light-travel time only rescales t - T_ej, so both lobes
should give the same T_ej. The exponent is fixed by the in-band lobe alpha (Epochs 6-8) and its +/-1 sigma; flux errors
include a 5 per cent scale error. A consistency check for the paper's \\note, not a measurement.

  python src/adiabatic_launch.py
"""

import numpy as np
from scipy.optimize import curve_fit

ALPHA, ALPHA_ERR = -1.064, 0.170  # alpha/inband_alpha_final.txt, mean lobe alpha (N+S)

lobes = {}
for line in open("alpha/inband_alpha_final.txt"):
    f = line.split()
    if len(f) >= 9 and not line.startswith("#") and f[1] in ("N", "S"):
        lobes.setdefault(f[1], []).append((float(f[2]), float(f[5]), float(f[6])))

for a in (ALPHA, ALPHA + ALPHA_ERR, ALPHA - ALPHA_ERR):
    n = 2 * (1 - 2 * a)
    out = []
    for lobe, v in lobes.items():
        t, s, e = np.array(v).T
        e = np.hypot(e, 0.05 * s)
        model = lambda t, ln_a, t_ej: ln_a - n * np.log(t - t_ej)
        p, c = curve_fit(model, t, np.log(s), p0=[n * np.log(400) + np.log(s[0]), 60400], sigma=e / s,
                         absolute_sigma=True, bounds=([-np.inf, 59500], [np.inf, t.min() - 1]))
        chi2 = np.sum(((np.log(s) - model(t, *p)) / (e / s)) ** 2)
        out.append(f"{lobe}: T_ej = {p[1]:.0f} +/- {np.sqrt(c[1, 1]):.0f} (chi2 {chi2:.1f}/{len(t) - 2})")
    print(f"alpha = {a:.3f} (exponent {n:.2f}):  " + "   ".join(out))
