"""
Epoch 3-4 limit radii from the empirical noise alone, without injection.

A knot of flux S counts as detectable at radius r if S >= threshold x sigma_emp(r), where sigma_emp(r) is the scatter of
fixed-position amplitudes at that radius in the post-peel image (postpeel_tools.PostPeelImage.sigma_emp: 36 PAs,
+/-30 deg around both jet directions left out). r_lim(S) = smallest r on the 1.0-4.0" grid (--step, default 0.05") where this holds
at r and every larger r. sigma_emp does not depend on the side, so NW and SE share one value.
sigma_pos = b / (2 SNR), b = sqrt(BMAJ BMIN), SNR = S / sigma_emp(r_lim).

Replaces injection_recovery.py for the limits: the free-position recovery there picks up positive residuals near the
injected knot and so overstates the detectability of faint knots. Like it, this is an image-plane limit only.

  python src/simple_limits.py [--thresholds 5 3] [--out data/peel_regions/limits_ep34_simple.txt]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postpeel_tools import EPOCHS, PostPeelImage

p = argparse.ArgumentParser()
p.add_argument("--thresholds", type=float, nargs="+", default=[5.0, 3.0])
p.add_argument("--s-factors", type=float, nargs="+", default=[1.0, 0.5, 0.25])
p.add_argument("--out", default="data/peel_regions/limits_ep34_simple.txt")
p.add_argument("--step", type=float, default=0.05, help="Radius step (arcsec); was 0.25 before 2026-10-27")
args = p.parse_args()

RADII = np.round(np.arange(1.0, 4.0 + args.step / 2, args.step), 3)

lines = [
    "Epoch 3-4 limit radii from the empirical noise (no injection; image plane, the peel itself is not tested).",
    "Detectable at r if S >= threshold x sigma_emp(r); r_lim = smallest r with this at r and every larger r in "
    f"{RADII[0]:.2f}-{RADII[-1]:.2f}\" ('>4.00' = none). sigma_emp is the same for NW and SE.",
    "sigma_pos = b / (2 SNR), b = sqrt(BMAJ BMIN), SNR = S / sigma_emp at r_lim.",
    f"{'thr':>4} {'epoch':>5} {'S/S_exp':>7} {'S_mJy':>6} {'r_lim':>6} {'sigma_pos':>9} {'sigma_emp':>9}",
]
sig_table = {}
for ep in EPOCHS:
    im = PostPeelImage(ep)
    sig = np.array([im.sigma_emp(r)[0] for r in RADII])
    sig_table[ep] = sig
    for thr in args.thresholds:
        for f in args.s_factors:
            s = f * im.s_expected
            ok = s >= thr * sig
            # first index from which every larger radius is detectable
            bad = np.where(~ok)[0]
            i = 0 if len(bad) == 0 else bad[-1] + 1
            if i >= len(RADII):
                lines.append(f"{thr:4.0f} {ep:5d} {f:7.2f} {s:6.2f} {'>4.00':>6} {'--':>9} {'--':>9}")
                continue
            snr = s / sig[i]
            lines.append(f"{thr:4.0f} {ep:5d} {f:7.2f} {s:6.2f} {RADII[i]:6.2f} {im.b / (2 * snr):9.3f} {sig[i]:9.3f}")

lines += ["", "sigma_emp (mJy/beam) by radius:", f"{'r':>5} " + " ".join(f"ep{ep}".rjust(8) for ep in EPOCHS)]
for j, r in enumerate(RADII):
    lines.append(f"{r:5.2f} " + " ".join(f"{sig_table[ep][j]:8.3f}" for ep in EPOCHS))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as fp:
    fp.write("\n".join(lines) + "\n")
print("\n".join(lines))
