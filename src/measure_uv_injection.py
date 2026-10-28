"""
Recovered flux of the uv-injected knot (src/inject_knot_uv.py) after the peel.

Fits the same beam-shaped component with a free core residual (postpeel_tools.fit_fixed) at the injected position in the
original post-peel image and in the post-peel image of the injected copy. Recovered fraction =
(amp_injected - amp_original) / S_injected, so whatever was already at that spot (e.g. the Ep4 ripple) cancels.
Also gives a fit with the position free within 1", to show whether the knot came back displaced.

  python src/measure_uv_injection.py --epoch 4 [--image FITS/peel/ep4_inj/...pbcor.fits] [--r 2.8 --pa 137.6 --flux-mjy 2.83]
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postpeel_tools import PostPeelImage, en

# Defaults as in inject_knot_uv.py
DEFAULTS = {3: dict(r=2.6, flux_mjy=4.13), 4: dict(r=2.8, flux_mjy=2.83)}

p = argparse.ArgumentParser()
p.add_argument("--epoch", type=int, choices=sorted(DEFAULTS), required=True)
p.add_argument("--image", default=None, help="Post-peel image of the injected copy (default: FITS/peel/ep<N>_inj/*postpeel-MFS-image.pbcor.fits)")
p.add_argument("--r", type=float, default=None)
p.add_argument("--pa", type=float, default=137.6)
p.add_argument("--flux-mjy", type=float, default=None)
args = p.parse_args()

r = args.r or DEFAULTS[args.epoch]["r"]
s_inj = args.flux_mjy or DEFAULTS[args.epoch]["flux_mjy"]
image = args.image
if image is None:
    found = sorted(glob.glob(f"FITS/peel/ep{args.epoch}_inj/*postpeel-MFS-image.pbcor.fits"))
    if len(found) != 1:
        raise SystemExit(f"Expected one injected post-peel image in FITS/peel/ep{args.epoch}_inj/, found {found}")
    image = found[0]

orig = PostPeelImage(args.epoch)
inj = PostPeelImage(args.epoch, image=image)
pos = en(r, args.pa)

a_orig, e_orig, _ = orig.fit_fixed([pos])
a_inj, e_inj, _ = inj.fit_fixed([pos])
sig = orig.sigma_emp(r)[0]
amp_free, pos_free = inj.fit_free(pos, inj.img, amp0=s_inj)
frac = (a_inj[0] - a_orig[0]) / s_inj

print(f"Epoch {args.epoch}: knot injected at r = {r:.2f}\", PA = {args.pa:.1f} deg, S = {s_inj:.2f} mJy")
print(f"  original image : {a_orig[0]:6.2f} +/- {e_orig[0]:.2f} mJy/beam at the knot position")
print(f"  injected image : {a_inj[0]:6.2f} +/- {e_inj[0]:.2f} mJy/beam   ({image})")
print(f"  recovered      : {a_inj[0] - a_orig[0]:6.2f} mJy/beam = {frac:.0%} of the injected flux "
      f"({(a_inj[0] - a_orig[0]) / sig:.1f} sigma_emp, sigma_emp = {sig:.2f} mJy/beam)")
d = np.hypot(*(np.asarray(pos_free) - np.asarray(pos)))
print(f"  free position  : {amp_free:6.2f} mJy/beam, {d:.2f}\" from the injected position")
