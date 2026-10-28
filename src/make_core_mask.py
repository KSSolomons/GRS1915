"""
Core-only 3GC clean mask for the peel: within --radius of the core, keep only a --disc disc of the original oxkat mask1
(everything else there set to 0); the rest of the mask is unchanged. Same edit as the Epoch 1, 3 and 4 masks in
data/peel_regions/new_mask1_core*/ (reproduces the Epoch 3 and 4 files byte for byte).

    python src/make_core_mask.py masks/epoch2/img_1738471273_..._mask1.fits --core "19:15:11.5419 +10:56:44.701" \
        --out data/peel_regions/new_mask1_core_epoch2/

The output keeps the original filename (oxkat takes the first *mask1.fits in IMAGES/), so copy it over the original there.
"""

import argparse
import hashlib
import os

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

p = argparse.ArgumentParser()
p.add_argument("mask", help="Original oxkat *mask1.fits")
p.add_argument("--core", required=True, help='Core position, "HH:MM:SS.ssss +DD:MM:SS.sss"')
p.add_argument("--radius", type=float, default=10.0, help="Radius around the core that is cleared (arcsec)")
p.add_argument("--disc", type=float, default=1.0, help="Radius of the core disc kept (arcsec)")
p.add_argument("--out", required=True, help="Output directory")
args = p.parse_args()

data, hdr = fits.getdata(args.mask, header=True)
wcs = WCS(hdr).celestial
core = SkyCoord(args.core, unit=(u.hourangle, u.deg))
x0, y0 = wcs.world_to_pixel(core)
pix = abs(hdr["CDELT2"]) * 3600.0

n = int(np.ceil(args.radius / pix)) + 2
ys, xs = np.mgrid[int(y0) - n:int(y0) + n + 1, int(x0) - n:int(x0) + n + 1]
r = np.hypot(xs - x0, ys - y0) * pix
plane = data[0, 0] if data.ndim == 4 else data
box = plane[ys[:, 0][0]:ys[:, 0][-1] + 1, xs[0][0]:xs[0][-1] + 1]
clear = (r <= args.radius) & (r > args.disc) & (box != 0)
box[clear] = 0
kept = int(((r <= args.disc) & (box != 0)).sum())

hdr.add_history(f'mask1 edited: within {args.radius:.1f}" of the core ({args.core}) only a {args.disc:.1f}" core disc is kept')

os.makedirs(args.out, exist_ok=True)
out = os.path.join(args.out, os.path.basename(args.mask))
fits.writeto(out, np.array(data), hdr, overwrite=True)
md5 = hashlib.md5(open(out, "rb").read()).hexdigest()
print(f"core pixel ({x0:.2f}, {y0:.2f}); {int(clear.sum())} px cleared, {kept} px kept in the {args.disc}\" disc")
print(f"wrote {out}\nmd5 {md5}")
