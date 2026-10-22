"""
Overlay the FITS peel masks (data/peel_regions/*_peel_mask.fits) on the pbcor MFS images of Epochs 1-4.

Each panel shows a cutout around the core with the mask outline, the synthesised beam and the jet axis
(PA 138/318 deg), so the mask size can be compared with the core and with where the knots should sit.
"""

import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Ellipse

HALF = 8.0  # cutout half-width (arcsec)
JET_PA = 138.0

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

masks = sorted(glob.glob("data/peel_regions/epoch*_peel_mask.fits"))
fig, axes = plt.subplots(1, len(masks), figsize=(4.2 * len(masks), 4.6))

for ax, mask_path in zip(np.atleast_1d(axes), masks):
    tag = os.path.basename(mask_path).split("_")
    epoch, block = tag[0], tag[1]
    image_path = glob.glob(f"FITS/img_{block}*MFS-image.pbcor.fits")[0]

    with fits.open(image_path, memmap=True) as himg, fits.open(mask_path, memmap=True) as hmask:
        header = himg[0].header
        pix = abs(header["CDELT2"]) * 3600.0
        mask = hmask[0].data[0, 0]
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        n = int(np.ceil(HALF / pix))
        sl = (slice(int(cy) - n, int(cy) + n + 1), slice(int(cx) - n, int(cx) + n + 1))
        img = np.array(himg[0].data[0, 0][sl], dtype=float)
        msk = np.array(mask[sl], dtype=float)

    # Offsets in arcsec about the mask centre (east to the left)
    ny, nx = img.shape
    x_off = -(np.arange(nx) + sl[1].start - cx) * pix
    y_off = (np.arange(ny) + sl[0].start - cy) * pix
    extent = [x_off[0] + 0.5 * pix, x_off[-1] - 0.5 * pix, y_off[0] - 0.5 * pix, y_off[-1] + 0.5 * pix]

    im = ax.imshow(img * 1e3, origin="lower", cmap="inferno", extent=extent,
                   norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=np.nanmax(img) * 1e3))
    ax.contour(x_off, y_off, msk, levels=[0.5], colors="cyan", linewidths=1.5)
    for pa in (JET_PA, JET_PA + 180.0):
        ax.plot([0, HALF * np.sin(np.radians(pa))], [0, HALF * np.cos(np.radians(pa))], color="white", ls="--", lw=0.8)

    bmaj, bmin, bpa = header["BMAJ"] * 3600.0, header["BMIN"] * 3600.0, header["BPA"]
    # Ellipse angle is measured from +x; with east to the left a PA (E of N) maps to 90 + PA
    ax.add_patch(Ellipse((HALF - 0.6 * bmaj, -HALF + 0.6 * bmaj), bmin, bmaj, angle=-bpa,
                         facecolor="none", edgecolor="white", lw=1.0))

    ax.set_xlim(HALF, -HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_xlabel("East offset (arcsec)")
    ax.set_title(f"{epoch.capitalize()} ({block})\nmask: {int(msk.sum())} px", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="mJy/beam")

np.atleast_1d(axes)[0].set_ylabel("North offset (arcsec)")
fig.tight_layout()
out = "figures/peel_masks_vs_pbcor.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
