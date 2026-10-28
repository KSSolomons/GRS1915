#!/usr/bin/env python3
"""
Per-epoch astrometric frame offsets of the S-band MFS images, from compact field sources.

Each campaign image (FITS/pcalmask/*pcalmask-MFS-image.pbcor.fits) can be shifted as a whole by a
phase-calibration error. This script fits a 2D Gaussian to a set of compact field sources
in every epoch, takes each source's median position over the epochs as its reference, and
writes the offset of every source in every epoch plus the weighted-mean frame offset of the
epoch. Subtracting that frame offset from all positions of the epoch (core and lobes) ties
the epochs to a common frame.

The reference sources are the compact sources brighter than MIN_PEAK_MJY within MAX_RADIUS
of GRS 1915+105 in the Epoch 6 image whose fitted size is close to the beam in every epoch.
The scatter of the individual sources about the frame offset tests whether the shift is a
uniform translation.

Run from the repository root: python src/measure_frame_offsets.py
Writes data/frame_offsets.txt.
"""

import glob
import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import maximum_filter
from scipy.optimize import curve_fit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IMAGES = sorted(glob.glob(os.path.join(ROOT, "FITS", "pcalmask", "*pcalmask-MFS-image.pbcor.fits")))
SELECT_EPOCH = "1749586570"  # Epoch 6: low core, typical beam
OUTPUT = os.path.join(ROOT, "data", "frame_offsets.txt")
TARGET = SkyCoord("19h15m11.558s", "+10d56m44.905s")
MAX_RADIUS = 20.0  # arcmin
MIN_RADIUS = 1.5   # arcmin, keeps clear of GRS 1915+105 and its jets
MIN_PEAK_MJY = 3.0
MAX_SIZE_RATIO = 1.3  # fitted FWHM / beam, in every epoch
HALF = 12  # cutout half-size (pixels)


def g2(xy, amp, x0, y0, sx, sy, th, z):
    x, y = xy
    a = np.cos(th) ** 2 / (2 * sx**2) + np.sin(th) ** 2 / (2 * sy**2)
    b = np.sin(2 * th) * (1 / (4 * sy**2) - 1 / (4 * sx**2))
    c = np.sin(th) ** 2 / (2 * sx**2) + np.cos(th) ** 2 / (2 * sy**2)
    return (amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)) + z).ravel()


def load(path):
    header = fits.getheader(path)
    data = fits.getdata(path, memmap=True)
    return header, WCS(header).celestial, data.reshape(data.shape[-2:])


def fit_source(data, wcs, coord, beam_pix):
    """Gaussian fit around coord. Returns (SkyCoord, err_arcsec, size_ratio, peak) or None."""
    px, py = (int(round(float(v))) for v in wcs.world_to_pixel(coord))
    cut = np.nan_to_num(np.array(data[py - HALF:py + HALF + 1, px - HALF:px + HALF + 1]))
    if cut.shape != (2 * HALF + 1, 2 * HALF + 1):
        return None
    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    j = np.unravel_index(np.argmax(cut), cut.shape)
    try:
        p, cov = curve_fit(g2, (xx, yy), cut.ravel(), p0=[cut[j], xx[j], yy[j], 2, 2, 0, 0], maxfev=20000)
    except RuntimeError:
        return None
    pos = wcs.pixel_to_world(px + p[1], py + p[2])
    pix = abs(wcs.wcs.cdelt[1]) * 3600.0
    err = pix * np.sqrt(0.5 * (cov[1, 1] + cov[2, 2]))
    size = 2.3548 * np.sqrt(abs(p[3] * p[4])) / beam_pix
    return pos, err, size, p[0]


def main():
    header, wcs, data = load(next(f for f in IMAGES if SELECT_EPOCH in f))
    pix = abs(wcs.wcs.cdelt[1]) * 3600.0
    cx, cy = (int(round(float(v))) for v in wcs.world_to_pixel(TARGET))
    r = int(MAX_RADIUS * 60 / pix)
    sub = np.nan_to_num(np.array(data[cy - r:cy + r, cx - r:cx + r]))
    peaks = (sub == maximum_filter(sub, size=41)) & (sub > MIN_PEAK_MJY * 1e-3)
    yy, xx = np.nonzero(peaks)
    rad = np.hypot(yy - r, xx - r) * pix / 60
    keep = rad > MIN_RADIUS
    candidates = [wcs.pixel_to_world(x + cx - r, y + cy - r) for x, y in zip(xx[keep], yy[keep])]

    epochs = []
    for path in IMAGES:
        h, w, d = load(path)
        ts = os.path.basename(path).split("img_")[1][:10]
        beam_pix = np.sqrt(h["BMAJ"] * h["BMIN"]) * 3600.0 / pix
        epochs.append((ts, int(ts) / 86400.0 + 40587.0, [fit_source(d, w, c, beam_pix) for c in candidates]))

    # Sources compact and fitted in every epoch
    good = [i for i in range(len(candidates))
            if all(e[2][i] is not None and e[2][i][2] < MAX_SIZE_RATIO for e in epochs)]
    cosd = np.cos(TARGET.dec.radian)
    ref = {}
    for i in good:
        # Median, not mean: the zero point of the frame then follows a typical epoch and is
        # not dragged by the strongly shifted Epochs 1 and 4, so positions stay comparable
        # with the catalogue position of GRS 1915+105
        ra = np.median([e[2][i][0].ra.deg for e in epochs])
        dec = np.median([e[2][i][0].dec.deg for e in epochs])
        ref[i] = (ra, dec)

    lines = [
        "# Per-epoch frame offsets of the S-band MFS images (written by src/measure_frame_offsets.py)",
        f"# Reference sources: {len(good)} compact field sources ({MIN_RADIUS}-{MAX_RADIUS} arcmin from GRS 1915+105, "
        f"peak > {MIN_PEAK_MJY} mJy/beam, fitted size < {MAX_SIZE_RATIO}x beam in every epoch); each source's",
        "# reference position is its median over the epochs. dx = dRA cos(dec), dy = dDec (arcsec): source offset in that epoch.",
        "# Frame offset = inverse-variance mean over sources; scatter = rms of the sources about it.",
        "# Subtract (dx_frame, dy_frame) from every position measured in that epoch.",
        "#",
        "# Sources:",
    ]
    for k, i in enumerate(good):
        c = SkyCoord(ref[i][0], ref[i][1], unit="deg")
        lines.append(f"#   S{k + 1}: {c.to_string('hmsdms', precision=2)}  {c.separation(TARGET).arcmin:5.2f}'  "
                     f"peak {np.mean([e[2][i][3] for e in epochs]) * 1e3:7.2f} mJy/beam")
    lines += ["#", "# FRAME", "# Epoch          MJD        dx_frame  dy_frame  err_x   err_y   scatter_x scatter_y  N"]
    rows = []
    for ts, mjd, fits_ in epochs:
        dx = np.array([(fits_[i][0].ra.deg - ref[i][0]) * 3600 * cosd for i in good])
        dy = np.array([(fits_[i][0].dec.deg - ref[i][1]) * 3600 for i in good])
        err = np.array([max(fits_[i][1], 1e-3) for i in good])
        w = 1 / err**2
        fx, fy = np.sum(w * dx) / np.sum(w), np.sum(w * dy) / np.sum(w)
        sx, sy = np.sqrt(np.mean((dx - fx) ** 2)), np.sqrt(np.mean((dy - fy) ** 2))
        # error of the frame offset: the larger of the formal error and the scatter / sqrt(N)
        ex = max(1 / np.sqrt(np.sum(w)), sx / np.sqrt(len(good)))
        ey = max(1 / np.sqrt(np.sum(w)), sy / np.sqrt(len(good)))
        lines.append(f"{ts:<14} {mjd:10.4f} {fx:+9.3f} {fy:+9.3f} {ex:7.3f} {ey:7.3f} {sx:9.3f} {sy:9.3f} {len(good):3d}")
        rows.append((ts, dx, dy))
    lines += ["#", "# PER SOURCE (dx, dy in arcsec)", "# Epoch          " + " ".join(f"{'S' + str(k + 1):>15}" for k in range(len(good)))]
    for ts, dx, dy in rows:
        lines.append(f"{ts:<16}" + " ".join(f"{x:+7.3f},{y:+7.3f}" for x, y in zip(dx, dy)))
    with open(OUTPUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
