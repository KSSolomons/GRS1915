"""
Overview figures of the 3GC peel set-up and products, for docs/peel_status/peel_status.tex.

  1. figures/peel/peel_masks_ep234.png: pre-peel (2GC pcalmask) image of Epochs 2-4 around the core with the original
     oxkat mask1 and the core-only mask1 (src/make_core_mask.py), the 3" peel region and the 10" cleared radius.
  2. figures/peel/peel_products_ep34.png: for Epochs 3 and 4, the pre-peel image, the pre-peel CLEAN model and the
     post-peel image, with the peel region.

Also prints the numbers quoted in the document (core peak before/after, residual peak and rms within 6").

    python src/plot_peel_overview.py
"""

import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle, Ellipse

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

HALF = 12.0  # cutout half-width (arcsec)
EPOCHS = {
    2: dict(obs="1738471273", core="19:15:11.5419 +10:56:44.701"),
    3: dict(obs="1740883517", core="19:15:11.5449 +10:56:44.612"),
    4: dict(obs="1742780174", core="19:15:11.5085 +10:56:43.507"),
}
OUT = "figures/peel"


def cutout(path, core):
    """Cutout (arcsec offsets, data) of the first image plane around the core; also returns the header."""
    with fits.open(path, memmap=True) as h:
        hdr = h[0].header
        w = WCS(hdr).celestial
        x0, y0 = w.world_to_pixel(core)
        pix = abs(hdr["CDELT2"]) * 3600.0
        n = int(np.ceil(HALF / pix)) + 1
        xi, yi = int(round(float(x0))), int(round(float(y0)))
        d = h[0].data
        plane = d[0, 0] if d.ndim == 4 else d
        sub = np.array(plane[yi - n:yi + n + 1, xi - n:xi + n + 1], dtype=float)
    # extent in arcsec, east to the left
    dx0, dy0 = (xi - x0) * pix, (yi - y0) * pix
    ext = [n * pix + pix / 2 - dx0, -n * pix - pix / 2 - dx0, -n * pix - pix / 2 + dy0, n * pix + pix / 2 + dy0]
    return sub, ext, hdr, pix


def radius_grid(sub, ext):
    ny, nx = sub.shape
    x = np.linspace(ext[0], ext[1], nx)
    y = np.linspace(ext[2], ext[3], ny)
    X, Y = np.meshgrid(x, y)
    return np.hypot(X, Y)


def decorate(ax, hdr=None, peel=True, cleared=False, half=HALF):
    if peel:
        ax.add_patch(Circle((0, 0), 3.0, fill=False, ls="--", lw=1.0, color="tab:cyan"))
    if cleared:
        ax.add_patch(Circle((0, 0), 10.0, fill=False, ls=":", lw=1.0, color="w"))
    if hdr is not None and "BMAJ" in hdr:
        ax.add_patch(Ellipse((HALF - 2.5, -HALF + 2.5), hdr["BMAJ"] * 3600, hdr["BMIN"] * 3600, angle=hdr["BPA"],
                             fc="none", ec="w", lw=1.0))
    ax.set_xlim(half, -half)
    ax.set_ylim(-half, half)
    ax.set_xlabel(r"$\Delta\alpha$ (arcsec)")


os.makedirs(OUT, exist_ok=True)

# 1. Masks
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
for ax, (ep, e) in zip(axes, EPOCHS.items()):
    core = SkyCoord(e["core"], unit=(u.hourangle, u.deg))
    img, ext, hdr, _ = cutout(glob.glob(f"FITS/pcalmask/img_{e['obs']}_*pcalmask-MFS-image.pbcor.fits")[0], core)
    m0, _, _, _ = cutout(glob.glob(f"masks/epoch{ep}/img_{e['obs']}_*mask1.fits")[0], core)
    m1, _, _, _ = cutout(glob.glob(f"data/peel_regions/new_mask1_core_epoch{ep}/img_{e['obs']}_*mask1.fits")[0], core)
    ax.imshow(img * 1e3, origin="lower", extent=ext, cmap="magma", norm=SymLogNorm(0.5, vmin=-1, vmax=img.max() * 1e3))
    ax.contour(m0, levels=[0.5], extent=ext, colors="0.8", linewidths=1.0)
    ax.contourf(m1, levels=[0.5, 1.5], extent=ext, colors="none", hatches=["////"])
    ax.contour(m1, levels=[0.5], extent=ext, colors="tab:green", linewidths=1.5)
    decorate(ax, hdr, cleared=True)
    ax.set_title(f"Epoch {ep}: {int((m0 > 0).sum())} $\\to$ {int((m1 > 0).sum())} mask px in view")
axes[0].set_ylabel(r"$\Delta\delta$ (arcsec)")
fig.savefig(f"{OUT}/peel_masks_ep234.png", dpi=200)
plt.close(fig)

# 2. Peel products
fig, axes = plt.subplots(2, 3, figsize=(13, 8.4), constrained_layout=True)
print("epoch  core_pre(mJy/b)  peak_post<6\"(mJy/b)  rms_post 3-6\"(mJy/b)  model_flux<3\"(mJy)  model_flux 3-10\"(mJy)")
for row, ep in zip(axes, (3, 4)):
    e = EPOCHS[ep]
    core = SkyCoord(e["core"], unit=(u.hourangle, u.deg))
    pre, ext, hpre, _ = cutout(glob.glob(f"FITS/pcalmask/img_{e['obs']}_*pcalmask-MFS-image.pbcor.fits")[0], core)
    mod, ext_m, _, _ = cutout(glob.glob(f"FITS/peel/ep{ep}/img_{e['obs']}_*prepeel-MFS-model.fits")[0], core)
    post, ext_p, hpost, _ = cutout(glob.glob(f"FITS/peel/ep{ep}/img_{e['obs']}_*postpeel-MFS-image.pbcor.fits")[0], core)
    m0, ext_k, _, _ = cutout(glob.glob(f"masks/epoch{ep}/img_{e['obs']}_*mask1.fits")[0], core)
    m1, _, _, _ = cutout(glob.glob(f"data/peel_regions/new_mask1_core_epoch{ep}/img_{e['obs']}_*mask1.fits")[0], core)
    r = radius_grid(post, ext_p)
    rm = radius_grid(mod, ext_m)
    ann = (r > 3) & (r <= 6)
    rms = 1.4826 * np.median(np.abs(post[ann] - np.median(post[ann])))
    print(f"{ep:5d}  {pre.max() * 1e3:15.1f}  {np.abs(post[r <= 6]).max() * 1e3:19.2f}  {rms * 1e3:19.3f}"
          f"  {mod[rm <= 3].sum() * 1e3:17.1f}  {mod[(rm > 3) & (rm <= 10)].sum() * 1e3:19.2f}")

    im = row[0].imshow(pre * 1e3, origin="lower", extent=ext, cmap="magma", norm=SymLogNorm(0.5, vmin=-1, vmax=pre.max() * 1e3))
    fig.colorbar(im, ax=row[0], shrink=0.8, label="mJy beam$^{-1}$")
    row[0].set_title(f"Epoch {ep}: pre-peel image")
    decorate(row[0], hpre)

    lim = np.abs(mod).max() * 1e3
    im = row[1].imshow(mod * 1e3, origin="lower", extent=ext_m, cmap="RdBu_r", norm=SymLogNorm(0.01, vmin=-lim, vmax=lim))
    fig.colorbar(im, ax=row[1], shrink=0.8, label="mJy pixel$^{-1}$")
    row[1].set_title(f"Epoch {ep}: CLEAN model (zoom)")
    decorate(row[1], half=4.0)

    vmax = np.abs(post[r <= 6]).max() * 1e3
    im = row[2].imshow(post * 1e3, origin="lower", extent=ext_p, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=row[2], shrink=0.8, label="mJy beam$^{-1}$")
    row[2].set_title(f"Epoch {ep}: post-peel image")
    decorate(row[2], hpost)
    row[0].contour(m0, levels=[0.5], extent=ext_k, colors="0.8", linewidths=1.0)
    for a in row:
        a.contour(m1, levels=[0.5], extent=ext_k, colors="tab:green", linewidths=1.5)
    for a in row[1:]:
        a.plot(0, 0, "k+", ms=8)
    row[0].set_ylabel(r"$\Delta\delta$ (arcsec)")
fig.savefig(f"{OUT}/peel_products_ep34.png", dpi=200)
plt.close(fig)
print(f"Saved {OUT}/peel_masks_ep234.png, {OUT}/peel_products_ep34.png")
