"""
Epoch 4 uv-injection check (src/inject_knot_uv.py, src/measure_uv_injection.py): original post-peel image, post-peel
image of the injected copy, their difference, and a cut along the jet axis through the core.

    python src/plot_uv_injection.py [--image FITS/peel/ep4_inj/...postpeel-MFS-image.fits]
"""

import argparse
import glob

import astropy.units as u
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Circle, Ellipse
from scipy.ndimage import map_coordinates

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

CORE = "19:15:11.5085 +10:56:43.507"  # data/peel_regions/epoch4_1742780174_peel_3as.reg
R_INJ, PA_INJ, S_INJ = 2.8, 137.6, 2.83  # arcsec, deg, mJy (inject_knot_uv.py defaults, Epoch 4)
HALF = 9.0
OUT = "figures/peel/ep4_uv_injection.png"

p = argparse.ArgumentParser()
p.add_argument("--image", default=None)
args = p.parse_args()
inj_path = args.image or glob.glob("FITS/peel/ep4_inj/*postpeel-MFS-image*.fits")[0]
orig_path = glob.glob("FITS/peel/ep4/*postpeel-MFS-image.pbcor.fits")[0]
psf_path = glob.glob("FITS/peel/ep4/*postpeel-MFS-psf.fits")[0]
core = SkyCoord(CORE, unit=(u.hourangle, u.deg))


def load(path):
    """Cut-out (mJy/beam) around the core with pixel offsets of the core, pixel size and header."""
    with fits.open(path, memmap=True) as h:
        hd = h[0].header
        x0, y0 = (float(v) for v in WCS(hd).celestial.world_to_pixel(core))
        pix = abs(hd["CDELT2"]) * 3600
        n = int(np.ceil(HALF / pix)) + 4
        xi, yi = int(round(x0)), int(round(y0))
        sub = np.array(h[0].data[0, 0][yi - n : yi + n + 1, xi - n : xi + n + 1], dtype=float) * 1e3
    return sub, (n + x0 - xi, n + y0 - yi), pix, hd


orig, (cx, cy), pix, hd = load(orig_path)
inj, _, _, _ = load(inj_path)
diff = inj - orig
ny, nx = orig.shape
ext = [cx * pix + pix / 2, -(nx - 1 - cx) * pix - pix / 2, -cy * pix - pix / 2, (ny - 1 - cy) * pix + pix / 2]
bmaj, bmin, bpa = hd["BMAJ"] * 3600, hd["BMIN"] * 3600, hd["BPA"]
e_inj, n_inj = R_INJ * np.sin(np.radians(PA_INJ)), R_INJ * np.cos(np.radians(PA_INJ))

# Cut along the jet axis: s > 0 towards the SE (PA_INJ), s < 0 towards the NW
s = np.linspace(-HALF, HALF, 721)
xs = cx - s * np.sin(np.radians(PA_INJ)) / pix  # East is towards -x
ys = cy + s * np.cos(np.radians(PA_INJ)) / pix
cut = {k: map_coordinates(v, [ys, xs], order=1) for k, v in (("orig", orig), ("inj", inj), ("diff", diff))}
# The knot is outside the CLEAN mask, so a knot that survived intact would appear as S_INJ x PSF (dirty beam)
with fits.open(psf_path, memmap=True) as h:
    psf = np.asarray(h[0].data[0, 0])
py, px = np.unravel_index(np.argmax(psf), psf.shape)
ds = s - R_INJ
psf_cut = map_coordinates(psf, [py + ds * np.cos(np.radians(PA_INJ)) / pix, px - ds * np.sin(np.radians(PA_INJ)) / pix], order=1)

fig = plt.figure(figsize=(13, 8.2), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.62])
vmax = max(np.abs(inj).max(), 1.0)
titles = ["(a) Original post-peel", "(b) Injected copy, post-peel", "(c) Difference (b) $-$ (a)"]
for i, (img, t) in enumerate(zip((orig, inj, diff), titles)):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlim(HALF, -HALF)
    ax.set_ylim(-HALF, HALF)
    ax.plot([-HALF * np.sin(np.radians(PA_INJ)), HALF * np.sin(np.radians(PA_INJ))],
            [-HALF * np.cos(np.radians(PA_INJ)), HALF * np.cos(np.radians(PA_INJ))], "k--", lw=0.7, alpha=0.5)
    ax.add_patch(Circle((0, 0), 3.0, fill=False, ls=":", lw=1.0, ec="0.3"))
    ax.plot(0, 0, "k+", ms=9)
    ax.plot(e_inj, n_inj, "o", ms=13, mfc="none", mec="k", mew=1.4)
    ax.add_patch(Ellipse((-HALF + 1.8, -HALF + 1.8), bmin, bmaj, angle=-bpa, fc="0.4", ec="k"))
    ax.set_title(t, fontsize=11)
    ax.set_xlabel(r"$\Delta\alpha$ (arcsec)")
    if i == 0:
        ax.set_ylabel(r"$\Delta\delta$ (arcsec)")
        ax.text(e_inj + 0.6, n_inj - 2.6, "injected\n2.83 mJy", fontsize=8, va="top", ha="center")
        ax.text(-1.6, -3.4, '3" peel region', fontsize=8, color="0.3", va="top")
fig.colorbar(im, ax=fig.axes[:3], shrink=0.85, label="mJy beam$^{-1}$")

ax = fig.add_subplot(gs[1, :])
ax.axhline(0, color="0.6", lw=0.8)
ax.axvspan(-3, 3, color="0.92", zorder=0)
ax.plot(s, cut["orig"], color="0.45", lw=2, label="(a) original")
ax.plot(s, cut["inj"], color="#1f4e8c", lw=2, label="(b) injected copy")
ax.plot(s, cut["diff"], color="#c0392b", lw=2, label="(c) difference")
ax.plot(s, S_INJ * psf_cut, color="k", lw=1.2, ls="--", label=f"injected knot if it survived intact ({S_INJ} mJy x PSF)")
ax.axvline(R_INJ, color="k", lw=0.6, alpha=0.4)
ax.set_xlim(-HALF, HALF)
ax.set_ylim(top=3.9)
ax.set_xlabel(r"Distance from the core along PA 137.6$^\circ$ (arcsec; $<0$ NW, $>0$ SE)")
ax.set_ylabel("mJy beam$^{-1}$")
ax.set_title('(d) Cut along the jet axis (shaded: 3" peel region)', fontsize=11)
ax.legend(fontsize=9, ncol=3, loc="upper left", frameon=False)
fig.suptitle("Epoch 4 uv injection: knot at r = 2.8\", PA 137.6$^\\circ$", fontsize=12)
fig.savefig(OUT, dpi=170)
print(f"Saved {OUT}")
