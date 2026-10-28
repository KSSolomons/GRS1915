"""
Shared tools for the post-peel Epoch 3 and 4 images: cut-out around the core, beam-shaped component fits and the
empirical noise (scatter of fixed-position amplitudes at the same radius, which includes the ripple).

Used by test_predicted_knots.py and injection_recovery.py. The fit follows fit_postpeel_knots.py: chi2 over the pixels
within FIT_RADIUS of the core, with the 2-8" MAD noise, a core-residual component (|amp| <= 5 mJy/beam, position within
1") and beam-shaped components of fixed BMAJ, BMIN, BPA.
"""

import os
import re
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epoch_limits import UPPER_LIMIT_BLOCKS, beam_fwhm_along_pa

FIT_RADIUS = 8.0  # arcsec
JET_PA = 138.0  # SE direction (deg); the NW direction is JET_PA + 180
PA_EXCLUDE = 30.0  # deg around both jet directions left out of the noise scan
N_SCAN = 36  # positions in the noise scan

EPOCHS = {
    3: dict(
        image="FITS/peel/ep3/img_1740883517_sdp_l0_1024ch_GRS1915+105.ms_postpeel-MFS-image.pbcor.fits",
        region="data/peel_regions/epoch3_1740883517_peel_3as.reg",
        mjd=60736.11, block=2,
    ),
    4: dict(
        image="FITS/peel/ep4/img_1742780174_sdp_l0_1024ch_GRS1915+105.ms_postpeel-MFS-image.pbcor.fits",
        region="data/peel_regions/epoch4_1742780174_peel_3as.reg",
        mjd=60758.07, block=3,
    ),
}


def en(r, pa_deg):
    """(E, N) offset of a point at separation r and position angle pa (East of North)."""
    return r * np.sin(np.radians(pa_deg)), r * np.cos(np.radians(pa_deg))


class PostPeelImage:
    def __init__(self, epoch, n=30, image=None):
        """image: another post-peel image of the same epoch (e.g. the uv-injection run) instead of EPOCHS[epoch]."""
        cfg = EPOCHS[epoch]
        self.epoch, self.mjd = epoch, cfg["mjd"]
        self.s_expected = UPPER_LIMIT_BLOCKS[cfg["block"]]["s_jet_jy"] * 1e3  # mJy
        m = re.search(r"circle\(([^,]+),([^,]+),", open(cfg["region"]).read())
        core = SkyCoord(m.group(1), m.group(2), unit=("hourangle", "deg"))
        h = fits.open(image or cfg["image"], memmap=True)[0]
        hd = h.header
        w = WCS(hd).celestial
        self.pix = abs(hd["CDELT2"]) * 3600
        x0, y0 = (float(v) for v in w.world_to_pixel_values(core.ra.deg, core.dec.deg))
        xi, yi = int(round(x0)), int(round(y0))
        self.img = np.array(h.data[0, 0][yi - n : yi + n + 1, xi - n : xi + n + 1]) * 1e3  # mJy/beam
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        self.E = -(xx - (x0 - xi)) * self.pix
        self.N = (yy - (y0 - yi)) * self.pix
        self.r = np.hypot(self.E, self.N)
        self.bmaj, self.bmin, self.bpa_deg = hd["BMAJ"] * 3600, hd["BMIN"] * 3600, hd["BPA"]
        self.bpa = np.radians(self.bpa_deg)
        self.f2s = 1 / (2 * np.sqrt(2 * np.log(2)))
        self.beam_px = 1.1331 * self.bmaj * self.bmin / self.pix**2
        mad = lambda v: 1.4826 * np.median(np.abs(v - np.median(v)))
        self.noise = mad(self.img[(self.r > 2) & (self.r < 8)])
        sel = self.r < FIT_RADIUS
        self.Es, self.Ns = self.E[sel], self.N[sel]
        self.sel = sel
        self.b = np.sqrt(self.bmaj * self.bmin)
        self.fwhm_jet = beam_fwhm_along_pa(self.bmaj, self.bmin, self.bpa_deg, JET_PA)[0]
        self.core0 = self._fit_core()
        self._scan_cache = {}

    def beam(self, e0, n0):
        de, dn = self.Es - e0, self.Ns - n0
        u = de * np.sin(self.bpa) + dn * np.cos(self.bpa)
        v = de * np.cos(self.bpa) - dn * np.sin(self.bpa)
        return np.exp(-0.5 * ((u / (self.bmaj * self.f2s)) ** 2 + (v / (self.bmin * self.f2s)) ** 2))

    def _fit_core(self):
        d = self.img[self.sel]
        f = least_squares(lambda p: (p[0] * self.beam(p[1], p[2]) - d) / self.noise,
                          [np.clip(self.img[30, 30], -4.9, 4.9), 0, 0], bounds=([-5, -1, -1], [5, 1, 1]))
        return f.x

    def fit_fixed(self, positions, img=None):
        """Core residual plus components at fixed (E, N); returns amplitudes (mJy/beam), formal errors, core params."""
        d = (self.img if img is None else img)[self.sel]
        b = [self.beam(*p) for p in positions]
        k = len(b)

        def res(q):
            m = q[0] * self.beam(q[1], q[2])
            for a, bb in zip(q[3:], b):
                m = m + a * bb
            return (m - d) / self.noise

        f = least_squares(res, list(self.core0) + [0.0] * k,
                          bounds=([-5, -1, -1] + [-20] * k, [5, 1, 1] + [20] * k))
        err = np.sqrt(np.diag(np.linalg.pinv(f.jac.T @ f.jac)))
        return f.x[3:], err[3:], f.x[:3]

    def fit_free(self, pos0, img, amp0=1.0, box=1.0):
        """Core residual plus one component free in amplitude and in position within +/-box of pos0."""
        d = img[self.sel]

        def res(q):
            return (q[0] * self.beam(q[1], q[2]) + q[3] * self.beam(q[4], q[5]) - d) / self.noise

        lo = [-5, -1, -1, -20, pos0[0] - box, pos0[1] - box]
        hi = [5, 1, 1, 20, pos0[0] + box, pos0[1] + box]
        f = least_squares(res, list(self.core0) + [amp0, *pos0], bounds=(lo, hi))
        return f.x[3], f.x[4:6]

    def sigma_emp(self, r):
        """Scatter of fixed-position amplitudes at radius r over N_SCAN evenly spaced PAs, away from the jet axes."""
        key = round(float(r), 3)
        if key not in self._scan_cache:
            pas = np.arange(N_SCAN) * 360.0 / N_SCAN
            dpa = lambda a, b: np.abs((a - b + 180) % 360 - 180)
            keep = [pa for pa in pas if min(dpa(pa, JET_PA), dpa(pa, JET_PA + 180)) > PA_EXCLUDE]
            amps = np.array([self.fit_fixed([en(r, pa)])[0][0] for pa in keep])
            self._scan_cache[key] = (amps.std(ddof=1), amps.mean(), len(amps))
        return self._scan_cache[key]
