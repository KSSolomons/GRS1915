# -*- coding: utf-8 -*-
"""
In-band spectral index per epoch for the MeerKAT monitoring of GRS 1915+105.

A stripped-down relative of getflux_MK_channels.py, intended to be run over the
whole monitoring archive rather than over the four DDT blocks. It fits the
per-channel images of each observation, then fits a single power law across the
band, and writes one row per epoch. There is no cross-band or joint fitting,
since the monitoring is L-band throughout.

Run inside CASA:

    CASA <1>: execfile('getflux_MK_inband_alpha.py')

Orchestration
-------------
On loading, the script asks how to select the data, exactly as
getflux_MK_Claude_DIR_or_IMG.py does:

    Process (d)irectories, (f)iles directly, or (a)utomatic search? [d/f/a]

  (d) Directories. Give a pattern such as 15* or a space-separated list. Each
      directory is one epoch, and its channel images are looked for in an
      IMAGES/ subdirectory first and then in the directory itself, which is the
      same convention as the band-averaged script.
  (f) Files. Give a pattern or a list of FITS files. They are grouped into
      epochs by the scheduling-block ID in the filename where one is present,
      and otherwise by the directory containing them.
  (a) Automatic. Walks ROOTS below, which may be plain directories or glob
      patterns, pruning measurement sets and stopping at MAX_DEPTH.

The three routes are also callable directly, which is convenient when
re-running part of a set:

    RESULTS = analyse_dirs(['1563825653', '1564248236'])
    RESULTS = analyse_files(glob.glob('IMAGES/img*-0*-image.fits'))
    RESULTS = main('/scratch3/users/sem/GRS1915/GRSKAT/15*')

The per-channel images are the WSClean products whose name carries a four-digit
channel index in place of "MFS":

    img_<blockid>_...ms_pcalmask-0000-image.fits ... -0008-image.fits

Note that the imaging run is selected by CHANNEL_GLOB, not by MASK_TOKEN. The
mask names are nested, 'pcalmask' being a substring of 'pcalmask_rep' and
'pcalmask_rep_2nd', so a substring test would match all three and yield several
images per channel. The hyphen after the mask name in the glob rules the longer
forms out, and any duplicate channel index that slips through is reported and
removed rather than double-counted.

Primary-beam-corrected products are excluded: the target sits close to the
phase centre, so the correction is small, and it is not applied per channel.

What it does per image
----------------------
A single Gaussian is fitted with the shape held fixed to that image's own
restoring beam, which widens across the band and is therefore read image by
image, the centroid held at the source position, and a constant zero level
fitted simultaneously. The last of these is not optional: with the shape fixed,
an unmodelled pedestal b is absorbed into the amplitude and inflates it by 2b,
not by b, because the ratio of the integrals of a Gaussian and of its square is
exactly two.

What it does per epoch
----------------------
S(nu) = S0 (nu/nu0)^alpha is fitted by weighted least squares in linear flux
space, so marginal and negative channels are kept without the bias that
discarding them would introduce. Note that a flux-scale error is a single
multiplicative factor common to every channel of a band, hence it cancels
exactly in an in-band spectral index and no systematic term is applied here.

Expect alpha to be useful only for the brighter epochs. Splitting an
observation into N channels raises the per-channel rms by sqrt(N), and for nine
channels across the L band the uncertainty is roughly

    sigma_alpha ~ 1.7 / (S/N per channel) ~ 5 / (S/N of the band-averaged image)

so about 0.05 for an epoch detected at 100 sigma, 0.17 at 30 sigma and 0.5 at
10 sigma. The output carries chi2/dof and a quality flag to make filtering
straightforward.

Author: prepared for S. Motta, 2026-08-02.
"""

import os
import re
import glob
import math

import numpy as np

# ===========================================================================
# USER-CONFIGURABLE CONSTANTS
# ===========================================================================

# --- Where the observations are ---------------------------------------------
ROOTS      = ['.']        # directories or glob patterns, one entry per line
RECURSIVE  = True         # walk below each root (handles IMAGES/ subdirs)
MAX_DEPTH  = 3            # how far below each root to descend; None for no limit

# Directory names matching these are never descended into. Measurement sets and
# CASA images are themselves directories containing thousands of table files,
# so walking into them costs minutes and can never find a FITS image.
SKIP_DIR_SUFFIXES = ('.ms', '.MS', '.im', '.image', '.model', '.residual',
                     '.psf', '.pb', '.sumwt', '.mask', '.flagversions',
                     '.tt0', '.tt1', '.workdirectory', '.cache', '.tmp')
SKIP_DIR_NAMES = ('.git', '__pycache__', 'CASA_LOGS', 'logs', 'caltables')

# Channel images: four digits in place of "MFS". The imaging run is selected
# here rather than through MASK_TOKEN, because the mask names are nested:
# 'pcalmask' is a substring of 'pcalmask_rep' and 'pcalmask_rep_2nd', so a
# substring test would keep all three and give two or three images per channel.
# The literal hyphen after the mask name in the glob excludes the longer forms,
# since those continue with an underscore.
CHANNEL_GLOB = 'img*pcalmask-[0-9][0-9][0-9][0-9]-image.fits'
# The band-averaged image is fitted too, purely as a reference column.
MFS_GLOB     = 'img*pcalmask-MFS-image.fits'
INCLUDE_MFS  = True
# Optional extra substring requirement, applied on top of the globs above.
# Normally leave this at None and put the imaging run in the globs.
MASK_TOKEN   = None

# --- Source position (GRS 1915+105) -----------------------------------------
SOURCE_RA  = '19h15m11.558s'
SOURCE_DEC = '+10d56m44.905s'

# Fixing the centroid makes each channel a forced photometry. At high signal to
# noise it makes no practical difference; at low signal to noise a free centroid
# locks onto noise peaks and biases the flux high, so True is the safe default.
FIX_POSITION = True

PEAK_ESTIMATE_JY = 0.001

# --- Fit region and noise ----------------------------------------------------
FIT_REGION_BEAM_MULTIPLE = 4.0
RMS_ANNULUS_INNER = '40arcsec'
RMS_ANNULUS_OUTER = '120arcsec'
RMS_ALGORITHM     = 'chauvenet'

# --- Zero level --------------------------------------------------------------
FIT_ZERO_LEVEL           = True
FIX_ZERO_LEVEL           = False
SEED_ZERO_LEVEL_FROM_MAP = True

# --- Spectral fit ------------------------------------------------------------
MIN_CHANNELS = 4          # epochs with fewer usable channels are not fitted
REF_FREQ_GHZ = None       # None -> geometric mean of the channels of that epoch
# The formal errors describe the map noise alone. Where the scatter about the
# power law exceeds it, chi2/dof > 1 indicates unmodelled systematics such as
# residual calibration or deconvolution errors, and the second uncertainty
# column rescales sigma_alpha by sqrt(chi2/dof). Both are reported; neither is
# imposed.
RESCALE_BY_CHI2 = True

OUTPUT_EPOCHS   = 'inband_alpha.txt'
OUTPUT_CHANNELS = 'inband_channel_flux.txt'

# ===========================================================================

_CASA_ROUTES = {}


def _resolve_casa(name, casac_factory=None):
    """Locate a CASA tool or task by name. Returns None if unavailable."""
    existing = globals().get(name)
    if existing is not None:
        _CASA_ROUTES[name] = 'CASA namespace'
        return existing
    for module_name in ('taskinit', 'casatasks', 'casatools', name):
        try:
            module = __import__(module_name, globals(), locals(), [name])
        except ImportError:
            continue
        obj = getattr(module, name, None)
        if obj is not None:
            _CASA_ROUTES[name] = '{}.{}'.format(module_name, name)
            return obj
    try:
        module = __import__('taskinit', globals(), locals(), [name + 'tool'])
        factory = getattr(module, name + 'tool', None)
        if factory is not None:
            _CASA_ROUTES[name] = 'taskinit.{}tool()'.format(name)
            return factory()
    except Exception:
        pass
    if casac_factory is not None:
        try:
            from casac import casac
            _CASA_ROUTES[name] = 'casac.{}()'.format(casac_factory)
            return getattr(casac, casac_factory)()
        except Exception:
            pass
    _CASA_ROUTES[name] = 'NOT FOUND'
    return None


me     = _resolve_casa('me', 'measures')
ia     = _resolve_casa('ia', 'image')
imhead = _resolve_casa('imhead')
imfit  = _resolve_casa('imfit')
imstat = _resolve_casa('imstat')

_MISSING = [n for n in ('me', 'ia', 'imhead', 'imfit', 'imstat')
            if globals()[n] is None]
CASA_READY = (len(_MISSING) == 0)
print("CASA components: " + ", ".join(
    "{}<-{}".format(n, _CASA_ROUTES.get(n, '?'))
    for n in ('me', 'ia', 'imhead', 'imfit', 'imstat')))
if not CASA_READY:
    print("=" * 70)
    print("ERROR: missing CASA components: " + ", ".join(_MISSING))
    print("       Run this inside CASA: execfile('getflux_MK_inband_alpha.py')")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Discovery and grouping
# ---------------------------------------------------------------------------

BLOCK_RE   = re.compile(r'img_(\d{8,12})_')
CHANNEL_RE = re.compile(r'-(\d{3,5})-image\.fits$')
MFS_RE     = re.compile(r'-MFS-image\.fits$')


def epoch_key(path):
    """Group images into epochs: block ID if present, else the directory."""
    base = os.path.basename(path)
    m = BLOCK_RE.search(base)
    if m:
        return m.group(1)
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.basename(parent).upper() == 'IMAGES':
        parent = os.path.dirname(parent)
    return os.path.basename(parent)


def channel_index(path):
    base = os.path.basename(path)
    if MFS_RE.search(base):
        return None
    m = CHANNEL_RE.search(base)
    return int(m.group(1)) if m else None


def _expand(root):
    hits = sorted(glob.glob(root))
    return [h for h in hits if os.path.isdir(h)] or (
        [root] if os.path.isdir(root) else [])


def _prune(dirnames):
    """Drop directories that cannot contain FITS images, in place."""
    keep = []
    for d in dirnames:
        if d in SKIP_DIR_NAMES or d.startswith('.'):
            continue
        if any(d.endswith(s) for s in SKIP_DIR_SUFFIXES):
            continue
        keep.append(d)
    dirnames[:] = keep


def _walk_dirs(base):
    """Directories at or below base, pruned and depth-limited, with progress."""
    base = os.path.abspath(base)
    yield base
    if not RECURSIVE:
        return
    base_depth = base.rstrip(os.sep).count(os.sep)
    scanned = 0
    for current, dirnames, _files in os.walk(base, topdown=True):
        _prune(dirnames)
        if MAX_DEPTH is not None:
            if current.rstrip(os.sep).count(os.sep) - base_depth >= MAX_DEPTH:
                dirnames[:] = []
        if current != base:
            yield current
        scanned += 1
        if scanned % 500 == 0:
            print("    ...scanned {} directories".format(scanned))


def find_images():
    """Return {epoch: [paths]}, channel images first, MFS last."""
    patterns = [CHANNEL_GLOB] + ([MFS_GLOB] if INCLUDE_MFS else [])
    print("Searching for images")
    print("  roots     : {}".format(', '.join(ROOTS)))
    print("  recursive : {}{}".format(
        RECURSIVE, "" if not RECURSIVE else
        " (max depth {}, measurement sets and CASA tables skipped)".format(
            MAX_DEPTH)))
    print("  patterns  : {}".format(', '.join(patterns)))
    if ROOTS == ['.']:
        print("  NOTE: ROOTS is still the default '.', so this searches the "
              "current directory.\n        Set ROOTS if the images are "
              "elsewhere, or the search may take a long time.")

    files, ndirs = [], 0
    for root in ROOTS:
        bases = _expand(root)
        if not bases:
            print("  WARNING: nothing matches root {!r}".format(root))
        for base in bases:
            for d in _walk_dirs(base):
                ndirs += 1
                for pat in patterns:
                    files.extend(glob.glob(os.path.join(d, pat)))
    print("  scanned {} director(ies), matched {} file(s)".format(
        ndirs, len(files)))

    files = [f for f in set(files) if '.pbcor.' not in os.path.basename(f)]
    if MASK_TOKEN:
        before = len(files)
        files = [f for f in files if MASK_TOKEN in os.path.basename(f)]
        print("  MASK_TOKEN {!r} kept {} of {}".format(
            MASK_TOKEN, len(files), before))

    epochs = {}
    for f in files:
        epochs.setdefault(epoch_key(f), []).append(f)
    for key in epochs:
        epochs[key].sort(key=lambda p: (channel_index(p) is None,
                                        channel_index(p) or 0))
    return epochs


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

_RAD2DEG = 180.0 / math.pi


def _angle_to_rad(value, unit):
    u = (unit or 'rad').strip().lower()
    if u.startswith('rad'):
        return float(value)
    if u.startswith('deg'):
        return float(value) / _RAD2DEG
    if u.startswith('arcmin') or u == "'":
        return float(value) / (_RAD2DEG * 60.0)
    if u.startswith('arcsec') or u == '"':
        return float(value) / (_RAD2DEG * 3600.0)
    raise ValueError("Unsupported angular unit: {!r}".format(unit))


def _rad_to_angle(value_rad, unit):
    u = (unit or 'rad').strip().lower()
    if u.startswith('rad'):
        return float(value_rad)
    if u.startswith('deg'):
        return float(value_rad) * _RAD2DEG
    if u.startswith('arcmin') or u == "'":
        return float(value_rad) * _RAD2DEG * 60.0
    if u.startswith('arcsec') or u == '"':
        return float(value_rad) * _RAD2DEG * 3600.0
    raise ValueError("Unsupported angular unit: {!r}".format(unit))


def _source_pixel(fits_file):
    """Pixel coordinates of the source, via the image's own coordinate system.

    The full reference world vector is retrieved and only its direction
    elements replaced. Passing a bare [ra, dec, 0, 0] to topixel fails, because
    0 is not a valid Stokes code and CASA rejects it with "Stokes value ?? is
    not contained in this StokesCoordinate".
    """
    csys = None
    try:
        ia.open(fits_file)
        try:
            csys = ia.coordsys()
            shape = ia.shape()
        finally:
            ia.close()

        try:
            found = csys.findcoordinate('direction')
            world_idx = [int(i) for i in found['world']]
            pixel_idx = [int(i) for i in found['pixel']]
        except Exception:
            world_idx = pixel_idx = [0, 1]

        try:
            units = list(csys.units())
            u_ra, u_dec = units[world_idx[0]], units[world_idx[1]]
        except Exception:
            u_ra = u_dec = 'rad'

        direction = me.direction('J2000', SOURCE_RA, SOURCE_DEC)
        ra_rad = float(direction['m0']['value'])
        dec_rad = float(direction['m1']['value'])

        world = list(csys.referencevalue(format='n')['numeric'])
        ref_ra_rad = _angle_to_rad(world[world_idx[0]], u_ra)
        while ra_rad - ref_ra_rad > math.pi:
            ra_rad -= 2.0 * math.pi
        while ref_ra_rad - ra_rad > math.pi:
            ra_rad += 2.0 * math.pi

        world[world_idx[0]] = _rad_to_angle(ra_rad, u_ra)
        world[world_idx[1]] = _rad_to_angle(dec_rad, u_dec)

        pix = csys.topixel(world)['numeric']
        x, y = float(pix[pixel_idx[0]]), float(pix[pixel_idx[1]])
        nx, ny = int(shape[pixel_idx[0]]), int(shape[pixel_idx[1]])
        if not (0 <= x < nx and 0 <= y < ny):
            return None
        return x, y
    except Exception:
        return None
    finally:
        if csys is not None:
            try:
                csys.done()
            except Exception:
                pass


def _frequency_GHz(fits_file, header):
    for i in (3, 4, 2):
        key, unit = 'crval{}'.format(i), 'cunit{}'.format(i)
        if key in header and str(header.get(unit, '')).lower().startswith('hz'):
            try:
                return float(header[key]) / 1e9
            except Exception:
                pass
    try:
        ia.open(fits_file)
        cs = ia.coordsys()
        ref = cs.referencevalue(format='n')['numeric']
        names = [n.lower() for n in cs.names()]
        cs.done()
        ia.close()
        for n, v in zip(names, ref):
            if 'freq' in n:
                return float(v) / 1e9
    except Exception:
        try:
            ia.close()
        except Exception:
            pass
    return float('nan')


def _mjd(header):
    try:
        date_obs = header['date-obs']
        casa_date = date_obs.replace('-', '/').replace('T', '/')
        return float(me.epoch('UTC', casa_date)['m0']['value'])
    except Exception:
        return float('nan')


def _measure_local_noise(fits_file):
    region = 'annulus[[{}, {}], [{}, {}]] coord=J2000'.format(
        SOURCE_RA, SOURCE_DEC, RMS_ANNULUS_INNER, RMS_ANNULUS_OUTER)
    try:
        stats = imstat(imagename=fits_file, region=region,
                       algorithm=RMS_ALGORITHM)
        if not stats or float(stats.get('npts', [0])[0]) < 100:
            return None, None
        background = None
        if len(stats.get('median', [])) > 0:
            cand = float(stats['median'][0])
            if np.isfinite(cand):
                background = cand
        for key, scale in (('medabsdevmed', 1.4826), ('sigma', 1.0), ('rms', 1.0)):
            if len(stats.get(key, [])) > 0:
                value = float(stats[key][0]) * scale
                if np.isfinite(value) and value > 0.0:
                    return value, background
        return None, background
    except Exception:
        return None, None


def fit_image(fits_file, estimates_filename):
    """Fit one channel or MFS image. Returns a dict, or None if unusable."""
    base = os.path.basename(fits_file)
    try:
        header = imhead(imagename=fits_file, mode='list')
        if not all(k in header for k in ('beammajor', 'beamminor', 'beampa')):
            return None
        bmaj = header['beammajor']['value']
        bmin = header['beamminor']['value']
        bpa = header['beampa']['value']
        nu = _frequency_GHz(fits_file, header)
        if not np.isfinite(nu):
            return None

        pixel = _source_pixel(fits_file)
        if pixel is None:
            return None

        with open(estimates_filename, 'w') as fh:
            fh.write("{:.6f},{:.3f},{:.3f}, {:.4f}arcsec, {:.4f}arcsec, "
                     "{:.3f}deg, {}\n".format(
                         PEAK_ESTIMATE_JY, pixel[0], pixel[1], bmaj, bmin, bpa,
                         'xyabp' if FIX_POSITION else 'abp'))

        rms, background = _measure_local_noise(fits_file)
        if not rms:
            return None   # a fully flagged channel gives no usable noise

        radius = (FIT_REGION_BEAM_MULTIPLE or 4.0) * max(bmaj, bmin)
        region = ('ellipse[[{}, {}], [{:.4f}arcsec, {:.4f}arcsec], 90deg] '
                  'coord=J2000'.format(SOURCE_RA, SOURCE_DEC, radius, radius))

        kwargs = dict(imagename=fits_file, region=region,
                      estimates=estimates_filename,
                      logfile='imfit_inband.log', append=True, rms=rms)
        if FIT_ZERO_LEVEL:
            kwargs['dooff'] = True
            kwargs['offset'] = (background if SEED_ZERO_LEVEL_FROM_MAP
                                and background is not None else 0.0)
            kwargs['fixoffset'] = bool(FIX_ZERO_LEVEL)

        result = imfit(**kwargs)
        if not (result and result.get('results', {}).get('component0')):
            return None
        comp = result['results']['component0']
        flux = float(np.ravel(comp['flux']['value'])[0])
        err = float(np.ravel(comp['flux']['error'])[0])
        if err < rms:
            err = rms
        if not (np.isfinite(flux) and np.isfinite(err) and err > 0):
            return None

        zero = float('nan')
        for key in ('zerooff', 'offset'):
            entry = result['results'].get(key)
            if isinstance(entry, dict) and 'value' in entry:
                try:
                    zero = float(np.ravel(entry['value'])[0])
                except Exception:
                    pass
                break

        return dict(file=base, channel=channel_index(fits_file), nu_GHz=nu,
                    bmaj=bmaj, rms=rms, zero=zero, flux=flux, err=err,
                    mjd=_mjd(header))
    except Exception:
        return None
    finally:
        if os.path.exists(estimates_filename):
            os.remove(estimates_filename)


# ---------------------------------------------------------------------------
# Power-law fit, in linear flux space
# ---------------------------------------------------------------------------

# Reason the last call to fit_power_law gave up, so that a failure is reported
# rather than silently swallowed.
FIT_FAILURE_REASON = ''


def fit_power_law(nu, S, sigma, nu0, max_iter=200):
    """Weighted non-linear least squares for S = S0 (nu/nu0)^alpha.

    Fitting in linear space, rather than fitting ln S, keeps marginal and
    negative channels usable and unbiased. The minimisation is
    Levenberg-Marquardt, which is stable when the amplitude is poorly
    determined.

    Note that np.linalg.lstsq is deliberately avoided: numpy changed the
    default of its rcond argument in 1.14, and passing rcond=None raises
    TypeError on the older numpy that ships with CASA 5.x. The normal
    equations are solved directly instead.
    """
    global FIT_FAILURE_REASON
    FIT_FAILURE_REASON = ''

    nu, S, sigma = (np.asarray(a, float) for a in (nu, S, sigma))
    good = np.isfinite(nu) & np.isfinite(S) & np.isfinite(sigma) & (sigma > 0)
    nu, S, sigma = nu[good], S[good], sigma[good]
    if len(nu) < 3:
        FIT_FAILURE_REASON = "only {} usable channel(s)".format(len(nu))
        return None
    if np.ptp(nu) <= 0:
        FIT_FAILURE_REASON = "all channels at the same frequency"
        return None

    x = np.log(nu / nu0)
    w = 1.0 / sigma

    def model_and_jac(p):
        m = p[0] * np.exp(p[1] * x)
        return m, np.column_stack([np.exp(p[1] * x), m * x])

    def chi2_of(p):
        m, _ = model_and_jac(p)
        return float(np.sum(((S - m) / sigma) ** 2))

    p = np.array([np.sum(S / sigma ** 2) / np.sum(1.0 / sigma ** 2), 0.0])
    if not np.isfinite(p[0]) or p[0] == 0:
        p[0] = float(np.mean(np.abs(S))) or 1e-6

    lam = 1e-3
    chi2 = chi2_of(p)
    for _ in range(max_iter):
        m, J = model_and_jac(p)
        A = J * w[:, None]
        b = (S - m) * w
        AtA = A.T.dot(A)
        Atb = A.T.dot(b)
        diag = np.diag(np.maximum(np.diag(AtA), 1e-300))
        stepped = done = False
        for _damp in range(40):
            try:
                dp = np.linalg.solve(AtA + lam * diag, Atb)
            except Exception:
                lam *= 10.0
                continue
            if not np.all(np.isfinite(dp)):
                lam *= 10.0
                continue
            trial = p + dp
            chi2_trial = chi2_of(trial)
            if np.isfinite(chi2_trial) and chi2_trial <= chi2:
                rel = np.max(np.abs(dp) / np.maximum(np.abs(p), 1e-30))
                p, chi2 = trial, chi2_trial
                lam = max(lam / 10.0, 1e-12)
                stepped = True
                done = rel < 1e-8
                break
            lam *= 10.0
        if done or not stepped:
            break

    m, J = model_and_jac(p)
    A = J * w[:, None]
    try:
        cov = np.linalg.inv(A.T.dot(A))
    except Exception:
        FIT_FAILURE_REASON = "singular covariance; the spectrum is not constrained"
        return None
    if not np.all(np.isfinite(cov)):
        FIT_FAILURE_REASON = "non-finite covariance"
        return None

    chi2 = float(np.sum(((S - m) / sigma) ** 2))
    dof = len(x) - 2
    red = chi2 / dof if dof > 0 else float('nan')
    sa = math.sqrt(abs(cov[1, 1]))
    out = dict(n=len(x), nu0=nu0, S0=p[0], S0_err=math.sqrt(abs(cov[0, 0])),
               alpha=p[1], alpha_err=sa,
               alpha_err_scaled=(sa * math.sqrt(red)
                                 if RESCALE_BY_CHI2 and np.isfinite(red)
                                 and red > 1 else sa),
               chi2=chi2, dof=dof, chi2_red=red)
    if p[0] <= 0:
        out['warning'] = "amplitude not positive; alpha is not meaningful"
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_epoch(key, files, estimates):
    chans, mfs = [], None
    for path in files:
        row = fit_image(path, estimates)
        if row is None:
            continue
        if row['channel'] is None:
            mfs = row
        else:
            chans.append(row)
    if len(chans) < MIN_CHANNELS:
        print("  {:<16} only {} usable channel(s), not fitted".format(
            key, len(chans)))
        return None, chans

    nu = [c['nu_GHz'] for c in chans]
    nu0 = REF_FREQ_GHZ or math.sqrt(min(nu) * max(nu))
    fit = fit_power_law(nu, [c['flux'] for c in chans],
                        [c['err'] for c in chans], nu0)
    mjd = mfs['mjd'] if mfs else np.nanmedian([c['mjd'] for c in chans])
    snr_mfs = (mfs['flux'] / mfs['rms']) if mfs and mfs['rms'] else float('nan')

    if fit is None:
        print("  {:<16} power-law fit failed ({})".format(
            key, FIT_FAILURE_REASON or 'reason not recorded'))
        return None, chans

    flag = 'good' if fit['alpha_err_scaled'] < 0.2 else (
        'fair' if fit['alpha_err_scaled'] < 0.5 else 'poor')
    print("  {:<16} MJD {:9.3f}  N={:2d}  nu0={:.3f} GHz  S0={:9.1f} uJy  "
          "alpha={:+.2f} +/- {:.2f}  chi2/dof={:5.2f}  [{}]".format(
              key, mjd, fit['n'], fit['nu0'], fit['S0'] * 1e6, fit['alpha'],
              fit['alpha_err_scaled'], fit['chi2_red'], flag))
    fit.update(epoch=key, mjd=mjd, snr_mfs=snr_mfs,
               mfs_flux=(mfs['flux'] if mfs else float('nan')),
               mfs_err=(mfs['err'] if mfs else float('nan')),
               nu_min=min(nu), nu_max=max(nu), flag=flag)
    return fit, chans


def write_outputs(epochs, channels):
    epochs = sorted(epochs, key=lambda e: (not np.isfinite(e['mjd']), e['mjd']))
    with open(OUTPUT_EPOCHS, 'w') as fh:
        fh.write("# In-band spectral index per epoch, S(nu) = S0 (nu/nu0)^alpha\n")
        fh.write("# Fitted in linear flux space over the channel images; all "
                 "channels retained, including marginal ones.\n")
        fh.write("# A flux-scale error is common to the channels of a band and "
                 "cancels in alpha, so no systematic term is applied.\n")
        fh.write("# Err_a is the formal error; Err_a_sc rescales it by "
                 "sqrt(chi2/dof) where that exceeds one.\n")
        fh.write("# {:<14} {:>10} {:>3} {:>8} {:>8} {:>11} {:>10} {:>8} {:>9} "
                 "{:>8} {:>7} {:>6}\n".format(
                     "Epoch", "MJD", "N", "nu_min", "nu_max", "S0(uJy)",
                     "S0err", "alpha", "Err_a", "Err_a_sc", "chi2/dof", "Flag"))
        for e in epochs:
            fh.write("{:<16} {:>10.4f} {:>3d} {:>8.4f} {:>8.4f} {:>11.2f} "
                     "{:>10.2f} {:>8.3f} {:>9.3f} {:>8.3f} {:>7.2f} {:>6}\n"
                     .format(e['epoch'], e['mjd'], e['n'], e['nu_min'],
                             e['nu_max'], e['S0'] * 1e6, e['S0_err'] * 1e6,
                             e['alpha'], e['alpha_err'], e['alpha_err_scaled'],
                             e['chi2_red'], e['flag']))
    print("Wrote {} ({} epoch(s))".format(OUTPUT_EPOCHS, len(epochs)))

    with open(OUTPUT_CHANNELS, 'w') as fh:
        fh.write("# Per-channel measurements behind {}\n".format(OUTPUT_EPOCHS))
        fh.write("# shape fixed to the beam; position {}; zero level {}\n".format(
            'fixed' if FIX_POSITION else 'free',
            'fitted (dooff=True)' if FIT_ZERO_LEVEL else 'NOT fitted'))
        fh.write("# {:<14} {:>4} {:>10} {:>8} {:>10} {:>10} {:>10} {:>10}\n"
                 .format("Epoch", "Chan", "Freq(GHz)", "Beam(\")", "rms(uJy)",
                         "Zero(uJy)", "S(uJy)", "err(uJy)"))
        for key, rows in sorted(channels.items()):
            for r in rows:
                fh.write("{:<16} {:>4d} {:>10.5f} {:>8.2f} {:>10.2f} {:>10.2f} "
                         "{:>10.2f} {:>10.2f}\n".format(
                             key, r['channel'], r['nu_GHz'], r['bmaj'],
                             r['rms'] * 1e6, r['zero'] * 1e6, r['flux'] * 1e6,
                             r['err'] * 1e6))
    print("Wrote {}".format(OUTPUT_CHANNELS))


def deduplicate(epochs_files):
    """Ensure each epoch has at most one image per channel.

    Two imaging runs of the same observation sitting in one directory would
    otherwise contribute two images with the same channel index, which the
    spectral fit would treat as independent measurements. Rather than
    double-count, the extras are reported and dropped, keeping the first
    filename in sorted order, and the glob should then be tightened.
    """
    for key in sorted(epochs_files):
        seen, keep, dropped = {}, [], []
        for path in sorted(epochs_files[key]):
            idx = channel_index(path)
            if idx in seen:
                dropped.append(path)
            else:
                seen[idx] = path
                keep.append(path)
        if dropped:
            print("WARNING: {} has more than one image per channel, so {} "
                  "file(s) were dropped.".format(key, len(dropped)))
            print("         Kept   : {}".format(
                os.path.basename(seen[channel_index(dropped[0])])))
            for d in dropped[:3]:
                print("         Dropped: {}".format(os.path.basename(d)))
            if len(dropped) > 3:
                print("         ...and {} more".format(len(dropped) - 3))
            print("         Tighten CHANNEL_GLOB if this is not what you want.")
        epochs_files[key] = sorted(
            keep, key=lambda p: (channel_index(p) is None,
                                 channel_index(p) or 0))
    return epochs_files


def run(epochs_files):
    """Fit every epoch in {epoch: [paths]} and write the outputs."""
    if not CASA_READY:
        print("CASA components missing; nothing done.")
        return []
    if not epochs_files:
        print("Nothing to process.")
        return []
    epochs_files = deduplicate(epochs_files)
    total = sum(len(v) for v in epochs_files.values())
    print("\nProcessing {} image(s) in {} epoch(s)\n".format(
        total, len(epochs_files)) + "-" * 100)

    estimates = os.path.abspath('estimates_inband.txt')
    results, channels = [], {}
    for key in sorted(epochs_files):
        fit, chans = process_epoch(key, epochs_files[key], estimates)
        if chans:
            channels[key] = chans
        if fit:
            results.append(fit)

    print("-" * 100)
    if not results:
        print("No epoch could be fitted.")
        return []
    good = [r for r in results if r['flag'] == 'good']
    print("{} epoch(s) fitted, {} with sigma_alpha < 0.2".format(
        len(results), len(good)))
    if good:
        a = np.array([r['alpha'] for r in good])
        e = np.array([r['alpha_err_scaled'] for r in good])
        w = 1.0 / e ** 2
        print("weighted mean alpha over the 'good' epochs: {:+.3f} +/- {:.3f} "
              "(scatter {:.3f})".format(np.sum(a * w) / np.sum(w),
                                        1.0 / math.sqrt(np.sum(w)), a.std()))
    write_outputs(results, channels)
    return results


def images_in_dir(directory):
    """Channel images, and the MFS image, for one observation directory.

    An IMAGES/ subdirectory is searched first and the directory itself second,
    which is the convention used by getflux_MK_Claude_DIR_or_IMG.py.
    """
    patterns = [CHANNEL_GLOB] + ([MFS_GLOB] if INCLUDE_MFS else [])
    for base in (os.path.join(directory, 'IMAGES'), directory):
        if not os.path.isdir(base):
            continue
        found = []
        for pat in patterns:
            found.extend(glob.glob(os.path.join(base, pat)))
        found = [f for f in set(found)
                 if '.pbcor.' not in os.path.basename(f)]
        if MASK_TOKEN:
            found = [f for f in found if MASK_TOKEN in os.path.basename(f)]
        if found:
            return sorted(found, key=lambda p: (channel_index(p) is None,
                                                channel_index(p) or 0))
    return []


def analyse_dirs(dirs):
    """One epoch per directory, as in the band-averaged script's (d) mode."""
    epochs_files = {}
    for d in dirs:
        d = d.rstrip(os.sep)
        if not os.path.isdir(d):
            print("Warning: {!r} is not a directory. Skipping.".format(d))
            continue
        files = images_in_dir(d)
        if not files:
            print("No channel images matching {!r} in {} "
                  "(checked IMAGES/ and the directory itself)".format(
                      CHANNEL_GLOB, d))
            continue
        # Label with the block ID where every image in the directory agrees on
        # one, since that is the more informative name, and with the directory
        # otherwise. Either way there is one row per directory.
        ids = set(m.group(1) for m in
                  (BLOCK_RE.search(os.path.basename(f)) for f in files) if m)
        key = ids.pop() if len(ids) == 1 else os.path.basename(d)
        print("{:<16} {} image(s) from {}".format(
            key, len(files), os.path.dirname(files[0]) or '.'))
        epochs_files[key] = files
    return run(epochs_files)


def analyse_files(files):
    """Group an explicit list of files into epochs and fit them."""
    files = [f for f in files if os.path.isfile(f)]
    epochs_files = {}
    for f in files:
        if '.pbcor.' in os.path.basename(f):
            continue
        if MASK_TOKEN and MASK_TOKEN not in os.path.basename(f):
            continue
        epochs_files.setdefault(epoch_key(f), []).append(f)
    for key in epochs_files:
        epochs_files[key].sort(key=lambda p: (channel_index(p) is None,
                                              channel_index(p) or 0))
    return run(epochs_files)


def main(*roots):
    """Automatic mode: walk ROOTS, or the roots given as arguments."""
    global ROOTS
    if roots:
        ROOTS = [r for r in roots]
    epochs_files = find_images()
    if not epochs_files:
        print("\nNo channel images found under: {}".format(", ".join(ROOTS)))
        print("Check, in order: that ROOTS points at the right place; that "
              "CHANNEL_GLOB\n({!r}) matches your filenames; and that "
              "MAX_DEPTH is deep enough.".format(CHANNEL_GLOB))
        return []
    return run(epochs_files)


def _ask(prompt):
    """Read a line, on Python 2 (CASA 5) or Python 3 (CASA 6) alike."""
    try:
        return raw_input(prompt)          # noqa: F821  (Python 2)
    except NameError:
        return input(prompt)


def _expand_items(text):
    """Expand a space-separated list of names or glob patterns."""
    out = []
    for item in text.split():
        hits = sorted(glob.glob(item))
        out.extend(hits if hits else [item])
    return sorted(set(out))


if __name__ == '__main__':
    if not CASA_READY:
        raise RuntimeError(
            "CASA components missing ({}). Run this script inside CASA.".format(
                ", ".join(_MISSING)))

    print("Source position: {} {}".format(SOURCE_RA, SOURCE_DEC))
    print("Position in the channel fits: {}".format(
        'fixed' if FIX_POSITION else 'free'))
    print("Zero level: {}".format(
        'fitted (dooff=True)' if FIT_ZERO_LEVEL else 'NOT fitted'))
    print("Channel glob: {}{}".format(
        CHANNEL_GLOB,
        "   MASK_TOKEN={!r}".format(MASK_TOKEN) if MASK_TOKEN else ""))

    mode = _ask("\nProcess (d)irectories, (f)iles directly, or "
                "(a)utomatic search? [d/f/a]: ").strip().lower()

    if mode == 'f':
        text = _ask("Enter file pattern (e.g., IMAGES/img*-0[0-9][0-9][0-9]"
                    "-image.fits) or space-separated filenames: ")
        items = _expand_items(text)
        if items:
            RESULTS = analyse_files(items)
        else:
            print("No files found matching the input: {}".format(text))
            RESULTS = []
    elif mode == 'a':
        RESULTS = main()
    else:
        text = _ask("Enter directory pattern (e.g., 156*) or space-separated "
                    "list of directories: ")
        items = [i for i in _expand_items(text) if os.path.isdir(i)]
        if items:
            print("\n{} director(ies) selected".format(len(items)))
            RESULTS = analyse_dirs(items)
        else:
            print("No directories found matching the input: {}".format(text))
            RESULTS = []

