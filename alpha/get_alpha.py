# -*- coding: utf-8 -*-
"""
In-band spectral index per epoch for the MeerKAT monitoring of GRS 1915+105.

A stripped-down relative of getflux_MK_channels.py. It fits the per-channel
images of each observation, then fits a single power law across the band, and
writes one row per epoch. There is no cross-band or joint fitting: the JetKAT
campaign is S-band (2.68-3.44 GHz in the channel images) throughout.

What alpha describes: the core fit is a beam-shaped Gaussian held at the core
position measured in that epoch's MFS image (row C of LOBE_POSITIONS_FILE, or
the catalogue position if the epoch has none), so in Epochs 1-4, where the
ejecta are blended with the core, alpha is that of the core and ejecta together,
and from Epoch 5 it is that of the core. With LOBE_POSITIONS_FILE set (see
below) the resolved lobes are fitted at the same time as extra fixed-position
components, which removes their flux from the core region and gives each lobe
its own spectral index.

Run inside CASA:

    CASA <1>: execfile('get_alpha.py')

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
observation into N channels raises the per-channel rms by sqrt(N). The lever arm
of the S-band channels (2.68-3.34 GHz, ln(nu_max/nu_min) = 0.22) is shorter than
at L-band, so for seven channels the uncertainty is roughly

    sigma_alpha ~ 5 / (S/N per channel) ~ 13 / (S/N of the band-averaged image)

so about 0.13 for a component detected at 100 sigma, 0.45 at 30 sigma and 1.3 at
10 sigma (this matches the formal errors of the core fits). For the lobes (MFS
S/N 24-43 in Epochs 6-7, 12-17 in Epoch 8) a single epoch gives sigma_alpha of
roughly 0.3-1, so the weighted mean over epochs printed for each lobe, about 0.25
per lobe, is the useful number. The output carries chi2/dof and a quality flag
to make filtering straightforward.

Lobe components
---------------
LOBE_POSITIONS_FILE lists, per scheduling block, the positions measured in the
MFS image (written by make_lobe_positions.py): the core (C), the resolved lobes
(N, S) and the reference sources (REF1, ...). Every component is held at its
listed position in the channel images. This matters: the images are offset from
the catalogue frame by 0.3-0.9 arcsec, and Epoch 4 by 1.6 arcsec, and a
beam-shaped Gaussian held at the wrong position loses flux, more so at high
frequency because the beam shrinks, which biases the flux low and alpha steep.
For an epoch with lobes listed, every channel image is fitted with the core
plus one beam-shaped Gaussian per lobe, and the fit region is enlarged to cover
them. The lobe spectra are written to OUTPUT_LOBES; the core
rows of OUTPUT_EPOCHS keep their format. Set LOBE_POSITIONS_FILE = None to
reproduce the core-only fits.

Reference sources
-----------------
REFERENCE_SOURCES are compact, constant field sources measured in every channel
image in the same way, each in its own fit region. Their per-channel fluxes go
to OUTPUT_REFERENCE; inband_systematics.py (run locally, no CASA) uses them to
measure the channel-to-channel calibration systematic and to test the core
spectra for curvature. Set REFERENCE_SOURCES = [] to skip them.

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
# --- JetKAT campaign layout (mode c, the default) ----------------------------
# One folder per epoch named <blockid>_S<n> directly under CAMPAIGN_ROOT, with
# the channel images in its IMAGES/ subfolder. Only S numbers in
# CAMPAIGN_EPOCHS are used; S13 and later (and names such as *_S1_old) are
# ignored because the number must be the whole suffix.
CAMPAIGN_ROOT   = '.'               # the GRS1915 folder on ilifu
CAMPAIGN_EPOCHS = range(1, 13)      # S1 to S12

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

# Fixing the centroid makes each channel a forced photometry: at low signal to
# noise a free centroid locks onto noise peaks and biases the flux high, so True
# is the safe default. The position held is the one measured in the epoch's MFS
# image (LOBE_POSITIONS_FILE); SOURCE_RA/SOURCE_DEC are the fallback and the
# centre of the fit region.
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
# Channels left out of the power-law fits (core and lobes); they are still measured
# and written to the channel files. Channel 0 (2.68 GHz) reads ~3% low in every
# source and epoch, a gain error shared with the reference source.
DROP_CHANNELS = (0,)
# A channel whose flux error exceeds this multiple of the epoch's median channel
# error is left out of the fits (a nearly fully flagged channel, e.g. Epoch 9
# channel 7 at 3.44 GHz with a 42 arcsec beam).
MAX_ERR_RATIO = 5.0
REF_FREQ_GHZ = None       # None -> geometric mean of the channels of that epoch
# The formal errors describe the map noise alone. Where the scatter about the
# power law exceeds it, chi2/dof > 1 indicates unmodelled systematics such as
# residual calibration or deconvolution errors, and the second uncertainty
# column rescales sigma_alpha by sqrt(chi2/dof). Both are reported; neither is
# imposed.
RESCALE_BY_CHI2 = True

OUTPUT_EPOCHS   = 'inband_alpha.txt'
OUTPUT_CHANNELS = 'inband_channel_flux.txt'

# --- Component positions -----------------------------------------------------
# Per-epoch positions from make_lobe_positions.py: core (C), lobes (N, S) and
# reference sources. Epochs with lobes listed are fitted with the core and the
# lobes together; None (or a missing file) fits the core alone at SOURCE_RA/DEC
# and the reference sources at their REFERENCE_SOURCES positions.
LOBE_POSITIONS_FILE = 'lobe_positions.txt'
OUTPUT_LOBES        = 'inband_alpha_lobes.txt'
# Lobe spectra with sigma_alpha (scaled) below this enter the weighted mean.
LOBE_MEAN_MAX_ERR   = 1.0

# --- Reference sources (channel-systematics check) ---------------------------
# Compact, constant field sources measured in every channel image the same way
# as the core (beam-shaped Gaussian at a fixed position, zero level fitted), in
# their own fit region. Their scatter about a power law measures the
# channel-to-channel calibration systematic that GRS 1915+105 shares. The
# channel images are not primary-beam corrected, so the raw fluxes written here
# still carry the frequency-dependent beam attenuation; inband_systematics.py
# corrects for it using the separation from the pointing centre.
#   REF1: 6.5 mJy, unresolved and constant to 0.7% rms across the 12 MFS images,
#         6.70 arcmin from GRS 1915+105 (primary-beam attenuation ~0.89 at 3.06 GHz).
REFERENCE_SOURCES = [
    ('REF1', '19h14m44.703s', '+10d57m56.151s'),
]
OUTPUT_REFERENCE = 'inband_reference_flux.txt'

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
    print("       Run this inside CASA: execfile('get_alpha.py')")
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


def _source_pixel(fits_file, ra=SOURCE_RA, dec=SOURCE_DEC):
    """Pixel coordinates of a position (default: the source), via the image's own coordinate system.

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

        direction = me.direction('J2000', ra, dec)
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


def _measure_local_noise(fits_file, ra=SOURCE_RA, dec=SOURCE_DEC):
    region = 'annulus[[{}, {}], [{}, {}]] coord=J2000'.format(
        ra, dec, RMS_ANNULUS_INNER, RMS_ANNULUS_OUTER)
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


LOBES = {}   # {block ID: [(label, ra, dec), ...]}, filled by run(); all components


def _load_lobe_positions():
    """Read LOBE_POSITIONS_FILE into {block ID: [(label, ra, dec), ...]}."""
    if not LOBE_POSITIONS_FILE:
        return {}
    if not os.path.isfile(LOBE_POSITIONS_FILE):
        print("NOTE: {} not found, so the core is fitted alone.".format(
            LOBE_POSITIONS_FILE))
        return {}
    lobes = {}
    with open(LOBE_POSITIONS_FILE) as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0].startswith('#') or len(parts) < 4:
                continue
            lobes.setdefault(parts[0], []).append(tuple(parts[1:4]))
    print("Component positions for {} epoch(s) from {}".format(
        len(lobes), LOBE_POSITIONS_FILE))
    return lobes


def _positions_for(key):
    """Components listed for an epoch key, which starts with the block ID."""
    m = re.match(r'(\d{8,12})', key)
    return LOBES.get(m.group(1), []) if m else []


def _lobes_for(key):
    """Lobes listed for an epoch (every component that is not the core or a reference)."""
    refs = set(label for label, _ra, _dec in REFERENCE_SOURCES)
    return [row for row in _positions_for(key) if row[0] != 'C' and row[0] not in refs]


def _core_for(key):
    """Core position measured in the epoch's MFS image, or the catalogue position."""
    for label, ra, dec in _positions_for(key):
        if label == 'C':
            return ra, dec, True
    return SOURCE_RA, SOURCE_DEC, False


def _references_for(key):
    """REFERENCE_SOURCES with each position replaced by the epoch's measured one where listed."""
    listed = dict((label, (ra, dec)) for label, ra, dec in _positions_for(key))
    return [(label,) + listed.get(label, (ra, dec)) for label, ra, dec in REFERENCE_SOURCES]


def _separation_arcsec(ra, dec):
    """Angular separation of (ra, dec) from the source position, in arcsec."""
    try:
        sep = me.separation(me.direction('J2000', ra, dec),
                            me.direction('J2000', SOURCE_RA, SOURCE_DEC))
        value = float(sep['value'])
        unit = sep.get('unit', 'deg')
        return _rad_to_angle(_angle_to_rad(value, unit), 'arcsec')
    except Exception:
        return 0.0


def _fit_reference(fits_file, estimates_filename, ra, dec, bmaj, bmin, bpa):
    """
    Forced photometry of one reference source: beam-shaped Gaussian at (ra, dec)
    with the zero level fitted, in a region of FIT_REGION_BEAM_MULTIPLE beams
    around it and with the noise from an annulus around it.
    Returns (flux, err, rms), or None if it cannot be measured.
    """
    pixel = _source_pixel(fits_file, ra, dec)
    if pixel is None:
        return None
    rms, background = _measure_local_noise(fits_file, ra, dec)
    if not rms:
        return None
    try:
        with open(estimates_filename, 'w') as fh:
            fh.write("{:.6f},{:.3f},{:.3f}, {:.4f}arcsec, {:.4f}arcsec, "
                     "{:.3f}deg, xyabp\n".format(PEAK_ESTIMATE_JY, pixel[0],
                                                 pixel[1], bmaj, bmin, bpa))
        radius = (FIT_REGION_BEAM_MULTIPLE or 4.0) * max(bmaj, bmin)
        region = ('ellipse[[{}, {}], [{:.4f}arcsec, {:.4f}arcsec], 90deg] '
                  'coord=J2000'.format(ra, dec, radius, radius))
        kwargs = dict(imagename=fits_file, region=region,
                      estimates=estimates_filename,
                      logfile='imfit_inband.log', append=True, rms=rms)
        if FIT_ZERO_LEVEL:
            kwargs['dooff'] = True
            kwargs['offset'] = (background if SEED_ZERO_LEVEL_FROM_MAP
                                and background is not None else 0.0)
            kwargs['fixoffset'] = bool(FIX_ZERO_LEVEL)
        result = imfit(**kwargs)
        comp = (result or {}).get('results', {}).get('component0')
        if not comp:
            return None
        flux = float(np.ravel(comp['flux']['value'])[0])
        err = max(float(np.ravel(comp['flux']['error'])[0]), rms)
        if not (np.isfinite(flux) and np.isfinite(err)):
            return None
        return flux, err, rms
    except Exception:
        return None
    finally:
        if os.path.exists(estimates_filename):
            os.remove(estimates_filename)


def fit_image(fits_file, estimates_filename, lobes=(), core=(SOURCE_RA, SOURCE_DEC),
              references=None):
    """
    Fit one channel or MFS image. Returns a dict, or None if unusable.

    core = (ra, dec) is where the core is held; references = [(label, ra, dec)]
    the reference sources (default REFERENCE_SOURCES).
    lobes = [(label, ra, dec), ...] are fitted together with the core as
    beam-shaped Gaussians held at the given positions; their fluxes are returned
    under 'lobes' as {label: (flux, err)}. A lobe that cannot be placed in the
    image is left out of that image's fit.
    """
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

        pixel = _source_pixel(fits_file, core[0], core[1])
        if pixel is None:
            return None

        # Core first (component0), then the lobes that can be placed, all with
        # the shape held at the beam; the lobes are always held at their positions.
        placed = []
        for label, ra, dec in lobes:
            lobe_pixel = _source_pixel(fits_file, ra, dec)
            if lobe_pixel is None:
                print("  {}: lobe {} could not be placed; fitted without it".format(
                    base, label))
                continue
            placed.append((label, lobe_pixel, _separation_arcsec(ra, dec)))

        line = ("{:.6f},{:.3f},{:.3f}, {:.4f}arcsec, {:.4f}arcsec, "
                "{:.3f}deg, {}\n")
        with open(estimates_filename, 'w') as fh:
            fh.write(line.format(PEAK_ESTIMATE_JY, pixel[0], pixel[1], bmaj,
                                 bmin, bpa, 'xyabp' if FIX_POSITION else 'abp'))
            for _label, lobe_pixel, _sep in placed:
                fh.write(line.format(PEAK_ESTIMATE_JY * 0.1, lobe_pixel[0],
                                     lobe_pixel[1], bmaj, bmin, bpa, 'xyabp'))

        rms, background = _measure_local_noise(fits_file)
        if not rms:
            return None   # a fully flagged channel gives no usable noise

        # The region must hold every component plus background around it
        radius = (FIT_REGION_BEAM_MULTIPLE or 4.0) * max(bmaj, bmin)
        if placed:
            radius = max(radius, max(sep for _l, _p, sep in placed)
                         + 2.0 * max(bmaj, bmin))
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

        lobe_fluxes = {}
        for i, (label, _p, _s) in enumerate(placed, start=1):
            entry = result['results'].get('component{}'.format(i))
            if not entry:
                continue
            l_flux = float(np.ravel(entry['flux']['value'])[0])
            l_err = max(float(np.ravel(entry['flux']['error'])[0]), rms)
            if np.isfinite(l_flux) and np.isfinite(l_err):
                lobe_fluxes[label] = (l_flux, l_err)

        ref_fluxes = {}
        for label, ra, dec in (REFERENCE_SOURCES if references is None else references):
            ref = _fit_reference(fits_file, estimates_filename + '.ref', ra,
                                 dec, bmaj, bmin, bpa)
            if ref is not None:
                ref_fluxes[label] = ref + (_separation_arcsec(ra, dec) / 60.0,)

        return dict(file=base, channel=channel_index(fits_file), nu_GHz=nu,
                    bmaj=bmaj, rms=rms, zero=zero, flux=flux, err=err,
                    mjd=_mjd(header), lobes=lobe_fluxes, refs=ref_fluxes)
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

def fit_lobes(key, chans, nu0, mjd):
    """Power-law fits to the lobe fluxes of one epoch, one dict per lobe."""
    out = []
    labels = sorted(set(l for c in chans for l in c.get('lobes', {})))
    for label in labels:
        rows = [c for c in chans if label in c['lobes']]
        if len(rows) < MIN_CHANNELS:
            print("    lobe {:<3} only {} usable channel(s), not fitted".format(
                label, len(rows)))
            continue
        nu = [c['nu_GHz'] for c in rows]
        fit = fit_power_law(nu, [c['lobes'][label][0] for c in rows],
                            [c['lobes'][label][1] for c in rows], nu0)
        if fit is None:
            print("    lobe {:<3} power-law fit failed ({})".format(
                label, FIT_FAILURE_REASON or 'reason not recorded'))
            continue
        fit.update(epoch=key, label=label, mjd=mjd, nu_min=min(nu),
                   nu_max=max(nu),
                   channels=[(c['channel'], c['nu_GHz']) + c['lobes'][label]
                             for c in rows])
        print("    lobe {:<3} S0={:9.1f} uJy  alpha={:+.2f} +/- {:.2f}  "
              "chi2/dof={:5.2f}".format(label, fit['S0'] * 1e6, fit['alpha'],
                                        fit['alpha_err_scaled'], fit['chi2_red']))
        out.append(fit)
    return out


def _fit_channels(key, chans):
    """Channels that enter the power-law fits: DROP_CHANNELS and error outliers removed."""
    kept = [c for c in chans if c['channel'] not in DROP_CHANNELS]
    if kept and MAX_ERR_RATIO:
        median_err = float(np.median([c['err'] for c in kept]))
        bad = [c for c in kept if c['err'] > MAX_ERR_RATIO * median_err]
        for c in bad:
            print("  {:<16} channel {} left out of the fits (error {:.1f} uJy, {:.0f}x the median)".format(
                key, c['channel'], c['err'] * 1e6, c['err'] / median_err))
        kept = [c for c in kept if c not in bad]
    return kept


def process_epoch(key, files, estimates):
    chans, mfs = [], None
    lobes = _lobes_for(key)
    core_ra, core_dec, measured = _core_for(key)
    if not measured:
        print("  {:<16} WARNING: no measured core position listed, so the core is held "
              "at the catalogue position".format(key))
    for path in files:
        row = fit_image(path, estimates, lobes, (core_ra, core_dec), _references_for(key))
        if row is None:
            continue
        if row['channel'] is None:
            mfs = row
        else:
            chans.append(row)
    used = _fit_channels(key, chans)
    if len(used) < MIN_CHANNELS:
        print("  {:<16} only {} usable channel(s), not fitted".format(
            key, len(used)))
        return None, chans, []

    nu = [c['nu_GHz'] for c in used]
    nu0 = REF_FREQ_GHZ or math.sqrt(min(nu) * max(nu))
    fit = fit_power_law(nu, [c['flux'] for c in used],
                        [c['err'] for c in used], nu0)
    mjd = mfs['mjd'] if mfs else np.nanmedian([c['mjd'] for c in chans])
    snr_mfs = (mfs['flux'] / mfs['rms']) if mfs and mfs['rms'] else float('nan')

    if fit is None:
        print("  {:<16} power-law fit failed ({})".format(
            key, FIT_FAILURE_REASON or 'reason not recorded'))
        return None, chans, fit_lobes(key, used, nu0, mjd)

    flag = 'good' if fit['alpha_err_scaled'] < 0.2 else (
        'fair' if fit['alpha_err_scaled'] < 0.5 else 'poor')
    print("  {:<16} MJD {:9.3f}  N={:2d}  nu0={:.3f} GHz  S0={:9.1f} uJy  "
          "alpha={:+.2f} +/- {:.2f}  chi2/dof={:5.2f}  [{}]{}".format(
              key, mjd, fit['n'], fit['nu0'], fit['S0'] * 1e6, fit['alpha'],
              fit['alpha_err_scaled'], fit['chi2_red'], flag,
              "  (core + {} lobe(s))".format(len(lobes)) if lobes else ""))
    fit.update(epoch=key, mjd=mjd, snr_mfs=snr_mfs,
               mfs_flux=(mfs['flux'] if mfs else float('nan')),
               mfs_err=(mfs['err'] if mfs else float('nan')),
               nu_min=min(nu), nu_max=max(nu), flag=flag)
    return fit, chans, fit_lobes(key, used, nu0, mjd)


def write_lobe_outputs(lobe_fits):
    """Per-lobe spectra, a weighted mean per lobe, and the channel fluxes behind them."""
    lobe_fits = sorted(lobe_fits, key=lambda e: (e['mjd'], e['label']))
    with open(OUTPUT_LOBES, 'w') as fh:
        fh.write("# In-band spectral index of the resolved lobes, S(nu) = S0 (nu/nu0)^alpha\n")
        fh.write("# Core and lobes fitted together in every channel image; lobes held at the positions in {}\n"
                 .format(LOBE_POSITIONS_FILE))
        fh.write("# N = north (receding), S = south (approaching). Err_a_sc rescales Err_a by sqrt(chi2/dof) "
                 "where that exceeds one.\n")
        fh.write("# {:<14} {:>5} {:>10} {:>3} {:>10} {:>9} {:>8} {:>9} {:>8} {:>7}\n".format(
            "Epoch", "Lobe", "MJD", "N", "S0(uJy)", "S0err", "alpha", "Err_a",
            "Err_a_sc", "chi2/dof"))
        for e in lobe_fits:
            fh.write("{:<16} {:>5} {:>10.4f} {:>3d} {:>10.2f} {:>9.2f} {:>8.3f} {:>9.3f} "
                     "{:>8.3f} {:>7.2f}\n".format(
                         e['epoch'], e['label'], e['mjd'], e['n'], e['S0'] * 1e6,
                         e['S0_err'] * 1e6, e['alpha'], e['alpha_err'],
                         e['alpha_err_scaled'], e['chi2_red']))

        fh.write("\n# Weighted mean alpha per lobe over epochs with Err_a_sc < {:.2f}\n".format(
            LOBE_MEAN_MAX_ERR))
        for label in sorted(set(e['label'] for e in lobe_fits)):
            use = [e for e in lobe_fits if e['label'] == label
                   and e['alpha_err_scaled'] < LOBE_MEAN_MAX_ERR]
            if not use:
                continue
            a = np.array([e['alpha'] for e in use])
            w = 1.0 / np.array([e['alpha_err_scaled'] for e in use]) ** 2
            mean, err = np.sum(a * w) / np.sum(w), 1.0 / math.sqrt(np.sum(w))
            chi2_red = (np.sum(w * (a - mean) ** 2) / (len(a) - 1)
                        if len(a) > 1 else float('nan'))
            text = ("lobe {}: alpha = {:+.3f} +/- {:.3f} from {} epoch(s) ({}); "
                    "chi2/dof about the mean = {:.2f}".format(
                        label, mean, err, len(use),
                        ", ".join(e['epoch'] for e in use), chi2_red))
            fh.write("# " + text + "\n")
            print(text)

        fh.write("\n# Channel fluxes\n# {:<14} {:>5} {:>4} {:>10} {:>10} {:>10}\n".format(
            "Epoch", "Lobe", "Chan", "Freq(GHz)", "S(uJy)", "err(uJy)"))
        for e in lobe_fits:
            for chan, nu, flux, err in e['channels']:
                fh.write("{:<16} {:>5} {:>4d} {:>10.5f} {:>10.2f} {:>10.2f}\n".format(
                    e['epoch'], e['label'], chan, nu, flux * 1e6, err * 1e6))
    print("Wrote {} ({} lobe spectra)".format(OUTPUT_LOBES, len(lobe_fits)))


def write_reference_outputs(channels):
    """Per-channel fluxes of the reference sources, for inband_systematics.py."""
    with open(OUTPUT_REFERENCE, 'w') as fh:
        fh.write("# Per-channel forced photometry of the reference sources (REFERENCE_SOURCES)\n")
        fh.write("# Fitted like the core: beam-shaped Gaussian held at the position measured in the epoch's\n"
                 "# MFS image ({}) where listed, zero level fitted.\n".format(LOBE_POSITIONS_FILE))
        fh.write("# NOT primary-beam corrected; Sep is the separation from GRS 1915+105 in arcmin.\n")
        fh.write("# {:<14} {:>5} {:>4} {:>10} {:>10} {:>10} {:>10} {:>7}\n".format(
            "Epoch", "Ref", "Chan", "Freq(GHz)", "S(uJy)", "err(uJy)", "rms(uJy)", "Sep(')"))
        n = 0
        for key, rows in sorted(channels.items()):
            for r in rows:
                for label, (flux, err, rms, sep) in sorted(r.get('refs', {}).items()):
                    fh.write("{:<16} {:>5} {:>4d} {:>10.5f} {:>10.2f} {:>10.2f} {:>10.2f} {:>7.3f}\n"
                             .format(key, label, r['channel'], r['nu_GHz'], flux * 1e6,
                                     err * 1e6, rms * 1e6, sep))
                    n += 1
    print("Wrote {} ({} reference measurements)".format(OUTPUT_REFERENCE, n))


def write_outputs(epochs, channels):
    epochs = sorted(epochs, key=lambda e: (not np.isfinite(e['mjd']), e['mjd']))
    with open(OUTPUT_EPOCHS, 'w') as fh:
        fh.write("# In-band spectral index per epoch, S(nu) = S0 (nu/nu0)^alpha\n")
        if LOBES:
            fh.write("# Core held at the position measured in each epoch's MFS image (row C of {}); epochs\n"
                     "# with lobes listed there were fitted with the core and the lobes together.\n"
                     .format(LOBE_POSITIONS_FILE))
        else:
            fh.write("# Core held at the catalogue position {} {}.\n".format(SOURCE_RA, SOURCE_DEC))
        fh.write("# Fitted in linear flux space over the channel images; channels {} left out, and any channel "
                 "whose error exceeds {}x the epoch median.\n".format(list(DROP_CHANNELS), MAX_ERR_RATIO))
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
    global LOBES
    LOBES = _load_lobe_positions()
    total = sum(len(v) for v in epochs_files.values())
    print("\nProcessing {} image(s) in {} epoch(s)\n".format(
        total, len(epochs_files)) + "-" * 100)

    estimates = os.path.abspath('estimates_inband.txt')
    results, channels, lobe_results = [], {}, []
    for key in sorted(epochs_files):
        fit, chans, lobe_fits = process_epoch(key, epochs_files[key], estimates)
        if chans:
            channels[key] = chans
        if fit:
            results.append(fit)
        lobe_results.extend(lobe_fits)

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
    if lobe_results:
        write_lobe_outputs(lobe_results)
    if REFERENCE_SOURCES:
        write_reference_outputs(channels)
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


CAMPAIGN_DIR_RE = re.compile(r'_S(\d+)$', re.IGNORECASE)


def campaign_dirs(root=None):
    """
    The <blockid>_S<n> folders under root with n in CAMPAIGN_EPOCHS, in epoch
    order. Folders for other epochs are listed as skipped; a missing epoch or
    two folders for one epoch are reported, and a duplicated epoch is left out.
    """
    root = root or CAMPAIGN_ROOT
    found, skipped = {}, []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        m = CAMPAIGN_DIR_RE.search(name)
        if not (m and os.path.isdir(path)):
            continue
        n = int(m.group(1))
        if n not in CAMPAIGN_EPOCHS:
            skipped.append(name)
            continue
        found.setdefault(n, []).append(path)

    dirs = []
    for n in CAMPAIGN_EPOCHS:
        paths = found.get(n, [])
        if not paths:
            print("WARNING: no folder for epoch S{} under {}".format(n, root))
        elif len(paths) > 1:
            print("WARNING: {} folders for epoch S{} ({}); epoch skipped, "
                  "rename or move the extra one(s)".format(
                      len(paths), n, ", ".join(os.path.basename(p) for p in paths)))
        else:
            dirs.append(paths[0])
    print("Campaign folders ({} of {}): {}".format(
        len(dirs), len(CAMPAIGN_EPOCHS),
        ", ".join(os.path.basename(d) for d in dirs)))
    if skipped:
        print("Ignored (outside S{}-S{}): {}".format(
            min(CAMPAIGN_EPOCHS), max(CAMPAIGN_EPOCHS), ", ".join(skipped)))
    return dirs


def analyse_campaign(root=None):
    """Fit every epoch folder S1-S12 (CAMPAIGN_EPOCHS) under root in one run."""
    dirs = campaign_dirs(root)
    if not dirs:
        print("No campaign folders found under {}".format(root or CAMPAIGN_ROOT))
        return []
    return analyse_dirs(dirs)


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

    mode = _ask("\nProcess the (c)ampaign folders S{}-S{} under {!r}, "
                "(d)irectories, (f)iles directly, or (a)utomatic search? "
                "[c/d/f/a, Enter = c]: ".format(
                    min(CAMPAIGN_EPOCHS), max(CAMPAIGN_EPOCHS),
                    CAMPAIGN_ROOT)).strip().lower()

    if mode in ('', 'c'):
        RESULTS = analyse_campaign()
    elif mode == 'f':
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

