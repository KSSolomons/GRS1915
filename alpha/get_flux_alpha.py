# -*- coding: utf-8 -*-
"""
THIS SHOULD BE KEPT SAFE SOMEWHERE
This script is intended to be run in a CASA environment (v5.x).
It robustly calculates the flux of a point source in a series of radio maps
by iterating through observation directories or a list of FITS files directly.

Key Features:
- Prompts the user to choose between processing directories or FITS files directly.
- Uses only CASA-native tools (no astropy dependency).
- Fixes the Gaussian shape to the restoring beam from the image header.
- Seeds the fit at a fixed sky position (converted to pixels per-image via the
  image coordinate system), rather than a hardcoded pixel pair.
- Measures the local image RMS per epoch and uses it (a) inside imfit and
  (b) as a floor on the reported flux uncertainty.
- Adds a flux-scale systematic in quadrature.
- Avoids using os.chdir() for more stable file path management.
- Uses glob to reliably find the correct FITS file in each directory.
- Implements try-except blocks to gracefully handle missing files or failed fits.
- Uses the CASA 'me' tool for accurate conversion of the observation date to MJD.

Usage:
1.  Run the script within CASA using: execfile('your_script_name.py')
2.  When prompted, choose (d) to process directories or (f) to process FITS files directly.
3.  Enter a pattern (e.g., 156* or img*pcalmask-MFS-image.fits) or a space-separated list.
4.  Check the USER-CONFIGURABLE CONSTANTS block below before running.
"""

import os
import glob
import math
import numpy as np

# ===========================================================================
# USER-CONFIGURABLE CONSTANTS
# ===========================================================================

# --- Source position (GRS 1915+105) -----------------------------------------
# J2000 position adopted for the MeerKAT data.
# Agrees with the VLBA parallax position of Reid et al. (2014),
# 19:15:11.5473 +10:56:44.704, to ~0.25 arcsec -- far inside the L-band beam.
SOURCE_RA  = '19h15m11.558s'
SOURCE_DEC = '+10d56m44.905s'

# If True, the fit position is held fixed at SOURCE_RA/SOURCE_DEC ('xyabp').
# If False, the position is only used as the starting estimate ('abp').
# Fixing the position is the safer choice at low S/N; leaving it free is
# standard for a bright, well-detected source.
FIX_POSITION = False

# --- Initial peak estimate, in image brightness units (normally Jy/beam) ----
# Used only as a fallback: with PEAK_ESTIMATE_FROM_IMAGE the starting peak is
# taken from the brightest pixel in the fit region instead, which matters
# because this dataset spans four orders of magnitude in flux and a single
# fixed value cannot suit both ends (see the note in _fit_single_image).
PEAK_ESTIMATE_JY = 0.001
PEAK_ESTIMATE_FROM_IMAGE = True

# If a fit with the position free fails, retry once with the position held at
# the source coordinates. A runaway centroid is the usual failure mode, and the
# retry is reported in the log so it is never silent.
RETRY_WITH_FIXED_POSITION = True

# --- Fit region --------------------------------------------------------------
# Built from SOURCE_RA/SOURCE_DEC so the region centre and the fit estimate
# can never drift apart. Hardcode the centre here if you need them decoupled.
FIT_ELLIPSE_SEMIMAJOR = '16.2621arcsec'
FIT_ELLIPSE_SEMIMINOR = '16.0657arcsec'
FIT_ELLIPSE_PA        = '90deg'

FIT_REGION = 'ellipse[[{}, {}], [{}, {}], {}] coord=J2000'.format(
    SOURCE_RA, SOURCE_DEC,
    FIT_ELLIPSE_SEMIMAJOR, FIT_ELLIPSE_SEMIMINOR, FIT_ELLIPSE_PA)

# --- Local RMS measurement ---------------------------------------------------
MEASURE_RMS       = True
RMS_ANNULUS_INNER = '40arcsec'    # must comfortably exclude the source + sidelobes
RMS_ANNULUS_OUTER = '120arcsec'   # keep small enough that the noise is "local"
RMS_ALGORITHM     = 'chauvenet'   # robust outlier rejection; 'classic' to disable
RMS_TO_IMFIT      = True          # pass the measured RMS to imfit's rms argument
APPLY_RMS_FLOOR   = True          # sigma_flux >= local RMS

# --- Zero level (background) -------------------------------------------------
# imfit's default is dooff=False, i.e. no additive constant in the model. Any
# pedestal in the data must then be absorbed by the Gaussian itself, and for a
# shape-fixed Gaussian of unit peak g fitted to d = S g + b the least-squares
# amplitude is
#       A = sum(g d) / sum(g^2) = S + b [Int(g)/Int(g^2)] = S + 2 b
# because Int(g) = pi th^2 / (4 ln2) and Int(g^2) = pi th^2 / (8 ln2). A uniform
# background therefore inflates the fitted flux by TWICE its value, not once.
# On the July 2026 DDT images this amounted to 25 to 40 per cent, so fitting the
# zero level is not optional.
FIT_ZERO_LEVEL           = True   # imfit dooff
FIX_ZERO_LEVEL           = False  # imfit fixoffset; True holds it at the seed
SEED_ZERO_LEVEL_FROM_MAP = True   # seed with the annulus median rather than 0.0

# --- Fit-region size ---------------------------------------------------------
# With the zero level free, the fit needs enough background-dominated pixels to
# constrain it. Setting this to a number rebuilds the region per image as that
# multiple of the beam FWHM, which keeps the constraint uniform across bands;
# set to None to use the fixed FIT_ELLIPSE_* values above. At the fixed 16.3
# arcsec radius the L-band region reached only 2.1 beam FWHM, against 4.0 at
# S-band, so the zero level was markedly better constrained at S-band.
FIT_REGION_BEAM_MULTIPLE = 4.0

# --- Flux-scale systematic (fractional), added in quadrature -----------------
# MeerKAT L-band is typically quoted at a few per cent. Note that this term is
# common to every epoch, hence it does not average down when epochs are
# combined and should not be folded into inverse-variance weights. The Letter
# currently adopts 10 per cent; reconcile before submission.
FLUX_SCALE_FRAC = 0.05

OUTPUT_FILENAME = 'point_source_flux.txt'

# ===========================================================================

# --- CASA tool/task resolution ---------------------------------------------
# A single "from taskinit import me, ia, qa" is fragile: if any one name is
# absent the whole import fails and every component silently becomes unusable.
# Each component is therefore resolved independently, trying in order:
#   1. the interactive CASA namespace (present when run via execfile)
#   2. the 'taskinit' module / the same-named task module
#   3. the 'taskinit' <name>tool factory, e.g. iatool()
#   4. the casac factory, e.g. casac.image()

def _resolve_casa(name, casac_factory=None):
    """Locate a CASA tool or task by name. Returns None if unavailable."""
    # 1. Already defined in the enclosing namespace.
    existing = globals().get(name)
    if existing is not None:
        _CASA_ROUTES[name] = 'CASA namespace'
        return existing

    # 2. taskinit, then a task module of the same name.
    for module_name in ('taskinit', name):
        try:
            module = __import__(module_name, globals(), locals(), [name])
        except ImportError:
            continue
        obj = getattr(module, name, None)
        if obj is not None:
            _CASA_ROUTES[name] = '{}.{}'.format(module_name, name)
            return obj

    # 3. taskinit tool factory, e.g. iatool().
    try:
        module = __import__('taskinit', globals(), locals(), [name + 'tool'])
        factory = getattr(module, name + 'tool', None)
        if factory is not None:
            obj = factory()
            _CASA_ROUTES[name] = 'taskinit.{}tool()'.format(name)
            return obj
    except Exception:
        pass

    # 4. casac factory.
    if casac_factory is not None:
        try:
            from casac import casac
            obj = getattr(casac, casac_factory)()
            _CASA_ROUTES[name] = 'casac.{}()'.format(casac_factory)
            return obj
        except Exception:
            pass

    _CASA_ROUTES[name] = 'NOT FOUND'
    return None


_CASA_ROUTES = {}

me     = _resolve_casa('me', 'measures')
ia     = _resolve_casa('ia', 'image')
imhead = _resolve_casa('imhead')
imfit  = _resolve_casa('imfit')
imstat = _resolve_casa('imstat')

# A resolved object is only useful if it exposes the methods we call.
_REQUIRED_METHODS = {'me': ('epoch', 'direction'), 'ia': ('open', 'coordsys', 'close')}
for _name, _methods in _REQUIRED_METHODS.items():
    _obj = globals()[_name]
    if _obj is not None and not all(hasattr(_obj, m) for m in _methods):
        print("Warning: resolved '{}' lacks expected methods; treating as missing.".format(_name))
        globals()[_name] = None
        _CASA_ROUTES[_name] = 'UNUSABLE ({})'.format(_CASA_ROUTES.get(_name))

_MISSING = [n for n in ('me', 'ia', 'imhead', 'imfit', 'imstat')
            if globals()[n] is None]

CASA_READY = (len(_MISSING) == 0)

print("CASA components: " + ", ".join(
    "{}<-{}".format(n, _CASA_ROUTES.get(n, '?'))
    for n in ('me', 'ia', 'imhead', 'imfit', 'imstat')))

if not CASA_READY:
    print("=" * 70)
    print("ERROR: the following CASA components could not be loaded:")
    print("       " + ", ".join(_MISSING))
    print("       This script must be run inside CASA, e.g.")
    print("       CASA <1>: execfile('getflux_MK_Claude_DIR_or_IMG.py')")
    print("=" * 70)
    # No stub fallbacks: fabricated headers and beams would produce
    # plausible-looking but meaningless fluxes.


_RAD2DEG = 180.0 / math.pi


def _angle_to_rad(value, unit):
    """Convert an angle expressed in a CASA unit string to radians."""
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
    """Convert radians to an angle in the given CASA unit string."""
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


def _angular_separation(ra1, dec1, ra2, dec2):
    """Great-circle separation in radians (Vincenty form, stable at small sep)."""
    d_ra = ra1 - ra2
    num = math.sqrt((math.cos(dec1) * math.sin(d_ra)) ** 2 +
                    (math.cos(dec2) * math.sin(dec1) -
                     math.sin(dec2) * math.cos(dec1) * math.cos(d_ra)) ** 2)
    den = (math.sin(dec2) * math.sin(dec1) +
           math.cos(dec2) * math.cos(dec1) * math.cos(d_ra))
    return math.atan2(num, den)


def _peak_value_in_region(fits_file, region):
    """Brightest pixel value inside the region, in image brightness units.

    Used to seed the amplitude of the fit, so that the starting guess tracks
    the actual source brightness across the four orders of magnitude spanned by
    this dataset.

    Returns:
        The maximum, or None on failure.
    """
    try:
        stats = imstat(imagename=fits_file, region=region)
        if not stats or len(stats.get('max', [])) == 0:
            return None
        value = float(stats['max'][0])
        return value if np.isfinite(value) else None
    except Exception as e:
        print("  Peak-value lookup failed: {}".format(e))
        return None


def _peak_pixel_in_region(fits_file, region):
    """
    Fallback position estimate: the brightest pixel inside the fit region,
    obtained from imstat's 'maxpos'. Uses CASA's own region parser, so it is
    immune to any coordinate-conversion convention issue.

    Returns:
        (x_pixel, y_pixel) as floats, or None on failure.
    """
    try:
        stats = imstat(imagename=fits_file, region=region)
        if not stats or 'maxpos' not in stats:
            return None
        maxpos = stats['maxpos']
        if len(maxpos) < 2:
            return None
        return (float(maxpos[0]), float(maxpos[1]))
    except Exception as e:
        print("  Fallback peak-pixel lookup failed: {}".format(e))
        return None


def _source_pixel(fits_file, fit_region=None, verbose=True):
    """
    Converts SOURCE_RA / SOURCE_DEC to pixel coordinates using the image's own
    coordinate system, so the estimate is correct regardless of image size,
    cell size or phase centre.

    The conversion is validated by round-tripping the resulting pixel back to
    world coordinates; if it does not land within one pixel of the requested
    position, the result is rejected and (if a region is supplied) the brightest
    pixel in the fit region is used instead.

    Returns:
        (x_pixel, y_pixel) as floats, or None on failure.
    """
    csys = None
    x = y = None
    try:
        ia.open(fits_file)
        try:
            csys = ia.coordsys()
            shape = ia.shape()
        finally:
            ia.close()

        # --- Identify the direction axes; world and pixel indices can differ ---
        try:
            found = csys.findcoordinate('direction')
            world_idx = [int(i) for i in found['world']]
            pixel_idx = [int(i) for i in found['pixel']]
        except Exception:
            world_idx = [0, 1]
            pixel_idx = [0, 1]

        # --- Native units of the direction axes (rad and deg both occur) ---
        try:
            units = list(csys.units())
            u_ra  = units[world_idx[0]]
            u_dec = units[world_idx[1]]
        except Exception:
            u_ra = u_dec = 'rad'

        # --- Requested position ---
        direction = me.direction('J2000', SOURCE_RA, SOURCE_DEC)
        ra_rad  = float(direction['m0']['value'])
        dec_rad = float(direction['m1']['value'])

        # --- Reference value, converted to radians for comparison ---
        world = list(csys.referencevalue(format='n')['numeric'])
        ref_ra_rad  = _angle_to_rad(world[world_idx[0]], u_ra)
        ref_dec_rad = _angle_to_rad(world[world_idx[1]], u_dec)

        # Put RA on the same 2*pi branch as the image, rather than assuming
        # [0, 2pi): images store either branch depending on their provenance.
        while ra_rad - ref_ra_rad > math.pi:
            ra_rad -= 2.0 * math.pi
        while ref_ra_rad - ra_rad > math.pi:
            ra_rad += 2.0 * math.pi

        # --- Pixel scale, for the validation tolerance ---
        try:
            inc = list(csys.increment(format='n')['numeric'])
            cell_rad = abs(_angle_to_rad(inc[world_idx[1]], u_dec))
        except Exception:
            cell_rad = 1.0 / (_RAD2DEG * 3600.0)   # 1 arcsec fallback

        # --- Convert and transform ---
        world[world_idx[0]] = _rad_to_angle(ra_rad, u_ra)
        world[world_idx[1]] = _rad_to_angle(dec_rad, u_dec)

        pix = csys.topixel(world)['numeric']
        x = float(pix[pixel_idx[0]])
        y = float(pix[pixel_idx[1]])

        if verbose:
            print("  csys units: {} / {}   ref: {:.6f} {:+.6f} deg".format(
                u_ra, u_dec, ref_ra_rad * _RAD2DEG, ref_dec_rad * _RAD2DEG))
            print("  source:     {:.6f} {:+.6f} deg   cell: {:.4f} arcsec".format(
                ra_rad * _RAD2DEG, dec_rad * _RAD2DEG, cell_rad * _RAD2DEG * 3600.0))
            print("  -> pixel:   ({:.2f}, {:.2f})".format(x, y))

        # --- Round-trip validation ---
        back = csys.toworld(pix, 'n')['numeric']
        back_ra  = _angle_to_rad(back[world_idx[0]], u_ra)
        back_dec = _angle_to_rad(back[world_idx[1]], u_dec)
        sep = _angular_separation(ra_rad, dec_rad, back_ra, back_dec)

        if sep > cell_rad:
            print("  Round-trip check FAILED: {:.3f} arcsec discrepancy.".format(
                sep * _RAD2DEG * 3600.0))
            x = y = None
        elif verbose:
            print("  Round-trip check OK ({:.4f} arcsec).".format(
                sep * _RAD2DEG * 3600.0))

    except Exception as e:
        print("  Coordinate conversion failed for {}: {}".format(
            os.path.basename(fits_file), e))
        x = y = None

    finally:
        if csys is not None:
            try:
                csys.done()
            except Exception:
                pass

    # --- Fallback: brightest pixel inside the fit region ---
    if x is None and fit_region is not None:
        print("  Falling back to the brightest pixel in the fit region.")
        peak = _peak_pixel_in_region(fits_file, fit_region)
        if peak is not None:
            x, y = peak
            print("  -> peak pixel: ({:.2f}, {:.2f})".format(x, y))
        if FIX_POSITION:
            print("  WARNING: FIX_POSITION is True but the position came from "
                  "the image peak, not from SOURCE_RA/SOURCE_DEC.")

    if x is None:
        return None

    try:
        nx = int(shape[pixel_idx[0]])
        ny = int(shape[pixel_idx[1]])
    except Exception:
        return (x, y)

    if not (0 <= x < nx and 0 <= y < ny):
        print("Warning: source position falls outside the image "
              "({:.1f}, {:.1f}) vs shape ({}, {}).".format(x, y, nx, ny))
        return None

    return (x, y)


def _measure_local_noise(fits_file):
    """
    Measures the local background RMS in an annulus centred on the source, and
    the local zero level, which is the median of the same annulus and is the
    quantity that biases a fit made without an offset term.

    Uses the median absolute deviation scaled to a Gaussian sigma as the primary
    estimator (least sensitive to residual sources in a crowded field), falling
    back to the clipped standard deviation and then to the plain RMS.

    Returns:
        (rms, background) in image brightness units (Jy/beam). Either element
        is None if it could not be measured.
    """
    if not MEASURE_RMS:
        return None, None

    region = 'annulus[[{}, {}], [{}, {}]] coord=J2000'.format(
        SOURCE_RA, SOURCE_DEC, RMS_ANNULUS_INNER, RMS_ANNULUS_OUTER)

    try:
        stats = imstat(imagename=fits_file, region=region,
                       algorithm=RMS_ALGORITHM)
        if not stats:
            print("Warning: imstat returned nothing for the RMS annulus.")
            return None, None

        npts = stats.get('npts', [0])
        if len(npts) == 0 or float(npts[0]) < 100:
            print("Warning: too few pixels in the RMS annulus; skipping RMS.")
            return None, None

        background = None
        if 'median' in stats and len(stats['median']) > 0:
            candidate = float(stats['median'][0])
            if np.isfinite(candidate):
                background = candidate
                print("Local zero level (annulus median): "
                      "{:.6f} Jy/beam".format(background))

        # 1.4826 * MAD -> Gaussian-equivalent sigma
        for key, scale in (('medabsdevmed', 1.4826), ('sigma', 1.0), ('rms', 1.0)):
            if key in stats and len(stats[key]) > 0:
                value = float(stats[key][0]) * scale
                if np.isfinite(value) and value > 0.0:
                    print("Local RMS ({}): {:.6f} Jy/beam".format(key, value))
                    if background is not None and value > 0.0:
                        print("  zero level is {:+.2f} x the local rms"
                              .format(background / value))
                    return value, background

        print("Warning: no usable RMS estimator returned by imstat.")
        return None, background

    except Exception as e:
        print("RMS measurement failed for {}: {}".format(
            os.path.basename(fits_file), e))
        return None, None


def _fit_single_image(fits_file, fit_region, estimates_filename, label):
    """
    Performs the header read, beam fix, RMS measurement and imfit call for a
    single FITS file.

    Args:
        fits_file (str):          Full path to the FITS image.
        fit_region (str):         The region string for the imfit task.
        estimates_filename (str): Full path to the temporary estimates file.
        label (str):              Label to use in the results table.

    Returns:
        A tuple (label, mjd, flux, err_fit, rms, err_total) on success,
        or None on failure.
    """
    try:
        header_dict = imhead(imagename=fits_file, mode='list')

        req_keys = ['date-obs', 'beammajor', 'beamminor', 'beampa']
        if not all(key in header_dict for key in req_keys):
            print("Warning: One or more required header keys (date-obs, beam) missing in {}".format(fits_file))
            return None

        # --- Convert Date to MJD ---
        date_obs = header_dict['date-obs']
        casa_date_str = date_obs.replace('-', '/').replace('T', '/')
        time_measure = me.epoch('UTC', casa_date_str)
        mjd = time_measure['m0']['value']

        # --- Source position in pixels for this image ---
        pixel = _source_pixel(fits_file, fit_region=fit_region)
        if pixel is None:
            print("Skipping {} (could not place the source).".format(
                os.path.basename(fits_file)))
            return None
        x_pix, y_pix = pixel

        # --- Beam, for the fixed Gaussian shape ---
        bmaj = header_dict['beammajor']['value']
        bmin = header_dict['beamminor']['value']
        bpa  = header_dict['beampa']['value']

        # --- Measure the local noise and zero level ---
        rms, background = _measure_local_noise(fits_file)

        # --- Optionally scale the fit region with the beam ---
        # With the zero level free the fit needs background-dominated pixels to
        # constrain it, and a region fixed in arcsec spans a different number of
        # beams in each band.
        region_used = fit_region
        if FIT_REGION_BEAM_MULTIPLE:
            radius = FIT_REGION_BEAM_MULTIPLE * max(bmaj, bmin)
            region_used = ('ellipse[[{}, {}], [{:.4f}arcsec, {:.4f}arcsec], '
                           '90deg] coord=J2000'.format(SOURCE_RA, SOURCE_DEC,
                                                       radius, radius))
            print("Fit region set to {:.1f} x beam = {:.2f} arcsec radius".format(
                FIT_REGION_BEAM_MULTIPLE, radius))

        # --- Prepare the estimates file ---
        # The initial peak matters more than it looks. With the shape fixed, the
        # derivative of the model with respect to position is proportional to
        # the amplitude, so if the starting peak badly underestimates the true
        # one the position Jacobian is near zero and the first step throws the
        # centroid a long way off. That is how a bright source ends up failing
        # with "Ellipsoid lies completely outside the lattice". Seeding from the
        # brightest pixel actually in the fit region removes the failure mode
        # and costs one imstat call.
        peak_estimate = PEAK_ESTIMATE_JY
        if PEAK_ESTIMATE_FROM_IMAGE:
            observed = _peak_value_in_region(fits_file, region_used)
            if observed is not None and np.isfinite(observed):
                if background is not None and np.isfinite(background):
                    observed -= background
                if observed > 0:
                    peak_estimate = observed
                    print("Peak estimate from the image: {:.6f} Jy/beam "
                          "(fixed default was {:.6f})".format(
                              peak_estimate, PEAK_ESTIMATE_JY))

        # 'abp'   fixes major axis, minor axis and position angle to the beam.
        # 'xyabp' additionally holds the position fixed at the source coords.
        fixed_params = 'xyabp' if FIX_POSITION else 'abp'

        def _write_estimates(peak, fixed):
            with open(estimates_filename, "w") as f:
                f.write("{:.8f},{:.3f},{:.3f}, {:.3f}arcsec, {:.3f}arcsec, "
                        "{:.3f}deg, {}\n".format(peak, x_pix, y_pix, bmaj, bmin,
                                                 bpa, fixed))

        _write_estimates(peak_estimate, fixed_params)

        # --- Perform Gaussian Fit ---
        imfit_kwargs = dict(
            imagename=fits_file,
            region=region_used,
            estimates=estimates_filename,
            logfile='imfit.log',
            append=True
        )
        # Passing rms makes imfit derive its uncertainties from the map noise
        # rather than from the residuals of the fit itself.
        if rms is not None and RMS_TO_IMFIT:
            imfit_kwargs['rms'] = rms

        # Fit an additive zero level. Without this the shape-fixed Gaussian
        # absorbs any pedestal and returns S + 2b rather than S; see the note
        # in the constants block above.
        if FIT_ZERO_LEVEL:
            seed = 0.0
            if SEED_ZERO_LEVEL_FROM_MAP and background is not None:
                seed = background
            imfit_kwargs['dooff'] = True
            imfit_kwargs['offset'] = seed
            imfit_kwargs['fixoffset'] = bool(FIX_ZERO_LEVEL)
            print("Fitting zero level (dooff=True, offset seed {:.6f} Jy/beam, "
                  "fixoffset={})".format(seed, bool(FIX_ZERO_LEVEL)))
        elif background is not None and rms:
            print("WARNING: dooff=False, so the fitted flux is expected to be "
                  "high by about 2 x {:.6f} = {:.6f} Jy".format(
                      background, 2.0 * background))

        def _converged(res):
            return bool(res and res.get('results', {}).get('component0')
                        and res.get('converged', [True])[0])

        fit_result = imfit(**imfit_kwargs)

        # A runaway centroid is the usual way this fails, and imfit reports it
        # as "Ellipsoid lies completely outside the lattice". Retrying with the
        # position held at the source coordinates recovers the measurement.
        if (not _converged(fit_result) and RETRY_WITH_FIXED_POSITION
                and fixed_params == 'abp'):
            print("Fit did not converge with the position free; retrying with "
                  "it fixed at the source coordinates.")
            _write_estimates(peak_estimate, 'xyabp')
            fit_result = imfit(**imfit_kwargs)
            if _converged(fit_result):
                print("  the retry succeeded: this epoch's position is FIXED, "
                      "not fitted.")

        # --- Extract Results ---
        if fit_result and 'results' in fit_result and 'component0' in fit_result['results']:
            component = fit_result['results']['component0']
            flux      = component['flux']['value'][0]
            err_fit   = component['flux']['error'][0]

            # Because the shape is fixed to the restoring beam, the integrated
            # flux in Jy equals the peak in Jy/beam, so the map RMS is directly
            # a floor on the flux uncertainty.
            err = err_fit
            if rms is not None and APPLY_RMS_FLOOR and rms > err:
                print("Applying RMS floor: {:.6f} -> {:.6f} Jy".format(err, rms))
                err = rms

            err_total = math.sqrt(err ** 2 + (FLUX_SCALE_FRAC * flux) ** 2)

            # Report the fitted zero level where imfit provides it, so that the
            # value actually removed from the data is on the record.
            zero_level = float('nan')
            for key in ('zerooff', 'offset'):
                entry = (fit_result['results'].get(key)
                         if 'results' in fit_result else None)
                if isinstance(entry, dict) and 'value' in entry:
                    try:
                        zero_level = float(np.ravel(entry['value'])[0])
                    except Exception:
                        pass
                    break
            if np.isfinite(zero_level):
                print("Fitted zero level: {:.6f} Jy/beam".format(zero_level))

            print("Fit successful. Flux: {:.6f} +/- {:.6f} Jy "
                  "(fit err {:.6f})".format(flux, err_total, err_fit))
            return (label, mjd, flux, err_fit,
                    rms if rms is not None else float('nan'), err_total,
                    background if background is not None else float('nan'),
                    zero_level)
        else:
            print("Fit failed or no component found for: {}".format(os.path.basename(fits_file)))
            return None

    except Exception as e:
        print("An error occurred while processing {}: {}".format(os.path.basename(fits_file), e))
        return None

    finally:
        if os.path.exists(estimates_filename):
            os.remove(estimates_filename)


def _write_results(results, output_file, label_header="Label"):
    """
    Writes the list of result tuples to a formatted text file, sorted by MJD.

    Args:
        results (list):     List of (label, mjd, flux, err_fit, rms, err_total,
                            background, zero_level).
        output_file (str):  Path to the output file.
        label_header (str): Column header for the label column.
    """
    print("\n--- Writing results to {} ---".format(output_file))
    results = sorted(results, key=lambda r: r[1])
    with open(output_file, 'w') as f:
        f.write("# Source: {} {}   fixed_params: {}   flux-scale systematic: {:.1%}\n".format(
            SOURCE_RA, SOURCE_DEC, 'xyabp' if FIX_POSITION else 'abp', FLUX_SCALE_FRAC))
        f.write("# Zero level: dooff={}  fixoffset={}  seeded from map: {}\n".format(
            FIT_ZERO_LEVEL, FIX_ZERO_LEVEL, SEED_ZERO_LEVEL_FROM_MAP))
        f.write("# Flux_Error_Total = sqrt(max(fit_err, local_rms)^2 + (frac*flux)^2)\n")
        f.write("# Bkg is the annulus median.\n")
        f.write("# {:<40} {:<12} {:<14} {:<14} {:<14} {:<14} {:<14}\n".format(
            label_header, "MJD", "Flux (Jy)", "Fit_Err (Jy)",
            "RMS (Jy/bm)", "Err_Total (Jy)", "Bkg (Jy/bm)"))
        for r in results:
            # The fitted zero level is still returned in element 7 of each
            # result tuple and printed to the console, but is not tabulated.
            row = list(r) + [float('nan')] * (8 - len(r))
            f.write("{:<40} {:<12.5f} {:<14.6f} {:<14.6f} {:<14.6f} {:<14.6f} "
                    "{:<14.6f}\n".format(
                        row[0], row[1], row[2], row[3], row[4], row[5],
                        row[6]))
    print("Done.")


def analyze_source_flux(observation_dirs, fit_region, output_file=OUTPUT_FILENAME):
    """
    Analyzes the flux of a point source by iterating over observation directories.
    For each directory, looks for FITS files first in an IMAGES/ subdirectory,
    then falls back to the directory root.

    Args:
        observation_dirs (list): A list of observation directory paths.
        fit_region (str):        The region string for the imfit task.
        output_file (str):       The name of the file to save the results.
    """
    if not observation_dirs:
        print("No directories provided to process.")
        return

    results = []
    base_dir = os.getcwd()
    estimates_filename = os.path.join(base_dir, "estimates.txt")

    for obs_dir in observation_dirs:
        full_obs_path = os.path.join(base_dir, obs_dir)
        if not os.path.isdir(full_obs_path):
            print("Warning: '{}' is not a valid directory. Skipping.".format(obs_dir))
            continue

        print("\n--- Processing Directory: {} ---".format(obs_dir))

        # Search in IMAGES/ subdirectory first, then fall back to the directory root.
        fits_files = []
        images_dir = os.path.join(full_obs_path, 'IMAGES')
        if os.path.isdir(images_dir):
            fits_files = sorted(glob.glob(os.path.join(images_dir, 'img*pcalmask-MFS-image.fits')))

        if not fits_files:
            fits_files = sorted(glob.glob(os.path.join(full_obs_path, 'img*pcalmask-MFS-image.fits')))

        if not fits_files:
            print("No FITS file matching '*pcalmask-MFS-image.fits' found in: {} "
                  "(checked IMAGES/ and root)".format(obs_dir))
            continue

        if len(fits_files) > 1:
            print("Warning: multiple matching FITS files found, using: {}".format(
                os.path.basename(fits_files[0])))

        fits_file = fits_files[0]
        print("Found image: {}".format(os.path.basename(fits_file)))

        result = _fit_single_image(fits_file, fit_region, estimates_filename, label=obs_dir)
        if result is not None:
            results.append(result)

    if results:
        _write_results(results, output_file, label_header="Directory")
    else:
        print("\nNo successful fits were performed. Output file not written.")


def analyze_source_flux_from_files(fits_files, fit_region, output_file=OUTPUT_FILENAME):
    """
    Analyzes the flux of a point source from an explicit list of FITS files,
    without any directory-walking logic.

    Args:
        fits_files (list): A list of FITS file paths.
        fit_region (str):  The region string for the imfit task.
        output_file (str): The name of the file to save the results.
    """
    if not fits_files:
        print("No files provided to process.")
        return

    results = []
    base_dir = os.getcwd()
    estimates_filename = os.path.join(base_dir, "estimates.txt")

    for fits_file in fits_files:
        if not os.path.isfile(fits_file):
            print("Warning: '{}' is not a valid file. Skipping.".format(fits_file))
            continue

        print("\n--- Processing File: {} ---".format(os.path.basename(fits_file)))

        label = os.path.basename(fits_file)
        result = _fit_single_image(fits_file, fit_region, estimates_filename, label=label)
        if result is not None:
            results.append(result)

    if results:
        _write_results(results, output_file, label_header="Filename")
    else:
        print("\nNo successful fits were performed. Output file not written.")


if __name__ == '__main__':

    if not CASA_READY:
        raise RuntimeError(
            "CASA components missing ({}). Run this script inside CASA.".format(
                ", ".join(_MISSING)))

    fit_region_str  = FIT_REGION
    output_filename = OUTPUT_FILENAME

    print("Source position: {} {}".format(SOURCE_RA, SOURCE_DEC))
    print("Peak estimate:   {} Jy/beam   ({} position)".format(
        PEAK_ESTIMATE_JY, 'fixed' if FIX_POSITION else 'free'))


    # --- Ask user for processing mode ---
    mode = raw_input("Process (d)irectories or (f)iles directly? [d/f]: ").strip().lower()

    if mode == 'f':
        user_input = raw_input("Enter file pattern (e.g., img*pcalmask-MFS-image.fits) "
                               "or space-separated filenames: ")
        input_items = user_input.split()
        fits_to_process = []
        for item in input_items:
            fits_to_process.extend(glob.glob(item))
        fits_to_process = sorted(fits_to_process)
        if fits_to_process:
            analyze_source_flux_from_files(fits_to_process, fit_region_str, output_filename)
        else:
            print("No files found matching the input: {}".format(user_input))

    else:
        user_input = raw_input("Enter directory pattern (e.g., 156*) "
                               "or space-separated list of directories: ")
        input_items = user_input.split()
        dirs_to_process = []
        for item in input_items:
            dirs_to_process.extend(glob.glob(item))
        dirs_to_process = sorted(dirs_to_process)
        if dirs_to_process:
            analyze_source_flux(dirs_to_process, fit_region_str, output_filename)
        else:
            print("No directories found matching the input: {}".format(user_input))