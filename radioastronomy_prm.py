import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse


# ============================================================
# I/O
# ============================================================

def read_cube(path):
    """
    Reads a FITS data cube.

    Parameters
    ----------
    path : str

    Returns
    -------
    cube : ndarray
        Data cube with singleton dimensions removed.
    head : fits.Header
        FITS header.
    """
    cube = np.squeeze(fits.getdata(path))
    head = fits.getheader(path)
    return cube, head


# ============================================================
# PIXEL SCALE AND AXES
# ============================================================

def get_pxscale(head):
    """
    Returns pixel scale in arcseconds/pixel.
    """
    return 3600.0 * np.mean([np.abs(head['CDELT1']), np.abs(head['CDELT2'])])


def _readpositionaxis(head, a=1):
    """
    Returns spatial axis in arcseconds.

    Parameters
    ----------
    a : int
        Axis number (1 or 2).

    Returns
    -------
    axis : ndarray
    """
    if a not in (1, 2):
        raise ValueError("'a' must be 1 or 2.")

    a_len = head[f'NAXIS{a}']
    a_del = head[f'CDELT{a}']
    a_pix = head[f'CRPIX{a}']

    pix = np.arange(a_len)
    axis = 3600.0 * a_del * (pix + 1 - a_pix)
    return axis


# ============================================================
# BEAM
# ============================================================

def calculate_beam_area_arcsec(head):
    """
    Beam area in square arcseconds.
    """
    bmaj = head['BMAJ'] * 3600.0
    bmin = head['BMIN'] * 3600.0
    return np.pi * bmaj * bmin / (4.0 * np.log(2.0))


def beams_per_pix(head):
    """
    Number of beams per pixel.
    """
    xaxis = _readpositionaxis(head, a=1)
    dpix = np.mean(np.abs(np.diff(xaxis)))
    return dpix**2 / calculate_beam_area_arcsec(head)


# ============================================================
# SPECTRA
# ============================================================

def integrated_spectrum(data, head):
    """
    Computes integrated spectrum corrected by beams per pixel.

    Returns
    -------
    spectrum : ndarray
    """
    collapsed = np.nansum(data, axis=(1, 2))
    collapsed *= beams_per_pix(head)
    return collapsed


def mean_spectrum(cube, ww):
    """
    Computes mean spectrum inside a mask.
    """
    spectrum = np.array([
        np.nanmean(cube[i, ww[0], ww[1]])
        for i in range(cube.shape[0])
    ])
    return spectrum


def stack_spectrum(cube, ww):
    """
    Computes stacked (summed) spectrum inside a mask.
    """
    spectrum = np.array([
        np.nansum(cube[i, ww[0], ww[1]])
        for i in range(cube.shape[0])
    ])
    return spectrum


# ============================================================
# RADIAL PROFILES
# ============================================================

def radial_profile(data, center, norm='none'):
    """
    Computes azimuthally averaged radial profile.
    """
    if norm == 'data':
        data = data / np.nanmax(data)

    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)

    mask = ~np.isnan(data)
    tbin = np.bincount(r[mask], data[mask])
    tbin2 = np.bincount(r[mask], data[mask]**2)
    nr = np.bincount(r[mask])

    radialprofile = tbin / nr
    var = (tbin2 - (tbin**2) / nr) / nr
    sigma = np.sqrt(var)

    if norm == 'rp':
        radialprofile /= np.nanmax(radialprofile)
        sigma /= np.nanmax(radialprofile)

    return radialprofile, sigma


def radialAverage(image, center=None, stddev=False, returnAz=False,
                  return_naz=False, binsize=1.0, weights=None,
                  steps=False, interpnan=False, left=None, right=None,
                  mask=None, symmetric=None):
    """
    Azimuthal average as a function of angle.
    """
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2.,
                           (y.max() - y.min()) / 2.])

    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2 * np.pi
    theta_deg = np.degrees(theta)
    maxangle = 360

    if weights is None:
        weights = np.ones_like(image)
    elif stddev:
        raise ValueError("Weighted standard deviation not defined.")

    if mask is None:
        mask = np.ones(image.size, dtype=bool)
    else:
        mask = mask.ravel()

    if symmetric == 2:
        theta_deg %= 90
        maxangle = 90
    elif symmetric == 1:
        theta_deg %= 180
        maxangle = 180

    nbins = int(np.round(maxangle / binsize))
    bins = np.linspace(0, nbins * binsize, nbins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    whichbin = np.digitize(theta_deg.ravel(), bins)
    nr = np.bincount(whichbin)[1:]

    if stddev:
        azprof = np.array([
            image.ravel()[mask & (whichbin == b)].std()
            for b in range(1, nbins + 1)
        ])
    else:
        azprof = np.array([
            (image * weights).ravel()[mask & (whichbin == b)].sum() /
            weights.ravel()[mask & (whichbin == b)].sum()
            for b in range(1, nbins + 1)
        ])

    if interpnan:
        good = np.isfinite(azprof)
        azprof = np.interp(bin_centers, bin_centers[good], azprof[good],
                           left=left, right=right)

    if steps:
        return np.repeat(bins[:-1], 2), np.repeat(azprof, 2)
    if returnAz:
        return bin_centers, azprof
    if return_naz:
        return nr, bin_centers, azprof
    return azprof


# ============================================================
# VELOCITY / FREQUENCY AXES
# ============================================================

def build_v_array(size, rval, rpix, delta):
    """
    Builds velocity array.
    """
    i = np.arange(size) + 1
    return rval + (i - rpix) * delta


def build_varr_head(header):
    """
    Builds velocity array from FITS header.
    """
    return build_v_array(header['NAXIS3'],
                         header['CRVAL3'],
                         header['CRPIX3'],
                         header['CDELT3'])


def build_f_array(size, rval, rpix, delta):
    """
    Builds frequency array.
    """
    i = np.arange(size) + 1
    return rval + (i - rpix) * delta


# ============================================================
# GEOMETRY / CONTOURS
# ============================================================

def fit_ellipse(cont, method):
    """
    Fits an ellipse to a contour.
    """
    x = cont[:, 0][:, None]
    y = cont[:, 1][:, None]

    D = np.hstack([x*x, x*y, y*y, x, y, np.ones_like(x)])
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1

    E, V = np.linalg.eig(np.linalg.inv(S) @ C)
    n = np.argmax(np.abs(E)) if method == 1 else np.argmax(E)
    a = V[:, n]

    b, c, d, f, g, a0 = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b - a0*c
    cx = (c*d - b*f) / num
    cy = (a0*f - b*d) / num

    angle = 0.5 * np.degrees(np.arctan2(2*b, a0 - c))
    up = 2 * (a0*f*f + c*d*d + g*b*b - 2*b*d*f - a0*c*g)
    down1 = num * ((c - a0) * np.sqrt(1 + 4*b*b / (a0 - c)**2) - (c + a0))
    down2 = num * ((a0 - c) * np.sqrt(1 + 4*b*b / (a0 - c)**2) - (c + a0))

    A = np.sqrt(np.abs(up / down1))
    B = np.sqrt(np.abs(up / down2))

    ell = Ellipse((cx, cy), 2*A, 2*B, angle)
    return [cx, cy, A, B, angle], ell.get_verts()


def get_conts(image, levels=None):
    """
    Retrieves longest contour from an image.
    """
    cs = plt.contour(image) if levels is None else plt.contour(image, levels)
    cont = max((p.to_polygons()[0] for c in cs.collections for p in c.get_paths()),
               key=len)
    plt.close()
    return np.array(cont)


def inside_polygon(map, poly, verbose=False):
    """
    Returns mask and indices inside polygon.
    """
    y, x = np.indices(map.shape)
    points = np.column_stack([x.ravel(), y.ravel()])
    mask = Path(poly).contains_points(points).reshape(map.shape)
    ww = np.where(mask)

    if verbose:
        plt.imshow(map * mask, origin='lower')
        plt.show()

    return mask, ww


# ============================================================
# FLUXES / AREAS
# ============================================================

def compute_cont_flux(map, ww, pixel_scale, beam_area, rms=0.0):
    scale = (pixel_scale**2) / beam_area
    flux = np.nansum(map[ww]) * scale
    e_flux = rms * np.sqrt((len(ww[0]) * pixel_scale**2) / beam_area)
    return flux, e_flux


def compute_cont_area_T(map, ww, pixel_scale, beam_area, rms=0.0):
    n_beams = (len(ww[0]) * pixel_scale**2) / beam_area
    area = np.nanmean(map[ww])
    e_area = rms / np.sqrt(n_beams)
    return area, e_area


# ============================================================
# MASKS
# ============================================================

def channel_mask(cube, fchan=0, lchan=-1):
    chans = np.arange(cube.shape[0])
    mask = (chans >= min(fchan, lchan)) & (chans <= max(fchan, lchan))
    return mask[:, None, None].astype(float)


def circular_mask(cube, xc=0, yc=0, rin=0, rout=3, px_scale=1e-6):
    y, x = np.indices(cube[0].shape)
    r = np.hypot(x - xc, y - yc)
    return ((r >= rin/px_scale) & (r <= rout/px_scale)).astype(int)


def elliptical_mask(cube, xc=0, yc=0, a_in=0, b_in=0,
                    a_out=3, b_out=2, px_scale=1e-6,
                    angle=0.0, angle_in_degrees=True):
    y, x = np.indices(cube[0].shape)
    x = (x - xc) * px_scale
    y = (y - yc) * px_scale

    th = np.deg2rad(angle) if angle_in_degrees else angle
    c, s = np.cos(th), np.sin(th)

    xr = x*c + y*s
    yr = -x*s + y*c

    r_out = (xr/a_out)**2 + (yr/b_out)**2
    if a_in <= 0 or b_in <= 0:
        return (r_out <= 1).astype(int)

    r_in = (xr/a_in)**2 + (yr/b_in)**2
    return ((r_out <= 1) & (r_in >= 1)).astype(int)


def make_elliptical_mask(ellipse_param, im_size_x=128, im_size_y=128):
    coords = make_ellipse(ellipse_param)
    x, y = np.meshgrid(np.arange(im_size_x), np.arange(im_size_y))
    points = np.column_stack([x.ravel(), y.ravel()])
    return Path(coords).contains_points(points).reshape(im_size_x, im_size_y)


def apply_mask(image, mask, exclude=True, fill=np.nan):
    out = np.array(image, copy=True)
    if exclude:
        out[~mask] = fill
    else:
        out[mask] = fill
    return out


def make_ellipse(param):
    """
    Returns ellipse coordinates from parameters.
    """
    cx, cy, a, b, angle = param
    ell = Ellipse((cx, cy), 2*a, 2*b, angle)
    return ell.get_verts()


# ============================================================
# NOISE / SMOOTHING
# ============================================================

def smooth_data(data, smooth=0, polyorder=0):
    """
    Smooths data along axis=0.
    """
    if smooth > 1:
        if polyorder > 0:
            from scipy.signal import savgol_filter
            smooth += 1 if smooth % 2 == 0 else 0
            return savgol_filter(data, smooth, polyorder, axis=0, mode='wrap')
        else:
            from scipy.ndimage import uniform_filter1d
            a = uniform_filter1d(data, smooth, axis=0, mode='wrap')
            b = uniform_filter1d(data[::-1], smooth, axis=0, mode='wrap')[::-1]
            return 0.5 * (a + b)
    return data.copy()


def rms_mom_map(rms_chan, nchan, wchan):
    """
    RMS of moment-0 map.
    """
    return rms_chan * wchan * np.sqrt(nchan)
