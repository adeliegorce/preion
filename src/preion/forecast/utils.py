import copy
import logging
import os
import warnings

import healpy as hp
import numpy as np
from astropy import constants, cosmology, units
from matplotlib import rc, colormaps, colors
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator

from plancklens import utils as plancklens_utils, qresp, nhl

from ..parameters import nu21_ref, telescope_specs

logger = logging.getLogger(__name__)

_SENSITIVITY_DIR = os.path.join(os.path.dirname(__file__), "sensitivity_21cm")


def get_lbins(tel_tau, lbin_edges=None):

    assert tel_tau in telescope_specs.keys()
    lmin, lmax = telescope_specs[tel_tau]['lmin'], telescope_specs[tel_tau]['lmax']
    dell = telescope_specs[tel_tau]['Delta_ell']
    if lbin_edges is None:
        if tel_tau.find('SAT') >= 0:
            lbin_edges = np.r_[lmin, np.arange(dell, lmax+dell, step=dell)]
        else:
            lbin_edges = np.arange(lmin, lmax+dell, step=dell)
            if lbin_edges[-1] > lmax:
                lbin_edges[-1] = lmax
        dells = np.ones(lbin_edges.size - 1) * dell
    else:
        lbin_edges = np.sort(lbin_edges)
        dells = np.diff(lbin_edges)
        if np.min(lbin_edges) < lmin:
            warnings.warn(f'Min l below {tel_tau} limit, replacing value.')
            lbin_edges[0] = lmin
        if np.max(lbin_edges) > lmax:
            warnings.warn(f'Max l above {tel_tau} limit, replacing value.')
            lbin_edges[-1] = lmax
        if not np.allclose(dells, dells):
            warnings.warn(f'Delta l diff from {tel_tau} spec ({dells} vs. {dell}).')
    ells = np.array(0.5 * (lbin_edges[1:] + lbin_edges[:-1]), dtype=int)
    return ells, lbin_edges, dells


def _survey_label(telescope_21, sensitivity_case, ska_array=None, n_fields=None):
    """Assemble the on-disk sensitivity-file prefix, e.g.
    "hera_moderate" or "ska_aast_optimistic_3fields", from the separate,
    explicit config pieces. This is the one place such a filename gets
    constructed -- nobody types it directly into a config. The `n_fields`
    suffix is nested inside the 'ska' branch (not appended unconditionally
    afterward), so passing `n_fields` for 'hera'/'mwa' is naturally a no-op,
    matching the real files on disk (which have no such suffix for those
    telescopes)."""
    if telescope_21 == "ska":
        if ska_array is None:
            raise ValueError("ska_array is required when telescope_21='ska'")
        label = f"{telescope_21}_{ska_array}_{sensitivity_case}"
        if n_fields is not None:
            label += f"_{n_fields}fields"
    else:
        label = f"{telescope_21}_{sensitivity_case}"
    return label


def get_sensitivity(telescope_21, sensitivity_case, ska_array=None, n_fields=None,
                     zs=(5.5, 6., 6.5, 7., 8., 9., 10., 11., 12., 14.),
                     error_type='thermal', sensitivity_dir=None, debug=False):
    """Load the 21cm noise-power sensitivity curves for a survey
    (`telescope_21`/`sensitivity_case`/`ska_array`/`n_fields`, see
    `_survey_label`) across a grid of redshifts `zs`, and return a
    RegularGridInterpolator over (z, k)."""
    sensitivity_dir = sensitivity_dir or _SENSITIVITY_DIR
    label = _survey_label(telescope_21, sensitivity_case, ska_array=ska_array, n_fields=n_fields)

    zs = np.sort(zs)[::-1]
    freq_bands = np.round(nu21_ref / (zs + 1.0), decimals=1)  # MHz

    sensitivity_at_z_arr = []
    sensitivity_at_z_kbins = []
    for iz in range(zs.size):
        if error_type == 'both':
            out1 = np.loadtxt(os.path.join(sensitivity_dir, f"{label}_thermal_{freq_bands[iz]}.txt"))
            out2 = np.loadtxt(os.path.join(sensitivity_dir, f"{label}_sample_{freq_bands[iz]}.txt"))
            out = out1 + out2
        else:
            out = np.loadtxt(os.path.join(sensitivity_dir, f"{label}_{error_type}_{freq_bands[iz]}.txt"))
        sensitivity_at_z_arr.append(out[:, 1])
        sensitivity_at_z_kbins.append(out[:, 0])

    sensitivity_at_z = RegularGridInterpolator(
        (zs, sensitivity_at_z_kbins[0]),
        np.array(sensitivity_at_z_arr),
        method='linear',
        fill_value=np.inf,
        bounds_error=False
    )

    if debug:
        logger.debug(f"NaN count: {np.isnan(sensitivity_at_z_arr).sum()}")
        logger.debug(f"k range: {np.min(sensitivity_at_z_kbins)}, {np.max(sensitivity_at_z_kbins)}")

    return sensitivity_at_z


def get_fsky_21cm(telescope_21, n_fields=None):
    """Sky fraction observed by a 21cm survey, as a plain (dimensionless)
    float. A genuine lookup keyed on the bare telescope name (not the old
    string-substring-matching hack this replaces). `n_fields` is only ever
    read for `telescope_21='ska'`, so it's a structural no-op for
    'hera'/'mwa' -- no separate check needed.

    Note: the source this was ported from returned a bare float for 'hera'
    but an astropy Quantity for 'ska'/'mwa' (an FOV-in-deg**2-to-rad**2
    conversion happens to leave one behind) -- an inconsistency that never
    surfaced there since nothing downstream called `.value` on it
    explicitly. `.value` is called here so this function always returns
    the same (plain float) type regardless of telescope."""
    if telescope_21 == "hera":
        return 0.5 * (np.cos(55 * np.pi / 180.) - np.cos(65 * np.pi / 180.))
    if telescope_21 == "ska":
        fsky = (((9.10 * units.deg) ** 2).to(units.rad ** 2) / (4. * np.pi * units.rad ** 2)).value
        return fsky * n_fields if n_fields is not None else fsky
    if telescope_21 == "mwa":
        return (((24.6 * units.deg) ** 2).to(units.rad ** 2) / (4. * np.pi * units.rad ** 2)).value
    raise ValueError(f"Unknown 21cm telescope {telescope_21!r}; expected 'hera', 'ska', or 'mwa'.")


def get_cl21_noise(noise_function, z21, ells, Dells=True, delta_nu=0., nz=100,
                    freq_res=100. * units.kHz, cos=cosmology.Planck18):
    """
    Compute angular cross spectrum of the 21cm noise power at ell centred
    on z21, in uK^2.

    Parameters
    ----------
        noise_function: callable
            21cm noise-power interpolator, e.g. from `get_sensitivity`.
        z21: float
            Redshift of the observed 21cm signal.
        ells: array of floats
            Angular multipole to compute the spectrum at.
        Dells: boolean
            If True, give the results in terms of
            D(l) = l(l+1)Cl/2/pi.
            Default is False.
        delta_nu: float or Quantity
            Width of the top-hat function representing the
            frequency resolution of the interferometer, in MHz.
            Default is zero (Dirac delta). Accepts either a bare
            float (assumed MHz) or an astropy Quantity.
    Outputs
    ------
        Array of angular noise-power for ells, in uK2.
    """

    ells = np.atleast_1d(ells)
    try:
        dnu = delta_nu.to(units.MHz)
    except AttributeError:
        dnu = delta_nu * units.MHz
    nu21_ref_unit = nu21_ref * units.MHz

    if dnu.value == 0.:
        zlin = np.r_[np.linspace(z21 - 0.1, z21, nz // 2), np.linspace(z21, z21 + 0.1, nz // 2)[1:]]
    else:
        nu21 = nu21_ref_unit / (1. + z21)  # MHz
        numax = nu21 + dnu / 2.
        numin = nu21 - dnu / 2.
        zmax = (nu21_ref_unit / numin) - 1.
        zmin = (nu21_ref_unit / numax) - 1.
        zlin = np.linspace(zmin.value - 0.1, zmax.value + 0.2, nz)

    W21_zlin = np.zeros_like(zlin) * 1. / units.MHz
    iz21 = np.argmin(np.abs(zlin - z21))
    if delta_nu == 0.:
        W21_zlin[iz21] = 1. / units.MHz
    else:
        nu_integ = nu21_ref_unit / (1. + zlin)  # MHz
        if np.abs(np.diff(nu_integ)[iz21]) > dnu:
            raise ValueError('dnu smaller than zlin resolution '
                              f'({dnu:.2f}, {np.abs(np.diff(nu_integ)[iz21]):.2f}), '
                              'increase nz.')
        mask = (zlin < zmax) & (zlin >= zmin)
        W21_zlin[mask] = 1. / (numax - numin)  # unit T
    W21_zlin *= nu21_ref_unit * cos.H(zlin).si / (1 + zlin) ** 2 / constants.c.si  # units L-1
    W21_zlin = W21_zlin.to(1. / units.Mpc)
    W21_zlin /= np.trapz(W21_zlin, cos.comoving_distance(zlin))

    eta_lin = cos.comoving_distance(zlin)
    C_ell_21_integrand = []
    for j, ell in enumerate(ells):
        k_zlin = ell / eta_lin  # [Mpc-1]
        N21 = noise_function(np.c_[zlin, k_zlin.value]) * units.mK ** 2
        N21 = N21 / k_zlin ** 3 * 2 * np.pi ** 2
        N21[np.isnan(N21) | np.isinf(N21)] = 0.
        C_ell_21_integrand.append(
            N21
            / eta_lin ** 2
            * W21_zlin.to(1. / units.Mpc) ** 2
        )
    C_ell_21_integrand = np.array(C_ell_21_integrand) * units.mK ** 2 / units.Mpc
    C_ells = np.trapz(C_ell_21_integrand, eta_lin, axis=1)
    C_ells[C_ells == 0.] = np.inf * units.mK ** 2
    if not Dells:
        return C_ells
    else:
        return ells * (ells + 1.) * C_ells / 2. / np.pi


def lower_limit(param, lolim):
    """Flat lower-limit prior: -inf if param < lolim, else 0 (no-op)."""
    if lolim is None:
        return 0.
    elif param < lolim:
        return -np.inf
    return 0.


def gaussian_prior(param, gaus_params):
    """Gaussian log-prior term: -0.5*(param-mu)^2/sigma^2, or 0 (no-op) if
    gaus_params is None. gaus_params is [mu, sigma]."""
    if gaus_params is None:
        return 0.
    mu, sigma = gaus_params
    return -0.5 * ((param - mu) ** 2) / (sigma ** 2)


def invert_covariance(cov):
    """General inverse of a covariance matrix that may contain non-finite
    (e.g. inf) or non-positive marginal variances on its diagonal --
    entries with no real information. Those rows/columns get exactly zero
    in the returned inverse (not merely a small value from inverting a
    large-but-finite sentinel), so they drop out of a `resid @ inv_cov @
    resid` chi2 sum entirely, regardless of what they might otherwise
    correlate with. The well-constrained sub-block is inverted with a
    genuine `np.linalg.inv`, correctly accounting for any off-diagonal
    covariance between finite-variance entries -- this does not assume
    `cov` is diagonal, even though every covariance in this package is
    today."""
    diag = np.diagonal(cov)
    finite = np.isfinite(diag) & (diag > 0)
    inv_cov = np.zeros_like(cov)
    inv_cov[np.ix_(finite, finite)] = np.linalg.inv(cov[np.ix_(finite, finite)])
    return inv_cov


def sample_var(ls, dl, telescope):
    if type(telescope) is dict:
        fsky = telescope['fsky']
    else:
        fsky = float(telescope)
    if np.shape(ls) != np.shape(dl):
        raise ValueError('ls and dl must have the same shape.')
    dDl = dl * np.sqrt(2./fsky/(2.*ls+1.))
    return dDl


def noise(ls, telescope, pol=False, is_cl=False):

    ls = np.atleast_1d(ls)
    sig0 = telescope['noise'] / 60.0 * np.pi / 180.0 # arcmin to rad
    fwhm = telescope['fwhm'] / 60.0 * np.pi / 180.0 # arcmin to rad
    nl = sig0**2 * np.exp(ls*(ls+1.)*fwhm**2/8./np.log(2.))
    if pol:
        nl *= 2.
    if not is_cl:
        nl *= ls*(ls+1.)/2./np.pi
    return nl


def get_cls_for_tau_noise(preion_model):
    ls_temp = np.arange(25001)
    primary_cls = preion_model.get_primary_spectra(ells=ls_temp, Dells=False, type='unlensed_total')
    primary_cls = primary_cls[:, [0, 1, 2, 4, 3]]
    lensed_cls = preion_model.get_primary_spectra(ells=ls_temp, Dells=False)
    lensed_cls = lensed_cls[:, [0, 1, 2, 4, 3]]
    tau_cls = np.r_[0., np.sum(preion_model.get_tau(ells=ls_temp[1:], signal='both', Dells=False), axis=1)]
    return primary_cls, lensed_cls, tau_cls


def get_cls_dict(primary_cls, lensed_cls):
    primary_cls_dict = {}
    lensed_cls_dict = {}
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        primary_cls_dict[k] = np.copy(primary_cls[:, i+1])
        lensed_cls_dict[k] = np.copy(lensed_cls[:, i+1])
    primary_cls_dict['et'] = np.copy(primary_cls_dict['te'])
    lensed_cls_dict['et'] = np.copy(lensed_cls_dict['te'])
    for k in ['eb', 'tb', 'be', 'bt']:
        # zero in standard physics
        primary_cls_dict[k] = np.zeros_like(primary_cls[:, 0])
        lensed_cls_dict[k] = np.zeros_like(lensed_cls[:, 0])
    return primary_cls_dict, lensed_cls_dict


def tau_noise(telescope, primary_cls, lensed_cls, tau_cls, is_cl=False, key='f_eb', use_lensed=False, existing_telescope=True):

    assert np.shape(primary_cls)[-1] == np.shape(lensed_cls)[-1] == 5
    assert np.shape(primary_cls)[0] == np.shape(lensed_cls)[0] == np.shape(tau_cls)[0]
    primary_cls_dict, lensed_cls_dict = get_cls_dict(primary_cls, lensed_cls)

    # telescope specs
    if existing_telescope:
        tel = telescope_specs[telescope]
    else:
        tel = copy.copy(telescope)
    nlev_t = tel['noise']  # Temperature noise level in muK.arcmin
    nlev_p = nlev_t*np.sqrt(2.)  # Polarisation noise level in muK.arcmin
    beam_fwhm_amin = tel['fwhm']  # Full width at half maximum of the gaussian beam of our instrument, in arcmin
    lmin_ivf = int(tel['lmin'])
    if tel['lmax'] + 1 > primary_cls.shape[0]:
        warnings.warn('Input Cls do not go up to telescope lmax...')
    lmax_ivf = min(int(tel['lmax']), primary_cls.shape[0]-1)
    assert lmax_ivf > lmin_ivf
    nside = int(2**np.ceil(np.log(lmax_ivf)/np.log(2)))
    if nside > 4096:
        warnings.warn(
            'Too large nside ({nside}) required by lmax={lmax_ivf}... '
            'changing back to healpy max of 8192.'
        )
        nside = 8192
    # Noise spectrum with Gaussian beam
    cl_beam = hp.gauss_beam(fwhm=beam_fwhm_amin*np.pi/180/60, lmax=lmax_ivf)

    # Simulate maps from Cls
    primary_t_map, primary_q_map, primary_u_map = hp.synfast(
        cls=primary_cls[:, 1:].T,
        nside=nside,
        pol=True,
        new=True
    )
    tau_map = hp.synfast(tau_cls, nside=nside,)
    screened_maps = {
        't': primary_t_map * np.exp(-tau_map),
        'q': primary_q_map * np.exp(-tau_map),
        'u': primary_u_map * np.exp(-tau_map)
    }
    out = hp.anafast(
        [screened_maps[k] for k in ['t', 'q', 'u']],
        lmax=lmax_ivf, pol=True
    )
    screened_cls_from_maps = {
        'tt': out[0], 'ee': out[1], 'bb': out[2],
        'te': out[3], 'eb': out[4], 'tb': out[5],
        'et': out[3], 'be': out[4], 'bt': out[5],
    }

    # This is the inverse variance filter, ie 1/(C_ell^obs + N_ell^noise), eq 22
    # here, observed cls = screened primary + lensed + noise
    # It also sets the minimum and maximum scales of the CMB maps used in the reconstruction
    ftl = {}
    for k in ['t', 'e', 'b']:
        if use_lensed:
            ftl[k] = lensed_cls_dict[k+k][:lmax_ivf + 1]
        else:
            ftl[k] = screened_cls_from_maps[k+k][:lmax_ivf + 1]
        if k == 't':
            ftl[k] += (nlev_t / 60. / 180. * np.pi / cl_beam) ** 2
        else:
            ftl[k] += (nlev_p / 60. / 180. * np.pi / cl_beam) ** 2
        ftl[k] = plancklens_utils.cli(ftl[k])
        ftl[k][:lmin_ivf] *= 0.

    # cls entering the weight W of the estimator (eq 20, f_EB in our case)
    # here, primary cls
    # makes almost no difference if using primary, screened, or lensed Cls here (for SO-SAT)
    cls_weight = {}
    for k in ['tt', 'ee', 'bb', 'te']:
        cls_weight[k] = np.copy(primary_cls_dict[k])
    for k in ['eb', 'tb']:
        cls_weight[k] = np.zeros(primary_cls.shape[0])  # zero in standard physics

    # Spectra of the inverse-variance filtered maps
    # In general cls_ivfs = fal * dat_cls * fal^t, with a matrix product in T, E, B space
    # here, dat cls = screened primary + lensed + noise
    cls_ivf = {}
    for k1 in ['t', 'e', 'b']:
        nlev1 = nlev_t if k1 == 't' else nlev_p
        for k2 in ['t', 'e', 'b']:
            cls_ivf[k1+k2] = np.copy(screened_cls_from_maps[k1+k2][:lmax_ivf + 1])
            if k1 == k2:
                # the cross-correlation removes the noise
                cls_ivf[k1+k2] += (nlev1 / 60. / 180. * np.pi / cl_beam) ** 2
            cls_ivf[k1+k2] *= ftl[k1] * ftl[k2]

    # equation 22, normalisation of the QE
    resp = qresp.get_response(
        qe_key=key,
        lmax_ivf=lmax_ivf,
        # modulation field
        source=key[0],
        # cls entering the weight W of the estimator (eq 20, f_EB in our case)
        # here, primary cls
        cls_weight=cls_weight,
        # cls entering f_EB (eq 18, identical to cls_weight in our case)
        # here, primary cls
        cls_cmb=cls_weight,
        # 1/cls used in inverse-variance filter (eq 22)
        # here, observed cls (screened + lensed + noise)
        fal=ftl,
        lmax_qlm=lmax_ivf,
    )[0]
    # Estimator normalization is the inverse response:
    qnorm = plancklens_utils.cli(resp)
    # Noise bias
    N0 = nhl.get_nhl(
        key, key,
        # cls entering the weight W of the estimator (eq 20, f_EB in our case)
        # here, primary cls
        cls_weights=cls_weight,
        # Spectra of the inverse-variance filtered maps
        # In general cls_ivfs = fal * dat_cls * fal^t, with a matrix product in T, E, B space
        cls_ivfs={'ee': ftl['e'], 'bb': ftl['b']},
        lmax_ivf1=lmax_ivf,
        lmax_ivf2=lmax_ivf
    )[0]
    ell = np.arange(min(N0.size, qnorm.size))
    noise_out = N0[ell] * qnorm[ell]**2

    if is_cl:
        return noise_out
    else:
        return noise_out * ell * (ell+1.) / 2./np.pi
