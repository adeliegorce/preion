import copy
import warnings

import healpy as hp
import numpy as np
from astropy import cosmology, units
from matplotlib import rc, colormaps, colors
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from plancklens import utils as plancklens_utils, qresp, nhl

from ..parameters import telescope_specs


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
