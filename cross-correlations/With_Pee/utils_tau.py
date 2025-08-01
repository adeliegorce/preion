import os
import glob
import healpy as hp
import numpy as np
from os.path import join as opj
import warnings
from scipy.interpolate import interp1d

from plancklens import utils
from plancklens import qresp, nhl


from parameters import telescope_specs
from utils import *


def tau_noise(telescope, cls, is_cl=False, key='f_eb', overwrite=True, folder='/data/glx-calcul3/data1/agorce/cross-correlations/cross-correlations/With_Pee/'):

    assert np.shape(cls)[-1] == 6

    CMB_Cells = cls[:, :-1]  # l range, TT, EE, BB, TE
    tau_ps = cls[:, -1]

    # telescope specs
    tel = telescope_specs[telescope]
    nlev_t = tel['noise']  # Temperature noise level in muK.arcmin
    nlev_p = nlev_t*np.sqrt(2.)  # Polarisation noeise level in muK.arcmin
    beam_fwhm_amin = tel['fwhm']  # Full width at half maximum of the gaussian beam of our instrument, in arcmin
    lmin_ivf = int(tel['lmin'])
    if tel['lmax'] + 1 > CMB_Cells.shape[0]:
        warnings.warn('Input Cls do not go up to telescope lmax...')
    lmax_ivf = min(int(tel['lmax']), CMB_Cells.shape[0]-1)
    assert lmax_ivf > lmin_ivf
    nside = int(2**np.ceil(np.log(lmax_ivf)/np.log(2)))
    if nside > 4096:
        warnings.warn(
            'Too large nside ({nside}) required by lmax={lmax_ivf}... '
            'changing back to healpy max of 8192.'
        )
        nside = 8192

    primary_cls = {}
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        primary_cls[k] = np.copy(CMB_Cells[:, i+1])
    primary_cls['et'] = np.copy(primary_cls['te'])
    for k in ['eb', 'tb', 'be', 'bt']:
        primary_cls[k] = np.zeros_like(CMB_Cells[:, 0])  # zero in standard physics

    # Simulate maps from Cls
    primary_t_map, primary_q_map, primary_u_map = hp.synfast(
        cls=CMB_Cells[:, 1:].T,
        nside=nside,
        pol=True,
        new=True
    )
    tau_map = hp.synfast(tau_ps, nside=nside,)
    scattered_maps = {
        't': primary_t_map * np.exp(-tau_map),
        'q': primary_q_map * np.exp(-tau_map),
        'u': primary_u_map * np.exp(-tau_map)
    }
    out = hp.anafast(
        [scattered_maps[k] for k in ['t', 'q', 'u']],
        lmax=lmax_ivf, pol=True
    )
    scattered_cls_from_maps = {
        'tt': out[0], 'ee': out[1], 'bb': out[2],
        'te': out[3], 'eb': out[4], 'tb': out[5]
    }

    # Noise spectrum
    cl_beam = hp.gauss_beam(fwhm=beam_fwhm_amin*np.pi/180/60, lmax=lmax_ivf)
    # Gaussian beam

    # Estimate the lensing potential field
    ftl = {}
    for i, k in enumerate(['t', 'e', 'b']):
        if k == 't':
            ftl[k] = utils.cli(CMB_Cells[:, i+1][:lmax_ivf + 1] + (nlev_t / 60. / 180. * np.pi / cl_beam) ** 2)
        else:
            ftl[k] = utils.cli(CMB_Cells[:, i+1][:lmax_ivf + 1] + (nlev_p / 60. / 180. * np.pi / cl_beam) ** 2)
        ftl[k][:lmin_ivf] *= 0.
    # This is the inverse variance filter

    # for lensing, these are the lensed ps, so here we take the ps of the screened maps (could also use analytical derivation with preion)
    cls_weight = {}
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls_weight[k] = np.copy(CMB_Cells[:, i+1])
    for k in ['eb', 'tb']:
        cls_weight[k] = np.zeros(CMB_Cells.shape[0])  # zero in standard physics

    cls_ivf = {}
    for k1 in ['t', 'e', 'b']:
        nlev1 = nlev_t if k1 == 't' else nlev_p
        for k2 in ['t', 'e', 'b']:
            cls_ivf[k1+k2] = np.copy(primary_cls[k1+k2][:lmax_ivf + 1] )
            if k1 == k2:
                # the cross-correlation removes the noise
                cls_ivf[k1+k2] += (nlev1 / 60. / 180. * np.pi / cl_beam) ** 2
            cls_ivf[k1+k2] *= ftl[k1] * ftl[k2]

    if overwrite:
        tempfiles = glob.glob(f'{folder}/tau_snr/qresp_tau_{key}/*')
        for file in tempfiles:
            os.remove(file)
    # The function below performs the quadratic estimation.
    qresp_dd = qresp.resp_lib_simple(
        os.path.join(opj('./', 'tau_snr'), f'qresp_tau_{key}'),
        lmax_ivf, cls_weight, scattered_cls_from_maps,
        ftl, lmax_ivf
    )
    # Lensing response according to the fiducial cosmology:
    resp = qresp_dd.get_response(key, key[0])
    # Estimator normalization is the inverse response:
    qnorm = utils.cli(resp)
    # Noise bias
    N0s = nhl.get_nhl(key, key, cls_weight, cls_ivf, lmax_ivf, lmax_ivf)
    ell = np.arange(min(N0s[0].size, qnorm.size))
    noise = N0s[0][ell] * qnorm[ell]**2

    if is_cl:
        return noise
    else:
        return noise * ell * (ell+1.) / 2./np.pi
