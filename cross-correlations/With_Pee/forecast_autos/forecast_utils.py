import sys, os
import numpy as np
from astropy import cosmology
import warnings

from theory import Pee_model
from utils import sample_var, noise
from parameters import telescope_specs

theta_labels = [r'$z_\mathrm{re}$', r'd$z$', r'log$\alpha_0$', r'$\kappa$']
prior_bounds = [(5.5, 10.), (0., 4.), (0, 6.), (0, 0.4)]

def dobs_dz(theta, idz, lrange, low_lrange=None, dev=0.05, cos=cosmology.Planck18):

    lrange = np.atleast_1d(lrange)
    if low_lrange is None:
        low_lrange = np.copy(lrange)
    assert np.size(theta) == 4
    assert (idz >= 0) and (idz < np.size(theta))
    assert dev < 1

    # get bounds
    theta_max = np.copy(theta)
    theta_max[idz] = theta[idz]*(1+dev)
    preion_model_max = Pee_model(
        zre_h=theta_max[0],
        dz_h=theta_max[1],
        alpha0=theta_max[2],
        kappa=theta_max[3],
        h=cos.h,
        Ob_0=cos.Ob0,
        Om_0=cos.Om0,
        verbose=False, run_camb=True)
    ksz_ps_max = preion_model_max.get_ksz(ells=lrange, signal='both', Dells=True)[:, 0]
    tau_ps_max = preion_model_max.get_tau(ells=lrange, signal='both', Dells=True)[:, 0]
    scattering_bb = preion_model_max.get_scattering_B_modes(ells=low_lrange, signal='both', Dells=True)
    screening_bb = preion_model_max.get_screening_B_modes(ells=low_lrange, Dells=True)
    total_bb_max = np.sum(scattering_bb, axis=1)+screening_bb

    theta_min = np.copy(theta)
    theta_min[idz] = theta[ia0]*(1-dev)
    preion_model_min = Pee_model(
        zre_h=theta_min[0],
        a0_h=theta_min[1],
        alpha0=theta_min[2],
        kappa=theta_min[3],
        h=cos.h,
        Ob_0=cos.Ob0,
        Om_0=cos.Om0,
        verbose=False, run_camb=True)
    ksz_ps_min = preion_model_min.get_ksz(ells=lrange, signal='both', Dells=True)[:, 0]
    tau_ps_min = preion_model_min.get_tau(ells=lrange, signal='both', Dells=True)[:, 0]
    scattering_bb = preion_model_min.get_scattering_B_modes(ells=low_lrange, signal='both', Dells=True)
    screening_bb = preion_model_min.get_screening_B_modes(ells=low_lrange, Dells=True)
    total_bb_min = np.sum(scattering_bb, axis=1)+screening_bb

    deriv_ksz = (ksz_ps_max-ksz_ps_min) / (theta_max[ia0]-theta_min[idz])
    deriv_tau = (tau_ps_max-tau_ps_min) / (theta_max[idz]-theta_min[idz])
    deriv_bb = (total_bb_max-total_bb_min) / (theta_max[idz]-theta_min[idz])

    return deriv_ksz, deriv_tau, deriv_bb


def dtau_dz(theta, idz, dev=0.05, cos=cosmology.Planck18):

    # get bounds
    theta_max = np.copy(theta)
    theta_max[idz] = theta[idz]*(1+dev)
    preion_model_max = Pee_model(
        zre_h=theta_max[0],
        dz_h=theta_max[1],
        alpha0=theta_max[2],
        kappa=theta_max[3],
        h=cos.h,
        Ob_0=cos.Ob0,
        Om_0=cos.Om0,
        verbose=False,
        run_camb=False)
    tau_max = preion_model_max.tau

    theta_min = np.copy(theta)
    theta_min[idz] = theta[idz]*(1-dev)
    preion_model_min = Pee_model(
        zre_h=theta_min[0],
        dz_h=theta_min[1],
        alpha0=theta_min[2],
        kappa=theta_min[3],
        h=cos.h,
        Ob_0=cos.Ob0,
        Om_0=cos.Om0,
        verbose=False,
        run_camb=False)
    tau_min = preion_model_min.tau

    return (tau_max-tau_min) / (theta_max[idz]-theta_min[idz])


def make_datapoints(
        theta, ells, tel=telescope_specs['CMB-HD'], 
        low_ells=None, tel_lowl=None,
        cos=cosmology.Planck18, save=None):

    ells = np.atleast_1d(ells)
    if low_ells is None:
        low_ells = np.copy(ells)
        tel_lowl = tel.copy()
    assert np.size(theta) >= 4

    preion_model = Pee_model(
        zre_h=theta[0], dz_h=theta[1],
        alpha0=theta[2], kappa=theta[3],
        h=cos.h, Ob_0=cos.Ob0, Om_0=cos.Om0,
        verbose=False, run_camb=True)
    ksz_ps = preion_model.get_ksz(ells=ells, signal='patchy', Dells=True)[:, 0]
    tau_ps = preion_model.get_tau(ells=ells, signal='patchy', Dells=True)[:, 0]
    scattering_bb = preion_model.get_scattering_B_modes(ells=low_ells, signal='both', Dells=True)
    screening_bb = preion_model.get_screening_B_modes(ells=low_ells, Dells=True)
    total_bb = np.sum(scattering_bb, axis=1)+screening_bb

    cov_tau = np.diag( sample_var(ells, tau_ps[:, 0], tel)**2 \
                      + (noise(ells, tel, pol=True)/np.sqrt(np.diff(ells).mean()))**2 )
    cov_ksz = np.diag( sample_var(ells, ksz_ps[:, 0], tel)**2 \
                      + (noise(ells, tel, pol=False)/np.sqrt(np.diff(ells).mean()))**2 )
    cov_bb = np.diag( sample_var(low_ells, total_bb, tel_lowl)**2 \
                     + (noise(low_ells, tel_lowl, pol=True)/np.sqrt(np.diff(low_ells).mean()))**2 )

    if save is not None:
        np.savetxt(f'data/{str(save)}_lowell_datapoints.txt', np.c_[low_ells, total_bb], header='ell, BB total [uK2]')
        np.savetxt(f'data/{str(save)}_highell_datapoints.txt', np.c_[ells, ksz_ps, tau_ps], header='ell, ksz [uK2], tautau')
        np.savetxt(f'data/{str(save)}_cov_ksz.txt', cov_ksz)
        np.savetxt(f'data/{str(save)}_cov_tau.txt', cov_tau)
        np.savetxt(f'data/{str(save)}_cov_bb.txt', cov_bb)
    else:
        return ksz_ps, tau_ps, total_bb, cov_ksz, cov_tau, cov_bb


def get_derivatives(theta_true, ells, low_ells=None, run_derivatives=False, dev=0.05):

    ells = np.atleast_1d(ells)
    if low_ells is None:
        low_ells = np.copy(ells)
    assert np.size(theta_true) >= 4
    run = bool(run_derivatives)

    deriv_files = ['data/deriv_zre.txt', 'data/deriv_dz.txt', 'data/deriv_a0.txt', 'data/deriv_kappa.txt']
    deriv_list = []
    for idz, deriv_file in enumerate(deriv_files):
        if not run and os.path.exists(deriv_file):
            deriv = np.loadtxt(deriv_file, usecols=(1, 2, 3)).T  # shape [3, len(ells)]
            if ells.size != deriv.shape[1]:
                warnings.warn('Pre-saved derivatives are not compatible with ell range. Re-computing...')
                run = True
        if run:
            deriv = np.array([dobs_dz(theta_true, idz=idz, dev=dev, lrange=ells, low_lrange=low_ells)])
        deriv_list.append(deriv)

    return deriv_list


def individual_forecast(
        theta_true, ells, low_ells=None,
        tel=telescope_specs['CMB-HD'], cos=cosmology.Planck18,
        run_derivatives=False, dev=0.05
    ):

    # mock data and covariance
    _, _, _, cov_ksz, cov_tau, cov_bb = make_datapoints(
        theta_true, ells, low_ells, tel, cos, save=False)
    icov_list = [np.linalg.inv(cov_ksz), np.linalg.inv(cov_tau), np.linalg.inv(cov_bb)]

    # derivatives
    deriv_list = get_derivatives(theta_true, ells, low_ells, run_derivatives, dev)
    # deriv_ksz = [d[0] for d in deriv_list]

    # fisher
    fishcov_list = []
    for io, obs in enumerate(['kSZ', 'tau', 'BB']):
        fishmat = np.zeros((len(theta_true), len(theta_true)))
        for i, deriv in enumerate([d[io] for d in deriv_list]):
            for j, deriv2 in enumerate([d[io] for d in deriv_list]):
                fishmat[i, j] = np.transpose(deriv).dot(icov_list[io]).dot(deriv2)
        fishcov = np.linalg.inv(fishmat)
        assert np.allclose(np.dot(fishmat, fishcov), np.eye(len(theta_true)))
        fishcov_list.append(fishcov)

    return fishcov_list


def combined_forecast(
        theta_true, ells, low_ells=None,
        index_list=[0, 1, 2],
        tel=telescope_specs['CMB-HD'], cos=cosmology.Planck18,
        run_derivatives=False, dev=0.05
    ):

    titles = ['kSZ', 'tau', 'BB']
    print('Results for combining ')
    for i in index_list:
        print(titles[i])

    # mock data and covariance
    _, _, _, cov_ksz, cov_tau, cov_bb = make_datapoints(
        theta_true, ells, low_ells, tel, cos, save=False)
    cov_list = [[cov_ksz, cov_tau, cov_bb][i] for i in index_list]

    cov_tot = np.diag( np.hstack([np.diag(cov) for cov in cov_list[index_list]]) )
    icov_tot = np.linalg.inv(cov_tot)

    # derivatives
    deriv_list_temp = get_derivatives(theta_true, ells, low_ells, run_derivatives, dev)
    # remove obs not used
    deriv_list = []
    for l in deriv_list_temp:
        deriv_list.append([l[i] for i in index_list])
        # l = [deriv_tau_zre, deriv_ksz_zre, deriv_bb_zre]

    fish_all = np.zeros((len(theta_true), len(theta_true)))
    for i, deriv in enumerate([np.r_[l] for l in deriv_list]):
        for j, deriv2 in enumerate([np.r_[l] for l in deriv_list]):
            fish_all[i, j] = np.transpose(deriv).dot(icov_tot).dot(deriv2)
    fishcov_all = np.linalg.inv(fish_all)

    return fishcov_all
