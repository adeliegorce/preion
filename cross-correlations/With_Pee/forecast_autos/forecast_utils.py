import sys, os
import numpy as np
from astropy import cosmology
import warnings
from copy import copy
from scipy.interpolate import interp1d

from theory import Pee_model
from utils import sample_var, noise, get_lbins
from utils_tau import tau_noise, get_cls_for_tau_noise

from parameters import telescope_specs

theta_labels = [r'$z_\mathrm{re}$', r'd$z$', r'log$\alpha_0$', r'$\kappa$']
prior_bounds = [(5.5, 10.), (0., 4.), (0, 6.), (0, 0.4)]


def dobs_dz(theta, idz, ells, dev=0.05, cos=cosmology.Planck18, use_ksz_emulator=False):

    assert len(ells) == 3
    ells_tau = ells[0]
    ells_ksz = ells[1]
    ells_bb = ells[2]
    # lrange = np.atleast_1d(lrange)
    # if low_lrange is None:
    #     low_lrange = np.copy(lrange)
    assert np.size(theta) == 4
    assert (idz >= 0) and (idz < np.size(theta))
    assert dev < 1

    preion_model = Pee_model(
        zre_h=theta[0],
        dz_h=theta[1],
        alpha0=theta[2],
        kappa=theta[3],
        h=cos.h,
        Ob_0=cos.Ob0,
        Om_0=cos.Om0,
        verbose=False, run_camb=True,
        use_ksz_emulator=use_ksz_emulator)

    # get bounds
    theta_max = np.copy(theta)
    theta_max[idz] = theta[idz]*(1+dev)
    preion_model_max = copy(preion_model)
    preion_model_max.zre_h = theta_max[0]
    preion_model_max.dz_h = theta_max[1]
    preion_model_max.alpha0 = theta_max[2]
    preion_model_max.kappa = theta_max[3]
    preion_model_max.init_reionisation_history()
        #     Pee_model(
        # zre_h=theta_max[0],
        # dz_h=theta_max[1],
        # alpha0=theta_max[2],
        # kappa=theta_max[3],
        # h=cos.h,
        # Ob_0=cos.Ob0,
        # Om_0=cos.Om0,
        # verbose=False, run_camb=True,
        # use_ksz_emulator=use_ksz_emulator)
    ksz_ps_max = preion_model_max.get_ksz(ells=ells_ksz, signal='both', Dells=True)[:, 0]
    tau_ps_max = preion_model_max.get_tau(ells=ells_tau, signal='both', Dells=True)[:, 0]
    scattering_bb = preion_model_max.get_scattering_B_modes(ells=ells_bb, signal='both', Dells=True)
    screening_bb = preion_model_max.get_screening_B_modes(ells=ells_bb, Dells=True)
    total_bb_max = np.sum(scattering_bb, axis=1)+screening_bb

    theta_min = np.copy(theta)
    theta_min[idz] = theta[idz]*(1-dev)
    # preion_model_min = Pee_model(
    #     zre_h=theta_min[0],
    #     a0_h=theta_min[1],
    #     alpha0=theta_min[2],
    #     kappa=theta_min[3],
    #     h=cos.h,
    #     Ob_0=cos.Ob0,
    #     Om_0=cos.Om0,
    #     verbose=False, run_camb=True)
    preion_model_min = copy(preion_model)
    preion_model_min.zre_h = theta_min[0]
    preion_model_min.dz_h = theta_min[1]
    preion_model_min.alpha0 = theta_min[2]
    preion_model_min.kappa = theta_min[3]
    preion_model_min.init_reionisation_history()
    ksz_ps_min = preion_model_min.get_ksz(ells=ells_ksz, signal='both', Dells=True)[:, 0]
    tau_ps_min = preion_model_min.get_tau(ells=ells_tau, signal='both', Dells=True)[:, 0]
    scattering_bb = preion_model_min.get_scattering_B_modes(ells=ells_bb, signal='both', Dells=True)
    screening_bb = preion_model_min.get_screening_B_modes(ells=ells_bb, Dells=True)
    total_bb_min = np.sum(scattering_bb, axis=1)+screening_bb

    deriv_ksz = (ksz_ps_max-ksz_ps_min) / (theta_max[idz]-theta_min[idz])
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
        theta, telescopes, lbin_edges=None,
        use_ksz_emulator=False, randomness=False,
        cos=cosmology.Planck18, save=None):

    assert np.size(theta) >= 4
    if telescopes is not None:
        tel_tau, tel_ksz, tel_bb = telescopes
        ells = []
        dells = []
        for i, tel in enumerate(telescopes):
            assert tel in telescope_specs.keys()
            ls, _, dls = get_lbins(tel, lbin_edges=lbin_edges[i] if lbin_edges is not None else None)
            ells.append(ls)
            dells.append(dls)
    ells_tau, ells_ksz, ells_bb = ells
    dells_tau, dells_ksz, dells_bb = dells

    preion_model = Pee_model(
        zre_h=theta[0], dz_h=theta[1],
        alpha0=theta[2], kappa=theta[3],
        h=cos.h, Ob_0=cos.Ob0, Om_0=cos.Om0,
        verbose=False, run_camb=True,
        use_ksz_emulator=use_ksz_emulator)
    ksz_ps = preion_model.get_ksz(ells=ells_ksz, signal='patchy', Dells=True)[:, 0]
    tau_ps = preion_model.get_tau(ells=ells_tau, signal='both', Dells=True).sum(axis=1)
    total_bb = preion_model.get_B_modes(ells=ells_bb, Dells=True, Qrms=17.0)

    if telescopes is not None:

        ksz_obs = ksz_ps+noise(ells_ksz, telescope_specs[tel_ksz], pol=False, is_cl=False)
        cov_ksz = np.diag(
            1./(2. * ells_ksz+1.) / telescope_specs[tel_ksz]['fsky'] * 2. * ksz_obs**2
            / dells_ksz
        )
        if use_ksz_emulator == 'RF':
            print(f'Adding emulator reconstruction errors for {use_ksz_emulator}')
            cov_ksz += np.diag(np.ones_like(ells_ksz) * 0.01**2) # emulator error (~ constant with multipole)
        cov_bb = np.diag(
            1./(2. * ells_bb+1.) / telescope_specs[tel_bb]['fsky'] * 2. * (total_bb+noise(ells_bb, telescope_specs[tel_bb], pol=True, is_cl=False))**2            
            / dells_bb
        )
        primary_cls2, lensed_cls2, tau_cls2 = get_cls_for_tau_noise(preion_model)
        tau_noise_res = tau_noise(tel_tau, primary_cls2, lensed_cls2, tau_cls2, is_cl=False)
        cov_tau = np.diag(
            1./(2. * ells_tau+1.) / telescope_specs[tel_tau]['fsky'] * 2. * (tau_ps+interp1d(np.arange(tau_noise_res.size), tau_noise_res)(ells_tau))**2
            / dells_tau
        )

    else:
        cov_tau = np.diag(np.ones_like(tau_ps))
        cov_ksz = np.diag(np.ones_like(ksz_ps))
        cov_bb = np.diag(np.ones_like(total_bb))

    if randomness:
        ksz_ps = np.random.normal(ksz_ps, np.sqrt(np.diag(cov_ksz)))
        total_bb = np.random.normal(total_bb, np.sqrt(np.diag(cov_bb)))
        tau_ps = np.random.normal(tau_ps, np.sqrt(np.diag(cov_tau)))

    if (save is not None) and (not os.path.exists(f'data/{str(save)}_bb_datapoints.txt')):
        np.savetxt(f'data/{str(save)}_bb_datapoints.txt', np.c_[ells_bb, total_bb], header='ell, Dl BB total [uK2]')
        np.savetxt(f'data/{str(save)}_ksz_datapoints.txt', np.c_[ells_ksz, ksz_ps], header='ell, Dl kSZ [uK2]')
        np.savetxt(f'data/{str(save)}_tau_datapoints.txt', np.c_[ells_tau, tau_ps], header='ell, Dl tautau')
        np.savetxt(f'data/{str(save)}_cov_ksz.txt', cov_ksz)
        np.savetxt(f'data/{str(save)}_cov_tau.txt', cov_tau)
        np.savetxt(f'data/{str(save)}_cov_bb.txt', cov_bb)

    return tau_ps, ksz_ps, total_bb, cov_tau, cov_ksz, cov_bb


def get_derivatives(theta_true, ells, run_derivatives=False, dev=0.05, use_ksz_emulator=False, verbose=False):

    # ells = np.atleast_1d(ells)
    assert len(ells) == 3
    ells_tau = ells[0]
    ells_ksz = ells[1]
    ells_bb = ells[2]
    # if low_ells is None:
    #     low_ells = np.copy(ells)
    assert np.size(theta_true) >= 4
    run = bool(run_derivatives)

    deriv_files = ['data/deriv_zre.txt', 'data/deriv_dz.txt', 'data/deriv_a0.txt', 'data/deriv_kappa.txt']
    deriv_list = []
    for idz, deriv_file in enumerate(deriv_files):
        print(idz)
        if not run and os.path.exists(deriv_file):
            deriv = np.loadtxt(deriv_file, usecols=(1, 2, 3)).T  # shape [3, len(ells)]
            if ells.size != deriv.shape[1]:
                warnings.warn('Pre-saved derivatives are not compatible with ell range. Re-computing...')
                run = True
        else:
            deriv_list.append(dobs_dz(theta_true, idz=idz, dev=dev, ells=ells, use_ksz_emulator=use_ksz_emulator))

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
