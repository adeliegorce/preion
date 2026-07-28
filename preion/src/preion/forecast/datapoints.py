import logging
import os

import numpy as np
from astropy import cosmology
from scipy.interpolate import interp1d

from ..theory import Pee_model
from .utils import sample_var, noise, get_lbins, tau_noise, get_cls_for_tau_noise
from ..parameters import telescope_specs

logger = logging.getLogger(__name__)


def make_datapoints(
        theta, telescopes, lbin_edges=None, ells=None,
        use_ksz_emulator=False, randomness=False,
        cos=cosmology.Planck18, save=None):

    assert np.size(theta) >= 4
    use_ells = False if ells is None else True
    if telescopes is not None:
        tel_tau, tel_ksz, tel_bb = telescopes
        ells = []
        dells = []
        for i, tel in enumerate(telescopes):
            assert tel in telescope_specs.keys()
            if not use_ells:
                ls, _, dls = get_lbins(tel, lbin_edges=lbin_edges[i] if lbin_edges is not None else None)
            else:
                ls = np.copy(ells[i])
                dls = np.diff(ells[i]).mean()
            ells.append(ls)
            dells.append(dls)
    else:
        assert len(ells) == 3
        dells = [np.diff(ells[i]).mean() for i in range(3)]
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
            logger.info(f'Adding emulator reconstruction errors for {use_ksz_emulator}')
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
        cov_tau = np.diag(sample_var(ells[0], tau_ps, 1.)**2)
        cov_ksz = np.diag(sample_var(ells[1], ksz_ps, 1.)**2)
        if use_ksz_emulator:
            cov_ksz += np.diag(np.ones_like(ells[1]) * 0.01**2) # emulator error (~ constant with multipole)
        cov_bb = np.diag(sample_var(ells[2], total_bb, 1.)**2)

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


def load_datapoints(data_dir, label, ells=None):
    """Load tau/kSZ/BB datapoints + covariances for `label` from `data_dir`
    (as written by make_datapoints(..., save=label)). If `ells` (a
    [ells_tau, ells_ksz, ells_bb] list) is given, assert the loaded
    multipole grids match it.

    Returns a dict with keys "tau"/"ksz"/"bb", "cov_tau"/"cov_ksz"/"cov_bb",
    and "ells_tau"/"ells_ksz"/"ells_bb".
    """
    ells_tau, tau_data = np.loadtxt(os.path.join(data_dir, f"{label}_tau_datapoints.txt"), unpack=True)
    ells_ksz, ksz_data = np.loadtxt(os.path.join(data_dir, f"{label}_ksz_datapoints.txt"), unpack=True)
    ells_bb, bb_data = np.loadtxt(os.path.join(data_dir, f"{label}_bb_datapoints.txt"), unpack=True)
    cov_tau = np.loadtxt(os.path.join(data_dir, f"{label}_cov_tau.txt"))
    cov_ksz = np.loadtxt(os.path.join(data_dir, f"{label}_cov_ksz.txt"))
    cov_bb = np.loadtxt(os.path.join(data_dir, f"{label}_cov_bb.txt"))

    if ells is not None:
        assert np.allclose(ells[0], ells_tau), "ells do not match for tau!"
        assert np.allclose(ells[1], ells_ksz), "ells do not match for ksz!"
        assert np.allclose(ells[2], ells_bb), "ells do not match for bb!"
    for loaded_ells, cov in zip([ells_tau, ells_ksz, ells_bb], [cov_tau, cov_ksz, cov_bb]):
        assert cov.shape == (loaded_ells.size, loaded_ells.size), "ells do not match cov!"

    return {
        "tau": tau_data, "ksz": ksz_data, "bb": bb_data,
        "cov_tau": cov_tau, "cov_ksz": cov_ksz, "cov_bb": cov_bb,
        "ells_tau": ells_tau, "ells_ksz": ells_ksz, "ells_bb": ells_bb,
    }


_PACKAGED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_PACKAGED_LABEL = "mcmc_tutorial"


def load_packaged_datapoints():
    """Load the pre-computed cosmic-variance-limited mock tau/kSZ/BB
    datapoints + covariances shipped with the package (generated from the
    defaults in configs/config_tutorial_mcmc.yaml: theta_true=[7.0, 1.5, 3.7, 0.10],
    telescopes=None, use_ksz_emulator=False). Useful for examples/tests that
    just need *some* realistic mock data without re-running CAMB."""
    return load_datapoints(_PACKAGED_DATA_DIR, _PACKAGED_LABEL)
