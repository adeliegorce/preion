import logging
import os

import numpy as np
from astropy import cosmology, units
from scipy.interpolate import interp1d

from ..theory import Pee_model
from .utils import (
    sample_var, noise, get_lbins, tau_noise, get_cls_for_tau_noise,
    get_sensitivity, get_cl21_noise, get_fsky_21cm,
)
from ..parameters import telescope_specs

logger = logging.getLogger(__name__)


def _build_fiducial_model(theta, cos, use_ksz_emulator):
    """Build the fiducial Pee_model (with a fresh CAMB run) from
    theta = [zre_h, dz_h, alpha0, kappa]. Shared by make_autos_datapoints
    and make_cross_datapoints."""
    return Pee_model(
        zre_h=theta[0], dz_h=theta[1],
        alpha0=theta[2], kappa=theta[3],
        h=cos.h, Ob_0=cos.Ob0, Om_0=cos.Om0,
        verbose=False, run_camb=True,
        use_ksz_emulator=use_ksz_emulator)


def _tautau_signal_and_noise(preion_model, tel_tau, ells):
    """The tau-tau signal Dl (both patchy+late-time) and its telescope
    reconstruction-noise curve at `ells`, for `tel_tau`. Returns
    (tautau_model, nl_tautau) -- both outputs, hence the name. Shared by
    make_autos_datapoints (its telescope-limited branch) and
    make_cross_datapoints (which always needs this piece for its
    cross-covariance)."""
    tautau = preion_model.get_tau(ells=ells, signal='both', Dells=True).sum(axis=1)
    primary_cls, lensed_cls, tau_cls = get_cls_for_tau_noise(preion_model)
    tau_noise_res = tau_noise(tel_tau, primary_cls, lensed_cls, tau_cls, is_cl=False)
    nl_tautau = interp1d(np.arange(tau_noise_res.size), tau_noise_res)(ells)
    return tautau, nl_tautau


def make_autos_datapoints(
        theta, telescopes, lbin_edges=None, ells=None,
        use_ksz_emulator=False, randomness=False,
        cos=cosmology.Planck18, save=None):

    assert np.size(theta) >= 4
    use_ells = False if ells is None else True
    if telescopes is not None:
        tel_tau, tel_ksz, tel_bb = telescopes
        given_ells = ells  # preserve the caller-supplied grids -- `ells` is reused as the
                           # accumulator below, so reading `ells[i]` there would otherwise
                           # index into the (empty, still-being-built) accumulator, not the
                           # original argument (a pre-existing bug, dormant until a caller
                           # actually passes both `telescopes` and `ells` together)
        ells = []
        dells = []
        for i, tel in enumerate(telescopes):
            assert tel in telescope_specs.keys()
            if not use_ells:
                ls, _, dls = get_lbins(tel, lbin_edges=lbin_edges[i] if lbin_edges is not None else None)
            else:
                ls = np.copy(given_ells[i])
                dls = np.diff(given_ells[i]).mean()
            ells.append(ls)
            dells.append(dls)
    else:
        assert len(ells) == 3
        dells = [np.diff(ells[i]).mean() for i in range(3)]
    ells_tau, ells_ksz, ells_bb = ells
    dells_tau, dells_ksz, dells_bb = dells

    preion_model = _build_fiducial_model(theta, cos, use_ksz_emulator)
    ksz_ps = preion_model.get_ksz(ells=ells_ksz, signal='patchy', Dells=True)[:, 0]
    total_bb = preion_model.get_B_modes(ells=ells_bb, Dells=True)

    if telescopes is not None:
        tau_ps, nl_tautau = _tautau_signal_and_noise(preion_model, tel_tau, ells_tau)

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
        cov_tau = np.diag(
            1./(2. * ells_tau+1.) / telescope_specs[tel_tau]['fsky'] * 2. * (tau_ps+nl_tautau)**2
            / dells_tau
        )

    else:
        tau_ps = preion_model.get_tau(ells=ells_tau, signal='both', Dells=True).sum(axis=1)
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


def load_autos_datapoints(data_dir, label, ells=None):
    """Load tau/kSZ/BB datapoints + covariances for `label` from `data_dir`
    (as written by make_autos_datapoints(..., save=label)). If `ells` (a
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
_PACKAGED_LABEL = "mcmc_tutorial_autos"
_PACKAGED_CROSS_LABEL = "mcmc_tutorial_cross"
_PACKAGED_CROSS_Z21 = [7.0]


def make_cross_datapoints(
        theta, tel_tau, telescope_21, sensitivity_case, z21,
        ska_array=None, n_fields=None, delta_nu=0., lbin_edges=None, ells=None,
        sensitivity_dir=None, use_ksz_emulator=False, randomness=False,
        cos=cosmology.Planck18, save=None):
    """Build mock tau x 21cm cross-spectrum data (+ the auxiliary tau-tau
    and 21cm auto-spectra needed for its noise), for one or more 21cm
    redshifts z21.

    Mirrors make_autos_datapoints's `ells`-vs-`telescope` priority exactly,
    for a single telescope instead of three: if `ells` is given, it's used
    as-is (with per-bin width approximated by the mean spacing of its
    entries -- exact for the uniform arange/linspace grids the config
    schema can produce, same convention as make_autos_datapoints); else the
    grid is derived from `get_lbins(tel_tau, lbin_edges=lbin_edges)`.

    Returns a dict:
      "ells": 1D array of bin-centre multipoles (shared across z21 and the
              tau-tau/21cm-auto spectra -- there is only one telescope here).
      "tau21": (nz, nell) array, the tau x 21cm cross Dl [uK].
      "cov_tau21": (nz, nell, nell) array, its diagonal covariance matrix
              per z21 (may contain real `inf` entries where the 21cm survey
              has no k-coverage; see mcmc.invert_covariance for how those
              are excluded from the likelihood).
      "tautau": (nell,) array, the tau-tau Dl (noiseless model).
      "cl21": (nz, nell) array, the 21cm auto Dl (noiseless model).
      "tau": float, the fiducial model's integrated optical depth.
    """
    assert np.size(theta) >= 4
    if ells is None:
        ells, lbin_edges, dells = get_lbins(tel_tau, lbin_edges=lbin_edges)
    else:
        ells = np.copy(ells)
        dells = np.diff(ells).mean()

    preion_model = _build_fiducial_model(theta, cos, use_ksz_emulator)
    tautau, nl_tautau = _tautau_signal_and_noise(preion_model, tel_tau, ells)

    fsky = get_fsky_21cm(telescope_21, n_fields=n_fields)
    sensitivity = get_sensitivity(
        telescope_21, sensitivity_case, ska_array=ska_array, n_fields=n_fields,
        error_type='thermal', sensitivity_dir=sensitivity_dir)

    z21 = list(z21)
    nz = len(z21)
    nell = ells.size
    tau21 = np.zeros((nz, nell))
    cl21 = np.zeros((nz, nell))
    cov_tau21 = np.zeros((nz, nell, nell))
    for iz, z in enumerate(z21):
        cl21_z = preion_model.get_cl21(z, ells, Dells=True, delta_nu=delta_nu * units.MHz).to(units.uK**2).value
        nl_21_z = get_cl21_noise(sensitivity, z, ells, Dells=True, delta_nu=delta_nu).to(units.uK**2).value
        tau21_z = preion_model.get_tau_21_cross(z, ells, Dells=True, delta_nu=delta_nu).to(units.uK).value
        var_tau21_z = (
            1. / (2. * ells + 1.) / fsky / dells
            * (tau21_z**2 + (cl21_z + nl_21_z) * (tautau + nl_tautau))
        )
        tau21[iz] = tau21_z
        cl21[iz] = cl21_z
        cov_tau21[iz] = np.diag(var_tau21_z)

    if randomness:
        for iz in range(nz):
            tau21[iz] = np.random.normal(tau21[iz], np.sqrt(np.diag(cov_tau21[iz])))

    if (save is not None) and (not os.path.exists(f'data/{str(save)}_tautau_datapoints.txt')):
        np.savetxt(f'data/{str(save)}_tautau_datapoints.txt', np.c_[ells, tautau, nl_tautau],
                   header='ell, Dl_tautau, Nl_tautau (noise only)')
        np.savetxt(f'data/{str(save)}_tau.txt', [preion_model.tau], header='fiducial tau')
        for iz, z in enumerate(z21):
            np.savetxt(f'data/{str(save)}_tau21_z{z:.1f}_datapoints.txt',
                       np.c_[ells, tau21[iz], np.sqrt(np.diag(cov_tau21[iz]))],
                       header='ell, Dl [uK], err [uK]')
            np.savetxt(f'data/{str(save)}_cl21_z{z:.1f}_datapoints.txt', np.c_[ells, cl21[iz]],
                       header='ell, Dl_21 [uK2]')

    return {
        "ells": ells, "tau21": tau21, "cov_tau21": cov_tau21,
        "tautau": tautau, "cl21": cl21, "tau": preion_model.tau,
    }


def load_cross_datapoints(data_dir, label, z21, ells=None):
    """Load tau x 21cm cross datapoints + auxiliary tau-tau/21cm-auto
    spectra for `label` from `data_dir` (as written by
    make_cross_datapoints(..., save=label)). Mirrors
    load_autos_datapoints's minimal signature -- like that function, this
    only needs `label` (plus `z21`, to know how many per-z files to expect)
    to reconstruct the on-disk paths, not the telescope/survey details that
    produced them.

    Returns the same dict shape as make_cross_datapoints.
    """
    z21 = list(z21)
    ells_tautau, tautau, nl_tautau = np.loadtxt(
        os.path.join(data_dir, f"{label}_tautau_datapoints.txt"), unpack=True)
    tau = float(np.loadtxt(os.path.join(data_dir, f"{label}_tau.txt")))

    tau21_list, cov_tau21_list, cl21_list = [], [], []
    for z in z21:
        ells_z, tau21_z, err_z = np.loadtxt(
            os.path.join(data_dir, f"{label}_tau21_z{z:.1f}_datapoints.txt"), unpack=True)
        ells_cl21_z, cl21_z = np.loadtxt(
            os.path.join(data_dir, f"{label}_cl21_z{z:.1f}_datapoints.txt"), unpack=True)
        if ells is not None:
            assert np.allclose(ells, ells_z), f"ells do not match for z21={z}!"
        tau21_list.append(tau21_z)
        cov_tau21_list.append(np.diag(err_z**2))
        cl21_list.append(cl21_z)

    return {
        "ells": ells_tautau,
        "tau21": np.array(tau21_list), "cov_tau21": np.array(cov_tau21_list),
        "tautau": tautau, "cl21": np.array(cl21_list), "tau": tau,
    }
