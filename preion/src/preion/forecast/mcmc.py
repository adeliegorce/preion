import argparse
import contextlib
import logging
import os
import time
from multiprocessing import Pool

import emcee
import numpy as np
from astropy import cosmology, units

from ..parameters import props, telescope_specs
from ..theory import Pee_model
from .config import build_ells, load_config, run_label, setup_logging, CROSS_DATA
from .datapoints import (
    load_autos_datapoints, make_autos_datapoints,
    load_cross_datapoints, make_cross_datapoints,
)
from .utils import get_lbins, invert_covariance, lower_limit, gaussian_prior

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _chdir(path):
    """Temporarily chdir into `path` (creating it if needed) so the relative
    data/backends/figures paths below resolve under cfg['output_dir'], the way
    the original script resolved them under whatever directory it was run from."""
    prev = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_ells(cfg, ells):
    """Turn `build_ells(cfg)`'s result (possibly `None`, meaning "derive from
    telescopes") into concrete multipole-grid array(s), resolving via
    get_lbins per telescope when needed. Returns a [ells_tau, ells_ksz,
    ells_bb] list for an auto config, or a single array for a cross one."""
    telescopes = cfg["telescopes"]
    is_cross = cfg["data"] in CROSS_DATA
    if is_cross:
        if ells is not None:
            return ells[0]
        ells_tau21, _, _ = get_lbins(telescopes[0], lbin_edges=cfg.get("lbin_edges"))
        return ells_tau21
    if ells is not None:
        return ells
    lbin_edges_cfg = cfg.get("lbin_edges")
    resolved = []
    for i, tel in enumerate(telescopes):
        ls, _, _ = get_lbins(tel, lbin_edges=lbin_edges_cfg[i] if lbin_edges_cfg is not None else None)
        resolved.append(ls)
    return resolved


def get_or_make_datapoints(cfg):
    """Get the mock datapoints and covariances for this config, reading them
    back from `{output_dir}/data/` if already present, else generating them
    with `make_autos_datapoints`/`make_cross_datapoints` (and writing them
    there for next time). `cfg` is a dict as returned by
    preion.forecast.config.load_config.

    For an auto (`bb`/`ksz`/`tau`/`all`) config, returns a dict with keys
    "tau"/"ksz"/"bb", "cov_tau"/"cov_ksz"/"cov_bb",
    "ells_tau"/"ells_ksz"/"ells_bb", plus "inv_cov_tau"/"inv_cov_ksz"/
    "inv_cov_bb". For a cross (`tau21`) config, returns the dict shape from
    `make_cross_datapoints`, plus "z21"/"delta_nu" and "inv_cov_tau21".
    """
    setup_logging(cfg)
    cos = cosmology.Planck18
    telescopes = cfg["telescopes"]
    use_ksz_emulator = cfg["use_ksz_emulator"]
    theta_true = list(cfg["theta_true"])
    label1 = cfg["label"]
    data = cfg["data"]
    is_cross = data in CROSS_DATA

    ells = _resolve_ells(cfg, build_ells(cfg))

    with _chdir(cfg["output_dir"]):
        os.makedirs("data", exist_ok=True)
        if is_cross:
            if os.path.exists(f"data/{label1}_tautau_datapoints.txt"):
                logger.info("Reading mock data points from file...")
                datapoints = load_cross_datapoints("data", label1, cfg["z21"], ells=ells)
            else:
                logger.info("Generating mock data points...")
                datapoints = make_cross_datapoints(
                    theta_true, telescopes[0], cfg["telescope_21"], cfg["sensitivity_case"], cfg["z21"],
                    ska_array=cfg.get("ska_array"), n_fields=cfg.get("n_fields"),
                    delta_nu=cfg["delta_nu"], ells=ells,
                    sensitivity_dir=cfg.get("sensitivity_dir"),
                    use_ksz_emulator=use_ksz_emulator, randomness=False,
                    cos=cos, save=label1,
                )
            datapoints["z21"] = list(cfg["z21"])
            datapoints["delta_nu"] = cfg["delta_nu"]
            datapoints["inv_cov_tau21"] = np.array(
                [invert_covariance(datapoints["cov_tau21"][iz]) for iz in range(len(datapoints["z21"]))]
            )
        else:
            if os.path.exists(f"data/{label1}_cov_ksz.txt"):
                logger.info("Reading mock data points from file...")
                datapoints = load_autos_datapoints("data", label1, ells=ells)
            else:
                logger.info("Generating mock data points...")
                tau_data, ksz_data, bb_data, cov_tau, cov_ksz, cov_bb = make_autos_datapoints(
                    theta_true,
                    ells=ells,
                    telescopes=telescopes, randomness=False,
                    cos=cos, save=label1, use_ksz_emulator=use_ksz_emulator,
                )
                datapoints = {
                    "tau": tau_data, "ksz": ksz_data, "bb": bb_data,
                    "cov_tau": cov_tau, "cov_ksz": cov_ksz, "cov_bb": cov_bb,
                    "ells_tau": ells[0], "ells_ksz": ells[1], "ells_bb": ells[2],
                }
            for name in ("tau", "ksz", "bb"):
                datapoints[f"inv_cov_{name}"] = invert_covariance(datapoints[f"cov_{name}"])
        logger.info("Done.")

    return datapoints


def plot_autos_obs(cfg, datapoints=None):
    """Plot the tau/kSZ/BB mock data points and error bars for a forecast
    config, with the noiseless truth tau curve overlaid.

    `cfg` is a config dict (as returned by preion.forecast.config.load_config)
    or a path to a YAML config file. `datapoints` is a dict as returned by
    `get_or_make_datapoints`/`load_autos_datapoints`; if not given, it is
    obtained with `get_or_make_datapoints(cfg)` (reading it from disk if
    already generated, else generating it).
    """
    import matplotlib.pyplot as plt

    if isinstance(cfg, str):
        cfg = load_config(cfg)
    if datapoints is None:
        datapoints = get_or_make_datapoints(cfg)

    telescopes = cfg["telescopes"]

    ells = [datapoints["ells_tau"], datapoints["ells_ksz"], datapoints["ells_bb"]]
    data = [datapoints["tau"], datapoints["ksz"], datapoints["bb"]]
    errs = [np.sqrt(np.diag(datapoints[f"cov_{k}"])) for k in ("tau", "ksz", "bb")]
    titles = ["Optical depth", "kSZ signal", r"$B$-modes"]
    ylabels = [
        r"$\ell(\ell+1)C_\ell^{\tau\tau}/2\pi$",
        r"$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu$K$^2$]",
        r"$\ell(\ell+1)C_\ell^{BB}/2\pi$ [$\mu$K$^2$]",
    ]
    tels = telescopes if telescopes is not None else [None, None, None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, ax in enumerate(axes):
        ax.errorbar(ells[i], data[i], yerr=errs[i], lw=1., marker='.', capsize=2.)
        ax.text(0.95, 0.95, titles[i], transform=ax.transAxes, bbox=props, ha='right', va='top')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(ylabels[i])
        ax.grid()
        if tels[i] is not None:
            ax.set_title(tels[i])
            ax.set_xlim(telescope_specs[tels[i]]['lmin'], telescope_specs[tels[i]]['lmax'])

    axes[0].set_yscale('log')
    axes[0].set_ylim(bottom=1e-8)
    axes[2].set_yscale('log')

    fig.tight_layout()
    return fig, axes


def plot_cross_obs(cfg, datapoints=None):
    """Plot the tau x 21cm mock cross-spectrum data points and error bars
    (one panel per z21), plus the noiseless tau-tau model curve.

    `cfg` is a config dict or a path to a YAML config file. `datapoints` is
    a dict as returned by `get_or_make_datapoints`/`load_cross_datapoints`;
    if not given, it is obtained with `get_or_make_datapoints(cfg)`.
    """
    import matplotlib.pyplot as plt

    if isinstance(cfg, str):
        cfg = load_config(cfg)
    if datapoints is None:
        datapoints = get_or_make_datapoints(cfg)

    ells = datapoints["ells"]
    z21 = datapoints["z21"]
    nz = len(z21)

    fig, axes = plt.subplots(1, nz + 1, figsize=(5 * (nz + 1), 4))
    axes = np.atleast_1d(axes)

    for iz, z in enumerate(z21):
        ax = axes[iz]
        err = np.sqrt(np.diag(datapoints["cov_tau21"][iz]))
        ax.errorbar(ells, datapoints["tau21"][iz], yerr=err, lw=1., marker='.', capsize=2.)
        ax.text(0.95, 0.95, rf'$\tau\times$21cm, $z={z:.1f}$', transform=ax.transAxes, bbox=props, ha='right', va='top')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\tau,21}/2\pi$ [$\mu$K]')
        ax.grid()

    axes[-1].plot(ells, datapoints["tautau"])
    axes[-1].text(0.95, 0.95, "Optical depth (noiseless)", transform=axes[-1].transAxes, bbox=props, ha='right', va='top')
    axes[-1].set_xlabel(r'Multipole $\ell$')
    axes[-1].set_ylabel(r'$\ell(\ell+1)C_\ell^{\tau\tau}/2\pi$')
    axes[-1].set_yscale('log')
    axes[-1].grid()

    fig.tight_layout()
    return fig, axes


def get_model(theta, preion_model, datapoints, log_kappa, data):
    """Evaluate the model at `theta`, branching on `data`. Returns a
    6-tuple in both cases: `(model1, model2, model3, tau, dksz)` --
    `(tau_ps, ksz_ps, total_bb, tau, dksz)` for an auto `data`, or
    `(tau21_models, cl21_models, tautau_model, tau, dksz)` for `data in
    CROSS_DATA`. `tau` (the fiducial model's integrated optical depth) and
    `dksz` (the kSZ Dl at l=3000) are returned in both branches so that
    `zend_prior`/`tau_prior` (via `lnprob`) and the kSZ diagnostic are
    available regardless of `data`."""
    preion_model.zre_h = theta[0]
    preion_model.dz_h = theta[1]
    preion_model.alpha0 = theta[2]
    preion_model.kappa = 10**theta[3] if log_kappa else theta[3]
    preion_model.init_reionisation_history()

    dksz = preion_model.get_ksz(ells=[3000], signal='both', Dells=True)[0]

    if data in CROSS_DATA:
        ells = datapoints["ells"]
        z21 = datapoints["z21"]
        delta_nu = datapoints["delta_nu"]
        tautau_model = preion_model.get_tau(ells=ells, signal='both', Dells=True).sum(axis=1)
        tau21_model = np.array([
            preion_model.get_tau_21_cross(z, ells, Dells=True, delta_nu=delta_nu).to(units.uK).value
            for z in z21
        ])
        cl21_model = np.array([
            preion_model.get_cl21(z, ells, Dells=True, delta_nu=delta_nu * units.MHz).to(units.uK**2).value
            for z in z21
        ])
        return tau21_model, cl21_model, tautau_model, preion_model.tau, dksz

    tau_ps = preion_model.get_tau(ells=datapoints["ells_tau"], signal='both', Dells=True).sum(axis=1)
    ksz_ps = preion_model.get_ksz(ells=datapoints["ells_ksz"], signal='patchy', Dells=True)[:, 0]
    total_bb = preion_model.get_B_modes(ells=datapoints["ells_bb"], Dells=True)
    return tau_ps, ksz_ps, total_bb, preion_model.tau, dksz


def lnprior(theta, priors):
    for i, p in enumerate(priors):
        low, high = p
        if not (low <= theta[i] <= high):
            return -np.inf
    return 0.


def lnlike(theta, preion_model, datapoints, log_kappa, data):
    model1, model2, model3, tau, dksz = get_model(theta, preion_model, datapoints, log_kappa, data)

    if data in CROSS_DATA:
        resid = model1 - datapoints["tau21"]
        chi2 = -0.5 * sum(
            resid[iz].T.dot(datapoints["inv_cov_tau21"][iz]).dot(resid[iz])
            for iz in range(resid.shape[0])
        )
        return chi2, model1, model2, model3, tau, dksz

    tau_data, ksz_data, bb_data = datapoints["tau"], datapoints["ksz"], datapoints["bb"]
    like = {}
    like['tau'] = (model1 - tau_data).T.dot(datapoints["inv_cov_tau"]).dot(model1 - tau_data)
    like['ksz'] = (model2 - ksz_data).T.dot(datapoints["inv_cov_ksz"]).dot(model2 - ksz_data)
    like['bb'] = (model3 - bb_data).T.dot(datapoints["inv_cov_bb"]).dot(model3 - bb_data)
    if data == 'all':
        chi2 = -0.5 * (like['tau']+like['ksz']+like['bb'])
    else:
        chi2 = -0.5 * like[data]
    return chi2, model1, model2, model3, tau, dksz


def _zero_blobs(datapoints, data):
    """Zero-filled (model1, model2, model3) matching the shapes `lnlike`
    would otherwise return for this `data`, used by `lnprob` when the flat
    prior already rejects `theta` (so `get_model` is never called, but the
    blob shapes must still match the declared `blobs_dtype`)."""
    if data in CROSS_DATA:
        nz = len(datapoints["z21"])
        nell = np.size(datapoints["ells"])
        return np.zeros((nz, nell)), np.zeros((nz, nell)), np.zeros(nell)
    return (
        np.zeros(np.size(datapoints["ells_tau"])),
        np.zeros(np.size(datapoints["ells_ksz"])),
        np.zeros(np.size(datapoints["ells_bb"])),
    )


def lnprob(theta, priors, preion_model, datapoints, log_kappa, data, zend_prior=None, tau_prior=None):
    """`zend_prior` (lower-limit prior on zre-dz) and `tau_prior` ([mu,
    sigma] for a Gaussian prior on the model's tau) are optional and apply
    regardless of `data` -- both default to `None`/no-op via
    `lower_limit`/`gaussian_prior`'s own no-op convention, so existing
    callers that don't pass them see no behavior change."""
    lp = lnprior(theta, priors)
    lp += lower_limit(theta[0] - theta[1], zend_prior)
    if not np.isfinite(lp):
        model1, model2, model3 = _zero_blobs(datapoints, data)
        return -np.inf, model1, model2, model3, 0., np.zeros(2)
    ln, model1, model2, model3, tau, dksz = lnlike(theta, preion_model, datapoints, log_kappa, data)
    lp += gaussian_prior(tau, tau_prior)
    return lp + ln, model1, model2, model3, tau, dksz


_worker_model = None
_worker_extra = None


def _init_worker(preion_model, priors, datapoints, log_kappa, data, zend_prior, tau_prior):
    """Pool(initializer=...) target: runs once per worker process at Pool
    creation. Under the default 'fork' start method this doesn't pickle
    `preion_model` (workers inherit it via fork's memory copy), unlike
    passing it through `args=` to EnsembleSampler, which gets pickled on
    every step and fails on the model's cached (unpicklable) CAMB results."""
    global _worker_model, _worker_extra
    _worker_model = preion_model
    _worker_extra = {
        "priors": priors, "datapoints": datapoints, "log_kappa": log_kappa,
        "data": data, "zend_prior": zend_prior, "tau_prior": tau_prior,
    }


def lnprob_worker(theta):
    e = _worker_extra
    return lnprob(
        theta, e["priors"], _worker_model, e["datapoints"], e["log_kappa"], e["data"],
        zend_prior=e["zend_prior"], tau_prior=e["tau_prior"],
    )


def run_mcmc(datapoints, cfg):
    """Run an emcee MCMC fitting the model selected by `cfg['data']`
    (bb/ksz/tau/all or tau21) to `datapoints` (as returned by
    `get_or_make_datapoints`).

    Writes an emcee HDFBackend chain to `{output_dir}/backends/mcmc_{label}_backend.h5`.
    """
    setup_logging(cfg)
    data = cfg["data"]
    is_cross = data in CROSS_DATA
    label = run_label(cfg, data)
    logger.info(label)

    cos = cosmology.Planck18
    overwrite = cfg["overwrite"]
    use_ksz_emulator = cfg["use_ksz_emulator"]
    log_kappa = cfg["log_kappa"]
    zend_prior = cfg.get("zend_prior")
    tau_prior_cfg = cfg.get("tau_prior")

    niterations = cfg["niterations"]
    nwalkers = cfg["nwalkers"]
    progress = cfg["progress"]

    theta_true = list(cfg["theta_true"])

    with _chdir(cfg["output_dir"]):
        os.makedirs("backends", exist_ok=True)

        ndim = len(theta_true)
        theta_labels = [r'$z_\mathrm{re}$', r'd$z$', r'log$\alpha_0$', r'$\kappa$']
        priors = [(5., 10.), (0.1, 4.5), (2.5, 4.5), (0.05, 0.4)]
        if log_kappa:
            theta_true[-1] = np.log10(theta_true[-1])
            priors[-1] = (-1.30, -0.40)
            theta_labels[-1] = r'$\log\kappa$'

        preion_model = Pee_model(
            h=cos.h, Ob_0=cos.Ob0, Om_0=cos.Om0,
            verbose=False, run_camb=True,
            use_ksz_emulator=use_ksz_emulator)
        preion_model.run_camb = False

        # `get_or_make_datapoints` already sets these; backfilled here too
        # so `datapoints` obtained another way (e.g. a notebook calling
        # load_packaged_*_datapoints() directly) also works with run_mcmc.
        if is_cross:
            datapoints.setdefault("z21", list(cfg["z21"]))
            datapoints.setdefault("delta_nu", cfg["delta_nu"])
            if "inv_cov_tau21" not in datapoints:
                datapoints["inv_cov_tau21"] = np.array(
                    [invert_covariance(cov) for cov in datapoints["cov_tau21"]])
        else:
            for name in ("tau", "ksz", "bb"):
                if f"inv_cov_{name}" not in datapoints:
                    datapoints[f"inv_cov_{name}"] = invert_covariance(datapoints[f"cov_{name}"])

        tau_prior = None
        if tau_prior_cfg is not None:
            # resolve mu from the fiducial model itself (never stored in the YAML)
            tau_prior = [preion_model.tau, tau_prior_cfg["sigma"]]

        t0 = time.time()
        logger.info(lnlike(theta_true, preion_model, datapoints, log_kappa, data)[0])
        t1 = time.time()
        logger.info(f'It takes {t1-t0:.1f} seconds to compute one model.')

        # blobs & backend
        if os.path.isfile(f'backends/mcmc_{label}_backend.h5') and overwrite:
            os.remove(f'backends/mcmc_{label}_backend.h5')
        backend = emcee.backends.HDFBackend(f'backends/mcmc_{label}_backend.h5')
        backend.reset(nwalkers, ndim)

        if is_cross:
            nz = len(datapoints["z21"])
            nell = np.size(datapoints["ells"])
            dtype = [
                ("tau21_models", float, (nz, nell)),
                ("cl21_models", float, (nz, nell)),
                ("tautau_models", float, (nell,)),
                ("tau", float),
                ("dksz", float, (2,)),
            ]
        else:
            dtype = [
                ("tau_models", float, (np.size(datapoints["ells_tau"]),)),
                ("ksz_models", float, (np.size(datapoints["ells_ksz"]),)),
                ("bb_models", float, (np.size(datapoints["ells_bb"]),)),
                ("tau", float),
                ("dksz", float, (2,)),
            ]

        p0 = [np.random.uniform(low, high, size=nwalkers) for low, high in priors]
        p0 = np.vstack(p0).T

        with Pool(
            nwalkers, initializer=_init_worker,
            initargs=(preion_model, priors, datapoints, log_kappa, data, zend_prior, tau_prior),
        ) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob_worker,
                backend=backend,
                pool=pool,
                blobs_dtype=dtype, )
            t0 = time.time()
            sampler.run_mcmc(p0, niterations, progress=progress)
            t1 = time.time()
        logger.info(f'It took {(t1-t0)/60./60.:.1f} hours to run {niterations} iterations with {nwalkers} walkers.')

        samples = sampler.get_chain(flat=False)

        # auto-correlation analysis to assess convergence and define burn-in
        taus = sampler.get_autocorr_time(tol=0)
        if np.isnan(taus).any():
            logger.info('NaN tau. Taking max.')
        endtau = np.nanmax(taus)
        converged = np.all(taus * 60 < sampler.iteration)
        logger.info('Auto-correlation time: %.2f. Converged: %s.' % (endtau, converged))
        burnin = int(max(0.1*samples.shape[0], 2.*np.nanmax(taus)))
        logger.info('burnin = %.1f' % (burnin/samples.shape[0]))

        flatsamples = sampler.get_chain(flat=True, discard=burnin)
        cov = np.cov(flatsamples.T)
        fom = np.sqrt(np.linalg.det(cov))
        logger.info(f'FoM = {fom:.1e}')

    return sampler


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the kSZ/tau/BB/tau21 MCMC forecast (Pee_model) from a YAML config.")
    parser.add_argument("config", help="Path to a YAML run config. "
                         "Its 'data' field (bb/ksz/tau/all/tau21) selects which data the MCMC fits.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    datapoints = get_or_make_datapoints(cfg)
    run_mcmc(datapoints, cfg)


if __name__ == "__main__":
    main()
