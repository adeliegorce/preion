import argparse
import contextlib
import logging
import os
import time
from multiprocessing import Pool

import emcee
import numpy as np
from astropy import cosmology

from ..parameters import props, telescope_specs
from ..theory import Pee_model
from ..plotting import corner as _corner_plot
from .config import VALID_DATA, build_ells, load_config, run_label, setup_logging
from .datapoints import load_datapoints, make_datapoints

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


def get_or_make_datapoints(cfg):
    """Get the kSZ/tau/BB mock datapoints and covariances for this config,
    reading them back from `{output_dir}/data/` if already present, else
    generating them with `make_datapoints` (and writing them there for next
    time). `cfg` is a dict as returned by preion.forecast.config.load_config.

    Returns a dict with keys "tau"/"ksz"/"bb" (the data),
    "cov_tau"/"cov_ksz"/"cov_bb" (their covariances), and
    "ells_tau"/"ells_ksz"/"ells_bb" (their multipole grids), ready to feed to
    `run_mcmc`.
    """
    setup_logging(cfg)
    cos = cosmology.Planck18
    ells = build_ells(cfg)
    telescopes = cfg["telescopes"]
    use_ksz_emulator = cfg["use_ksz_emulator"]
    theta_true = list(cfg["theta_true"])
    label1 = cfg["label"]

    with _chdir(cfg["output_dir"]):
        os.makedirs("data", exist_ok=True)
        if os.path.exists(f"data/{label1}_cov_ksz.txt"):
            logger.info("Reading mock data points from file...")
            datapoints = load_datapoints("data", label1, ells=ells)
        else:
            logger.info("Generating mock data points...")
            tau_data, ksz_data, bb_data, cov_tau, cov_ksz, cov_bb = make_datapoints(
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
        logger.info("Done.")

    return datapoints


def plot_obs(cfg, datapoints=None):
    """Plot the tau/kSZ/BB mock data points and error bars for a forecast
    config, with the noiseless truth tau curve overlaid.

    `cfg` is a config dict (as returned by preion.forecast.config.load_config)
    or a path to a YAML config file. `datapoints` is a dict as returned by
    `get_or_make_datapoints`/`load_datapoints`; if not given, it is obtained
    with `get_or_make_datapoints(cfg)` (reading it from disk if already
    generated, else generating it).
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


def get_model(theta, preion_model, datapoints, log_kappa):
    preion_model.zre_h = theta[0]
    preion_model.dz_h = theta[1]
    preion_model.alpha0 = theta[2]
    preion_model.kappa = 10**theta[3] if log_kappa else theta[3]
    preion_model.init_reionisation_history()

    tau_ps = preion_model.get_tau(ells=datapoints["ells_tau"], signal='both', Dells=True).sum(axis=1)
    ksz_ps = preion_model.get_ksz(ells=datapoints["ells_ksz"], signal='patchy', Dells=True)[:, 0]
    total_bb = preion_model.get_B_modes(ells=datapoints["ells_bb"], Dells=True)
    return tau_ps, ksz_ps, total_bb


def lnprior(theta, priors):
    for i, p in enumerate(priors):
        low, high = p
        if not (low <= theta[i] <= high):
            return -np.inf
    return 0.


def lnlike(theta, preion_model, datapoints, log_kappa, data):
    tau_data, ksz_data, bb_data = datapoints["tau"], datapoints["ksz"], datapoints["bb"]
    cov_tau, cov_ksz, cov_bb = datapoints["cov_tau"], datapoints["cov_ksz"], datapoints["cov_bb"]
    model, like = {}, {}
    model['tau'], model['ksz'], model['bb'] = get_model(theta, preion_model, datapoints, log_kappa)
    like['tau'] = (model['tau'] - tau_data).T.dot(np.linalg.inv(cov_tau)).dot(model['tau'] - tau_data)
    like['ksz'] = (model['ksz'] - ksz_data).T.dot(np.linalg.inv(cov_ksz)).dot(model['ksz'] - ksz_data)
    like['bb'] = (model['bb'] - bb_data).T.dot(np.linalg.inv(cov_bb)).dot(model['bb'] - bb_data)
    if data == 'all':
        chi2 = -0.5 * (like['tau']+like['ksz']+like['bb'])
    else:
        chi2 = -0.5 * like[data]
    return chi2, model['tau'], model['ksz'], model['bb']


def lnprob(theta, priors, preion_model, datapoints, log_kappa, data):
    lp = lnprior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf, 0., 0., 0.
    ln, tau_model, ksz_model, bb_model = lnlike(theta, preion_model, datapoints, log_kappa, data)
    return lp + ln, tau_model, ksz_model, bb_model


_worker_model = None
_worker_extra = None


def _init_worker(preion_model, priors, datapoints, log_kappa, data):
    """Pool(initializer=...) target: runs once per worker process at Pool
    creation. Under the default 'fork' start method this doesn't pickle
    `preion_model` (workers inherit it via fork's memory copy), unlike
    passing it through `args=` to EnsembleSampler, which gets pickled on
    every step and fails on the model's cached (unpicklable) CAMB results."""
    global _worker_model, _worker_extra
    _worker_model = preion_model
    _worker_extra = (priors, datapoints, log_kappa, data)


def lnprob_worker(theta):
    priors, datapoints, log_kappa, data = _worker_extra
    return lnprob(theta, priors, _worker_model, datapoints, log_kappa, data)


def run_mcmc(datapoints, cfg):
    """Run a cosmic-variance-limited (or telescope-limited) emcee MCMC fitting
    kSZ/tau/BB data with a Pee_model. `datapoints` is a dict with keys
    "tau"/"ksz"/"bb"/"cov_tau"/"cov_ksz"/"cov_bb"/"ells_tau"/"ells_ksz"/"ells_bb"
    (as returned by `get_or_make_datapoints`), and `cfg` is a dict as
    returned by preion.forecast.config.load_config.

    Writes an emcee HDFBackend chain to `{output_dir}/backends/mcmc_{label}_backend.h5`.
    """
    setup_logging(cfg)
    data = cfg["data"]
    label = run_label(cfg, data)
    logger.info(label)

    cos = cosmology.Planck18
    overwrite = cfg["overwrite"]
    use_ksz_emulator = cfg["use_ksz_emulator"]
    log_kappa = cfg["log_kappa"]

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

        t0 = time.time()
        logger.info(lnlike(theta_true, preion_model, datapoints, log_kappa, data)[0])
        t1 = time.time()
        logger.info(f'It takes {t1-t0:.1f} seconds to compute one model.')

        # blobs & backend
        if os.path.isfile(f'backends/mcmc_{label}_backend.h5') and overwrite:
            os.remove(f'backends/mcmc_{label}_backend.h5')
        backend = emcee.backends.HDFBackend(f'backends/mcmc_{label}_backend.h5')
        backend.reset(nwalkers, ndim)
        dtype = [("tau_models", float, (np.size(datapoints["ells_tau"]),)),
                 ("ksz_models", float, (np.size(datapoints["ells_ksz"]),)),
                 ("bb_models", float, (np.size(datapoints["ells_bb"]),)), ]

        p0 = [np.random.uniform(low, high, size=nwalkers) for low, high in priors]
        p0 = np.vstack(p0).T

        with Pool(
            nwalkers, initializer=_init_worker,
            initargs=(preion_model, priors, datapoints, log_kappa, data),
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
        description="Run the kSZ/tau/BB MCMC forecast (Pee_model) from a YAML config.")
    parser.add_argument("config", help="Path to a YAML run config (see configs/cv_limited_new.yaml).")
    parser.add_argument("--data", choices=VALID_DATA, default=None,
                         help="Override the config's 'data' field (bb/ksz/tau/all).")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.data is not None:
        cfg["data"] = args.data

    datapoints = get_or_make_datapoints(cfg)
    run_mcmc(datapoints, cfg)


if __name__ == "__main__":
    main()
