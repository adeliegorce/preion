import argparse
import contextlib
import os
import time
from multiprocessing import Pool

import emcee
import numpy as np
from astropy import cosmology

from ..theory import Pee_model
from ..plotting import corner as _corner_plot
from .config import VALID_DATA, build_ells, load_config, run_label
from .datapoints import make_datapoints


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
    cos = cosmology.Planck18
    ells = build_ells(cfg)
    telescopes = cfg["telescopes"]
    use_ksz_emulator = cfg["use_ksz_emulator"]
    theta_true = list(cfg["theta_true"])
    label1 = cfg["label"]

    with _chdir(cfg["output_dir"]):
        os.makedirs("data", exist_ok=True)
        if os.path.exists(f"data/{label1}_cov_ksz.txt"):
            print("Reading mock data points from file...")
            ells_bb, bb_data = np.loadtxt(f"data/{label1}_bb_datapoints.txt", unpack=True)
            ells_ksz, ksz_data = np.loadtxt(f"data/{label1}_ksz_datapoints.txt", unpack=True)
            ells_tau, tau_data = np.loadtxt(f"data/{label1}_tau_datapoints.txt", unpack=True)
            assert np.allclose(ells[0], ells_tau), "ells do not match for tau!"
            assert np.allclose(ells[1], ells_ksz), "ells do not match for ksz!"
            assert np.allclose(ells[2], ells_bb), "ells do not match for bb!"
            cov_bb = np.loadtxt(f"data/{label1}_cov_bb.txt", unpack=True)
            cov_ksz = np.loadtxt(f"data/{label1}_cov_ksz.txt", unpack=True)
            cov_tau = np.loadtxt(f"data/{label1}_cov_tau.txt", unpack=True)
            for i, cov in enumerate([cov_tau, cov_ksz, cov_bb]):
                assert cov.shape == (ells[i].size, ells[i].size), "ells do not match cov!"

        else:
            print("Generating mock data points...")
            tau_data, ksz_data, bb_data, cov_tau, cov_ksz, cov_bb = make_datapoints(
                theta_true,
                ells=ells,
                telescopes=telescopes, randomness=False,
                cos=cos, save=label1, use_ksz_emulator=use_ksz_emulator,
            )
        print("Done.")

    return {
        "tau": tau_data, "ksz": ksz_data, "bb": bb_data,
        "cov_tau": cov_tau, "cov_ksz": cov_ksz, "cov_bb": cov_bb,
        "ells_tau": ells[0], "ells_ksz": ells[1], "ells_bb": ells[2],
    }


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


def run_mcmc(datapoints, cfg):
    """Run a cosmic-variance-limited (or telescope-limited) emcee MCMC fitting
    kSZ/tau/BB data with a Pee_model. `datapoints` is a dict with keys
    "tau"/"ksz"/"bb"/"cov_tau"/"cov_ksz"/"cov_bb"/"ells_tau"/"ells_ksz"/"ells_bb"
    (as returned by `get_or_make_datapoints`), and `cfg` is a dict as
    returned by preion.forecast.config.load_config.

    Writes an emcee HDFBackend chain to `{output_dir}/backends/mcmc_{label}_backend.h5`.
    """
    data = cfg["data"]
    label = run_label(cfg, data)
    print(label)

    cos = cosmology.Planck18
    overwrite = cfg["overwrite"]
    use_ksz_emulator = cfg["use_ksz_emulator"]
    log_kappa = cfg["log_kappa"]

    niterations = cfg["niterations"]
    nwalkers = cfg["nwalkers"]

    theta_true = list(cfg["theta_true"])
    plot = cfg["plot"]

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
        print(lnlike(theta_true, preion_model, datapoints, log_kappa, data)[0])
        t1 = time.time()
        print(f'It takes {t1-t0:.1f} seconds to compute one model.')

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

        with Pool(nwalkers) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob,
                args=(priors, preion_model, datapoints, log_kappa, data),
                backend=backend,
                pool=pool,
                blobs_dtype=dtype, )
            t0 = time.time()
            sampler.run_mcmc(p0, niterations, progress=False)
            t1 = time.time()
        print(f'It took {(t1-t0)/60./60.:.1f} hours to run {niterations} iterations with {nwalkers} walkers.')

        samples = sampler.get_chain(flat=False)

        # auto-correlation analysis to assess convergence and define burn-in
        taus = sampler.get_autocorr_time(tol=0)
        if np.isnan(taus).any():
            print('NaN tau. Taking max.')
        endtau = np.nanmax(taus)
        converged = np.all(taus * 60 < sampler.iteration)
        print('Auto-correlation time: %.2f. Converged: %s.' % (endtau, converged))
        burnin = int(max(0.1*samples.shape[0], 2.*np.nanmax(taus)))
        print('burnin = %.1f' % (burnin/samples.shape[0]))

        flatsamples = sampler.get_chain(flat=True, discard=burnin)
        cov = np.cov(flatsamples.T)
        fom = np.sqrt(np.linalg.det(cov))
        print(f'FoM = {fom:.1e}')

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
