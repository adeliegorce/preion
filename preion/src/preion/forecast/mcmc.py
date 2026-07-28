import argparse
import contextlib
import os
import time
from multiprocessing import Pool

import emcee
import numpy as np
from astropy import cosmology

from ..theory import Pee_model
from .utils import sample_var
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


def run_mcmc_cv_limited_new(cfg):
    """Run a cosmic-variance-limited (or telescope-limited) emcee MCMC fitting
    kSZ/tau/BB mock data with a Pee_model. `cfg` is a dict as returned by
    preion.forecast.config.load_config.

    Writes an emcee HDFBackend chain to `{output_dir}/backends/mcmc_{label}_backend.h5`
    and (if not already present) mock data/covariances to `{output_dir}/data/`.
    """
    data = cfg["data"]
    label = run_label(cfg, data)
    print(label)

    cos = cosmology.Planck18
    ells = build_ells(cfg)
    telescopes = cfg["telescopes"]
    fsky = cfg["fsky"]
    randomness = False
    overwrite = cfg["overwrite"]
    use_ksz_emulator = cfg["use_ksz_emulator"]
    log_kappa = cfg["log_kappa"]

    niterations = cfg["niterations"]
    nwalkers = cfg["nwalkers"]

    theta_true = list(cfg["theta_true"])
    plot = cfg["plot"]
    label1 = cfg["label"]

    with _chdir(cfg["output_dir"]):
        os.makedirs("data", exist_ok=True)
        os.makedirs("backends", exist_ok=True)

        # mock data
        if os.path.exists(f"data/{label1}_cov_ksz.txt"):
            print("Reading mock data points from file...")
            ells_bb, bb_data = np.loadtxt(f"data/{label1}_bb_datapoints.txt", unpack=True)
            ells_ksz, ksz_data = np.loadtxt(f"data/{label1}_ksz_datapoints.txt", unpack=True)
            ells_tau, tau_data = np.loadtxt(f"data/{label1}_tau_datapoints.txt", unpack=True)
            assert np.allclose(ells[0], ells_tau), "ells do not match for tau!"
            assert np.allclose(ells[1], ells_ksz), "ells do not match for ksz!"
            assert np.allclose(ells[2], ells_bb), "ells do not match for bb!"
        else:
            print("Generating mock data points...")
            tau_data, ksz_data, bb_data, _, _, _ = make_datapoints(
                theta_true,
                ells=ells,
                telescopes=telescopes, randomness=randomness,
                cos=cos, save=label1, use_ksz_emulator=use_ksz_emulator,
            )
        cov_tau = np.diag(sample_var(ells[0], tau_data, fsky)**2)
        cov_ksz = np.diag(sample_var(ells[1], ksz_data, fsky)**2)
        if use_ksz_emulator:
            cov_ksz += np.diag(np.ones_like(ells[1]) * 0.01**2)  # emulator error (~ constant with multipole)
        cov_bb = np.diag(sample_var(ells[2], bb_data, fsky)**2)
        np.savetxt(f"data/{label1}_cov_ksz.txt", cov_ksz)
        np.savetxt(f"data/{label1}_cov_tau.txt", cov_tau)
        np.savetxt(f"data/{label1}_cov_bb.txt", cov_bb)
        print("Done.")

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

        def get_model(theta, ells=ells):

            preion_model.zre_h = theta[0]
            preion_model.dz_h = theta[1]
            preion_model.alpha0 = theta[2]
            if log_kappa:
                preion_model.kappa = 10**theta[3]
            else:
                preion_model.kappa = theta[3]
            preion_model.init_reionisation_history()

            tau_ps = preion_model.get_tau(ells=ells[0], signal='both', Dells=True).sum(axis=1)
            ksz_ps = preion_model.get_ksz(ells=ells[1], signal='patchy', Dells=True)[:, 0]
            total_bb = preion_model.get_B_modes(ells=ells[2], Dells=True)

            return tau_ps, ksz_ps, total_bb

        def lnprior(theta, priors=priors):
            for i, p in enumerate(priors):
                low, high = p
                if not (low <= theta[i] <= high):
                    return -np.inf
            return 0.

        def lnlike(theta):
            model, like = {}, {}
            model['tau'], model['ksz'], model['bb'] = get_model(theta)
            like['tau'] = (model['tau'] - tau_data).T.dot(np.linalg.inv(cov_tau)).dot(model['tau'] - tau_data)
            like['ksz'] = (model['ksz'] - ksz_data).T.dot(np.linalg.inv(cov_ksz)).dot(model['ksz'] - ksz_data)
            like['bb'] = (model['bb'] - bb_data).T.dot(np.linalg.inv(cov_bb)).dot(model['bb'] - bb_data)
            if data == 'all':
                chi2 = -0.5 * (like['tau']+like['ksz']+like['bb'])
            else:
                chi2 = -0.5 * like[data]
            return chi2, model['tau'], model['ksz'], model['bb']

        def lnprob(theta):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf, 0., 0., 0.
            ln, tau_model, ksz_model, bb_model = lnlike(theta)
            return lp + ln, tau_model, ksz_model, bb_model

        t0 = time.time()
        print(lnlike(theta_true)[0])
        t1 = time.time()
        print(f'It takes {t1-t0:.1f} seconds to compute one model.')

        # blobs & backend
        if os.path.isfile(f'backends/mcmc_{label}_backend.h5') and overwrite:
            os.remove(f'backends/mcmc_{label}_backend.h5')
        backend = emcee.backends.HDFBackend(f'backends/mcmc_{label}_backend.h5')
        backend.reset(nwalkers, ndim)
        dtype = [("tau_models", float, (np.size(ells[0]),)),
                 ("ksz_models", float, (np.size(ells[1]),)),
                 ("bb_models", float, (np.size(ells[2]),)), ]

        p0 = [np.random.uniform(low, high, size=nwalkers) for low, high in priors]
        p0 = np.vstack(p0).T

        with Pool(nwalkers) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob,
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

        if plot:
            os.makedirs("figures", exist_ok=True)
            import matplotlib.pyplot as plt

            fig = _corner_plot(
                flatsamples,
                truths=theta_true,
                labels=theta_labels, truth_color='k',
                sigmas=[1, 2], plot_datapoints=False,
                lw=2., smooth=1.,
            )
            axes = np.asarray(fig.axes)  # Access all the axes in the corner plot
            axes = axes.reshape(np.size(theta_true), np.size(theta_true))
            for i in range(1, ndim):       # Loop over rows (starting at 1 to skip diagonal)
                for j in range(i):         # Loop over columns (lower triangle only)
                    axes[i, j].scatter(p0[:, j], p0[:, i], color='red', s=10, alpha=0.5)
            fig.tight_layout()
            fig.savefig(f'figures/mcmc_{label}_corner.png', dpi=220)

            fig, ax = plt.subplots()
            im = ax.imshow(
                cov, cmap='coolwarm',
                extent=(0, 2, 2, 0))
            ax.set_xticks(np.arange(0.25, 2, step=.5), labels=theta_labels)
            ax.set_yticks(np.arange(0.25, 2, step=.5), labels=theta_labels)
            plt.colorbar(im, fraction=0.045, ax=ax, label=r'$\mathcal{C}_{ij}$')
            fig.tight_layout()
            fig.savefig(f'figures/mcmc_{label}_covmat.png', dpi=220)

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

    run_mcmc_cv_limited_new(cfg)


if __name__ == "__main__":
    main()
