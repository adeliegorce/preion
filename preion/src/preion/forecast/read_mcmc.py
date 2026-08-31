import argparse
import logging
import os

import arviz as az
import emcee
import h5py
import numpy as np
from astropy import cosmology
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import corner
from scipy.stats import qmc
from tqdm import tqdm

from . import datapoints, mcmc
from .config import cfg_title, load_config, run_label, setup_logging, truncate_log_at_marker, CROSS_DATA
from .utils import gaussian_prior, lower_limit
from ..parameters import priors
from ..theory import Pee_model

PARAM_NAMES = ["zre", "dz", "alpha0", "kappa"]

logger = logging.getLogger(__name__)


def _theta_labels(cfg):
    labels = [r'$z_\mathrm{re}$', r'd$z$', r'log$\alpha_0$', r'$\kappa$']
    if cfg["log_kappa"]:
        labels[-1] = r'$\log\kappa$'
    return labels


def _theta_true(cfg):
    theta_true = list(cfg["theta_true"])
    if cfg["log_kappa"]:
        theta_true[-1] = np.log10(theta_true[-1])
    return theta_true


def load_mock_data(cfg):
    """Load the mock datapoints + covariances written by
    preion.forecast.mcmc.get_or_make_datapoints for this config, dispatching
    on cfg['data'] to the matching datapoints.py loader -- the single,
    data-type-branching entry point (like get_or_make_datapoints/lnlike/
    lnprob/get_model), not a separate loader per family."""
    data_dir = os.path.join(cfg["output_dir"], "data")
    if cfg["data"] in CROSS_DATA:
        dp = datapoints.load_cross_datapoints(data_dir, cfg["label"], cfg["z21"])
        dp.setdefault("z21", list(cfg["z21"]))
        dp.setdefault("delta_nu", cfg["delta_nu"])
        return dp
    return datapoints.load_autos_datapoints(data_dir, cfg["label"])


def load_chain(cfg, data=None):
    """Open the emcee HDFBackend chain (read-only) for this config."""
    backend_dir = os.path.join(cfg["output_dir"], "backends")
    label = run_label(cfg, data)
    path = os.path.join(backend_dir, f"mcmc_{label}_backend.h5")
    return emcee.backends.HDFBackend(path, read_only=True)


def convergence_diagnostics(sampler):
    """Autocorrelation time, burn-in estimate, convergence flag, and Gelman-Rubin
    R-hat (via arviz) — same formulas as read_mcmc.ipynb /
    compare_forecast_cv_limited_new.py."""
    samples = sampler.get_chain(flat=False)
    taus = sampler.get_autocorr_time(tol=0)
    if np.isnan(taus).any():
        logger.info("NaN autocorrelation time. Taking max of the finite values.")
    endtau = np.nanmax(taus)
    converged = bool(np.all(taus * 60 < sampler.iteration))
    burnin = int(max(0.1 * samples.shape[0], 2. * endtau))
    idata = az.from_emcee(sampler, var_names=PARAM_NAMES)
    rhat = az.rhat(idata)
    return {
        "taus": taus,
        "endtau": endtau,
        "converged": converged,
        "burnin": burnin,
        "rhat": rhat,
        "nwalkers": samples.shape[1],
        "nsteps": samples.shape[0],
    }


def get_flat_samples(sampler, burnin, tau_prior=False):
    """Flatten the post-burn-in chain; optionally append the `tau` blob column
    (only present for runs that included a tau Gaussian prior)."""
    flatsamples = sampler.get_chain(flat=True, discard=burnin)
    logps = sampler.get_log_prob(flat=True, discard=burnin)
    if tau_prior:
        tau_samples = sampler.get_blobs(flat=True, discard=burnin)["tau"]
        flatsamples = np.c_[flatsamples, tau_samples]
    return flatsamples, logps


def plot_corner(flatsamples, truths, labels, title=None, **kwargs):
    fig = corner.corner(
        flatsamples, truths=truths, labels=labels, truth_color="k",
        sigmas=[1, 2], plot_datapoints=False, lw=2.,
        show_titles=True, **kwargs,
    )
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_corner_comparison(chains, labels, chain_labels, truths=None, colors=None, fig=None, **kwargs):
    """Overlay N sets of flat samples on one corner plot, one colour per
    chain, with a legend built from `chain_labels`. `truths` (if given) is
    only drawn once, on the first chain, since corner.corner draws truth
    lines unconditionally on every call and would otherwise duplicate them.
    `fig`, if given (e.g. a grey prior-background layer already drawn by
    the caller), is drawn into rather than starting a fresh figure."""
    colors = colors if colors is not None else [f"C{i}" for i in range(len(chains))]
    for i, (chain, color) in enumerate(zip(chains, colors)):
        fig = corner.corner(
            chain, fig=fig, color=color, labels=labels,
            truths=truths if i == 0 else None, truth_color="k",
            levels=[0.393, 0.864], plot_datapoints=False,
            plot_density=False, no_fill_contours=True, lw=2.,
            smooth=1., **kwargs,
        )
    handles = [mlines.Line2D([], [], color=c, label=name) for c, name in zip(colors, chain_labels)]
    fig.legend(handles=handles, loc=(0.6, 0.75), frameon=False)
    fig.tight_layout()
    return fig


def plot_trace(sampler, truths, labels, burnin, title=None):

    samples = sampler.get_chain(flat=False)
    logps = sampler.get_log_prob(flat=False)
    ndim = samples.shape[-1]
    nwalkers = samples.shape[1]
    fig, axes = plt.subplots(ndim + 1, 1, figsize=(8, 2 * (ndim + 1)), sharex=True)
    for i in range(ndim):
        for j in range(nwalkers):
            axes[i].plot(samples[:, j, i], alpha=0.6, lw=0.5, color=f'C{j}')
        axes[i].axvline(burnin, color="k", ls="--", lw=1.)
        axes[i].set_ylabel(labels[i])
        axes[i].axhline(truths[i], color='k', ls=':', lw=1.5)
    for j in range(nwalkers):
        axes[-1].plot(logps[:, j], alpha=0.6, lw=0.5, color=f'C{j}', label=f'Walker {j}')
    axes[-1].axvline(burnin, color="k", ls="--", lw=1.)
    axes[-1].set_ylabel("log prob")
    axes[-1].set_xlabel("step")
    axes[-1].legend(ncol=2, frameon=False)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_autos_models(sampler, datapoints, burnin, n_draws=500, title=None):
    """Posterior-predictive plot for an auto (bb/ksz/tau/all) run:
    `datapoints` is a dict as returned by `load_mock_data`/
    `get_or_make_datapoints` for an auto config."""

    ells_tau, ells_ksz, ells_bb = datapoints["ells_tau"], datapoints["ells_ksz"], datapoints["ells_bb"]
    tau_models = sampler.get_blobs(flat=True, discard=burnin)["tau_models"]
    ksz_models = sampler.get_blobs(flat=True, discard=burnin)["ksz_models"]
    bb_models = sampler.get_blobs(flat=True, discard=burnin)["bb_models"]

    rng = np.random.default_rng()
    idx = rng.choice(tau_models.shape[0], size=min(n_draws, tau_models.shape[0]), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, ells_obs, obs_data, obs_cov, models, panel_title in zip(
        axes,
        [ells_tau, ells_ksz, ells_bb],
        [datapoints["tau"], datapoints["ksz"], datapoints["bb"]],
        [datapoints["cov_tau"], datapoints["cov_ksz"], datapoints["cov_bb"]],
        [tau_models, ksz_models, bb_models],
        ["Optical depth", "kSZ signal", r"$B$-modes"],
    ):
        for i in idx:
            ax.plot(ells_obs, models[i], color="C0", alpha=0.05, lw=0.5)
        ax.errorbar(
            ells_obs, obs_data, yerr=np.sqrt(np.diag(obs_cov)),
            lw=0., elinewidth=0.8, marker='.', capsize=2., color='k')
        ax.set_title(panel_title)
        ax.set_xlabel(r"Multipole $\ell$")
        ax.grid()
    axes[0].set_ylabel(r'$\mathcal{D}_\ell$ [$\mu\mathrm{K}^2$]')
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cross_models(sampler, datapoints, burnin, n_draws=500, title=None):
    """Posterior-predictive plot for a cross (tau21) run, one panel per
    z21: `datapoints` is a dict as returned by `load_mock_data`/
    `get_or_make_datapoints` for a cross config."""

    ells = datapoints["ells"]
    z21 = datapoints["z21"]
    nz = len(z21)
    tau21_models = sampler.get_blobs(flat=True, discard=burnin)["tau21_models"]

    rng = np.random.default_rng()
    idx = rng.choice(tau21_models.shape[0], size=min(n_draws, tau21_models.shape[0]), replace=False)

    fig, axes = plt.subplots(1, nz, figsize=(5 * nz, 4))
    axes = np.atleast_1d(axes)
    for iz, (ax, z) in enumerate(zip(axes, z21)):
        for i in idx:
            ax.plot(ells, tau21_models[i, iz], color="C0", alpha=0.05, lw=0.5)
        ax.errorbar(
            ells, datapoints["tau21"][iz], yerr=np.sqrt(np.diag(datapoints["cov_tau21"][iz])),
            lw=0., elinewidth=0.8, marker='.', capsize=2., color='k')
        ax.set_title(rf'$z={z:.1f}$')
        ax.set_xlabel(r"Multipole $\ell$")
        ax.grid()
    axes[0].set_ylabel(r'$\mathcal{D}_\ell^{\tau,21}$ [$\mu\mathrm{K}$]')
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def summarize(flatsamples, truths, paramnames):
    """Print & return mean/std/bias/percent-error per parameter, as in
    read_mcmc.ipynb."""
    summary = {}
    logger.info('ML parameters:')
    for j, name in enumerate(paramnames):
        mean, std = np.mean(flatsamples[:, j]), np.std(flatsamples[:, j])
        logger.info(f' {name} = {mean:.2f} +/- {std:.3f} vs {truths[j]:.2f}')
        summary[name] = {"mean": mean, "std": std, "truth": truths[j]}
    logger.info('Biases:')
    for j, name in enumerate(paramnames):
        bias_sigma = (summary[name]["mean"] - truths[j]) / summary[name]["std"]
        pct_error = summary[name]["std"] / truths[j] * 100. if truths[j] != 0 else np.nan
        logger.info(f' {name} measured with {bias_sigma:.2f}sig bias and {pct_error:.2f}% error')
        summary[name]["bias_sigma"] = bias_sigma
        summary[name]["percent_error"] = pct_error
    return summary


def generate_prior_cache(cfg, nrand=1000, out_path=None):
    """LHC-sample `nrand` draws of theta=[zre,dz,alpha0,kappa] from the same
    box priors used by `mcmc.run_mcmc`, drop any with `zre-dz <=
    cfg.get('zend_prior')` (a hard cutoff, if set -- matching `lnprob`'s
    own flat check), evaluate the cross model (tau21/cl21/tautau/tau/dksz)
    for each surviving draw via `mcmc.get_model`, and cache the result to
    `out_path` (default f"{cfg['output_dir']}/prior_distributions_{cfg['label']}.hdf5")
    for `compare_main`'s prior-background overlay.

    `tau_prior` (a soft Gaussian term, not a hard cutoff) can't be applied
    as a cutoff -- instead each draw gets an importance `weight =
    exp(gaussian_prior(tau, [mu, sigma]))` (else `weight=1.0` for every
    draw when `tau_prior` isn't set), computed here and stored alongside
    the other fields, so a consumer doesn't need to know `tau_prior`'s
    value at all to use it.

    Only meaningful for `data in CROSS_DATA` configs today."""
    out_path = out_path or os.path.join(cfg["output_dir"], f"prior_distributions_{cfg['label']}.hdf5")
    zend_prior = cfg.get("zend_prior")
    tau_prior_cfg = cfg.get("tau_prior")
    log_kappa = cfg["log_kappa"]
    use_ksz_emulator = cfg["use_ksz_emulator"]

    dp = mcmc.get_or_make_datapoints(cfg)

    cos = cosmology.Planck18
    preion_model = Pee_model(
        h=cos.h, Ob_0=cos.Ob0, Om_0=cos.Om0,
        verbose=False, run_camb=True, use_ksz_emulator=use_ksz_emulator)
    preion_model.run_camb = False

    box = [list(p) for p in priors]
    if log_kappa:
        box[-1] = [-1.30, -0.40]
    lo = np.array([p[0] for p in box])
    hi = np.array([p[1] for p in box])

    theta_true = _theta_true(cfg)
    _, _, _, fiducial_tau, _ = mcmc.get_model(theta_true, preion_model, dp, log_kappa, "tau21")
    tau_prior = None if tau_prior_cfg is None else [fiducial_tau, tau_prior_cfg["sigma"]]

    sample = qmc.LatinHypercube(d=4).random(n=nrand)
    thetas = qmc.scale(sample, lo, hi)

    kept_params, tau21_list, cl21_list, tautau_list, tau_list, dksz_list, weight_list = (
        [], [], [], [], [], [], [])
    for theta in tqdm(thetas, desc="Sampling prior"):
        zend = theta[0] - theta[1]
        if not np.isfinite(lower_limit(zend, zend_prior)):
            continue
        tau21_model, cl21_model, tautau_model, tau, dksz = mcmc.get_model(
            theta, preion_model, dp, log_kappa, "tau21")
        weight = np.exp(gaussian_prior(tau, tau_prior))
        kept_params.append(theta)
        tau21_list.append(tau21_model)
        cl21_list.append(cl21_model)
        tautau_list.append(tautau_model)
        tau_list.append(tau)
        dksz_list.append(dksz)
        weight_list.append(weight)

    with h5py.File(out_path, "w") as f:
        f["params"] = np.array(kept_params)
        f["tau21"] = np.array(tau21_list)
        f["cl21"] = np.array(cl21_list)
        f["tautau"] = np.array(tautau_list)
        f["tau"] = np.array(tau_list)
        f["dksz"] = np.array(dksz_list)
        f["weight"] = np.array(weight_list)
        f["ells"] = dp["ells"]
        f["z21"] = np.array(cfg["z21"])

    logger.info(f"Saved prior cache ({len(kept_params)}/{nrand} draws survived zend_prior) to {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read back and diagnose an emcee MCMC chain produced by preion-run-mcmc.")
    parser.add_argument("config", help="Path to the same YAML run config used with preion-run-mcmc. ")
    parser.add_argument("--with-zend", action="store_true", help="Whether to include zend contours.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    truncate_log_at_marker(cfg, "Results of chain analysis:")
    setup_logging(cfg)

    dp = load_mock_data(cfg)
    sampler = load_chain(cfg)
    diagnostics = convergence_diagnostics(sampler)
    logger.info("\nResults of chain analysis:")
    logger.info(f"Ran {diagnostics['nwalkers']} walkers for {diagnostics['nsteps']} steps each.")
    logger.info(f"Auto-correlation time: {diagnostics['endtau']:.2f}. Converged: {diagnostics['converged']}.")
    logger.info(f"burnin = {diagnostics['burnin']} ({diagnostics['burnin']/sampler.iteration:.1%})")
    logger.info("Gelman-Rubin R-hat:")
    logger.info(diagnostics["rhat"])

    flatsamples, _ = get_flat_samples(sampler, diagnostics["burnin"])
    truths = _theta_true(cfg)
    labels = _theta_labels(cfg)
    if args.with_zend:
        flatsamples = np.c_[flatsamples, flatsamples[:, 0] - flatsamples[:, 1]]
        labels.append(r'$z_\mathrm{end}$')
        truths.append(truths[0]-truths[1])
        PARAM_NAMES.append('zend')
    summarize(flatsamples, truths, PARAM_NAMES)

    title = cfg_title(cfg)
    fig_corner = plot_corner(flatsamples, truths, labels, title=title, color='C0', smooth=1.)
    fig_trace = plot_trace(sampler, truths, labels, diagnostics["burnin"], title=title)
    if cfg["data"] in CROSS_DATA:
        fig_pp = plot_cross_models(sampler, dp, diagnostics["burnin"], title=title)
    else:
        fig_pp = plot_autos_models(sampler, dp, diagnostics["burnin"], title=title)

    figure_dir = os.path.join(cfg["output_dir"], "figures")
    os.makedirs(figure_dir, exist_ok=True)
    label = run_label(cfg)
    fig_corner.savefig(os.path.join(figure_dir, f"mcmc_{label}_corner.png"), dpi=220)
    fig_trace.savefig(os.path.join(figure_dir, f"mcmc_{label}_trace.png"), dpi=220)
    fig_pp.savefig(os.path.join(figure_dir, f"mcmc_{label}_models.png"), dpi=220)


def compare_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Overlay corner plots from multiple preion-run-mcmc chains.")
    parser.add_argument("configs", nargs="+",
                         help="Paths to two or more YAML run configs, each already run with preion-run-mcmc.")
    parser.add_argument("-o", "--output", default="mcmc_comparison_corner.png",
                         help="Filename for the saved overlaid corner plot (default: %(default)s).")
    parser.add_argument("--with-prior-background", action="store_true",
                         help="Overlay a grey Latin-Hypercube-sampled prior background behind the chain "
                              "contours (generated via generate_prior_cache if not already cached). "
                              "Requires every config to have data=tau21.")
    args = parser.parse_args(argv)

    if len(args.configs) < 2:
        parser.error("give at least 2 config files to compare.")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfgs = [load_config(path) for path in args.configs]
    ref_cfg = cfgs[0]
    for cfg in cfgs[1:]:
        if cfg["log_kappa"] != ref_cfg["log_kappa"]:
            raise ValueError(
                "All configs must agree on 'log_kappa' (kappa vs log-kappa axes aren't comparable)."
            )
        if list(cfg["theta_true"]) != list(ref_cfg["theta_true"]):
            logger.info(f"Warning: theta_true differs from {args.configs[0]} for {cfg['label']}; "
                        "using the first config's truths for the overlay.")

    truths = _theta_true(ref_cfg)
    truths += [truths[0] - truths[1]]  # zend
    priors2 = priors + [(3., 8.5)]
    if ref_cfg["log_kappa"]:
        priors2[-2] = (-1.30, -0.40)
    theta_labels = _theta_labels(ref_cfg) + [r'$z_\mathrm{end}$']
    chain_labels = [cfg_title(cfg) for cfg in cfgs]
    flatsamples_list = []
    for cfg, name in zip(cfgs, chain_labels):
        sampler = load_chain(cfg)
        diagnostics = convergence_diagnostics(sampler)
        logger.info(f"{name}: converged={diagnostics['converged']}, "
                    f"burnin={diagnostics['burnin']} ({diagnostics['burnin']/sampler.iteration:.1%})")
        flatsamples, _ = get_flat_samples(sampler, diagnostics["burnin"])
        flatsamples = np.c_[flatsamples, flatsamples[:, 0] - flatsamples[:, 1]] #zend
        flatsamples_list.append(flatsamples)

    fig = None
    if args.with_prior_background:
        if not all(cfg["data"] in CROSS_DATA for cfg in cfgs):
            parser.error("--with-prior-background requires every config to have data=tau21.")
        cache_path = os.path.join(ref_cfg["output_dir"], f"prior_distributions_{ref_cfg['label']}.hdf5")
        if not os.path.exists(cache_path):
            cache_path = generate_prior_cache(ref_cfg)
        with h5py.File(cache_path, "r") as f:
            prior_params = f["params"][:]
            prior_weight = f["weight"][:]
        prior_zend = prior_params[:, 0] - prior_params[:, 1]
        prior_samples = np.c_[prior_params, prior_zend]
        fig = corner.corner(
            prior_samples, weights=prior_weight, range=priors2,
            color='grey', plot_datapoints=False, plot_density=False,
            no_fill_contours=True, levels=[0.393, 0.864], smooth=1.,
        )

    fig = plot_corner_comparison(
        flatsamples_list[::-1],
        theta_labels, chain_labels[::-1],
        truths=truths, range=priors2, fig=fig)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=220)
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
