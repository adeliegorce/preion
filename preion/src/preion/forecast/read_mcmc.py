import argparse
import os

import arviz as az
import emcee
import numpy as np

from ..plotting import corner as _corner_plot
from .config import load_config, run_label

PARAM_NAMES = ["zre", "dz", "alpha0", "kappa"]


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
    """Load the mock tau/kSZ/BB data points + covariances written by
    preion.forecast.mcmc.run_mcmc_cv_limited_new for this config."""
    data_dir = os.path.join(cfg["output_dir"], "data")
    label1 = cfg["label"]
    ells_bb, bb_data = np.loadtxt(os.path.join(data_dir, f"{label1}_bb_datapoints.txt"), unpack=True)
    ells_ksz, ksz_data = np.loadtxt(os.path.join(data_dir, f"{label1}_ksz_datapoints.txt"), unpack=True)
    ells_tau, tau_data = np.loadtxt(os.path.join(data_dir, f"{label1}_tau_datapoints.txt"), unpack=True)
    cov_ksz = np.loadtxt(os.path.join(data_dir, f"{label1}_cov_ksz.txt"))
    cov_tau = np.loadtxt(os.path.join(data_dir, f"{label1}_cov_tau.txt"))
    cov_bb = np.loadtxt(os.path.join(data_dir, f"{label1}_cov_bb.txt"))
    ells = [ells_tau, ells_ksz, ells_bb]
    data = {"tau": tau_data, "ksz": ksz_data, "bb": bb_data}
    cov = {"tau": cov_tau, "ksz": cov_ksz, "bb": cov_bb}
    return ells, data, cov


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
        print("NaN autocorrelation time. Taking max of the finite values.")
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


def plot_corner(flatsamples, truths, labels, **kwargs):
    fig = _corner_plot(
        flatsamples, truths=truths, labels=labels, truth_color="k",
        sigmas=[1, 2], plot_datapoints=False, lw=2., smooth=1., **kwargs,
    )
    fig.tight_layout()
    return fig


def plot_trace(sampler, paramnames, burnin):
    import matplotlib.pyplot as plt

    samples = sampler.get_chain(flat=False)
    logps = sampler.get_log_prob(flat=False)
    ndim = samples.shape[-1]
    fig, axes = plt.subplots(ndim + 1, 1, figsize=(8, 2 * (ndim + 1)), sharex=True)
    for i in range(ndim):
        axes[i].plot(samples[:, :, i], alpha=0.6, lw=0.5)
        axes[i].axvline(burnin, color="k", ls="--", lw=1.)
        axes[i].set_ylabel(paramnames[i])
    axes[-1].plot(logps, alpha=0.6, lw=0.5)
    axes[-1].axvline(burnin, color="k", ls="--", lw=1.)
    axes[-1].set_ylabel("log prob")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    return fig


def plot_posterior_predictive(sampler, ells, data, cov, burnin, n_draws=500):
    import matplotlib.pyplot as plt

    ells_tau, ells_ksz, ells_bb = ells
    tau_models = sampler.get_blobs(flat=True, discard=burnin)["tau_models"]
    ksz_models = sampler.get_blobs(flat=True, discard=burnin)["ksz_models"]
    bb_models = sampler.get_blobs(flat=True, discard=burnin)["bb_models"]

    rng = np.random.default_rng()
    idx = rng.choice(tau_models.shape[0], size=min(n_draws, tau_models.shape[0]), replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, ells_obs, obs_data, obs_cov, models, title in zip(
        axes,
        [ells_tau, ells_ksz, ells_bb],
        [data["tau"], data["ksz"], data["bb"]],
        [cov["tau"], cov["ksz"], cov["bb"]],
        [tau_models, ksz_models, bb_models],
        ["Optical depth", "kSZ signal", "B-modes"],
    ):
        for i in idx:
            ax.plot(ells_obs, models[i], color="C0", alpha=0.05, lw=0.5)
        ax.errorbar(
            ells_obs, obs_data, yerr=np.sqrt(np.diag(obs_cov)),
            lw=0., elinewidth=0.8, marker='.', capsize=2., color='k')
        ax.set_title(title)
        ax.set_xlabel(r"Multipole $\ell$")
    fig.tight_layout()
    return fig


def summarize(flatsamples, truths, paramnames):
    """Print & return mean/std/bias/percent-error per parameter, as in
    read_mcmc.ipynb."""
    summary = {}
    print('ML parameters:')
    for j, name in enumerate(paramnames):
        mean, std = np.mean(flatsamples[:, j]), np.std(flatsamples[:, j])
        print(f' {name} = {mean:.2f} +/- {std:.3f} vs {truths[j]:.2f}')
        summary[name] = {"mean": mean, "std": std, "truth": truths[j]}
    print('Biases:')
    for j, name in enumerate(paramnames):
        bias_sigma = (summary[name]["mean"] - truths[j]) / summary[name]["std"]
        pct_error = summary[name]["std"] / truths[j] * 100. if truths[j] != 0 else np.nan
        print(f' {name} measured with {bias_sigma:.2f}sig bias and {pct_error:.2f}% error')
        summary[name]["bias_sigma"] = bias_sigma
        summary[name]["percent_error"] = pct_error
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read back and diagnose an emcee MCMC chain produced by preion-run-mcmc.")
    parser.add_argument("config", help="Path to the same YAML run config used with preion-run-mcmc.")
    parser.add_argument("--data", choices=("bb", "ksz", "tau", "all"), default=None,
                         help="Override the config's 'data' field (bb/ksz/tau/all).")
    parser.add_argument("--save-figures", action="store_true",
                         help="Save corner/trace/posterior-predictive figures under {output_dir}/figures.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.data is not None:
        cfg["data"] = args.data

    ells, data, cov = load_mock_data(cfg)
    sampler = load_chain(cfg)
    diagnostics = convergence_diagnostics(sampler)
    print(f"Auto-correlation time: {diagnostics['endtau']:.2f}. Converged: {diagnostics['converged']}.")
    print(f"burnin = {diagnostics['burnin']} ({diagnostics['burnin']/sampler.iteration:.1%})")
    print("Gelman-Rubin R-hat:")
    print(diagnostics["rhat"])

    flatsamples, _ = get_flat_samples(sampler, diagnostics["burnin"])
    truths = _theta_true(cfg)
    labels = _theta_labels(cfg)
    summarize(flatsamples, truths, PARAM_NAMES)

    fig_corner = plot_corner(flatsamples, truths, labels)
    fig_trace = plot_trace(sampler, PARAM_NAMES, diagnostics["burnin"])
    fig_pp = plot_posterior_predictive(sampler, ells, data, cov, diagnostics["burnin"])

    if args.save_figures:
        figure_dir = os.path.join(cfg["output_dir"], "figures")
        os.makedirs(figure_dir, exist_ok=True)
        label = run_label(cfg, cfg["data"])
        fig_corner.savefig(os.path.join(figure_dir, f"mcmc_{label}_corner.png"), dpi=220)
        fig_trace.savefig(os.path.join(figure_dir, f"mcmc_{label}_trace.png"), dpi=220)
        fig_pp.savefig(os.path.join(figure_dir, f"mcmc_{label}_posterior_predictive.png"), dpi=220)


if __name__ == "__main__":
    main()
