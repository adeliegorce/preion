import sys
sys.path.insert(0, '../')

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from astropy import cosmology
import triangleme2 as triangleme
import time
import emcee
from multiprocessing import Pool
import os

from forecast_utils import make_datapoints
from parameters import telescope_specs
from theory import Pee_model

cos = cosmology.Planck18
# ells = [np.linspace(10, 1000, 50), np.arange(1000, 8000, step=500), np.linspace(10, 1000, 50)]
telescopes = ['CMB-HD', 'CMB-HD', 'CMB-HD']
ells = []
for tel in telescopes:
    ells.append(np.arange(telescope_specs[tel]['lmin'], telescope_specs[tel]['lmax'], step=telescope_specs[tel]['Delta_ell']))
label = 'CMB-HD-full_RF'
randomness = False
overwrite = True
use_ksz_emulator = 'RF'

niterations = 20000
nwalkers = 8

theta_true = [7.5, 2., 4., 0.13]#[7.0, 1.5, 3.7, 0.10]
ndim = len(theta_true)
theta_labels = [r'$z_\mathrm{re}$', r'd$z$', r'log$\alpha_0$', r'$\kappa$']
priors = [(5., 10.), (0.1, 4.5), (2.5, 4.5), (0.05, 0.4)]

plot = True

# mock data
tau_data, ksz_data, bb_data, cov_tau, cov_ksz, cov_bb = make_datapoints(
    theta_true,
    ells=ells,
    telescopes=telescopes, randomness=randomness,
    cos=cos, save=label, use_ksz_emulator=use_ksz_emulator,
)

preion_model = Pee_model(
    h=cos.h, Ob_0=cos.Ob0, Om_0=cos.Om0,
    verbose=False, run_camb=True,
    use_ksz_emulator=use_ksz_emulator)
preion_model.run_camb = False


def get_model(theta, ells=ells):

    preion_model.zre_h = theta[0]
    preion_model.dz_h = theta[1]
    preion_model.alpha0 = theta[2]
    preion_model.kappa = theta[3]
    preion_model.init_reionisation_history()

    tau_ps = preion_model.get_tau(ells=ells[0], signal='both', Dells=True).sum(axis=1)
    ksz_ps = preion_model.get_ksz(ells=ells[1], signal='patchy', Dells=True)[:, 0]
    total_bb = preion_model.get_B_modes(ells=ells[2], Dells=True, Qrms=17.0)

    return tau_ps, ksz_ps, total_bb


def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, 0., 0., 0.
    ln, tau_model, ksz_model, bb_model = lnlike(theta)
    return lp + ln, tau_model, ksz_model, bb_model


def lnlike(theta):
    tau_ps, ksz_ps, bb_ps = get_model(theta)
    tau_like = (tau_ps - tau_data).T.dot(np.linalg.inv(cov_tau)).dot(tau_ps - tau_data)
    ksz_like = (ksz_ps - ksz_data).T.dot(np.linalg.inv(cov_ksz)).dot(ksz_ps - ksz_data)
    bb_like = (bb_ps - bb_data).T.dot(np.linalg.inv(cov_bb)).dot(bb_ps - bb_data)
    return -0.5 * (ksz_like+tau_like+bb_like), tau_ps, ksz_ps, bb_ps


def lnprior(theta, priors=priors):
    for i, p in enumerate(priors):
        low, high = p
        if not (low <= theta[i] <= high):
            return -np.inf
    return 0.


t0 = time.time()
print(lnlike(theta_true)[0])
t1 = time.time()
print(f'It takes {t1-t0:.1f} seconds to compute one model.')
# blobs & backend
if os.path.isfile(f'backends/mcmc_{label}_backend.h5') and overwrite:
    os.remove(f'backends/mcmc_{label}_backend.h5')
backend = emcee.backends.HDFBackend(f'backends/mcmc_{label}_backend.h5')
if overwrite:
    backend.reset(nwalkers, ndim)
dtype = [("tau_models", float, (np.size(ells[0]),)),
         ("ksz_models", float, (np.size(ells[1]),)),
         ("bb_models", float, (np.size(ells[2]),)),]

p0 = [np.random.uniform(low, high, size=nwalkers) for low, high in priors]
p0 = np.vstack(p0).T

# with Pool(6) as pool:
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, lnprob,
    backend=backend,
    # pool=pool,
    blobs_dtype=dtype,)
t0 = time.time()
state = sampler.run_mcmc(p0, niterations, progress=False)
t1 = time.time()
print(f'It took {(t1-t0)/60./60.:.1f} hours to run {niterations} iterations with {nwalkers} walkers.')
# sampler = emcee.backends.HDFBackend('mcmc_backend.h5', read_only=True)

samples = sampler.get_chain(flat=False)

# auto-correlation analysis to assess convergence and define burn-in
taus = sampler.get_autocorr_time(tol=0)
if np.isnan(taus).any():
    print('NaN tau. Taking max.')
endtau = np.nanmax(taus)
converged = np.all(taus * 60 < sampler.iteration)
print('Auto-correlation time: %.2f. Converged: %s.' %(endtau, converged))
burnin = int(max(0.1*samples.shape[0],2.*np.max(taus)))
print('burnin = %.1f' %(burnin/samples.shape[0]))

flatsamples = sampler.get_chain(flat=True, discard=burnin)
logps = sampler.get_log_prob(flat=True, discard=burnin)
cov = np.cov(flatsamples.T)
fom = np.sqrt(np.linalg.det(cov))
print(f'FoM = {fom:.1e}')

if plot:

    fig = triangleme.corner(
        flatsamples,
        truths=theta_true,
        labels=theta_labels, truth_color='k',
        sigmas=[1, 2], plot_datapoints=False, 
        # cmap='Reds', color='red', 
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
        # norm=colors.LogNorm(),
        extent=(0, 2, 2, 0))
    ax.set_xticks(np.arange(0.25, 2, step=.5), labels=theta_labels)
    ax.set_yticks(np.arange(0.25, 2, step=.5), labels=theta_labels)
    cbar = plt.colorbar(im, fraction=0.045, ax=ax, label=r'$\mathcal{C}_{ij}$')
    fig.tight_layout()
    fig.savefig(f'figures/mcmc_{label}_covmat.png', dpi=220)
