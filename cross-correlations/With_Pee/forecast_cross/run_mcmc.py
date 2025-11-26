import sys
sys.path.insert(0, '../')

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from astropy import cosmology, units
import triangleme2 as triangleme
import time
import emcee
import multiprocessing as mp
import os
import copy

from forecast_utils import make_datapoints, get_lbins, gaussian_prior, flat_prior, lower_limit
from parameters import telescope_specs
from theory import Pee_model

cos = cosmology.Planck18

randomness = False
overwrite = False
use_ksz_emulator = 'RF'

zend_prior = 4.5
tau_prior = [0.055, 0.007]  # effectively use the tau from true model

tel_tau = sys.argv[1]  # 'CMB-S4-LAT'
label21 = sys.argv[2]  # 'hera_moderate'
delta_nu = float(sys.argv[3])*units.MHz
zs = np.atleast_1d(np.array(sys.argv[4].split(','), dtype=float))  # e.g., '7.0,8.0'
label = f'taux21_{tel_tau}_{label21}'
if zend_prior is not None:
    label += '_zendprior'
if tau_prior is not None:
    label += '_tauprior'
print(label)

niterations = 50000
nwalkers = 8
ncores = mp.cpu_count()
print(f'\nRunning on {ncores} CPUs with {nwalkers} walkers for {niterations} iterations.')

theta_true = [7.0, 1.5, 3.7, 0.10]
ndim = len(theta_true)
theta_labels = [r'$z_\mathrm{re}$', r'd$z$', r'log$\alpha_0$', r'$\kappa$']
priors = [(5., 10.), (0.1, 4.5), (2.5, 4.5), (0.05, 0.4)]

plot = True

# mock data
ells, tau21_data, tau21_err = make_datapoints(
    theta_true, tel_tau,
    delta_nu=delta_nu, label21=label21,
    zs=zs,
    randomness=randomness,
    cos=cos, save=label, use_ksz_emulator=use_ksz_emulator,
)

preion_model = Pee_model(
    h=cos.h, Ob_0=cos.Ob0, Om_0=cos.Om0,
    verbose=False, run_camb=True,
    use_ksz_emulator=use_ksz_emulator)
preion_model.run_camb = False
if tau_prior is not None:
    tau_prior[0] = copy.copy(preion_model.tau)


def get_model(theta, ells=ells):

    preion_model.zre_h = theta[0]
    preion_model.dz_h = theta[1]
    preion_model.alpha0 = theta[2]
    preion_model.kappa = theta[3]
    preion_model.init_reionisation_history()

    model = []
    ps21 = []
    for z21 in zs:
        model.append(preion_model.get_tau_21_cross(z21, ells, Dells=True, delta_nu=delta_nu.value).to(units.uK).value)
        ps21.append(preion_model.get_cl21(z21, ells, Dells=True, delta_nu=delta_nu).to(units.uK**2).value)
    tau_ps = np.sum(preion_model.get_tau(ells=ells, signal='both', Dells=True), axis=1)
    DkSZ = preion_model.get_ksz(ells=[3000], signal='both', Dells=True)

    return np.array(model), tau_ps, np.array(ps21), preion_model.tau, DkSZ


def lnprob(theta):
    lp = flat_prior(theta, priors)
    lp += lower_limit(theta[0] - theta[1], zend_prior)
    if not np.isfinite(lp):
        return -np.inf, 0., 0., 0., 0., 0.
    ln, model, tau_ps, ps21, tau, DkSZ = lnlike(theta)
    lp += gaussian_prior(tau, tau_prior)
    return lp + ln, model, tau_ps, ps21, tau, DkSZ


def lnlike(theta):
    model, tau_ps, ps21, tau, DkSZ = get_model(theta)
    m_inf = ~np.isinf(tau21_err)
    like = (model[m_inf] - tau21_data[m_inf])**2 / tau21_err[m_inf]**2
    return -0.5 * like.sum(), model, tau_ps, ps21, tau, DkSZ


t0 = time.time()
print(f'Likelihood of true parameters: {lnprob(theta_true)[0]:.2f}')
t1 = time.time()
print(f'It takes {t1-t0:.1f} seconds to compute one model.')
# blobs & backend
if os.path.isfile(f'backends/mcmc_{label}_backend.h5') and overwrite:
    os.remove(f'backends/mcmc_{label}_backend.h5')
backend = emcee.backends.HDFBackend(f'backends/mcmc_{label}_backend.h5')
if overwrite:
    backend.reset(nwalkers, ndim)
dtype = [("Dl_tau21", float, (np.size(zs), np.size(ells),)),
         ("Dl_tautau", float, (np.size(ells),)),
         ("Dl_21", float, (np.size(zs), np.size(ells),)),
         ("taus", float, (1,)),
         ("DkSZ", float, (2,)),
         ]

lp0 = np.inf
ip0 = 0
while np.isinf(lp0).any():
    p0 = [np.random.uniform(low, high, size=nwalkers) for low, high in priors]
    p0 = np.vstack(p0).T
    lp0 = [lnprob(pset)[0] for pset in p0]
    ip0 += 1
    if ip0 == 1000:
        raise ValueError('Cannot find initial positions compatible with the priors.')
print(f'Likelihood at initial positions: {lp0}.')

with mp.Pool(ncores) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob,
        backend=backend,
        pool=pool,
        blobs_dtype=dtype,)
    t0 = time.time()
    state = sampler.run_mcmc(p0, niterations, progress=False)
    t1 = time.time()
print(f'It took {(t1-t0)/60./60.:.1f} hours to run {niterations} iterations with {nwalkers} walkers.')

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
