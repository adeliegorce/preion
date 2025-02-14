import sys
sys.path.insert(0, '../')

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
from astropy import cosmology
import triangleme2 as triangleme

from forecast_utils import make_datapoints, get_derivatives
from parameters import telescope_specs

cos = cosmology.Planck18
ells = np.arange(500, 10000, step=500)
low_ells = np.copy(ells)  # np.logspace(1, 2.5, 10)

nsamples = 10000
theta_true = [7.0, 1.5, 3.7, 0.10]
theta_labels = [r'$z_\mathrm{re}$', r'd$z$', r'log$\alpha_0$', r'$\kappa$']
dev = 0.05
index_list = [0, 1, 2]  # combine all observables
tel = telescope_specs['CMB-HD']
plot = True

# mock data
ksz_ps, tau_ps, total_bb, cov_ksz, cov_tau, cov_bb = make_datapoints(
    theta_true, ells, low_ells,
    tel=tel,
    cos=cos, save=None,
)
icov_list = [
    np.linalg.inv(cov_ksz),
    np.linalg.inv(cov_tau),
    np.linalg.inv(cov_bb)
]

# derivatives
deriv_list = get_derivatives(
    theta_true, ells, low_ells,
    run_derivatives=False, dev=dev
)

# fisher
fishcov_list = []
for io, obs in enumerate(['kSZ', 'tau', 'BB']):
    fishmat = np.zeros((len(theta_true), len(theta_true)))
    for i, deriv in enumerate([d[io] for d in deriv_list]):
        for j, deriv2 in enumerate([d[io] for d in deriv_list]):
            fishmat[i, j] = np.transpose(deriv).dot(icov_list[io]).dot(deriv2)
    fishcov = np.linalg.inv(fishmat)
    assert np.allclose(np.dot(fishmat, fishcov), np.eye(len(theta_true)))
    fishcov_list.append(fishcov)

# combination
cov_list = [[cov_ksz, cov_tau, cov_bb][i] for i in index_list]
cov_tot = np.diag(np.hstack([np.diag(cov) for cov in cov_list]))
icov_tot = np.linalg.inv(cov_tot)
# remove obs not used from derivatives
deriv_list2 = []
for l in deriv_list:
    deriv_list2.append([l[i] for i in index_list])
fish_all = np.zeros((len(theta_true), len(theta_true)))
for i, deriv in enumerate([np.hstack(l) for l in deriv_list2]):
    for j, deriv2 in enumerate([np.hstack(l) for l in deriv_list2]):
        fish_all[i, j] = np.transpose(deriv).dot(icov_tot).dot(deriv2)
fishcov_all = np.linalg.inv(fish_all)

if plot:

    plt.ion()
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    rc('font', **{'family': 'serif', 'serif': ['times new roman'], 'size': 16})
    rc('text', usetex=True)
    rc('axes', linewidth=1.5)

    icov_tau = np.linalg.inv(cov_tau)
    icov_ksz = np.linalg.inv(cov_ksz)
    icov_bb = np.linalg.inv(cov_bb)

    # mock gaussian datasets
    data_ksz = np.random.multivariate_normal(theta_true, fishcov_list[0], size=nsamples)
    data_tau = np.random.multivariate_normal(theta_true, fishcov_list[1], size=nsamples)
    data_bb = np.random.multivariate_normal(theta_true, fishcov_list[2], size=nsamples)
    data_all = np.random.multivariate_normal(theta_true, fishcov_all, size=nsamples)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    # tau
    axes[0].errorbar(
        ells, tau_ps, yerr=np.sqrt(np.diag(cov_tau)),
        lw=1., marker='.', capsize=2.)
    axes[0].set_xlabel(r'Multipole $\ell$')
    axes[0].set_ylabel(r'$\ell(\ell+1)C_\ell^{\tau\tau}/2\pi$')
    axes[0].set_yscale('log')
    axes[0].set_ylim(bottom=1e-8)
    axes[0].set_xlim(ells.min(), ells.max())
    # ksz
    axes[1].errorbar(
        ells, ksz_ps, yerr=np.sqrt(np.diag(cov_ksz)),
        lw=1., marker='.', capsize=2.)
    axes[1].set_xlabel(r'Multipole $\ell$')
    axes[1].set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu$K$^2$]')
    # B-modes
    axes[2].errorbar(
        low_ells, total_bb, yerr=np.sqrt(np.diag(cov_bb)),
        lw=1., marker='.', capsize=2.)
    axes[2].set_xlabel(r'Multipole $\ell$')
    axes[2].set_ylabel(r'$\ell(\ell+1)C_\ell^{BB}/2\pi$ [$\mu$K$^2$]')
    axes[2].set_ylim(bottom=1e-6)
    axes[2].set_yscale('log')

    for i, (letter, title) in enumerate(zip(['a', 'b', 'c'],
                                            ['Optical\n depth', 'kSZ signal', r'$B$-modes'])):
        axes[i].text(0.05, 0.05, title, transform=axes[i].transAxes, fontsize=17)# bbox=props)
    fig.tight_layout()

    # bounds = [(0, 50), (0, 15), (0, 10), (0, 0.3)]
    zorders = [0, 1, 2, 3]
    fig, axes = plt.subplots(4, 4, figsize=(12, 9))
    triangleme.corner(data_tau,
                    labels=theta_labels,
                    sigmas=[1, 2], plot_datapoints=False, 
                    cmap='Blues', color='C0', lw=2., smooth=1., alpha=.5,
                    #   extents=bounds, 
                    zorder=zorders[0],truth_color='k', 
                    fig=fig)
    triangleme.corner(data_ksz, #corrcoef=True,
                    sigmas=[1, 2], plot_datapoints=False, 
                    cmap='Oranges', color='orange', lw=2., smooth=1., alpha=.5,
                    #   extents=bounds, 
                    zorder=zorders[1],
                    fig=fig)
    triangleme.corner(data_bb,
                    labels=theta_labels,
                    sigmas=[1, 2], plot_datapoints=False, 
                    cmap='Greens', color='green', lw=2., smooth=1., alpha=.5,
                    #   extents=bounds, 
                    zorder=zorders[2],
                    fig=fig)
    triangleme.corner(data_all, plot_hist=False,
                    labels=theta_labels,
                    sigmas=[1, 2], plot_datapoints=False, 
                    cmap='Greys', color='k', lw=2., smooth=1., alpha=1,
                    #   extents=bounds, 
                    zorder=zorders[3],
                    truths=theta_true, truth_color='k',
                    fig=fig)
    plt.plot([], [], color='C0', label='Optical depth')
    plt.plot([], [], color='orange', label='kSZ')
    plt.plot([], [], color='green', label=r'$B$-modes')
    plt.plot([], [], color='k', label=r'All')
    fig.legend(loc=(0.3, 0.8), frameon=False)
    fig.tight_layout()


    bounds = [(0, 20), (0, 10), (0, 10), (0, 0.3)]
    fig, axes = plt.subplots(1, len(theta_labels)+1, figsize=(12, 3), gridspec_kw={'width_ratios':(1,1,1,1,0.7)}, sharey=True)
    for i, (obs, fishmat, color) in enumerate(zip(
        ['kSZ', 'Optical \ndepth', r'$B$-modes', 'All'],
        fishcov_list+[fishcov_all],
        ['orange', 'C0', 'green', 'k'])):
        sigma_err = np.sqrt(np.diag(fishmat))
        for u, (label, sig) in enumerate(zip(theta_labels,sigma_err)):
            xlin = np.linspace(theta_true[u]*(-10), theta_true[u]*10, 500)
            gauss = np.exp(- (xlin-theta_true[u])**2/2./sig**2)
            # gauss /= (sig * np.sqrt(2.*np.pi))
            axes[u].plot(xlin, gauss/gauss.max(), color=color, lw=2.)
            axes[u].set_xlabel(label)
            axes[u].set_xlim(bounds[u])
        axes[-1].plot([], [], color=color, label=obs, lw=2.)
    axes[0].set_ylabel('PDF')
    axes[-1].set_visible(False)
    fig.legend(loc=(0.85, 0.3), frameon=False)
    fig.tight_layout()
    # fig.savefig('figures/fisher_forecast_ideal_1D_posteriors.pdf', dpi=200)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (obs, fishmat, data, color) in enumerate(zip(
        ['kSZ', r'Optical depth', r'$B$-modes', 'All'],
        fishcov_list+[fishcov_all],
        [data_ksz, data_tau, data_bb, data_all],
        ['orange', 'C0', 'green', 'k'])):

        xlin = np.linspace(theta_true[0]*(-10), theta_true[0]*10, 500)
        r = fishmat[0, 1] / fishmat[0, 0] #/ fishmat[1, 1]
        if i != 3:
            axes[0].plot(xlin, xlin * r, color=color, label=obs)
        axes[0].scatter(data[::10, 0]-theta_true[0], data[::10, 1]-theta_true[1], color=color, alpha=0.05, zorder=i)
        print(np.cov(data.T)[0, 1],  r, fishmat[0,1])

        xlin = np.linspace(theta_true[2]*(-10), theta_true[2]*10, 500)
        r = fishmat[2, 3] / fishmat[2, 2] #/ fishmat[3, 3]
        if i != 3:
            axes[1].plot(xlin, xlin * r, color=color, label=obs)
        axes[1].scatter(data[::10, 2]-theta_true[2], data[::10, 3]-theta_true[3], color=color, alpha=0.05, zorder=i)
    axes[0].set_ylabel(theta_labels[1])
    axes[0].set_xlabel(theta_labels[0])
    axes[0].set_xlim(-10, 19)
    axes[0].set_ylim(-10, 10)
    axes[1].set_ylabel(theta_labels[3])
    axes[1].set_xlabel(theta_labels[2])
    axes[1].set_xlim(-8, 8)
    axes[1].set_ylim(-0.2, 0.2)
    axes[0].legend(frameon=False)
    for i in range(2):
        axes[i].axvline(0, color='k', ls=':')
        axes[i].axhline(0, color='k', ls=':')
    fig.tight_layout()
