import numpy as np
from main import *
import sys, glob, os, time
from multiprocessing import Pool, cpu_count
import emcee
import matplotlib.pyplot as plt
from give_RF import get_ksz
from parameters import *

ncpu = cpu_count()

########################################## DATA
print("Reading data...")
ells_data, ksz_data, ksz_err = np.loadtxt("ksz_data.txt", unpack=True)
k_data, z_data, data_21, err_21 = np.loadtxt("ps21_data.txt", unpack=True)

krange = np.unique(np.r_[np.logspace(-3, 1, 100), k_data])
lrange = np.unique(np.r_[np.linspace(100, 10000, 100), ells_data])
zrange = np.unique(np.r_[np.arange(0, 20), z_data])

def generate_model(values):
    res = KSZ_power(
        dz=values[1],
        zre=values[0],
        alpha0=alpha0,
        kappa=kappa,
        h=None,
        theta=thetaMC / 100.0,
        Ob_0=omegabh2,
        Om_0=omegach2,
        A_s=logA,
        n_s=n_s,
        z_early=z_early,
        xe_recomb=xe_recomb,
        verbose=False,
        cosmomc=True,
        run_CMB=False,
    )
    res.init_reionisation_history()
    res.run_camb(kmax_pk=kmax_pk)
    pksz = get_ksz(values, ells_data)
    p21 = np.array([res.P21(k, z, mK=True, units=False) for (k, z) in zip(k_data, z_data)])

    return np.array(pksz.flatten()), np.array(p21), get_ksz(values, lrange), np.array([res.P21(krange, z, mK=True, units=False) for z in zrange]), res.tau


# likelihood
def lnlike(theta):

    pksz, p21, model_ksz, model_21, tau = generate_model(theta)

    # ksz
    chi2s_ksz = -0.5 * ((ksz_data - pksz.flatten()) ** 2 / ksz_err ** 2)
    chi2_ksz = np.sum(chi2s_ksz)
    if np.any(pksz <= 0.0):
        chi2_ksz = -np.inf

    # 21cm
    chi2s_21 = []
    for mod21, dat21, err21 in zip(p21, data_21, err_21):
        if err21 == 0.0:
            if mod21 > dat21:
                chi2s_21.append(-np.inf)
            else:
                chi2s_21.append(0.0)
        else:
            chi2s_21.append(-0.5 * ((data_21 - p21.flatten()) ** 2 / err_21 ** 2))
    chi2_21 = np.sum(chi2s_21)

    return chi2_ksz, chi2_21, pksz, p21, model_ksz, model_21, tau


def lnprior(theta):
    for i, param in enumerate(theta):
        if (param < priors[i][0]) or (param > priors[i][1]):
            return -np.inf
    if theta[0] - theta[1] < zend_min:
        return -np.inf
    return 0.0


def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf, 0.0, 0.0, 0.0, 0., 0., 0., 0.
    chi2_ksz, chi2_21, pksz, p21, model_ksz, model_21, tau = lnlike(theta)
    return lp + chi2_21 + chi2_ksz, chi2_ksz, chi2_21, pksz, p21, model_ksz, model_21, tau


def main():

    #################################################### MAIN

    ########################################## INI
    ndim = len(bf_values)
    nwalkers = 4 * ndim
    nsteps = 10000

    # backend
    filename = "chains_%s.h5" % outroot
    backend = emcee.backends.HDFBackend(filename)
    if os.path.isfile(filename):
        print("Loading backend...")
        pos = None
    else:
        print("Initialising run...")
        backend.reset(nwalkers, ndim)
        # initial position
        pos = [bf_values + np.random.randn(ndim) * [0.1, 0.1] for i in range(nwalkers)]
        # check if initial positions are compatible with the priors
        test_pos = [lnprior(theta) for theta in pos]
        i = 0
        while np.any(np.isinf(test_pos)):
            pos = [
                bf_values + np.random.randn(ndim) * [0.1, 0.1] for i in range(nwalkers)
            ]
            test_pos = [lnprior(theta) for theta in pos]
            i += 1
            if i > 100:
                raise ValueError("Priors and initial walker positions uncompatible.")
        print("Likelihood for input values: %.1f" % np.sum(lnlike(bf_values)[:2]))

    # blobs
    dtype = [
        ("chi2_ksz", float, (1, )),
        ("chi2_21", float, (1, )),
        ("pksz_point", float, (np.size(ells_data), )),
        ("p21_point", float, (data_21.size, )),
        ("model_ksz", float, (lrange.size, )),
        ("model_21", float, (zrange.size, krange.size, )),
        ("tau", float, (1,)),
    ]

    # run
    print("Running MCMC....")
    # with Pool(ncpu) as pool:
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
    #                                     pool=pool,
    #                                     backend=backend,
    #                                     blobs_dtype=dtype)
    #     t0 = time.time()
    #     sampler.run_mcmc(pos, nsteps)#, progress=True)
    #     t1 = time.time()
    #     print('Done in %.1fhrs' %((t1-t0)/60./60.))

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, backend=backend, blobs_dtype=dtype
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    print(
        "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    )
    samples = sampler.get_chain()
    like = sampler.get_log_prob()

    # plot walkers
    labels = ["zre", "dz"]
    fig, axes = plt.subplots(ndim + 1, figsize=(ndim * 4, ndim * 2), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        for j in range(samples.shape[1]):
            ax.plot(samples[:, j, i], "k", alpha=0.3)
        ax.set_xlim(0, samples.shape[0])
        ax.set_ylabel(labels[i])
    for j in range(samples.shape[1]):
        axes[-1].plot(like[:, j], "k", alpha=0.3)
    axes[-1].set_ylabel(r"log$\mathcal{L}$")
    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    plt.savefig("walkers_%s.png" % outroot)


if __name__ == "__main__":
    # freeze_support()
    main()
