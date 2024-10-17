#!/usr/bin/env python
# coding: utf-8

# # Reionisation observables from the electron power spectrum

import time
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps, colors
import numpy as np
from astropy import cosmology, constants, units
import os

from theory import Pee_model
from utils import compute_cross_angular_spectrum
from simulations import gaussian_box_from_ps, gaussian_box_from_cl
from tqdm import tqdm

cos = cosmology.Planck18

klog = np.logspace(-3, 2, 500)
z = np.linspace(0, 20, 300)
ells = np.linspace(200, 10000, 20)
zarr = np.linspace(5., 15, 10)
karr = np.logspace(-2, 2, 30)
colorlist = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c','#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5','#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f','#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

# L = 500
N = 512
fov = 5. * units.deg

CL = 68
percentile1 = (100.-CL)/2.
percentile2 = CL+(100.-CL)/2.

nrand = 25

datafiles = ['data/Cl21_tau_stats.npy', 'data/Cl21_ksz_stats.npy', 'data/Cltau_ksz_stats.npy']
errfiles = ['data/Cl21_tau_err.npy', 'data/Cl21_ksz_err.npy', 'data/Cltau_ksz_err.npy']
lfile = 'data/cross_stats_l.txt'

if np.all([os.path.exists(file) for file in datafiles]):
    print('Loading from files...')
    Cl21_tau_list, Cl21_tau_err_list = np.load(datafiles[0]), np.load(errfiles[0])
    Cl21_ksz_list, Cl21_ksz_err_list = np.load(datafiles[1]), np.load(errfiles[1])
    Cltau_ksz_list, Cltau_ksz_err_list = np.load(datafiles[2]), np.load(errfiles[2])
    l_Cl21_tau, l_Cl21_ksz, l_Cltau_ksz = np.loadtxt(lfile, unpack=True)

else:

    print('Generating global model...')
    model = Pee_model(
        h=cos.h,
        Ob_0=cos.Ob0,
        Om_0=cos.Om0,
        verbose=False,
        run_camb=True
    )

    # observables
    ksz_ps = model.get_ksz(ells=ells, signal='both', Dells=True)
    tau_ps = model.get_tau(ells=ells, signal='both', Dells=True)
    ps_21 = model.get_p21(karr, zarr[:, None], mK=True, log=False, pk_units=True)

    Cl21_tau_list, Cl21_tau_err_list = [], []
    Cltau_ksz_list, Cltau_ksz_err_list = [], []
    Cl21_ksz_list, Cl21_ksz_err_list = [], []

    print('Simulating Gaussian realisations...')
    for it in tqdm(range(nrand)):

        # gaussian boxes
        iz = 4
        L21 = np.tan(fov.to(units.rad).value/2.) * 2. * cos.comoving_distance(zarr[iz]).value
        pk_array = np.c_[karr, ps_21[4]].T
        box_21 = gaussian_box_from_ps(N, L21, pk_array, ndim=3)

        pk_array = np.c_[ells, ksz_ps[:, 0]/ells/(ells + 1)*2.0*np.pi/(model.T_cmb * 1e6) ** 2].T
        ksz_box = gaussian_box_from_cl(
            N, fov.to(units.rad).value, 
            pk_array, ndim=3)

        pk_array = np.c_[ells, (tau_ps[:, 0]+tau_ps[:, 1])/ells/(ells + 1)*2.0*np.pi/(model.T_cmb * 1e6) ** 2].T
        tau_box = gaussian_box_from_cl(
            N, fov.to(units.rad).value, 
            pk_array, ndim=3)

        # cross spectra
        l_Cl21_ksz, Cl21_ksz, Cl21_ksz_err = compute_cross_angular_spectrum(ksz_box, box_21, fov.to(units.rad).value, nbins=30)
        Cl21_ksz_list.append(Cl21_ksz)
        Cl21_ksz_err_list.append(Cl21_ksz_err)

        l_Cl21_tau, Cl21_tau, Cl21_tau_err = compute_cross_angular_spectrum(tau_box, box_21, fov.to(units.rad).value, nbins=30)
        Cl21_tau_list.append(Cl21_tau)
        Cl21_tau_err_list.append(Cl21_tau_err)

        l_Cltau_ksz, Cltau_ksz, Cltau_ksz_err = compute_cross_angular_spectrum(ksz_box, tau_box, fov.to(units.rad).value, nbins=30)
        Cltau_ksz_list.append(Cltau_ksz)
        Cltau_ksz_err_list.append(Cltau_ksz_err)

    print('Saving...')
    np.save(datafiles[0], Cl21_tau_list)
    np.save(datafiles[1], Cl21_ksz_list)
    np.save(datafiles[2], Cltau_ksz_list)

    np.save(errfiles[0], Cl21_tau_err_list)
    np.save(errfiles[1], Cl21_ksz_err_list)
    np.save(errfiles[2], Cltau_ksz_err_list)

    np.savetxt(lfile, np.c_[l_Cl21_tau, l_Cl21_ksz, l_Cltau_ksz])
    print('Done.')

fig, ax = plt.subplots()

m = np.median(Cl21_ksz_list, axis=0) != 0.
print(np.sum(m)/Cl21_ksz_list[0].size)
prefac = l_Cl21_ksz[m] * (l_Cl21_ksz[m]+1.)/2./np.pi * (model.T_cmb * 1e6)
ax.fill_between(
    l_Cl21_ksz[m],
    np.percentile(Cl21_ksz_list, percentile2, axis=0)[m] * prefac,
    np.percentile(Cl21_ksz_list, percentile1, axis=0)[m] * prefac,
    color='plum', alpha=0.5
)
ax.plot(
    l_Cl21_ksz[m], np.median(Cl21_ksz_list, axis=0)[m] * prefac,
    color='plum', linestyle='-', linewidth=2,
    label=r'$Cl_{21\times \mathrm{kSZ}}$')

m = np.median(Cl21_tau_list, axis=0) != 0.
print(np.sum(m)/Cl21_tau_list[0].size)
prefac = l_Cl21_tau[m] * (l_Cl21_tau[m]+1.)/2./np.pi * (model.T_cmb * 1e6)
ax.fill_between(
    l_Cl21_tau[m],
    np.percentile(Cl21_tau_list, percentile2, axis=0)[m] * prefac,
    np.percentile(Cl21_tau_list, percentile1, axis=0)[m] * prefac,
    color='purple', alpha=0.5
)
ax.plot(
    l_Cl21_tau[m], np.median(Cl21_tau_list, axis=0)[m] * prefac,
    color='purple', linestyle='-', linewidth=2,
    label=r'$Cl_{21\times \tau}$')

m = np.median(Cltau_ksz_list, axis=0) != 0.
print(np.sum(m)/Cltau_ksz_list[0].size)
prefac = l_Cltau_ksz[m] * (l_Cltau_ksz[m]+1.)/2./np.pi * (model.T_cmb * 1e6)
ax.fill_between(
    l_Cltau_ksz[m],
    np.percentile(Cltau_ksz_list, percentile2, axis=0)[m] * prefac,
    np.percentile(Cltau_ksz_list, percentile1, axis=0)[m] * prefac,
    color='deeppink', alpha=0.5
)
ax.plot(
    l_Cltau_ksz[m], np.median(Cltau_ksz_list, axis=0)[m] * prefac,
    color='deeppink', linestyle='-', linewidth=2,
    label=r'$Cl_{\tau\times \mathrm{kSZ}}$')

ax.set_yscale('symlog', linthresh=1e-8)
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$\mathcal{D}_\ell$ [$\mu$K$^2$]')
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig('cross_stats.png', dpi=220)

