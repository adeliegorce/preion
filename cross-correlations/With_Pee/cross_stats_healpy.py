#!/usr/bin/env python
# coding: utf-8

# # Reionisation observables from the electron power spectrum

import time
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps, colors
import numpy as np
from astropy import cosmology, constants, units
import os
import healpy as hp
from scipy.interpolate import interp1d
from tqdm import tqdm

from theory import Pee_model
from utils import bin_spectrum

cos = cosmology.Planck18

# z = np.linspace(0, 20, 300)
ells = np.arange(200, 10000, step=250)
dell = np.diff(ells).mean()
ells_edges = np.arange(ells.min()-dell/2., ells.max()+dell, step=dell)

zarr = np.linspace(5., 15, 10)
karr = np.logspace(-3, 1, 100)

hp_ls_interp = np.arange(int(ells.max()))
ells_cross = np.arange(0, 2000, step=50)

nside = 2048
nrand = 25
CL = 68
percentile1=(100-CL)/2
percentile2=CL+(100-CL)/2

datafiles = ['data/P21_ksz_stats_healpy.npy', 'data/P21_2_ksz_stats_healpy.npy']
corrfiles = ['data/r21_ksz_stats_healpy.npy', 'data/r21_2_ksz_stats_healpy.npy']

if np.all([os.path.exists(file) for file in datafiles]):
    print('Loading from files...')
    binned_cl_ksz_21 = np.load(datafiles[0])
    binned_cl_ksz_21_2 = np.load(datafiles[1])
    # binned_rl_ksz_21 = np.load(corrfiles[0])
    # binned_rl_ksz_21_2 = np.load(corrfiles[1])

    imin = np.where(np.any(binned_cl_ksz_21[:, -1, :], axis=-1))[0][-1]
    print(imin/nrand)
    binned_cl_ksz_21 = binned_cl_ksz_21[:imin]
    binned_cl_ksz_21_2 = binned_cl_ksz_21_2[:imin]
    model = Pee_model(
        h=cos.h,
        Ob_0=cos.Ob0,
        Om_0=cos.Om0,
        verbose=False,
        run_camb=False
    )

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
    # tau_ps = model.get_tau(ells=ells, signal='both', Dells=True)
    ps_21 = model.get_p21(karr, zarr[:, None], mK=True, log=False, pk_units=True)

    print('Looping over redshifts...')
    binned_cl_ksz_21, binned_cl_ksz_21_2 = np.zeros((nrand, zarr.size, ells.size)),  np.zeros((nrand, zarr.size, ells.size))
    for j in tqdm(range(nrand)):

        cls_ksz = ksz_ps[:, 0]/ells/(ells+1)*2.*np.pi
        interp_ksz = interp1d(
            ells, cls_ksz,
            fill_value=0, bounds_error=False
        )  # mK2
        ksz_map = hp.synfast(interp_ksz(hp_ls_interp), nside=nside, ) / 1e3  # mK

        for iz in range(zarr.size):
            # gaussian boxes
            cls_21 = ps_21[iz]  # mK2
            ells_21 = karr * cos.comoving_distance(zarr[iz]).value
            interp_p21 = interp1d(ells_21, cls_21, fill_value=0, bounds_error=False)
            dTb_map = hp.synfast(interp_p21(hp_ls_interp), nside=nside, )  # mK

            # cross spectra
            cl_ksz_21 = hp.anafast(map1=ksz_map,  map2=dTb_map)
            lrange = np.arange(cl_ksz_21.size)
            binned_cl_ksz_21[j, iz] = bin_spectrum(
                lrange,
                cl_ksz_21*lrange*(lrange+1.)/2./np.pi,
                ells_edges,
            )
            np.save(datafiles[0], binned_cl_ksz_21)

            cl_ksz_21_2 = hp.anafast(map1=ksz_map,  map2=dTb_map**2)
            binned_cl_ksz_21_2[j, iz] = bin_spectrum(
                lrange,
                cl_ksz_21_2*lrange*(lrange+1.)/2./np.pi,
                ells_edges,
            )
            np.save(datafiles[1], binned_cl_ksz_21_2)

    print('Done.')

cmap = colormaps['viridis']
norm = colors.LogNorm(vmin=5e-4, vmax=1.)
nz = zarr.size

fig, axes = plt.subplots(2, nz//2, figsize=(15, 8), sharex=True, sharey=True)
for iz in range(nz):
    axes[iz//(nz//2), iz%(nz//2)].fill_between(
        ells,
        np.percentile(binned_cl_ksz_21[:, iz, :], percentile2, axis=0)*ells*(ells+1)/2./np.pi,
        np.percentile(binned_cl_ksz_21[:, iz, :], percentile1, axis=0)*ells*(ells+1)/2./np.pi,
        color=cmap(norm(model.xe(zarr[iz])[0]/model.f)), alpha=0.5
    )
    axes[iz//(nz//2), iz%(nz//2)].plot(
        ells,
        np.median(binned_cl_ksz_21[:, iz, :], axis=0)*ells*(ells+1)/2./np.pi,
        color=cmap(norm(model.xe(zarr[iz])[0]/model.f))
    )
    axes[iz//(nz//2), iz%(nz//2)].set_title(rf'$x_e={model.xe(zarr[iz])[0]:.4f}$')
    axes[iz//(nz//2), iz%(nz//2)].axhline(0, color='k', ls=':')
    if iz >= (nz//2):
        axes[iz//(nz//2), iz%(nz//2)].set_xlabel(r'Multipole $\ell$')
    if iz == 0 or iz == nz//2:
        axes[iz//(nz//2), iz%(nz//2)].set_ylabel(r'$\ell(\ell+1)C_\ell^{\mathrm{kSZ}\times \delta T_b}/2\pi$')
fig.tight_layout()
fig.savefig('figures/cl_kszx21_vs_z_stats.png', dpi=220)

fig, axes = plt.subplots(2, nz//2, figsize=(15, 8), sharex=True, sharey=True)
for iz in range(nz):
    axes[iz//(nz//2), iz%(nz//2)].fill_between(
        ells,
        np.percentile(binned_cl_ksz_21_2[:, iz, :], percentile2, axis=0)*ells*(ells+1)/2./np.pi,
        np.percentile(binned_cl_ksz_21_2[:, iz, :], percentile1, axis=0)*ells*(ells+1)/2./np.pi,
        color=cmap(norm(model.xe(zarr[iz])[0]/model.f)), alpha=0.5
    )
    axes[iz//(nz//2), iz%(nz//2)].plot(
        ells,
        np.median(binned_cl_ksz_21_2[:, iz, :], axis=0)*ells*(ells+1)/2./np.pi,
        color=cmap(norm(model.xe(zarr[iz])[0]/model.f))
    )
    axes[iz//(nz//2), iz%(nz//2)].set_title(rf'$x_e={model.xe(zarr[iz])[0]:.4f}$')
    axes[iz//(nz//2), iz%(nz//2)].axhline(0, color='k', ls=':')
    if iz >= (nz//2):
        axes[iz//(nz//2), iz%(nz//2)].set_xlabel(r'Multipole $\ell$')
    if iz == 0 or iz == nz//2:
        axes[iz//(nz//2), iz%(nz//2)].set_ylabel(r'$\ell(\ell+1)C_\ell^{\mathrm{kSZ}\times \delta T_b^2}/2\pi$')
fig.tight_layout()
fig.savefig('figures/cl_kszx212_vs_z_stats.png', dpi=220)
