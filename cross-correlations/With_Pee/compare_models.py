#!/usr/bin/env python
# coding: utf-8

# # Reionisation observables from the electron power spectrum

import time
import matplotlib.pyplot as plt
from matplotlib import rc, colormaps, colors
import numpy as np
from astropy import cosmology, constants, units

from theory import Pee_model
from utils import plot_field_theta, plot_field, compute_cross_spectrum
from simulations import gaussian_box_from_ps, gaussian_box_from_cl

cos = cosmology.Planck18

klog = np.logspace(-3, 2, 500)
z = np.linspace(0, 20, 300)
ells = np.linspace(200, 10000, 20)
zarr = np.linspace(5., 15, 10)
karr = np.logspace(-2, 2, 30)

L = 500.
N = 512
fov = 5. * units.deg

linestyles = ['-', '--']

fig, axes = plt.subplots(1, 2, figsize=(10,6))
axes[0].set_xlabel(r'Redshift')
axes[0].set_ylabel(r'Reionisation history')
# for alpha0 in [3., 4.]:
# for u, zre in enumerate([6.5, 8.]):
#     print(f'\nzre = {zre:.1f}')
for u, dz in enumerate([.5, 1.5]):
    print(f'\ndz = {dz:.1f}')
    model = Pee_model(
        h=cos.h,
        Ob_0=cos.Ob0,
        Om_0=cos.Om0,
        # zre_h=zre,
        dz_h=dz,
        verbose=True,
        run_camb=True
    )

    axes[0].plot(z, model.xe(z), color='C0', ls=linestyles[u])

    # observables
    ksz_ps = model.get_ksz(ells=ells, signal='both', Dells=True)
    tau_ps = model.get_tau(ells=ells, signal='both', Dells=True)
    ps_21 = model.get_p21(karr, zarr[:, None], mK=True, log=False, pk_units=True)

    # gaussian boxes
    iz = 4
    pk_array = np.c_[karr, ps_21[4]].T
    box_21 = gaussian_box_from_ps(N, L, pk_array, ndim=3)

    pk_array = np.c_[ells, ksz_ps[:, 0]/ells/(ells + 1)*2.0*np.pi/(model.T_cmb * 1e6) ** 2].T
    ksz_box = gaussian_box_from_cl(
        N, fov.to(units.rad).value, 
        pk_array, ndim=3)

    pk_array = np.c_[ells, (tau_ps[:, 0]+tau_ps[:, 1])/ells/(ells + 1)*2.0*np.pi/(model.T_cmb * 1e6) ** 2].T
    tau_box = gaussian_box_from_cl(
        N, fov.to(units.rad).value, 
        pk_array, ndim=3)

    # cross spectra
    k, P21_tau, P21_tau_err = compute_cross_spectrum(tau_box, box_21, L, nbins=30)
    k, P21_ksz, P21_ksz_erer = compute_cross_spectrum(ksz_box, box_21, L, nbins=30)
    k, Ptau_ksz, Ptau_ksz_err = compute_cross_spectrum(ksz_box, tau_box, L, nbins=30)

    axes[1].errorbar(k[P21_tau>0.], P21_tau[P21_tau>0.], yerr= P21_tau_err[P21_tau>0.], color='purple', ls=linestyles[u])
    axes[1].errorbar(k[P21_ksz>0.], P21_ksz[P21_ksz>0.], yerr=P21_ksz_erer[P21_ksz>0.], color='plum', ls=linestyles[u])
    axes[1].errorbar(k[Ptau_ksz>0.], Ptau_ksz[Ptau_ksz>0.], yerr=Ptau_ksz_err[Ptau_ksz>0], color='deeppink', ls=linestyles[u])

axes[1].loglog([], [], color='purple', label=r'$P_{21\times \tau}$')
axes[1].plot([], [], color='plum', label=r'$P_{21\times \mathrm{kSZ}}$')
axes[1].plot([], [], color='deeppink', label=r'$P_{\mathrm{kSZ}\times \tau}$')

axes[1].set_xlabel(r'$k$ [Mpc$^{-1}]$')
axes[1].set_ylabel(r'$P(k, z)$ [Mpc$^3]$')
axes[1].legend()

fig.tight_layout()
fig.savefig('compare_cross_dz.png', dpi=220)

