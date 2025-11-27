import sys, os, glob
sys.path.insert(0, '../')

import numpy as np
from astropy import cosmology, units, constants
from scipy.interpolate import RegularGridInterpolator, interp1d
import warnings
import emcee

from theory import Pee_model
from utils_tau import tau_noise

from parameters import telescope_specs, nu21_ref

theta_labels = [r'$z_\mathrm{re}$', r'd$z$', r'log$\alpha_0$', r'$\kappa$']
prior_bounds = [(5.5, 10.), (0., 4.), (0, 6.), (0, 0.4)]


def check_samples(key, filelist):
    has_samples = []
    for filename in filelist:
        try:
            _ = emcee.backends.HDFBackend(filename, read_only=True).get_blobs(flat=True)[key]
            has_samples.append(True)
        except ValueError:
            has_samples.append(False)
    print(f'Samples have {key}? {np.all(has_samples)}')
    return np.all(has_samples)


def flat_prior(theta, priors):
    for i, p in enumerate(priors):
        low, high = p
        if not (low <= theta[i] <= high):
            return -np.inf
    return 0.


def gaussian_prior(param, gaus_params):
    if gaus_params is None:
        return 0.
    [mu, sigma] = gaus_params
    return -0.5 * ((param - mu) ** 2) / (sigma ** 2)


def lower_limit(param, lolim):
    if lolim is None:
        return 0.
    elif param < lolim:
        return -np.inf
    return 0.


def get_sensitivity(label, zs=[5.5, 6., 6.5, 7., 8., 9., 10., 11., 12., 14.], error_type='thermal', debug=False):

    zs = np.sort(zs)[::-1]
    freq_bands = np.round(nu21_ref / (zs + 1.0), decimals=1)  # MHz

    sensitivity_at_z_arr = []
    sensitivity_at_z_kbins = []
    for iz in range(zs.size):
        if error_type == 'both':
            out1 = np.loadtxt(f"../sensitivity_21cm/{label}_thermal_{freq_bands[iz]}.txt")
            out2 = np.loadtxt(f"../sensitivity_21cm/{label}_sample_{freq_bands[iz]}.txt")
            out = out1 + out2
        else:
            out = np.loadtxt(f"../sensitivity_21cm/{label}_{error_type}_{freq_bands[iz]}.txt")
        sensitivity_at_z_arr.append(out[:, 1])
        sensitivity_at_z_kbins.append(out[:, 0])

    sensitivity_at_z = RegularGridInterpolator(
        (zs, sensitivity_at_z_kbins[0]),
        np.array(sensitivity_at_z_arr),
        method='linear',
        fill_value=np.inf,
        bounds_error=False
    )

    if debug:
        print(np.isnan(sensitivity_at_z_arr).sum())
        print(np.min(sensitivity_at_z_kbins), np.max(sensitivity_at_z_kbins))
        print(sensitivity_at_z([6.8, 1.1]))

    return sensitivity_at_z


def get_cl21_noise(noise_function, z21, ells, Dells=True, delta_nu=0., nz=100, freq_res=100.*units.kHz, cos=cosmology.Planck18):
    """
    Compute angular cross spectrum of 21cm at ell centred on z, in uK.

    Parameters
    ----------
        z: float
            Redshift of the observed 21cm signal.
        ells: array of floats
            Angular multipole to compute the spectrum at.
        Dells: boolean
            If True, give the results in terms of
            D(l) = l(l+1)Cl/2/pi.
            Default is False.
        delta_nu: float
            Width of the top-hat function representing the
            frequency resolution of the interferometer, in MHz.
            Default is zero (Dirac delta).
    Outputs
    ------
        Array of angular auto-power for ells, in uK2.
    """

    ells = np.atleast_1d(ells)
    delta_nu *= units.MHz
    nu21_ref_unit = nu21_ref * units.MHz

    if delta_nu == 0.:
        zlin = np.r_[np.linspace(z21-0.1, z21, nz//2), np.linspace(z21, z21+0.1, nz//2)[1:]]
    else:
        nu21 = nu21_ref_unit / (1. + z21)  # MHz
        numax = nu21 + delta_nu/2.
        numin = nu21 - delta_nu/2.
        zmax = (nu21_ref_unit / numin) - 1.
        zmin = (nu21_ref_unit / numax) - 1.
        zlin = np.linspace(zmin.value-0.1, zmax.value+0.2, nz)

    W21_zlin = np.zeros_like(zlin) * 1. / units.MHz
    iz21 = np.argmin(np.abs(zlin - z21))
    if delta_nu == 0.:
        W21_zlin[iz21] = 1. / units.MHz
    else:
        nu_integ = nu21_ref_unit / (1. + zlin)  # MHz
        # print(f'{delta_nu:.2f}, {np.abs(np.diff(nu_integ)[iz21]):.2f}')
        if np.abs(np.diff(nu_integ)[iz21]) > delta_nu:
            raise ValueError('dnu smaller than zlin resolution '
                             f'({delta_nu:.2f}, {np.abs(np.diff(nu_integ)[iz21]):.2f}), '
                             'increase nz.')
        mask = (zlin < zmax) & (zlin >= zmin)
        W21_zlin[mask] = 1./(numax-numin)  # unit T
    W21_zlin *= nu21_ref_unit * cos.H(zlin).si / (1+zlin)**2 / constants.c.si  # units L-1
    W21_zlin = W21_zlin.to(1./units.Mpc)
    W21_zlin /= np.trapz(W21_zlin, cos.comoving_distance(zlin))

    eta_lin = cos.comoving_distance(zlin).value
    # Compute C_tau(ell) integrand, unit 1
    C_ell_21_integrand = np.zeros((ells.size, zlin.size))
    for j, ell in enumerate(ells):
        k_zlin = ell / eta_lin  # [Mpc-1]
        N21 = (noise_function(np.c_[zlin, k_zlin])/k_zlin**3*2*np.pi**2)
        N21[np.isnan(N21) | np.isinf(N21)] = 0.
        C_ell_21_integrand[j] = (
            N21
            / eta_lin**2
            * W21_zlin.to(1./units.Mpc).value**2
        )
    # Compute C_kSZ(ell), no units
    C_ells = np.trapz(C_ell_21_integrand, eta_lin, axis=1)
    C_ells[C_ells == 0.] = np.inf
    # account for average over frequency channels
    C_ells /= np.sqrt(delta_nu.to(units.MHz).value / freq_res.to(units.MHz).value)

    if not Dells:
        return C_ells * units.mK**2
    else:
        return ells * (ells + 1.) * C_ells / 2. / np.pi * units.mK**2


def get_lbins(tel_tau, lbin_edges=None):

    assert tel_tau in telescope_specs.keys()
    lmin, lmax = telescope_specs[tel_tau]['lmin'], telescope_specs[tel_tau]['lmax']
    dell = telescope_specs[tel_tau]['Delta_ell']
    if lbin_edges is None:
        if tel_tau.find('SAT') >= 0:
            lbin_edges = np.r_[lmin, np.arange(dell, lmax+dell, step=dell)]
        else:
            lbin_edges = np.arange(lmin, lmax+dell, step=dell)
            if lbin_edges[-1] > lmax:
                lbin_edges[-1] = lmax
        dells = np.ones(lbin_edges.size - 1) * dell
    else:
        lbin_edges = np.sort(lbin_edges)
        dells = np.diff(lbin_edges)
        if np.min(lbin_edges) < lmin:
            warnings.warn(f'Min l below {tel_tau} limit, replacing value.')
            lbin_edges[0] = lmin
        if np.max(lbin_edges) > lmax:
            warnings.warn(f'Max l above {tel_tau} limit, replacing value.')
            lbin_edges[-1] = lmax
        if not np.allclose(dells, dells):
            warnings.warn(f'Delta l diff from {tel_tau} spec ({dells} vs. {dell}).')
    ells = 0.5 * (lbin_edges[1:] + lbin_edges[:-1])
    return ells, lbin_edges, dells


def make_datapoints(
        theta, tel_tau, label21='ska_aast_optimistic',
        delta_nu=100.*units.MHz, zs=[7.],
        use_ksz_emulator=False, randomness=False, lbin_edges=None,
        cos=cosmology.Planck18, save=None, verbose=True):

    if verbose:
        print(f'Computing sensitivity for {tel_tau} x {label21}, '
              f'with a {delta_nu:.1f} bandwidth.')

    # tau
    ells, lbin_edges, dells = get_lbins(tel_tau, lbin_edges=lbin_edges)
    llin = np.arange(lbin_edges[0], lbin_edges[-1]+1)
    # indlin = np.digitize(llin, lbin_edges)

    preion_model = Pee_model(
        zre_h=theta[0], dz_h=theta[1],
        alpha0=theta[2], kappa=theta[3],
        h=cos.h, Ob_0=cos.Ob0, Om_0=cos.Om0,
        verbose=False, run_camb=True,
        use_ksz_emulator=use_ksz_emulator)

    datafile = f'../data/dltau_{tel_tau}.txt'
    if os.path.exists(datafile):
        ells2, Dl_tautau, Nl_tautau = np.loadtxt(datafile, unpack=True)
        if not np.array_equal(ells2, ells):
            print(f'Warning: {datafile} does not match the current ells, inter(extra)polating...')
            m_inf = ~np.isinf(Nl_tautau)
            Dl_tautau = interp1d(ells2[m_inf], Dl_tautau[m_inf], bounds_error=False, fill_value='extrapolate')(ells)
            Nl_tautau = interp1d(ells2[m_inf], Nl_tautau[m_inf], bounds_error=False, fill_value='extrapolate')(ells)
        print(f'Loaded data from {datafile}')
    else:
        print(f'No data file found at {datafile}, computing...')
        Dl_tautau = np.sum(preion_model.get_tau(ells=ells, signal='both', Dells=True), axis=1)
        ls_temp = np.arange(int(telescope_specs[tel_tau]['lmax'])+1)
        CMB_Cells = preion_model.get_primary_spectra(ells=ls_temp, Dells=False)
        CMB_Cells = CMB_Cells[:, [0, 1, 2, 4, 3]]
        tau_temp = np.r_[0., np.sum(preion_model.get_tau(ells=ls_temp[1:], signal='both', Dells=False), axis=1)]
        tau_noise_res = tau_noise(tel_tau, np.c_[CMB_Cells, tau_temp], is_cl=False)
        Nl_tautau = interp1d(ls_temp, tau_noise_res)(ells)
        np.savetxt(datafile, np.c_[ells, Dl_tautau, Nl_tautau], fmt='%.6e', header='l, Dl_tautau, Nl_tautau (noise only)')
        print(f'Saved data to {datafile}.')
    if randomness:
        Dl_tautau = np.random.normal(Dl_tautau, Nl_tautau)

    # 21cm obs parameters
    if label21.find('ska') >= 0:
        fov_ska = (9.10*units.deg)**2
        fsky = fov_ska.to(units.rad**2)/(4.*np.pi*units.rad**2)
        if label21.find('fields') >= 0:
            fsky *= float(label21[label21.find('fields')-1])
    elif label21.find('hera') >= 0:
        fsky = 0.5*(np.cos(55*np.pi/180.)-np.cos(65*np.pi/180.))  # hera
    elif label21.find('mwa') >= 0:
        fov_mwa = (24.6*units.deg)**2
        fsky = fov_mwa.to(units.rad**2)/(4.*np.pi*units.rad**2)  # mwa
    sensitivity = get_sensitivity(label21, error_type='thermal')

    # cross
    Dl_tau21 = np.zeros((np.size(zs), np.size(ells)))
    Dltau21_err = np.zeros((np.size(zs), np.size(ells)))
    for iz, z21 in enumerate(zs):
        Dl_21 = preion_model.get_cl21(z21, ells, Dells=True, delta_nu=delta_nu).to(units.uK**2).value
        Nl_21 = get_cl21_noise(sensitivity, z21, ells, Dells=True, delta_nu=delta_nu.value).to(units.uK**2).value
        if np.isinf(Nl_21).any():
            print(f'Warning: 21cm noise values at l={ells[np.isinf(Nl_21)]} are infinite.')
        Dl_tau21[iz] = preion_model.get_tau_21_cross(z21, ells, Dells=True, delta_nu=delta_nu.value).to(units.uK).value
        # average over values inside multipole bin
        # siglin = preion_model.get_tau_21_cross(z21, llin, Dells=True, delta_nu=delta_nu.value).to(units.uK).value
        # Dl_tau21[iz] = np.array([np.mean(siglin[indlin == i+1]) for i in range(len(lbin_edges)-1)])
        Dltau21_err[iz] = 1. / (2.*ells+1) / fsky / dells * (
                Dl_tau21[iz]**2 + ((Dl_21 + Nl_21) * (Dl_tautau + Nl_tautau))
            )
        if verbose:
            cumu_snr2 = (Dl_tau21[iz]**2/Dltau21_err[iz]).sum()
            # cumu_snr = Dl_tau21[iz].mean() / np.sqrt(Dltau21_err[iz]).mean() * np.sqrt(ells.size)
            print(f'z = {z21:.1f}, cumulative SNR = {np.sqrt(cumu_snr2):.2f}')

    if (save is not None):# and (len(glob.glob(f'data/taux21_{tel_tau}_{label21}_z*_datapoints.txt'))==0):
        for iz, z21 in enumerate(zs):
            np.savetxt(
                f'data/taux21_{tel_tau}_{label21}_z{z21:.1f}_datapoints.txt',
                np.c_[ells, Dl_tau21[iz], np.sqrt(Dltau21_err[iz])],
                header='ell, Dl [uK], err [uK]'
            )

    return ells, Dl_tau21, np.sqrt(Dltau21_err)
