##############################################
######## Computes kSZ power spectrum #########
### Copyright Stephane Ilic & Adélie Gorce ###
##############################################

import camb
from camb import model, initialpower
import numpy as np
import sys
import multiprocessing
from scipy.integrate import simps, cumtrapz, quad, trapz
from scipy import interpolate
from astropy import cosmology, units, constants

#######################################
########### System settings ###########
#######################################

Mpcm = (1.0 * units.Mpc).to(units.m).value  # one Mpc in [m]
Mpckm = Mpcm / 1e3

#######################################
###### REIONISATION PARAMETERS ########
#######################################

# reionisation of Helium
helium_fullreion_redshift = 3.5
helium_fullreion_start = 5.0
helium_fullreion_deltaredshift = 0.5

########################################
#### Integration/precision settings ####
########################################
### Settings for theta integration
num_th = 50
th_integ = np.linspace(0.0001, np.pi * 0.9999, num_th)
mu = np.cos(th_integ)  # cos(k.k')
### Settings for k' (=kp) integration
# k' array in [Mpc-1] - over which you integrate
min_logkp = -5.0
max_logkp = 1.5
dlogkp = 0.05
kp_integ = np.logspace(min_logkp, max_logkp, int((max_logkp - min_logkp) / dlogkp) + 1)
# minimal and maximal values valid for CAMB interpolation of
kmax_camb = 6.0e3
kmin_camb = 7.2e-6
krange_camb = np.logspace(np.log10(kmin_camb), np.log10(kmax_camb), 500)
### Settings for z integration
z_recomb = 1100.0
z_min = 0.10
z_piv = 1.0
z_max = 20.0
dlogz = 0.1
dz = 0.15
z_integ = np.concatenate(
    (
        np.logspace(
            np.log10(z_min),
            np.log10(z_piv),
            int((np.log10(z_piv) - np.log10(z_min)) / dlogz) + 1,
        ),
        np.arange(z_piv, 10.0, step=dz),
        np.arange(10, z_max + 0.5, step=0.5),
    )
)
z3 = np.linspace(0, z_recomb, 10000)


class KSZ_power:
    def __init__(
        self,
        dz=1.5,
        zre=7.0,
        z_early=20.0,
        alpha0=3.7,
        kappa=0.10,
        helium=True,
        helium2=True,
        xe_recomb=1.0e-4,
        h=0.6774,
        theta=None,
        T_cmb=2.7255,
        Ob_0=0.049,
        Om_0=0.309,
        A_s=2.139e-9,
        n_s=0.9677,
        verbose=True,
        run_CMB=False,
        cosmomc=False,
    ):

        """
        Parameters:
            helium, helium2 : to include helium first and second reionisation or not
            zre : midpoint (when xe = 0.50) 
            z_end : redshift at wich reionisation ends
            z_early : redshift aroiund which the first sources form (taken to 20)

            if cosmomc==True, use syntax used in cosmomc, give an input
                omegabh2 for Ob_0
                omegach2 for Om_0
                logA for A_s

        """

        self.verbose = verbose
        self.run_CMB = run_CMB
        self.cosmomc = cosmomc

        assert (h is not None and theta is None) or (
            h is None and theta is not None
        ), "You must input h or theta but not both"
        if theta is not None:
            assert cosmomc, "If using theta, must also use cosmomc syntax"

        self.theta = theta

        self.T_cmb = T_cmb
        if cosmomc:
            self.obh2 = Ob_0
            self.och2 = Om_0
            if theta is not None:
                pars = camb.CAMBparams()
                pars.set_cosmology(
                    cosmomc_theta=self.theta, ombh2=self.obh2, omch2=self.och2
                )
                self.H0 = pars.H0
                self.h = pars.h
                if self.verbose:
                    print("h = %.4f" % self.h)
            else:
                self.h = h
                self.H0 = h * 100.0
            self.Ob_0 = self.obh2 / self.h ** 2
            self.Om_0 = (self.och2 + self.obh2) / self.h ** 2
            self.A_s = np.exp(A_s) / 1e10
        else:
            self.h = h
            self.H0 = self.h * 100.0
            self.Ob_0 = Ob_0
            self.Om_0 = Om_0
            self.obh2 = self.Ob_0 * self.h ** 2
            self.och2 = (self.Om_0 - self.Ob_0) * self.h ** 2
            self.A_s = A_s
        self.n_s = n_s
        self.tau = 0.0
        self.xe_recomb = xe_recomb
        cos = cosmology.FlatLambdaCDM(
            H0=self.H0, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )
        self.Yp = 0.2453
        mh = constants.m_n.value  # kg
        self.sigma8 = 0.0

        self.dz = dz
        self.zre = zre
        self.z_early = z_early
        self.z_end = self.zre - self.dz
        self.alpha = 0.0
        if self.verbose:
            print("zre = %.1f, zend = %.1f" % (self.zre, self.z_end))

        self.helium = helium
        self.helium2 = helium2
        self.f = 1.0
        if self.helium:
            self.fHe = self.Yp / (3.9715 * (1 - self.Yp))
            self.f += self.fHe
            if self.helium2:
                self.f += self.fHe
        else:
            self.fHe = 0.0
        if self.verbose:
            print("Late-time ionisation fraction: %.2f" % self.f)

        rhoc = cos.critical_density0.si.value  # kg m-3
        self.nh = (1.0 - self.Yp) * self.Ob_0 * rhoc / mh  # m-3

        self.alpha0 = alpha0
        self.kappa = kappa

        self.x_i_z_integ = np.zeros(z_integ.size)  # ionisation level of the IGM
        self.tau_z_integ = np.zeros(z_integ.size)  # thomson optical depth
        self.eta_z_integ = np.zeros(z_integ.size)  # comoving distance to z in [Mpc]
        self.detadz_z_integ = np.zeros(z_integ.size)  # Hubble parameter in SI units [m]
        self.f_z_integ = np.zeros(z_integ.size)  # growth rate, no units
        self.adot_z_integ = np.zeros(z_integ.size)  # in SI units [s-1]
        self.n_H_z_integ = np.zeros(
            z_integ.size
        )  # number density of baryons in SI units [m-3]
        self.Pk_lin_integ = np.zeros(z_integ.size)  # linear matter power spectrum
        self.b_del_e_integ = np.zeros(z_integ.size)  # electrons bias

    def xe(self, z):
        """
        Computes the redshift-asymmetric parameterisation of xe(z) in Douspis+2015
        Parameters:
            helium, helium2 : to include helium first and second reionisation or not
            zre : midpoint (when xe = 0.50) 
            z_end : redshift at wich reionisation ends
            z_early : redshift aroiund which the first sources form (taken to 20)

        """
        frac = 0.5 * (np.sign(self.z_end - z) + 1) + 0.5 * (
            np.sign(z - self.z_end) + 1
        ) * abs((self.z_early - z) / (self.z_early - self.z_end)) ** (self.alpha)
        xe = (1.0 + self.fHe - self.xe_recomb) * frac
        if self.helium2:
            assert (
                self.helium
            ), "Need to set both He reionisation to True, cannot have HeII without HeI"
            a = np.divide(1, z + 1.0)
            deltayHe2 = (
                1.5
                * np.sqrt(1 + helium_fullreion_redshift)
                * helium_fullreion_deltaredshift
            )
            VarMid2 = (1.0 + helium_fullreion_redshift) ** 1.5
            xod2 = (VarMid2 - 1.0 / a ** 1.5) / deltayHe2
            tgh2 = np.tanh(xod2)  # check if xod<100
            xe += (self.fHe - self.xe_recomb) * (tgh2 + 1.0) / 2.0
        x = np.where(z < self.z_early, xe + self.xe_recomb, self.xe_recomb)
        return x

    def xe2tau(self, z):
        """
        computes tau(z) integrated from xe(z) 
        Params:
            xe : ionization rate of the Universe
            z : list in descending order (from 30 to 0 for instance)
        """
        cos = cosmology.FlatLambdaCDM(
            H0=self.h * 100, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )
        z = np.sort(z)
        xe = np.sort(self.xe(z))[::-1]
        csurH = constants.c.value * 100 / cos.H(0).si.value  # cm
        Hdezsurc = cos.H(z).si.value / (constants.c.value * 100)  # cm-1

        eta = 1  # eta=1 for now
        integ2 = (
            constants.c.value
            * constants.sigma_T.value
            * self.nh
            * xe
            / cos.H(z).si.value
            * (1 + z) ** 2
            * (1 + eta * self.Yp / 4 / (1.0 - self.Yp))
        )
        taudez2 = cumtrapz(integ2[::-1], z, initial=0)[::-1]

        return taudez2

    # window function (power law) for early times
    def W(self, k, x):
        return 10 ** self.alpha0 * x ** (-0.2) / (1.0 + x * (k / self.kappa) ** 3.0)

    # electrons - matter bias after reionisation
    def bdH(self, k, z):
        kf = 9.4
        g = 0.5
        return 0.5 * (np.exp(-k / kf) + 1.0 / (1.0 + np.power(g * k / kf, 2.0)))

    # free electrons power spectrum
    def Pee(self, k, z):

        if np.sum(self.x_i_z_integ) == 0:
            self.init_reionisation_history()
        if np.sum(self.f_z_integ) == 0:
            if self.verbose:
                print(
                    "Calling Pee without having initialised the matter power spectrum. Running CAMB..."
                )
            self.run_camb()

        return (self.f - self.xe(z)) / self.f * self.W(k, self.xe(z)) + self.xe(
            z
        ) / self.f * self.bdH(k, z) * Pk(k, z)

    def P21(self, k, z, mK=True, units=False):
        P21 = self.xe(z) ** 2 * self.Pee(k, z) - (2.0 * self.xe(z) - 1.0) * self.bdH(
            k, z
        ) * Pk(k, z)
        if mK:
            T0 = (
                27.0
                * (self.f - self.xe(z))
                * self.Ob_0
                * (self.h ** 2)
                / 0.023
                * np.sqrt(0.15 * (1 + z) / (10 * self.Om_0 * self.h ** 2))
            )  # mK
            P21 *= T0 ** 2
        if not units:
            P21 *= k ** 3 / 2.0 / np.pi
        return P21

    def init_reionisation_history(self):

        self.alpha = np.log(1.0 / 2.0 / (1.0 + self.fHe)) / np.log(
            (self.z_early - self.zre) / (self.z_early - self.z_end)
        )

        self.tau = self.xe2tau(z3)[0]
        tauf = interpolate.interp1d(z3, self.xe2tau(z3))  # interpolation
        self.x_i_z_integ = self.xe(z_integ)  # reionisation history
        self.tau_z_integ = tauf(z_integ)  # thomson optical depth

        if self.verbose:
            print("tau = %.4f" % self.tau)

    def run_camb(self, force=False, kmax_pk=kmax_camb):

        global D_C, Pk, Pk_lin

        assert self.tau != 0.0, "Need to initialise reionisation history first"
        if (np.sum(self.f_z_integ) != 0) and (force is False):
            if self.verbose:
                print("CAMB already run. Set force to True if want to re-run anyway.")
            return

        if self.verbose:
            print("Running CAMB...")

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.H0, ombh2=self.obh2, omch2=self.och2, TCMB=self.T_cmb, tau=self.tau
        )
        pars.InitPower.set_params(ns=self.n_s, r=0, As=self.A_s)
        pars.WantTransfer = True
        pars.Reion.set_tau(self.tau)
        pars.set_for_lmax(2000, lens_potential_accuracy=0)
        pars.set_dark_energy()
        data = camb.get_background(pars)
        results = camb.get_results(pars)
        self.sigma8 = results.get_sigma8()

        #### CMB spectra
        results.calc_power_spectra(pars)
        if self.run_CMB:
            if self.verbose:
                print(" Computing CMB power spectra...")
            pars.set_for_lmax(10000, lens_potential_accuracy=0)
            powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", lmax=10000)
            # totCL=powers['total']
            unlensedCL = powers["unlensed_scalar"]
            ls = np.arange(unlensedCL.shape[0])
            self.CMB_Cells = np.c_[
                ls, unlensedCL[:, 0], unlensedCL[:, 1], unlensedCL[:, 3]
            ]  # tt, ee, te
            # np.savetxt('cmb_cells.txt',self.CMB_Cells,header="ell   TT   EE   TE    NB: dimensionless [l(l+1)/2pi] C_l's")

        ##############################################
        #### Cosmo functions & derived parameters ####
        ##############################################

        ### Hubble function (=adot/a) in SI units [s-1] (CAMB gives km/s/Mpc)
        H = np.vectorize(lambda z: results.hubble_parameter(z) / Mpckm)
        ### Growth rate f
        f = np.vectorize(
            lambda z: data.get_redshift_evolution(0.1, z, ["growth"]).flatten()
        )
        ### Comoving distance / conformal time in Mpc
        D_C = np.vectorize(lambda z: results.comoving_radial_distance(z))

        ## Hydrogen number density function in SI units [m-3]
        n_H = lambda z: self.nh * (1.0 + z) ** 3.0

        # self.Pk = np.vectorize(lambda k,z : interpolate.interp2d(krange_camb,z_integ,Pk(krange_camb,z_integ[:,None]))(k,z))
        # self.Pk_lin = np.vectorize(lambda k,z : interpolate.interp2d(krange_camb,z_integ,Pk_lin(krange_camb,z_integ[:,None]))(k,z))
        # self.D_C = interpolate.interp1d(z_integ,D_C(z_integ))

        self.eta_z_integ = D_C(z_integ)  # comoving distance to z in [Mpc]
        self.detadz_z_integ = constants.c.value / H(
            z_integ
        )  # Hubble parameter in SI units [m]
        self.f_z_integ = f(z_integ)  # growth rate, no units
        self.adot_z_integ = (1.0 / (1.0 + z_integ)) * H(z_integ)  # in SI units [s-1]
        self.n_H_z_integ = n_H(z_integ)  # number density of baryons in SI units [m-3]

        # mink, maxk = kmin_camb, kmax_camb
        # for i,ell in enumerate(ells):
        #     k_z_integ = ell / self.eta_z_integ
        #     k_min_kp = np.sqrt(k_z_integ[:, None, None]**2 + kp_integ[:, None]**2. - 2. * k_z_integ[:, None, None] * kp_integ[:, None] * mu)
        #     if (min(k_z_integ.min(),k_min_kp.min()) < mink):
        #         mink = min(k_z_integ.min(),k_min_kp.min())
        #     if (max(k_z_integ.max(),k_min_kp.max())> maxk):
        #         maxk = max(k_z_integ.max(),k_min_kp.max())
        # if (mink<kmin_camb) or (maxk>kmax_camb):
        #     extrap_kmax = True
        # else:
        #     extrap_kmax = False

        ## Linear matter power spectrum P(z,k) - no hubble units, result in (Mpc)^3
        # assert (kmax_pk <= kmax_camb),'k too large for P(k) extrapolation, need to modify either ell_max or z_min'
        interp_l = camb.get_matter_power_interpolator(
            pars,
            nonlinear=False,
            kmax=kmax_pk,
            hubble_units=False,
            k_hunit=False,
            zmax=z_max,
            var1=model.Transfer_nonu,
            var2=model.Transfer_nonu,
        )
        Pk_lin = np.vectorize(lambda k, z: interp_l.P(z, k))
        ### Non-linear matter power spectrum
        interp_nl = camb.get_matter_power_interpolator(
            pars,
            nonlinear=True,
            kmax=kmax_pk,
            hubble_units=False,
            k_hunit=False,
            zmax=z_max,
            var1=model.Transfer_nonu,
            var2=model.Transfer_nonu,
        )
        Pk = np.vectorize(lambda k, z: interp_nl.P(z, k))

        self.Pk_lin_integ = Pk_lin(
            kp_integ[:, None], z_integ[:, None, None]
        )  # linear matter power spectrum
        self.b_del_e_integ = np.sqrt(
            self.Pee(kp_integ[:, None], z_integ[:, None, None])
            / Pk(kp_integ[:, None], z_integ[:, None, None])
        )  # electrons bias

    ########################
    #### C_ell fonction ####
    ########################
    def C_ell_kSZ(self, ell, patchy):
        ### Preliminaries
        # in [Mpc-1]
        k_z_integ = ell / self.eta_z_integ

        # in [Mpc-1]
        k_min_kp = np.sqrt(
            k_z_integ[:, None, None] ** 2
            + kp_integ[:, None] ** 2.0
            - 2.0 * k_z_integ[:, None, None] * kp_integ[:, None] * mu
        )

        # if (min(k_z_integ.min(),k_min_kp.min())<kmin_camb) or (max(k_z_integ.max(),k_min_kp.max())>kmax_camb):
        #     raise Warning('Extrapolating the matter PK to too small or too large k')

        ### Compute I_tot1 and I_tot2, in [Mpc^2]
        Pee_min_kp = self.Pee(k_min_kp, z_integ[:, None, None])
        I_e = (Pee_min_kp / kp_integ[:, None] ** 2.0) - (
            np.sqrt(Pee_min_kp / Pk(k_min_kp, z_integ[:, None, None]))
            * self.b_del_e_integ
            * Pk_lin(k_min_kp, z_integ[:, None, None])
            / k_min_kp ** 2
        )

        ### Compute Delta_B^2 integrand, in [s-2.Mpc^2]
        Delta_B2_integrand = (
            k_z_integ[:, None, None] ** 3.0
            / 2.0
            / np.pi ** 2.0
            * (self.f_z_integ[:, None, None] * self.adot_z_integ[:, None, None]) ** 2.0
            * kp_integ[:, None] ** 3.0
            * np.log(10.0)
            * np.sin(th_integ)
            / (2.0 * np.pi) ** 2.0
            * self.Pk_lin_integ
            * (1.0 - mu ** 2.0)
            * I_e
        )
        ### Compute Delta_B^2, in [s-2.Mpc^2]
        Delta_B2 = simps(simps(Delta_B2_integrand, th_integ), np.log10(kp_integ))

        ### Compute C_kSZ(ell) integrand, unit 1
        C_ell_kSZ_integrand = (
            8.0
            * np.pi ** 2.0
            / (2.0 * ell + 1.0) ** 3.0
            * (constants.sigma_T.value / constants.c.value) ** 2.0
            * (self.n_H_z_integ * self.x_i_z_integ / (1.0 + z_integ)) ** 2.0
            * Delta_B2
            * np.exp(-2.0 * self.tau_z_integ)
            * self.eta_z_integ
            * self.detadz_z_integ
            * Mpcm ** 3.0
        )

        ### Compute C_kSZ(ell), no units
        result = trapz(C_ell_kSZ_integrand, z_integ)
        if patchy:
            g = z_integ >= self.z_end
            result_p = trapz(C_ell_kSZ_integrand[g], z_integ[g])
            return result_p, result
        else:
            return result

    def run_ksz(self, ells=[3000], n_threads=1, patchy=True, Dells=False):

        ells = np.array(ells)

        if np.sum(self.x_i_z_integ) == 0:
            self.init_reionisation_history()

        kmax_pk = np.max(ells) / D_C(z_min)
        if kmax_pk > 1e3:
            if self.verbose:
                print("Need to re-run CAMB for large k-values.")
            self.run_camb(force=True, kmax_pk=kmax_pk)
        if np.sum(self.f_z_integ) == 0:
            self.run_camb(kmax_pk=kmax_pk)

        if self.verbose:
            print(
                "Computing for %i l on range [%i,%i] with %i threads"
                % (len(ells), np.min(ells), np.max(ells), n_threads)
            )
        # self.ells = ells
        # self.C_ells = np.array(multiprocessing.Pool(n_threads).map(self.C_ell_kSZ, ells))
        if patchy:
            C_ells = np.zeros((len(ells), 2))
        else:
            C_ells = np.zeros((len(ells), 1))
        for i, ell in enumerate(ells):
            C_ells[i, :] = self.C_ell_kSZ(ell, patchy)

        if not Dells:
            return C_ells
        else:
            D_ells = (
                ells[:, None]
                * (ells[:, None] + 1)
                * C_ells
                / 2.0
                / np.pi
                * (self.T_cmb * 1e6) ** 2
            )
            return D_ells

    def save(self, outroot):

        assert isinstance(outroot, str), "outroot parameter needs to be a str"
        assert np.sum(self.C_ells) != 0, "No KSZ spectrum computed"

        ### Writing kSZ C_ells
        np.savetxt(
            outroot,
            np.c_[self.ells, self.C_ells],
            header="ell   kSZ tot   kSZ patchy    NB: dimensionless C_l's",
            delimiter=" ",
            fmt="%.5e",
        )

    def plot_ksz(self, filename=None):

        import matplotlib.pyplot as plt

        assert np.sum(self.C_ells) != 0, "No KSZ spectrum computed"

        T_CMB_uK = self.T_cmb * 1e6

        ######## results

        PS1 = (
            self.ells
            * (self.ells + 1)
            * self.C_ells[:, 0]
            / (2 * np.pi)
            * (T_CMB_uK ** 2)
        )  #
        PS2 = (
            self.ells
            * (self.ells + 1)
            * (self.C_ells[:, 1] - self.C_ells[:, 0])
            / (2 * np.pi)
            * (T_CMB_uK ** 2)
        )
        PS = PS1 + PS2

        if self.verbose:
            print("Results:")
            ls = np.linspace(min(self.ells), max(self.ells), 10000)
            homo = interpolate.interp1d(self.ells, PS2)
            print(" homogeneous... %.2f muK^2" % (homo(3000)))
            patchy = interpolate.interp1d(self.ells, PS1)
            print(" patchy... %.3f muK^2" % (patchy(3000)))
            pps = patchy(ls)
            print(" lmax... %.1f" % (ls[np.argmax(pps)]))
            print(" max ampl... %.2f " % (np.max(pps)))

        ######### plot

        fig, ax = plt.subplots(1, 1, figsize=(9, 8))

        ax.plot(self.ells, PS1, color="C0", lw=1.2, ls="--", label=r"Patchy signal")
        ax.plot(
            self.ells, PS2, color="C0", lw=1.2, ls="-.", label=r"Homogeneous signal"
        )
        ax.plot(self.ells, PS1 + PS2, color="C0", lw=1.5, label="Total signal")

        ######## data
        ax.errorbar(
            3000,
            1.1,
            yerr=[[0.7], [1.0]],
            lw=0.0,
            capsize=3.0,
            elinewidth=1.2,
            color="k",
            marker="o",
            ms=5,
            alpha=0.7,
        )  # reichardt+2021

        ax.set_xlabel(r"Angular multipole $\ell$", fontsize=16)
        ax.set_ylabel(r"$\mathcal{D}_\ell^{kSZ}$ [$\mu$K$^2$]", fontsize=16)
        ax.set_xlim(0, 1e4)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=14)
        ax.legend(loc="upper left", frameon=False, fontsize=16, ncol=1)
        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename)
