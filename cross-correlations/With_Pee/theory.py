##############################################
######## Computes kSZ power spectrum #########
### Copyright Stephane Ilic & Adélie Gorce ###
##############################################

import warnings

import camb
import numpy as np
from astropy import constants, cosmology, units
from camb import model

# import multiprocessing
from scipy.integrate import cumtrapz, simps, trapz
from scipy.interpolate import interp1d

from parameters import *
from utils import *
from simulations import *

##################################
# Integration/precision settings #
##################################

# minimal and maximal values valid for CAMB interpolation of
kmax_pk = 1570.5

# Settings for theta integration
num_th = 50
th_integ = np.linspace(0.0001, np.pi * 0.9999, num_th)
mu = np.cos(th_integ)  # cos(k.k')

# Settings for k' (=kp) integration
# k' array in [Mpc-1] - over which you integrate
min_logkp = -5.0
max_logkp = 1.5
dlogkp = 0.05
kp_integ = np.logspace(min_logkp, max_logkp, int((max_logkp - min_logkp) / dlogkp) + 1)
# minimal and maximal values valid for CAMB interpolation of
kmax_camb = 6.0e3
kmin_camb = 7.2e-6
krange_camb = np.logspace(np.log10(kmin_camb), np.log10(kmax_camb), 500)

# Settings for z integration
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

Mpcm = (1.0 * units.Mpc).to(units.m).value  # one Mpc in [m]
Mpckm = Mpcm / 1e3


class Pee_model:
    def __init__(
        self,
        dz_h=1.5,
        zre_h=7.0,
        z_early=20.0,
        asym_h_reion=True,
        alpha0=3.7,
        kappa=0.10,
        helium=True,
        helium2=True,
        xe_recomb=1.0e-4,
        h=0.6774,
        thetaMC=None,
        T_cmb=2.7255,
        Ob_0=0.049,
        Om_0=0.309,
        A_s=2.139e-9,
        n_s=0.9677,
        r=0.,
        verbose=False,
        run_CMB=False,
        cosmomc=False,
        run_camb=False,
    ):

        """Container for kSZ power spectrum models.

        This class computes a kSZ (and TT, TE, EE) power spectrum given
        a set of cosmological and reionisation parameters following the
        method described in Gorce et al. 2020, A&A 640 A90.

        FOR USER: IF ANY OF THE ATTRIBUTES IS CHANGED AFTER INITIALISING
        THE CLASS, YOU WILL NEED TO RE-INITIALISE THE REIONISATION
        HISTORY AND RE-RUN CAMB.

        Parameters
        ----------
            dz_h (float)
                Proxy for H reionisation duration.
                If asym parameterisation, dz = zre - zend.
                Else, dz is duration of instantaneous reion.
                Default is 1.5.
            zre_h (float)
                H reionisation midpoint (when xe = 0.50).
                Default is 7.0
            z_early (float)
                Redshift around which the first sources form.
                Default is 20.
            asym_h_reion (boolean)
                Whether the reionisation history use a symmetric (tanh, set
                to False) or asymmetric (power-law, set to True) model.
                Default is True.
            alpha0 (float)
                Large-scale amplitude of the electron power spectrum.
                See Gorce et al. 2020 for model.
                Default is 3.7.
            kappa (float)
                Minimum bubble size, in Mpc-1.
                See Gorce et al. 2020 for model.
                Default is 0.10.
            helium (boolean)
                If helium is True, the first reionisation of He is included by
                multiplying the H ionisation level by fH = 1.08.
                Default is True.
            helium2 (boolean)
                If helium2 is True, the second reionisation is He is included
                as a tanh whose parameters are read from the parameters.py
                module.
                Default is True.
            xe_recomb (float)
                IGM ionisation level at z = z_early.
                Default is 1.0e-4.
            h (float)
                Reduced Hubble constant.
                Default is 0.6774.
            thetaMC (float)
                Ratio of the sound horizon to the angular diameter distance
                at decoupling, scaled by 100.
                Default is None.
            Ob_0 (float)
                Baryon density at z = 0.
                Default is 0.049.
            Om_0 (float)
                Matter density at z = 0.
                Default is 0.309.
            A_s (float)
                Initial super-horizon amplitude of curvature perturbations
                at k = 0.05 Mpc-1.
                Default is 2.139e-9.
            n_s (float)
                Scalar spectral index.
                Default is 0.9677.
            r (float)
                Primordial GW.
                Default is 0.
            verbose (boolean)
                If True, run in verbose mode.
                Default is False.
            run_CMB (boolean)
                If True, the TT, TE and EE CMB power spectra are
                computed by CAMB.
                Default is False.
            cosmomc (boolean)
                Format of the cosmological parameters given as inputs.
                This is defined to match the CosmoMC parameterisation.
                If True, do not provide h but thetaMC instead, give for Ob_0
                its value multiplied by h**2, for Om_0 the value of Oc_0 * h**2
                and logA = exp(A_s) / 1e10 instead of A_s.
                Default is False.
            run_camb (boolean)
                If True, runs CAMB to obtain the matter power spectra.
                Default is False.

        """

        self.verbose = verbose
        self.run_CMB = run_CMB
        self.cosmomc = cosmomc
        self.asym_h_reion = asym_h_reion
        self.has_camb_run = False

        # INITIALISE COSMOLOGY

        # Ensure cosmomc value is consistent with input
        # cosmological parameters.
        assert (h is not None and thetaMC is None) or (
            h is None and thetaMC is not None
        ), "You must input h or theta but not both"
        if thetaMC is not None:
            assert cosmomc, "If using theta, must also use cosmomc syntax"

        self.thetaMC = thetaMC
        self.T_cmb = T_cmb
        self.n_s = n_s
        self.r = r

        if cosmomc:
            self.obh2 = Ob_0
            self.och2 = Om_0
            self.logA = A_s
            self.A_s = np.exp(A_s) / 1e10
            if thetaMC is not None:
                if self.thetaMC > 1.0:
                    self.thetaMC /= 100.0
                else:
                    self.thetaMC = self.thetaMC
                pars = camb.CAMBparams()
                pars.set_cosmology(
                    cosmomc_theta=self.thetaMC, ombh2=self.obh2, omch2=self.och2
                )
                self.H0 = pars.H0
                self.h = pars.h
                if self.verbose:
                    print("h = %.4f" % self.h)
            else:
                self.h = h
                self.H0 = h * 100.0
                pars = camb.CAMBparams()
                pars.set_cosmology(
                    H0=self.h * 100,
                    ombh2=self.obh2,
                    omch2=self.och2,
                )
                pars.Reion.redshift = zre_h
                if self.asym_h_reion:
                    pars.Reion.dz = dz_h
                else:
                    pars.Reion.delta_redshift = dz_h
                pars.InitPower.set_params(ns=self.n_s, r=self.r, As=self.A_s)
                results = camb.get_results(pars)
                self.thetaMC = results.cosmomc_theta()
            self.Ob_0 = self.obh2 / self.h**2
            self.Om_0 = (self.och2 + self.obh2) / self.h**2
        else:
            self.h = h
            self.H0 = self.h * 100.0
            self.Ob_0 = Ob_0
            self.Om_0 = Om_0
            self.obh2 = self.Ob_0 * self.h**2
            self.och2 = (self.Om_0 - self.Ob_0) * self.h**2
            self.A_s = A_s
            self.logA = np.log(self.A_s * 1e10)
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=self.h * 100,
                ombh2=self.obh2,
                omch2=self.och2,
            )
            pars.Reion.redshift = zre_h
            if self.asym_h_reion:
                pars.Reion.dz = dz_h
            else:
                pars.Reion.delta_redshift = dz_h
            pars.InitPower.set_params(ns=self.n_s, r=self.r, As=self.A_s)
            results = camb.get_results(pars)
            self.thetaMC = results.cosmomc_theta()

        cos = cosmology.FlatLambdaCDM(
            H0=self.H0, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )
        self.Yp = 0.2453
        self.nh = (
            (1.0 - self.Yp)
            * self.Ob_0
            * cos.critical_density0.si.value
            / constants.m_n.value
        )  # m-3

        # INITIALISE REIONISATION

        # H reionisation
        self.tau = 0.0
        self.xe_recomb = xe_recomb
        self.zre_h = zre_h
        self.dz_h = dz_h
        self.z_early = z_early
        if self.asym_h_reion:
            self.zend_h = self.zre_h - self.dz_h
        else:
            self.zend_h = self.zre_h - self.dz_h / 2.0
        self.alpha = 0.0
        if self.verbose:
            print(f"zre_h = {self.zre_h:.1f}, zend = {self.zend_h:.1f}")

        # He reionisation
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

        # KSZ shape parameters
        self.alpha0 = alpha0
        self.kappa = kappa

        # Initialise arrays for kSZ computation
        self.x_i_z_integ = np.zeros(z_integ.size)  # ionisation level of IGM
        self.tau_z_integ = np.zeros(z_integ.size)  # thomson optical depth
        self.eta_z_integ = np.zeros(z_integ.size)  # comoving dist in [Mpc]
        self.detadz_z_integ = np.zeros(z_integ.size)  # Hubble parameter [m]
        self.f_z_integ = np.zeros(z_integ.size)  # growth rate, unit 1
        self.adot_z_integ = np.zeros(z_integ.size)  # in SI units [s-1]
        self.n_H_z_integ = np.zeros(z_integ.size)  # baryons nb density [m-3]
        self.Pk_lin_integ = np.zeros(z_integ.size)  # linear matter ps
        self.b_del_e_integ = np.zeros(z_integ.size)  # electrons bias

        # Fill arrays related to reionisation
        self.init_reionisation_history()
        if run_camb:
            self.run_camb(kmax_pk=kmax_pk)

    def xe(self, z):
        """
        Computes model's reionisation history.

        The redshift-asymmetric parameterisation of xe(z) in Douspis+2015
        and the class parameters are used.

        Parameters
        ----------
            z: (array of) float(s)
                Redshift range used to compute the ionisation history.
        """
        z = np.atleast_1d(z)
        if self.asym_h_reion:
            xe_h = np.ones(np.shape(z))
            # H reionisation
            m = z > self.zend_h
            xe_h[m] = np.abs((self.z_early - z[m]) / (self.z_early - self.zend_h)) ** self.alpha
        else:
            deltay = 1.5 * np.sqrt(1 + self.zre_h) * self.dz_h
            xod = ((1 + self.zre_h) ** 1.5 - (1 + z) ** 1.5) / deltay
            xe_h = (np.tanh(xod) + 1.0) / 2.0

        # add first He reionisation if needed
        xe = (1.0 + self.fHe - self.xe_recomb) * xe_h

        # add second He reionisation
        if self.helium2:
            assert self.helium, (
                "Need to set both He reionisation "
                "to True, cannot have HeII without HeI"
            )
            a = np.divide(1, z + 1.0)
            deltayHe2 = (
                1.5
                * np.sqrt(1 + helium_fullreion_redshift)
                * helium_fullreion_deltaredshift
            )
            VarMid2 = (1.0 + helium_fullreion_redshift) ** 1.5
            xod2 = (VarMid2 - 1.0 / a**1.5) / deltayHe2
            tgh2 = np.tanh(xod2)  # check if xod<100
            xe += (self.fHe - self.xe_recomb) * (tgh2 + 1.0) / 2.0
        # Ensure continuity of the function
        x = np.where(z < self.z_early, xe + self.xe_recomb, self.xe_recomb)

        return x

    def xe2tau(self, z):
        """
        Computes redshift evolution of the model's optical depth.

        Parameters
        ----------
            z: (array of) float(s)
                Redshift range used to compute the optical depth.
        """
        cos = cosmology.FlatLambdaCDM(
            H0=self.h * 100, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )
        z = np.sort(z)
        xe = np.sort(self.xe(z))[::-1]

        integ = (
            constants.c.value
            * constants.sigma_T.value
            * self.nh
            * xe
            / cos.H(z).si.value
            * (1 + z) ** 2
        )
        tofz = cumtrapz(integ[::-1], z, initial=0)[::-1]

        return tofz

    def W(self, k, x):
        """
        Electron power spectrum at early times.

        Parameters
        ----------
            k: float, array of floats
                Fourier modes to compute the power spectrum at.
            x: float, array of floats
                Ionisation fraction at redshifts one want the
                power spectrum at.
        Outputs
        -------
            W: array of floats
                Power spectrum for k and z.
        """
        return 10**self.alpha0 * x ** (-0.2) / (1.0 + x * (k / self.kappa) ** 3.0)

    def bdH(self, k, z, kf=9.4, g=0.5):
        """
        Electrons - matter bias after reionisation.

        Use the parameterisation of Shaw+2012.

        Parameters
        ----------
            k: float, array of floats
                Fourier modes to compute the bias at.
            z: float, array of floats
                Redshift to compute the bias at.
            kf: float
                Mode [Mpc-1] where baryon power spectrum starts
                diverging from matter power spectrum.
                Default is 9.4.
            g: float
                Amplitude of the divergence.
                Default is 0.5 (no unit).
        Outputs
        -------
            b: array of floats
                Bias for k and z.
        """

        return 0.5 * (np.exp(-k / kf) + 1.0 / (1.0 + np.power(g * k / kf, 2.0)))

    def Pee(self, k, z):
        """
        Electron overdensity power spectrum.

        Note: Requires to initialise reionisation history and to
        run camb (will do it if it has not been previously done by
        running self.run_camb() and self.init_reionisation_history().)

        Parameters
        ----------
            k: float, array of floats
                Fourier modes to compute the power spectrum at.
            z: float, array of floats
                Redshift to compute the power spectrum at.
        Outputs
        -------
            Pee: array of floats
                Power spectrum for k and z.
        """

        if np.sum(self.x_i_z_integ) == 0:
            self.init_reionisation_history()
        if not self.has_camb_run:
            warnings.warn(
                "Calling Pee without having initialised the matter \
                          power spectrum. Running CAMB..."
            )
            self.run_camb(kmax_pk=kmax_pk)

        Pee = (self.f - self.xe(z)) / self.f * self.W(k, self.xe(z)) + self.xe(
            z
        ) / self.f * self.bdH(k, z) * self.Pk(k, z)

        return Pee

    def init_reionisation_history(self):
        """
        Initialise reionisation history.

        Running this method initialises reionisation
        parameters such as self.tau and self.alpha,
        as well as fills reionisation-related arrays
        for kSZ computation.
        """

        if self.asym_h_reion:
            self.zend_h = self.zre_h - self.dz_h
        else:
            self.zend_h = self.zre_h - self.dz_h / 2.0
        if self.asym_h_reion:
            self.alpha = np.log(.5) / np.log(
                (self.z_early - self.zre_h) / (self.z_early - self.zend_h)
            )

        self.tau = self.xe2tau(z3)[0]
        tauf = interp1d(z3, self.xe2tau(z3))  # interpolation
        self.x_i_z_integ = self.xe(z_integ)  # reionisation history
        self.tau_z_integ = tauf(z_integ)  # thomson optical depth

        if self.verbose:
            print("tau = %.4f" % self.tau)

    def def_camb(self):
        """
        Define CAMB.parameters object for KSZ_power model.
        """

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.H0, ombh2=self.obh2, omch2=self.och2, TCMB=self.T_cmb
        )  # ,tau=self.tau)
        pars.InitPower.set_params(ns=self.n_s, r=self.r, As=self.A_s)
        pars.WantTransfer = True
        # pars.Reion.set_tau(self.tau)
        pars.Reion.use_optical_depth = False
        if self.asym_h_reion:
            pars.Reion.dz = self.dz_h
        else:
            pars.delta_redshift = self.dz_h
        pars.Reion.redshift = self.zre_h
        pars.set_dark_energy()

        return pars

    def get_primary_spectra(self, ells=None, results=None, unit='muK', Dells=True, type='total'):
        """
        Compute primary power spectra for KSZ_power object.

        Parameters
        ----------
            ells: list of ints
                ells to compute the spectra over.
                If len(ells) == 1, use np.arange(0, ells[0]).
                If None, use np.arange(0, 2000).
            results: CAMB.results object
                Can be fed to avoid computing results twice.
            unit: str
                Unit to get results in. Similar to CAMB.
                Default is 'muK'.
            Dells: boolean
                Whether to compute Dl=l(l+1)Cl/2pi or raw Cls.
                Default is True.
            type: str
                Type of spectrum to compute. Options are
                ['total', 'unlensed_scalar', 'unlensed_total',
                'lensed_scalar', 'tensor', 'lens_potential']-
                similar to CAMB.
                Default is 'total'.

        Outputs
        -------
            CMB_Cells: array of floats of shape ((ells.size, 4).
            First column is ells.
            Second column is TT.
            Third column is EE.
            Fourth column is TE.
            Fifth column is BB.
        """
        ells = np.array(ells, dtype=int)
        if ells is None:
            lmax = 2000
        elif np.size(ells) == 1:
            lmax = ells[0]
        else:
            lmax = int(np.max(ells))

        pars = self.def_camb()
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        if results is None:
            results = camb.get_results(pars)
        else:
            assert isinstance(results, camb.CAMBdata)
        results.calc_power_spectra(pars)

        if self.verbose:
            print(" Computing CMB power spectra...")
        powers = results.get_cmb_power_spectra(
            pars, CMB_unit=unit, lmax=lmax, raw_cl=not Dells)
        CL = powers[type]
        ls = np.arange(CL.shape[0])
        CMB_Cells = np.c_[ls, CL[:, 0], CL[:, 1], CL[:, 3]]  # tt, ee, te
        if ells is None or np.size(ells) == 1:
            return CMB_Cells
        else:
            return np.c_[
                ells,
                interp1d(ls, CL[:, 0])(ells),
                interp1d(ls, CL[:, 1])(ells),
                interp1d(ls, CL[:, 3])(ells),
                interp1d(ls, CL[:, 2])(ells)
            ]

    # @profile
    def run_camb(self, force=False, kmax_pk=kmax_camb, CMB_ells=[10000]):
        """
        Compute matter power spectra and fill related arrays.

        Parameters
        ----------
            force: boolean
                For function to recompute all parameters and
                arrays even if they are already filled.
                Default is False.
            kmax_pk: float
                Maximum k used to compute the matter power
                spectra [Mpc-1].
                Default is read from parameters.py.
            CMB_ells: list of int
                List of multipoles used to compute the primary CMB spectra.


        Note: Requires to initialise reionisation history (will do so
        if it has not been previously done by running
        self.init_reionisation_history().)
        """

        assert self.tau != 0.0, "Need to initialise reionisation history first"
        if (self.has_camb_run) and (force is False):
            raise ValueError(
                "CAMB already run. " "Set force to True if want to re-run anyway."
            )
        if self.verbose:
            print("Running CAMB...")

        pars = self.def_camb()
        data = camb.get_background(pars)
        results = camb.get_results(pars)

        # CMB spectra
        if self.run_CMB:
            self.CMB_Cells = self.get_primary_spectra(ells=CMB_ells, results=results)

        # Arrays of cosmological parameters

        H_z_integ = np.array(
            [results.hubble_parameter(z) / Mpckm for z in z_integ]
        )  # Hubble parameter [m]
        self.eta_z_integ = np.array(
            [results.comoving_radial_distance(z) for z in z_integ]
        )  # comov dist to z [Mpc]
        self.detadz_z_integ = (
            constants.c.value / H_z_integ
        )  # Derivative of conformal time
        self.f_z_integ = np.array(
            [data.get_redshift_evolution(0.1, z, ["growth"])[0][0] for z in z_integ]
        )  # growth rate, unit 1
        self.adot_z_integ = (1.0 / (1.0 + z_integ)) * H_z_integ  # [s-1]
        self.n_H_z_integ = self.nh * (1.0 + z_integ) ** 3.0  # baryon nb density [m-3]

        # Linear matter power spectrum P(z,k) in Mpc^3
        assert (
            kmax_pk <= kmax_camb
        ), "k too large for P(k) extrapolation, modify ell_max or z_min"
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
        self.Pk_lin = np.vectorize(lambda k, z: interp_l.P(z, k))
        # Non-linear matter power spectrum
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
        self.Pk = np.vectorize(lambda k, z: interp_nl.P(z, k))

        self.Pk_lin_integ = self.Pk_lin(
            kp_integ[:, None], z_integ[:, None, None]
        )  # linear matter power spectrum
        self.check_ps(self.Pk_lin_integ)

        self.has_camb_run = True

        Pee_integ = self.Pee(kp_integ[:, None], z_integ[:, None, None])
        Pk_integ = self.Pk(kp_integ[:, None], z_integ[:, None, None])
        self.check_ps(Pee_integ)
        self.check_ps(Pk_integ, include_zero=False)
        self.b_del_e_integ = np.sqrt(Pee_integ / Pk_integ)  # electrons bias

    def Cl_to_Dl(self, ells, Cells):
        """
        Convert dimensionless C_ell to D_ell in muK2.

        Parameters
        ----------
            ells: array of floats
                Angular multipole(s).
            Cells: array of floats
                Corresponding dimensionless angular power Cell.

        Returns
        -------
            Dell: float
                Corresponding Dell.
        """
        ells = np.atleast_1d(ells)
        Cells = np.atleast_1d(Cells)
        # consistency checks
        assert np.size(ells) == np.shape(Cells)[0], "Inputs have different dimensions."
        # convert ells to array if float is given
        ells = np.array(ells)
        Cells = np.array(Cells)

        D_ells = (
            ells[:, None]
            * (ells[:, None] + 1)
            * Cells
            / 2.0
            / np.pi
            * (self.T_cmb * 1e6) ** 2
        )

        return D_ells

    # @profile
    def get_ksz(self, ells, signal="patchy", Dells=True):
        """
        Compute kSZ angular power spectrum for given model at ell.

        Parameters
        ----------
            ells: array of floats
                Angular multipole to compute the spectrum at.
            signal: str
                Wich kSZ signal to compute.
                Options are 'late-time', 'patchy' or 'both'.
                Default is patchy.
            Dells: boolean
                If True, give the results in terms of
                D(l) = Tcmb**2 * l(l+1)Cl/2/pi.
                Default is False.
        Outputs
        ------
            Tuple of (patchy, late-time) KSZ power at ell.
        """

        if signal == "late-time":
            g = z_integ < self.zend_h
        elif signal == "both":
            g = np.ones(self.x_i_z_integ.size, dtype=bool)
        elif signal == "patchy":
            g = z_integ >= self.zend_h
        else:
            raise ValueError("signal must be both, patchy, or late-time.")

        ells = np.atleast_1d(ells)

        if np.sum(self.x_i_z_integ) == 0:
            self.init_reionisation_history()

        cos = cosmology.FlatLambdaCDM(
            H0=self.H0, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )
        kmax_ells = np.max(ells) / cos.comoving_distance(z_min).value

        # Define integration ranges
        k_z_integ = ells[:, None] / self.eta_z_integ  # in [Mpc-1]

        k_min_kp = np.sqrt(
            k_z_integ[:, g, None, None] ** 2
            + kp_integ[None, :, None] ** 2.0
            - 2.0 * k_z_integ[:, g, None, None] * kp_integ[None, :, None] * mu[None, :]
        )  # in [Mpc-1]
        if (min(k_z_integ.min(), k_min_kp.min()) < kmin_camb) or (
            max(k_z_integ.max(), k_min_kp.max()) > kmax_camb
        ):
            raise ValueError(
                "Extrapolating the matter PK too far:"
                f"kmin = {min(k_z_integ.min(), k_min_kp.min()):.1e}, "
                f"kmax = {max(k_z_integ.max(), k_min_kp.max()):.1e}."
            )
        if not self.has_camb_run:
            self.run_camb(kmax_pk=np.min([kmax_pk, k_min_kp.max(), k_z_integ.max(), kp_integ.max()]))

        Pee_min_kp = self.Pee(k_min_kp, z_integ[None, g, None, None])

        I_e = (Pee_min_kp / kp_integ[None, :, None] ** 2.0) - (
            np.sqrt(Pee_min_kp / self.Pk(k_min_kp, z_integ[None, g, None, None]))
            * self.b_del_e_integ[None, g]
            * self.Pk_lin(k_min_kp, z_integ[None, g, None, None])
            / k_min_kp**2
        )

        # Compute Delta_B^2 integrand, in [s-2.Mpc^2]
        Delta_B2_integrand = (
            k_z_integ[:, g, None, None] ** 3.0
            / 2.0
            / np.pi**2.0
            * (
                self.f_z_integ[None, g, None, None]
                * self.adot_z_integ[None, g, None, None]
            )
            ** 2.0
            * kp_integ[None, :, None] ** 3.0
            * np.log(10.0)
            * np.sin(th_integ[None, :])
            / (2.0 * np.pi) ** 2.0
            * self.Pk_lin_integ[None, g]
            * (1.0 - mu[None, :] ** 2.0)
            * I_e
        )
        # Compute Delta_B^2, in [s-2.Mpc^2]
        Delta_B2 = simps(simps(Delta_B2_integrand, th_integ), np.log10(kp_integ))

        # Compute C_kSZ(ell) integrand, unit 1
        C_ell_kSZ_integrand = (
            8.0
            * np.pi**2.0
            / (2.0 * ells[:, None] + 1.0) ** 3.0
            * (constants.sigma_T.value / constants.c.value) ** 2.0
            * (
                self.n_H_z_integ[None, g]
                * self.x_i_z_integ[None, g]
                / (1.0 + z_integ[None, g])
            )
            ** 2.0
            * Delta_B2
            * np.exp(-2.0 * self.tau_z_integ[None, g])
            * self.eta_z_integ[None, g]
            * self.detadz_z_integ[None, g]
            * Mpcm**3.0
        )

        # Compute C_kSZ(ell), no units
        result = np.array(
            [trapz(C_ell_kSZ_integrand[i], z_integ[g]) for i in range(ells.size)]
        )
        self.check_result(result)

        if signal == "both":
            gp = z_integ >= self.zend_h
            pksz = np.array(
                [
                    trapz(C_ell_kSZ_integrand[i][gp], z_integ[gp])
                    for i in range(ells.size)
                ]
            )
            self.check_result(pksz)
            hksz = result - pksz
        elif signal == "patchy":
            pksz = np.copy(result)
            hksz = np.zeros(result.shape)
        else:
            pksz = np.zeros(result.shape)
            hksz = np.copy(result)
        C_ells = np.c_[pksz, hksz]

        if not Dells:
            return C_ells
        else:
            return self.Cl_to_Dl(ells, C_ells)

    def get_p21(self, k, z, mK=True, log=False, pk_units=True):
        P21 = (1.0 - 2.0 * self.xe(z) / self.f) * self.bdH(k, z) * self.Pk(k, z)
        if not pk_units:
            P21 *= k ** 3 / 2.0 / np.pi**2
        P21 += (self.xe(z) / self.f) ** 2 * self.Pee(k, z) 
        if mK:
            T0 = (
                27.0
                #* (self.f - self.xe(z)) # removed for the time being
                * self.Ob_0
                * (self.h ** 2)
                / 0.023
                * np.sqrt(0.15 * (1 + z) / (10 * self.Om_0 * self.h ** 2))
            )  # mK
            P21 *= T0 ** 2
        P21 = np.where(P21<=0, 1.0e-36, P21)
        return np.log10(P21) if log else P21

    def get_tau(self, ells, signal="patchy", Dells=True):
        """
        Compute tau angular power spectrum for given model at ell.

        Parameters
        ----------
            ells: array of floats
                Angular multipole to compute the spectrum at.
            signal: str
                Wich tau signal to compute.
                Options are 'late-time', 'patchy' or 'both'.
                Default is patchy.
            Dells: boolean
                If True, give the results in terms of
                D(l) = l(l+1)Cl/2/pi.
                Default is False.
        Outputs
        ------
            Tuple of (patchy, late-time) tau power at ell.
        """

        if signal == "late-time":
            g = z_integ < self.zend_h
        elif signal == "both":
            g = np.ones(z_integ.size, dtype=bool)
        elif signal == "patchy":
            g = z_integ >= self.zend_h
        else:
            raise ValueError("signal must be both, patchy, or late-time.")

        cos = cosmology.FlatLambdaCDM(
            H0=self.h * 100, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )

        ells = np.atleast_1d(ells)
        kmax_ells = np.max(ells) / cos.comoving_distance(z_integ.min()).value

        if not self.has_camb_run or kmax_ells > kmax_camb:
            warnings.warn('Re-running CAMB...')
            self.run_camb(kmax=kmax_ells)

        # Define integration ranges
        k_z_integ = ells[:, None] / cos.comoving_distance(z_integ).value  # [Mpc-1]
        if (k_z_integ.min() < kmin_camb) or (k_z_integ.max() > kmax_camb):
            raise ValueError("Extrapolating the matter PK too far.")

        Pee_z_integ = self.Pee(k_z_integ, z_integ)
        self.check_ps(Pee_z_integ)

        prefac = (constants.c.si  # speed of light
                  * constants.sigma_T.si**2  # Thomson cross section [m    2]
                  * (self.nh / units.m**3)**2  # baryon nb density [m-3]
                  )

        # Compute C_tau(ell) integrand, unit 1
        C_ell_tau_integrand = (
            self.xe(z_integ)**2
            / cos.H(z_integ).si
            * (1. + z_integ)**4
            / cos.comoving_distance(z_integ).si**2
            * Pee_z_integ * (units.Mpc).si**3
        ) * prefac

        # Compute C_kSZ(ell), no units
        result = np.array(
            [trapz(C_ell_tau_integrand[i], z_integ[g])
             for i in range(ells.size)]
        )
        self.check_result(result)

        if signal == "both":
            gp = z_integ >= self.zend_h
            ptau = np.array(
                [
                    trapz(C_ell_tau_integrand[i][gp], z_integ[gp])
                    for i in range(ells.size)
                ]
            )
            self.check_result(ptau)
            htau = result - ptau
        elif signal == "patchy":
            ptau = np.copy(result)
            htau = np.zeros(result.shape)
        else:
            ptau = np.zeros(result.shape)
            htau = np.copy(result)
        C_ells = np.c_[ptau, htau]

        if not Dells:
            return C_ells
        else:
            return self.Cl_to_Dl(ells, C_ells) / (self.T_cmb * 1e6) ** 2

    def get_B_modes(self, ells, Dells=True, Qrms=17.0):
        """
        Compute angular power spectrum of total EoR-induced B-modes at ell.

        Parameters
        ----------
            ells: array of floats
                Angular multipole to compute the spectrum at.
            Dells: boolean
                If True, give the results in terms of
                D(l) = l(l+1)Cl/2/pi.
                Default is False.
            Qrms: float
                rms of the primary quadrupole, in uK, taken constant
                with redshift. Default is 17uK.
        Outputs
        ------
            Array B-mode power at ells, in uK2.
        """
        return np.sum(self.get_scattering_B_modes(ells=ells, Dells=Dells, signal='both', Qrms=Qrms), axis=1) + self.get_screening_B_modes(ells=ells, Dells=Dells)

    def get_scattering_B_modes(self, ells, signal="patchy", Dells=True, Qrms=17.0):
        """
        Compute angular power spectrum of scattering B-modes for given model at ell.

        Parameters
        ----------
            ells: array of floats
                Angular multipole to compute the spectrum at.
            signal: str
                Wich signal to compute.
                Options are 'late-time', 'patchy' or 'both'.
                Default is patchy.
            Dells: boolean
                If True, give the results in terms of
                D(l) = l(l+1)Cl/2/pi.
                Default is False.
            Qrms: float
                rms of the primary quadrupole, in uK, taken constant
                with redshift. Default is 17uK.
        Outputs
        ------
            Tuple of (patchy, late-time) B-mode power at ell, in uK2.
        """

        if signal == "late-time":
            g = z_integ < self.zend_h
        elif signal == "both":
            g = np.ones(z_integ.size, dtype=bool)
        elif signal == "patchy":
            g = z_integ >= self.zend_h
        else:
            raise ValueError("signal must be both, patchy, or late-time.")

        cos = cosmology.FlatLambdaCDM(
            H0=self.h * 100, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )

        ells = np.atleast_1d(ells)
        kmax_ells = np.max(ells) / cos.comoving_distance(z_integ.min()).value

        if not self.has_camb_run or kmax_ells > kmax_camb:
            warnings.warn('Re-running CAMB...')
            self.run_camb(kmax=kmax_ells)

        # Define integration ranges
        k_z_integ = ells[:, None] / cos.comoving_distance(z_integ).value  # [Mpc-1]
        if (k_z_integ.min() < kmin_camb) or (k_z_integ.max() > kmax_camb):
            raise ValueError("Extrapolating the matter PK too far.")

        Pee_z_integ = self.Pee(k_z_integ, z_integ)
        self.check_ps(Pee_z_integ)

        prefac = (constants.c.si  # speed of light
                  * constants.sigma_T.si**2  # Thomson cross section [m-2]
                  * (self.nh / units.m**3)**2  # baryon nb density [m-3]
                  ) * 3. / 100.

        # Compute C_tau(ell) integrand, unit 1
        C_ell_BB_integrand = (
            prefac
            * self.xe(z_integ)**2
            / cos.H(z_integ).si
            * (1. + z_integ)**4
            / cos.comoving_distance(z_integ).si**2
            * Pee_z_integ * (units.Mpc).si**3
            * np.exp(-2. * self.tau_z_integ)
        )

        # Compute C_kSZ(ell), no units
        result = np.array(
            [trapz(C_ell_BB_integrand[i], z_integ[g])
             for i in range(ells.size)]
        ) * Qrms**2  # rms of the primary quadrupole [uK]
        self.check_result(result)

        if signal == "both":
            gp = z_integ >= self.zend_h
            pBB = np.array(
                [
                    trapz(C_ell_BB_integrand[i][gp], z_integ[gp])
                    for i in range(ells.size)
                ]
            )
            self.check_result(pBB)
            hBB = result - pBB
        elif signal == "patchy":
            pBB = np.copy(result)
            hBB = np.zeros(result.shape)
        else:
            pBB = np.zeros(result.shape)
            hBB = np.copy(result)
        C_ells = np.c_[pBB, hBB]

        if not Dells:
            return C_ells
        else:
            return self.Cl_to_Dl(ells, C_ells) / (self.T_cmb * 1e6) ** 2

    def get_screening_B_modes(self, ells, Dells=True):
        """
        Compute angular power spectrum of scattering B-modes for given model at ell.

        Parameters
        ----------
            ells: array of floats
                Angular multipole to compute the spectrum at.
            Dells: boolean
                If True, give the results in terms of
                D(l) = l(l+1)Cl/2/pi.
                Default is False.
        Outputs
        ------
            Tuple of (patchy, late-time) B-mode power at ell, in uK2.
        """
        ell_integ = np.arange(1, 3000)
        # primary EE modes
        Cell_EE_p = self.get_primary_spectra(ells=ell_integ, Dells=False, unit='muK')[:, 2]
        # tau tau ps (patchy+late-time)
        Cell_tt = np.sum(self.get_tau(ells=ell_integ, signal='both', Dells=False), axis=1)
        # cl bb
        prefac = 1. / 8. / np.pi**2 * np.exp(-2. * self.xe2tau(z=np.linspace(0, 20, 300))[0])
        CBB_screen = np.trapz(Cell_EE_p*Cell_tt, ell_integ) * prefac
        CBB_screen *= np.ones(ells.size)

        if not Dells:
            return CBB_screen
        else:
            return self.Cl_to_Dl(ells, CBB_screen)[:, 0] / (self.T_cmb * 1e6) ** 2

    def get_tau_21_cross(self, z21, ells, Dells=True, delta_nu=0.):
        """
        Compute angular cross spectrum of tau and 21cm at ell, in muK.

        Parameters
        ----------
            z21: float
                Redshift of the observed 21cm signal.
            ells: array of floats
                Angular multipole to compute the spectrum at.
            Dells: boolean
                If True, give the results in terms of
                D(l) = l(l+1)Cl/2/pi.
                Default is False.
            delta_nu: float
                Width of the top-hat function representing the
                frequency resolution of the interferometer.
                Default is zero (Dirac delta).
        Outputs
        ------
            Array of cross-power for ells, in uK2.
        """
        cos = cosmology.FlatLambdaCDM(
            H0=self.h * 100, Tcmb0=self.T_cmb, Ob0=self.Ob_0, Om0=self.Om_0
        )

        ells = np.atleast_1d(ells)
        kmax_ells = np.max(ells) / cos.comoving_distance(z_integ.min()).value

        if not self.has_camb_run or kmax_ells > kmax_camb:
            warnings.warn('Re-running CAMB...')
            self.run_camb(kmax=kmax_ells)

        # Define integration ranges
        k_z_integ = ells[:, None] / cos.comoving_distance(z_integ).value  # [Mpc-1]
        if (k_z_integ.min() < kmin_camb) or (k_z_integ.max() > kmax_camb):
            raise ValueError("Extrapolating the matter PK too far.")

        Pee_z_integ = self.Pee(k_z_integ, z_integ)
        self.check_ps(Pee_z_integ)

        Pbb_z_integ = self.bdH(k_z_integ, z_integ) * self.Pk(k_z_integ, z_integ)
        self.check_ps(Pbb_z_integ)

        W21_z_integ = np.zeros_like(z_integ)
        if delta_nu == 0.:
            iz21 = np.argmin(np.abs(z_integ - z21))
            W21_z_integ[iz21] = 1.
        else:
            nu21 = nu21_ref / (1. + z21) # MHz
            numax = nu21 + delta_nu/2.
            numin = nu21 - delta_nu/2.
            zmax = (nu21_ref / numin) - 1.
            zmin = (nu21_ref / numax) - 1.
            mask = (z_integ < zmax) & (z_integ >= zmin)
            W21_z_integ[mask] = (1.+zmin)*(1.+zmax)/(zmax-zmin)
        # print(W21_z_integ)

        T0 = (
            27.0 * 1e3  # muK
            * self.Ob_0
            * (self.h ** 2)
            / 0.023
            * np.sqrt(0.15 * (1 + z21) / (10 * self.Om_0 * self.h ** 2))
        )  # uK

        prefac = (1.  # speed of light
                  * constants.sigma_T.si  # Thomson cross section [m    2]
                  * (self.nh / units.m**3)  # baryon nb density [m-3]
                  ) * T0

        # Compute C_tau(ell) integrand, unit 1
        C_ell_tau21_integrand = (
            prefac.to(1./units.Mpc)
            * self.xe(z_integ)
            / (1. + z_integ) ** 2
            / cos.comoving_distance(z_integ)**2
            * [Pbb_z_integ - self.xe(z_integ) * Pee_z_integ] * (units.Mpc)**3
            * W21_z_integ
        )[0]
        # print(C_ell_tau21_integrand[12, iz21])
        # print(C_ell_tau21_integrand.unit)

        # Compute C_kSZ(ell), no units
        C_ells = np.array(
            [trapz(C_ell_tau21_integrand[i], z_integ)
             for i in range(ells.size)]
        )
        self.check_result(C_ells)

        if not Dells:
            return C_ells
        else:
            return ells * (ells + 1.) * C_ells / 2. / np.pi


    def check_ps(self, ps, include_zero=True):
        """
        Perform basic consistency checks on power spectrum array.

        Parameters
        ----------
        ps : array of floats
            Power spectrum values.
        include_zero: boolean
            Whether zero values are allowed for the ps.
        """
        if (include_zero and np.any(ps < 0.0)) or (
            np.any(ps <= 0.0) and not include_zero
        ):
            raise ValueError("Negative values in the power spectrum.")
        if np.isnan(ps).any():
            raise ValueError("NaN values in the power spectrum.")

    def check_result(self, res):
        """
        Perform basic consistency checks on result array.

        Parameters
        ----------
        res: array of floats
            Array of values to check.
        """
        if np.isnan(res).any():
            raise ValueError(f"NaN values: {res}.")
        return res
