import numpy as np
import pytest
from astropy import cosmology

pytest.importorskip("preion.forecast.mcmc")

from preion.forecast import mcmc
from preion.forecast.datapoints import make_cross_datapoints, _build_fiducial_model
from preion.forecast.utils import invert_covariance


@pytest.fixture(scope="module")
def cross_datapoints_and_model(theta_true, tiny_z21):
    """Built once and shared across this module's slow tests -- each of
    these needs the same (expensive, telescope-dependent) tau_noise
    computation, so building it once avoids re-paying that cost per test."""
    dp = make_cross_datapoints(
        theta_true, "CMB-S4-SAT", "hera", "moderate", tiny_z21,
        delta_nu=100., lbin_edges=[20, 100, 200, 330], use_ksz_emulator=False,
    )
    dp["z21"] = list(tiny_z21)
    dp["delta_nu"] = 100.
    dp["inv_cov_tau21"] = np.array([invert_covariance(dp["cov_tau21"][iz]) for iz in range(len(dp["z21"]))])
    cos = cosmology.Planck18
    preion_model = _build_fiducial_model(theta_true, cos, use_ksz_emulator=False)
    return dp, preion_model


@pytest.mark.slow
def test_get_model_cross_branch_shapes(theta_true, cross_datapoints_and_model):
    dp, preion_model = cross_datapoints_and_model
    log_kappa = True
    theta = list(theta_true)
    theta[-1] = np.log10(theta[-1])
    tau21_model, cl21_model, tautau_model, tau, dksz = mcmc.get_model(
        theta, preion_model, dp, log_kappa, "tau21")
    nz, nell = len(dp["z21"]), dp["ells"].size
    assert tau21_model.shape == (nz, nell)
    assert cl21_model.shape == (nz, nell)
    assert tautau_model.shape == (nell,)
    assert tau > 0.
    assert dksz.shape == (2,)


@pytest.mark.slow
def test_lnlike_cross_branch_finite_at_truth(theta_true, cross_datapoints_and_model):
    dp, preion_model = cross_datapoints_and_model
    log_kappa = True
    theta = list(theta_true)
    theta[-1] = np.log10(theta[-1])
    chi2, tau21_model, cl21_model, tautau_model, tau, dksz = mcmc.lnlike(
        theta, preion_model, dp, log_kappa, "tau21")
    assert np.isfinite(chi2)


@pytest.mark.slow
def test_lnprob_cross_out_of_prior_returns_zero_blobs(theta_true, cross_datapoints_and_model):
    dp, preion_model = cross_datapoints_and_model
    log_kappa = True
    priors = [(5., 10.), (0.1, 4.5), (2.5, 4.5), (-1.30, -0.40)]
    bad_theta = [20., 1.5, 3.7, np.log10(0.10)]  # zre way outside the box
    lp, m1, m2, m3, tau, dksz = mcmc.lnprob(bad_theta, priors, preion_model, dp, log_kappa, "tau21")
    assert lp == -np.inf
    assert m1.shape == (len(dp["z21"]), dp["ells"].size)
    assert tau == 0.
    assert dksz.shape == (2,)


def _make_auto_cv_limited_datapoints(theta_true, tiny_ells):
    from preion.forecast.datapoints import make_autos_datapoints
    tau_ps, ksz_ps, total_bb, cov_tau, cov_ksz, cov_bb = make_autos_datapoints(
        theta_true, telescopes=None, ells=tiny_ells, use_ksz_emulator=False,
    )
    dp = {
        "tau": tau_ps, "ksz": ksz_ps, "bb": total_bb,
        "cov_tau": cov_tau, "cov_ksz": cov_ksz, "cov_bb": cov_bb,
        "ells_tau": tiny_ells[0], "ells_ksz": tiny_ells[1], "ells_bb": tiny_ells[2],
    }
    for name in ("tau", "ksz", "bb"):
        dp[f"inv_cov_{name}"] = invert_covariance(dp[f"cov_{name}"])
    return dp


@pytest.mark.slow
def test_zend_prior_rejects_auto_theta_below_limit(theta_true, tiny_ells):
    """zend_prior (lower-limit on zre-dz) must be enforceable for an AUTO
    (data='tau') config too, not just cross -- confirming the
    generalization is real, not cross-only in practice."""
    cos = cosmology.Planck18
    preion_model = _build_fiducial_model(theta_true, cos, use_ksz_emulator=False)
    dp = _make_auto_cv_limited_datapoints(theta_true, tiny_ells)
    priors = [(5., 10.), (0.1, 4.5), (2.5, 4.5), (0.05, 0.4)]

    theta_ok = list(theta_true)  # zre=7.0, dz=1.5 -> zend=5.5
    lp_ok, *_ = mcmc.lnprob(theta_ok, priors, preion_model, dp, False, "tau", zend_prior=6.0)
    assert lp_ok == -np.inf  # zend=5.5 < zend_prior=6.0

    lp_no_cutoff, *_ = mcmc.lnprob(theta_ok, priors, preion_model, dp, False, "tau", zend_prior=None)
    assert np.isfinite(lp_no_cutoff)


@pytest.mark.slow
def test_tau_prior_shifts_auto_posterior(theta_true, tiny_ells):
    """tau_prior (Gaussian on the model's integrated tau) must measurably
    change lnprob for an AUTO config too."""
    cos = cosmology.Planck18
    preion_model = _build_fiducial_model(theta_true, cos, use_ksz_emulator=False)
    dp = _make_auto_cv_limited_datapoints(theta_true, tiny_ells)
    priors = [(5., 10.), (0.1, 4.5), (2.5, 4.5), (0.05, 0.4)]
    theta = list(theta_true)

    lp_no_prior, *_, tau_no_prior, _ = mcmc.lnprob(theta, priors, preion_model, dp, False, "tau")
    # A tau_prior centred far from the model's actual tau should pull lnprob down.
    far_tau_prior = [tau_no_prior + 1.0, 0.001]
    lp_with_prior, *_ = mcmc.lnprob(theta, priors, preion_model, dp, False, "tau", tau_prior=far_tau_prior)
    assert lp_with_prior < lp_no_prior
