import numpy as np
import pytest
from astropy import units

from preion.theory import Pee_model


def test_pee_model_construction(fast_pee_model):
    assert fast_pee_model.zre_h == 7.0
    assert fast_pee_model.tau > 0.


def test_xe_monotonic_in_redshift(fast_pee_model):
    z = np.linspace(0, 20, 50)
    xe = fast_pee_model.xe(z)
    # ionisation fraction should be ~1 well after reionisation (low z)
    # and drop towards 0 well before it (high z)
    assert xe[0] > xe[-1]
    assert 0. <= xe.min()
    assert xe.max() <= 1. + fast_pee_model.f + 1e-6  # allow for helium double-reionisation bump


def test_xe2tau_matches_init_tau(fast_pee_model):
    z = np.linspace(0, 30, 3000)
    tau_of_z = fast_pee_model.xe2tau(z)
    # xe2tau(z) is the *remaining* optical depth integrated from z out to
    # z.max(), so it starts at the model's total tau (z=0) and decreases
    # towards 0 as z grows past the reionisation era.
    assert tau_of_z[0] == pytest.approx(fast_pee_model.tau, rel=1e-6)
    assert tau_of_z[-1] == pytest.approx(0., abs=1e-6)
    assert np.all(np.diff(tau_of_z) <= 1e-12)


def test_check_ps_raises_on_negative():
    m = Pee_model(run_camb=False, verbose=False)
    with pytest.raises(ValueError):
        m.check_ps(np.array([1.0, -1.0, 2.0]))


def test_check_ps_raises_on_nan():
    m = Pee_model(run_camb=False, verbose=False)
    with pytest.raises(ValueError):
        m.check_ps(np.array([1.0, np.nan, 2.0]))


@pytest.mark.slow
def test_get_tau_and_ksz_with_camb(tiny_ells):
    m = Pee_model(
        zre_h=7.0, dz_h=1.5, alpha0=3.7, kappa=0.10,
        verbose=False, run_camb=True,
    )
    tau_ps = m.get_tau(ells=tiny_ells[0], signal='both', Dells=True)
    assert tau_ps.shape[0] == len(tiny_ells[0])
    assert np.all(np.isfinite(tau_ps))

    ksz_ps = m.get_ksz(ells=tiny_ells[1], signal='patchy', Dells=True)
    assert ksz_ps.shape[0] == len(tiny_ells[1])
    assert np.all(np.isfinite(ksz_ps))


@pytest.fixture(scope="module")
def camb_pee_model():
    """A single CAMB-backed Pee_model, shared across this module's slow
    tests to avoid re-running CAMB (~1 min) for each one."""
    return Pee_model(
        zre_h=7.0, dz_h=1.5, alpha0=3.7, kappa=0.10,
        verbose=False, run_camb=True,
    )


@pytest.mark.slow
def test_get_p21_positive_and_finite(camb_pee_model):
    k = np.logspace(-2, 0, 20)
    p21 = camb_pee_model.get_p21(k, 9.0, mK=True, log=False, pk_units=True)
    assert np.all(np.isfinite(p21))
    assert np.all(p21 > 0)


@pytest.mark.slow
def test_get_cl21_requires_quantity_delta_nu(camb_pee_model):
    # theory.py's get_cl21 requires delta_nu as an astropy Quantity, while
    # get_tau_21_cross/get_BB_21_cross accept a bare float -- a pre-existing
    # inconsistency, intentionally not "fixed" as part of the cross-forecast
    # port (out of scope, behavior-changing to tested physics). This test
    # documents/pins the asymmetry as expected current behavior.
    with pytest.raises(AttributeError):
        camb_pee_model.get_cl21(7.0, np.array([500., 1000.]), Dells=True, delta_nu=0.)


@pytest.mark.slow
def test_get_cl21_with_camb(camb_pee_model):
    ells = np.array([500., 1000., 2000.])
    out = camb_pee_model.get_cl21(7.0, ells, Dells=True, delta_nu=0. * units.MHz)
    assert out.shape == ells.shape
    out.to(units.uK**2)
    assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_get_cl21_nonzero_delta_nu_top_hat(camb_pee_model):
    ells = np.array([500., 1000., 2000.])
    out_zero = camb_pee_model.get_cl21(7.0, ells, Dells=True, delta_nu=0. * units.MHz)
    out_wide = camb_pee_model.get_cl21(7.0, ells, Dells=True, delta_nu=50. * units.MHz)
    assert np.all(np.isfinite(out_wide))
    assert np.all(out_wide > 0)
    assert not np.allclose(out_zero.value, out_wide.value)


@pytest.mark.slow
def test_get_tau_21_cross_with_camb(camb_pee_model):
    ells = np.array([500., 1000., 2000.])
    out = camb_pee_model.get_tau_21_cross(7.0, ells, Dells=True, delta_nu=0.)
    assert out.shape == ells.shape
    out.to(units.uK)
    assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_get_BB_21_cross_with_camb(camb_pee_model):
    # Regression test for two bugs fixed in this method: (1) it used
    # self.tau_z_integ (evaluated on a fixed, unrelated module-level z
    # grid) instead of self.xe2tau(zlin) (its own local z grid), causing a
    # shape mismatch; (2) prefac_scat/prefac_scr were folded into the trapz
    # operand but not into the unit reattached afterward, and the final
    # conversion used units.uK instead of units.uK**2 (contradicting the
    # method's own docstring). A clean run with finite uK2 output for every
    # mode confirms both are fixed.
    ells = np.array([500., 1000., 2000.])
    for mode in ("scattering", "screening", "both"):
        out = camb_pee_model.get_BB_21_cross(7.0, ells, Dells=True, delta_nu=0., mode=mode)
        assert out.shape == ells.shape
        out.to(units.uK**2)
        assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_get_BB_21_cross_multiple_z21(camb_pee_model):
    ells = np.array([500., 1000., 2000.])
    for z in (6.5, 7.5, 9.0):
        out = camb_pee_model.get_BB_21_cross(z, ells, Dells=True, delta_nu=0., mode='both')
        assert np.all(np.isfinite(out))
