import numpy as np
import pytest

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
