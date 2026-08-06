import os

import numpy as np
import pytest
from astropy import cosmology

pytest.importorskip("preion.forecast.datapoints")

from preion.forecast.datapoints import (
    make_autos_datapoints, load_cross_datapoints, make_cross_datapoints,
    _build_fiducial_model, _tautau_signal_and_noise,
)


def test_packaged_datapoints_shapes(packaged_datapoints):
    # configs/config_tutorial_mcmc_autos.yaml's ells grids: tau 49 points, ksz 14, bb 50.
    assert packaged_datapoints["tau"].shape == (49,)
    assert packaged_datapoints["ksz"].shape == (14,)
    assert packaged_datapoints["bb"].shape == (50,)
    assert packaged_datapoints["cov_tau"].shape == (49, 49)
    assert packaged_datapoints["cov_ksz"].shape == (14, 14)
    assert packaged_datapoints["cov_bb"].shape == (50, 50)
    assert packaged_datapoints["ells_tau"].shape == (49,)
    assert packaged_datapoints["ells_ksz"].shape == (14,)
    assert packaged_datapoints["ells_bb"].shape == (50,)
    for key in ["tau", "ksz", "bb", "cov_tau", "cov_ksz", "cov_bb"]:
        assert np.all(np.isfinite(packaged_datapoints[key]))


@pytest.mark.slow
def test_make_autos_datapoints_shapes_no_telescope(theta_true, tiny_ells):
    tau_ps, ksz_ps, total_bb, cov_tau, cov_ksz, cov_bb = make_autos_datapoints(
        theta_true, telescopes=None, ells=tiny_ells,
        use_ksz_emulator=False, randomness=False, save=None,
    )
    assert tau_ps.shape == (len(tiny_ells[0]),)
    assert ksz_ps.shape == (len(tiny_ells[1]),)
    assert total_bb.shape == (len(tiny_ells[2]),)
    assert cov_tau.shape == (len(tiny_ells[0]), len(tiny_ells[0]))
    assert cov_ksz.shape == (len(tiny_ells[1]), len(tiny_ells[1]))
    assert cov_bb.shape == (len(tiny_ells[2]), len(tiny_ells[2]))
    assert np.all(np.isfinite(tau_ps))
    assert np.all(np.isfinite(ksz_ps))
    assert np.all(np.isfinite(total_bb))


@pytest.mark.slow
def test_make_autos_datapoints_save_writes_files(theta_true, tiny_ells, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    make_autos_datapoints(
        theta_true, telescopes=None, ells=tiny_ells,
        use_ksz_emulator=False, randomness=False, save="unittest",
    )
    for suffix in ["bb_datapoints", "ksz_datapoints", "tau_datapoints", "cov_ksz", "cov_tau", "cov_bb"]:
        assert (tmp_path / "data" / f"unittest_{suffix}.txt").exists()


# --- shared helpers ---

@pytest.mark.slow
def test_build_fiducial_model_sets_theta(theta_true):
    cos = cosmology.Planck18
    m = _build_fiducial_model(theta_true, cos, use_ksz_emulator=False)
    assert m.zre_h == theta_true[0]
    assert m.dz_h == theta_true[1]
    assert m.alpha0 == theta_true[2]
    assert m.kappa == theta_true[3]
    assert m.tau > 0.


@pytest.mark.slow
def test_tautau_signal_and_noise_shapes(theta_true):
    cos = cosmology.Planck18
    m = _build_fiducial_model(theta_true, cos, use_ksz_emulator=False)
    # Must stay within CMB-S4-SAT's lmin/lmax (20/330) -- tau_noise's
    # reconstruction-noise curve is only computed up to the telescope's
    # own lmax, and a small-lmax telescope keeps this test fast.
    ells = np.array([50., 150., 300.])
    tautau, nl_tautau = _tautau_signal_and_noise(m, "CMB-S4-SAT", ells)
    assert tautau.shape == ells.shape
    assert nl_tautau.shape == ells.shape
    assert np.all(np.isfinite(tautau))
    assert np.all(np.isfinite(nl_tautau))


# --- cross datapoints ---

@pytest.mark.slow
def test_make_cross_datapoints_shapes(theta_true, tiny_z21):
    dp = make_cross_datapoints(
        theta_true, "CMB-S4-SAT", "hera", "moderate", tiny_z21,
        delta_nu=100., lbin_edges=[20, 100, 200, 330], use_ksz_emulator=False,
    )
    nz, nell = len(tiny_z21), dp["ells"].size
    assert dp["tau21"].shape == (nz, nell)
    assert dp["cl21"].shape == (nz, nell)
    assert dp["cov_tau21"].shape == (nz, nell, nell)
    assert dp["tautau"].shape == (nell,)
    assert dp["tau"] > 0.
    # The fixed tau_noise call path (get_cls_for_tau_noise + the new
    # 4-positional-arg tau_noise signature) is exactly what's exercised
    # here; a clean run without raising is a direct regression test for
    # the known bug in the old forecast_cross/forecast_utils.py.
    assert np.all(np.isfinite(dp["tautau"]))
    # cov entries may legitimately be inf (out of survey k-coverage) but
    # never nan.
    assert not np.any(np.isnan(dp["cov_tau21"]))


@pytest.mark.slow
def test_make_cross_datapoints_multi_z21(theta_true):
    z21 = [6.5, 9.5]
    dp = make_cross_datapoints(
        theta_true, "CMB-S4-SAT", "hera", "moderate", z21,
        delta_nu=100., lbin_edges=[20, 100, 200, 330], use_ksz_emulator=False,
    )
    assert dp["tau21"].shape[0] == 2
    assert dp["cov_tau21"].shape[0] == 2


@pytest.mark.slow
def test_make_cross_datapoints_save_and_reload_roundtrip(theta_true, tiny_z21, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    dp = make_cross_datapoints(
        theta_true, "CMB-S4-SAT", "hera", "moderate", tiny_z21,
        delta_nu=100., lbin_edges=[20, 100, 200, 330], use_ksz_emulator=False,
        save="unittest_cross",
    )
    dp2 = load_cross_datapoints("data", "unittest_cross", tiny_z21)
    np.testing.assert_allclose(dp["ells"], dp2["ells"])
    np.testing.assert_allclose(dp["tau21"], dp2["tau21"])
    np.testing.assert_allclose(dp["cl21"], dp2["cl21"])
    np.testing.assert_allclose(dp["tautau"], dp2["tautau"])
    assert np.isclose(dp["tau"], dp2["tau"])
    # cov reconstructed from the saved sqrt(diag) error column -- compare
    # diagonals only (the file format doesn't round-trip off-diagonals,
    # matching load_autos_datapoints's own convention).
    np.testing.assert_allclose(
        np.diagonal(dp["cov_tau21"], axis1=1, axis2=2),
        np.diagonal(dp2["cov_tau21"], axis1=1, axis2=2),
    )
