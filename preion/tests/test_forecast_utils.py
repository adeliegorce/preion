import numpy as np
import pytest
from astropy import units

pytest.importorskip("preion.forecast.utils")

from preion.parameters import telescope_specs
from preion.forecast.utils import (
    sample_var, noise, get_lbins, _survey_label, get_fsky_21cm,
    get_sensitivity, get_cl21_noise, invert_covariance,
)


def test_sample_var_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sample_var(np.array([1., 2.]), np.array([1., 2., 3.]), 0.5)


def test_sample_var_with_fsky_float():
    ls = np.array([10., 20., 30.])
    dl = np.array([1., 1., 1.])
    out = sample_var(ls, dl, 0.5)
    expected = dl * np.sqrt(2. / 0.5 / (2. * ls + 1.))
    np.testing.assert_allclose(out, expected)


def test_sample_var_with_telescope_dict():
    ls = np.array([10., 20.])
    dl = np.array([1., 1.])
    out_dict = sample_var(ls, dl, {'fsky': 0.4})
    out_float = sample_var(ls, dl, 0.4)
    np.testing.assert_allclose(out_dict, out_float)


def test_noise_positive_and_scales_with_pol():
    ls = np.array([100, 500, 1000])
    tel = telescope_specs['SO-LAT']
    nl = noise(ls, tel, pol=False)
    nl_pol = noise(ls, tel, pol=True)
    assert np.all(nl > 0)
    np.testing.assert_allclose(nl_pol, nl * 2.)


def test_get_lbins_within_telescope_range():
    ells, edges, _ = get_lbins('CMB-HD')
    tel = telescope_specs['CMB-HD']
    assert ells.min() >= tel['lmin']
    assert ells.max() <= tel['lmax']
    assert edges.size == ells.size + 1


# --- _survey_label ---

def test_survey_label_non_ska_without_n_fields():
    assert _survey_label("hera", "moderate") == "hera_moderate"


def test_survey_label_non_ska_ignores_n_fields():
    # n_fields is only ever meaningful for 'ska'; must not be appended here.
    assert _survey_label("hera", "moderate", n_fields=3) == "hera_moderate"


def test_survey_label_ska_without_n_fields():
    assert _survey_label("ska", "optimistic", ska_array="aast") == "ska_aast_optimistic"


def test_survey_label_ska_with_n_fields():
    # Matches a real file on disk: sensitivity_21cm/ska_aast_optimistic_3fields_*.
    assert _survey_label("ska", "optimistic", ska_array="aast", n_fields=3) == "ska_aast_optimistic_3fields"


def test_survey_label_ska_requires_ska_array():
    with pytest.raises(ValueError):
        _survey_label("ska", "optimistic")


# --- get_fsky_21cm ---

def test_get_fsky_21cm_known_telescopes_are_positive_and_small():
    for tel in ("hera", "ska", "mwa"):
        fsky = get_fsky_21cm(tel)
        assert 0. < float(fsky) < 1.


def test_get_fsky_21cm_ska_scales_with_n_fields():
    fsky1 = get_fsky_21cm("ska")
    fsky3 = get_fsky_21cm("ska", n_fields=3)
    assert np.isclose(float(fsky3), 3. * float(fsky1))


def test_get_fsky_21cm_unknown_telescope_raises():
    with pytest.raises(ValueError):
        get_fsky_21cm("bogus")


# --- get_sensitivity ---

def test_get_sensitivity_returns_callable_interpolator():
    sens = get_sensitivity("hera", "moderate")
    out = sens([6.8, 0.2])
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_get_sensitivity_out_of_range_is_inf():
    sens = get_sensitivity("hera", "moderate")
    out = sens([6.8, 1e6])  # k far outside the tabulated range
    assert np.isinf(out).all()


def test_get_sensitivity_bogus_combination_raises():
    with pytest.raises(OSError):
        get_sensitivity("hera", "bogus_case")


def test_get_sensitivity_custom_sensitivity_dir(tmp_path):
    from preion.forecast.utils import _SENSITIVITY_DIR
    import shutil
    for f in ("hera_moderate_thermal_94.7.txt",):
        shutil.copy(f"{_SENSITIVITY_DIR}/{f}", tmp_path / f)
    sens = get_sensitivity("hera", "moderate", zs=[14.0], sensitivity_dir=str(tmp_path))
    assert np.isfinite(sens([14.0, 0.2])).all()


# --- get_cl21_noise ---

def test_get_cl21_noise_shape_and_units():
    sens = get_sensitivity("hera", "moderate")
    ells = np.array([500., 1000., 2000.])
    out = get_cl21_noise(sens, 7.0, ells, Dells=True, delta_nu=0.)
    assert out.shape == ells.shape
    out.to(units.uK**2)  # raises if not convertible
    assert np.all(np.isfinite(out) | np.isinf(out))


def test_get_cl21_noise_accepts_float_or_quantity_delta_nu():
    sens = get_sensitivity("hera", "moderate")
    ells = np.array([500., 1000., 2000.])
    out_float = get_cl21_noise(sens, 7.0, ells, Dells=True, delta_nu=50.)
    out_quantity = get_cl21_noise(sens, 7.0, ells, Dells=True, delta_nu=50. * units.MHz)
    np.testing.assert_allclose(out_float.value, out_quantity.value)


# --- invert_covariance ---

def test_invert_covariance_zeroes_infinite_variance():
    cov = np.diag([1., 2., np.inf, 4.])
    inv_cov = invert_covariance(cov)
    np.testing.assert_allclose(np.diagonal(inv_cov), [1., 0.5, 0., 0.25])
    assert np.all(inv_cov[2, :] == 0.)
    assert np.all(inv_cov[:, 2] == 0.)
    resid = np.array([1., 1., 1e12, 1.])  # arbitrarily large residual at the inf-variance index
    chi2 = resid.T.dot(inv_cov).dot(resid)
    assert np.isfinite(chi2)
    assert chi2 == pytest.approx(1. + 0.5 + 0.25)


def test_invert_covariance_handles_off_diagonal_terms():
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    inv_cov = invert_covariance(cov)
    np.testing.assert_allclose(inv_cov, np.linalg.inv(cov))
