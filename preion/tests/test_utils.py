import numpy as np
import pytest

from preion.parameters import telescope_specs
from preion.utils import sample_var, noise, get_lbins


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
    tel = telescope_specs['CMB-HD']
    nl = noise(ls, tel, pol=False)
    nl_pol = noise(ls, tel, pol=True)
    assert np.all(nl > 0)
    np.testing.assert_allclose(nl_pol, nl * 2.)


def test_get_lbins_within_telescope_range():
    ells, edges, dells = get_lbins('CMB-HD')
    tel = telescope_specs['CMB-HD']
    assert ells.min() >= tel['lmin']
    assert ells.max() <= tel['lmax']
    assert edges.size == ells.size + 1
