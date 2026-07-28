import numpy as np
import pytest

from preion.theory import Pee_model


@pytest.fixture
def fast_pee_model():
    """A Pee_model instance built without running CAMB, for tests that only
    need the reionisation-history part of the physics (fast)."""
    return Pee_model(
        zre_h=7.0, dz_h=1.5, alpha0=3.7, kappa=0.10,
        verbose=False, run_camb=False,
    )


@pytest.fixture
def tiny_ells():
    """A small 3-array [ells_tau, ells_ksz, ells_bb] grid, coarse enough that
    CAMB/Pee_model calls in tests stay fast."""
    return [
        np.array([100, 500, 1000]),
        np.array([3000]),
        np.array([100, 500, 1000]),
    ]


@pytest.fixture
def theta_true():
    return [7.0, 1.5, 3.7, 0.10]


@pytest.fixture
def packaged_datapoints():
    """The pre-computed mock tau/kSZ/BB datapoints shipped with the
    package, for tests that need realistic data without running CAMB."""
    from preion.forecast.datapoints import load_packaged_datapoints
    return load_packaged_datapoints()
