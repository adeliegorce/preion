import numpy as np
import pytest

pytest.importorskip("preion.forecast.config")

from preion.forecast.config import build_ells, load_config, run_label

EXAMPLE_CONFIG = "configs/cv_limited_new.yaml"


@pytest.fixture
def cfg(pytestconfig):
    root = pytestconfig.rootpath
    return load_config(root / EXAMPLE_CONFIG)


def test_load_config_defaults(cfg):
    assert cfg["label"] == "cv_limited_new"
    assert cfg["data"] == "all"
    assert cfg["theta_true"] == [7.0, 1.5, 3.7, 0.10]
    assert cfg["nwalkers"] == 8


def test_load_config_rejects_bad_data(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data: bogus\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_build_ells_shapes(cfg):
    ells_tau, ells_ksz, ells_bb = build_ells(cfg)
    assert np.all(ells_tau == np.arange(100, 5000, step=100))
    assert np.all(ells_ksz == np.arange(1000, 8000, step=500))
    assert ells_bb.size == 50
    assert ells_bb[0] == 10
    assert ells_bb[-1] == 1000


@pytest.mark.parametrize("data,expected", [
    ("bb", "cv_limited_new_bb_only"),
    ("ksz", "cv_limited_new_ksz_only"),
    ("tau", "cv_limited_new_tau_only"),
    ("all", "cv_limited_new_all"),
])
def test_run_label(cfg, data, expected):
    assert run_label(cfg, data) == expected


def test_run_label_rejects_bad_data(cfg):
    with pytest.raises(ValueError):
        run_label(cfg, "bogus")
