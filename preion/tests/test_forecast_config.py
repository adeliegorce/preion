import numpy as np
import pytest
import yaml

pytest.importorskip("preion.forecast.config")

from preion.forecast.config import build_ells, load_config, run_label

EXAMPLE_CONFIG = "configs/cv_limited_new.yaml"


@pytest.fixture
def cfg(pytestconfig):
    root = pytestconfig.rootpath
    return load_config(root / EXAMPLE_CONFIG)


@pytest.fixture
def full_cfg_dict(pytestconfig):
    """A complete, valid config dict, loaded straight from YAML (no
    load_config validation applied yet) so tests can mutate it and re-dump it
    to check load_config's validation in isolation."""
    root = pytestconfig.rootpath
    with open(root / EXAMPLE_CONFIG) as f:
        return yaml.safe_load(f)


def test_load_config_defaults(cfg):
    assert cfg["label"] == "cv_limited_new"
    assert cfg["data"] == "all"
    assert cfg["theta_true"] == [7.0, 1.5, 3.7, 0.10]
    assert cfg["nwalkers"] == 8


def test_load_config_rejects_bad_data(tmp_path, full_cfg_dict):
    full_cfg_dict["data"] = "bogus"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(full_cfg_dict))
    with pytest.raises(ValueError):
        load_config(bad)


def test_load_config_requires_all_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data: all\n")
    with pytest.raises(KeyError):
        load_config(bad)


def test_load_config_requires_ells_entries(tmp_path, full_cfg_dict):
    del full_cfg_dict["ells"]["bb"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(full_cfg_dict))
    with pytest.raises(KeyError):
        load_config(bad)


def test_load_config_rejects_mixed_ells_keys(tmp_path, full_cfg_dict):
    # Has both 'step' and 'num': not one of the two valid key sets.
    full_cfg_dict["ells"]["bb"] = {
        "start": 100, "stop": 5000, "step": 100, "num": 50,
    }
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump(full_cfg_dict))
    with pytest.raises(ValueError):
        load_config(bad)


def test_load_config_accepts_either_ells_keyset(tmp_path, full_cfg_dict):
    # Either {start, stop, step} or {start, stop, num} is valid for any entry.
    full_cfg_dict["ells"]["bb"] = {"start": 100, "stop": 5000, "step": 100}
    ok = tmp_path / "ok.yaml"
    ok.write_text(yaml.dump(full_cfg_dict))
    load_config(ok)


def test_build_ells_shapes(cfg):
    ells_tau, ells_ksz, ells_bb = build_ells(cfg)
    assert np.all(ells_tau == np.arange(100, 5000, step=100))
    assert np.all(ells_ksz == np.arange(1000, 8000, step=500))
    assert ells_bb.size == 50
    assert ells_bb[0] == 10
    assert ells_bb[-1] == 1000


def test_build_ells_step_based_bb(tmp_path, full_cfg_dict):
    # bb given a {start, stop, step} spec instead of {start, stop, num}
    # should build with np.arange, same as tau/ksz.
    full_cfg_dict["ells"]["bb"] = {"start": 100, "stop": 5000, "step": 100}
    ok = tmp_path / "ok.yaml"
    ok.write_text(yaml.dump(full_cfg_dict))
    cfg = load_config(ok)
    _, _, ells_bb = build_ells(cfg)
    assert np.all(ells_bb == np.arange(100, 5000, step=100))


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
