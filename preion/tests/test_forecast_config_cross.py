import copy

import pytest
import yaml

pytest.importorskip("preion.forecast.config")

from preion.forecast.config import build_ells, load_config, run_label

# The existing tests/test_forecast_config.py fixtures point at a stale
# configs/cv_limited_new.yaml that no longer exists (pre-existing breakage,
# unrelated to the cross-correlation feature) -- build a base config dict
# inline here instead of depending on that file, so this file's tests are
# unaffected by that separate issue.
_BASE_AUTO_CFG = {
    "label": "unittest", "data": "all", "theta_true": [7.0, 1.5, 3.7, 0.10],
    "log_kappa": True, "niterations": 10, "nwalkers": 8, "fsky": 1.0,
    "use_ksz_emulator": False, "overwrite": True, "plot": False, "progress": True,
    "telescopes": None, "output_dir": ".",
    "ells": {
        "tau": {"start": 100, "stop": 5000, "step": 100},
        "ksz": {"start": 1000, "stop": 8000, "step": 500},
        "bb": {"start": 10, "stop": 1000, "num": 50},
    },
}

_BASE_CROSS_CFG = {
    "label": "unittest_cross", "data": "tau21", "theta_true": [7.0, 1.5, 3.7, 0.10],
    "log_kappa": True, "niterations": 10, "nwalkers": 8, "fsky": 1.0,
    "use_ksz_emulator": False, "overwrite": True, "plot": False, "progress": True,
    "telescopes": ["CMB-S4-LAT"], "output_dir": ".",
    "telescope_21": "hera", "sensitivity_case": "moderate",
    "delta_nu": 100.0, "z21": [7.0],
}


def _write(tmp_path, cfg_dict, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.dump(cfg_dict))
    return path


def test_cross_config_loads_without_ells_key(tmp_path):
    cfg = load_config(_write(tmp_path, _BASE_CROSS_CFG))
    assert cfg["data"] == "tau21"
    assert "ells" not in cfg


def test_cross_config_missing_required_key_raises(tmp_path):
    for key in ("telescope_21", "sensitivity_case", "delta_nu", "z21"):
        bad = copy.deepcopy(_BASE_CROSS_CFG)
        del bad[key]
        with pytest.raises(KeyError):
            load_config(_write(tmp_path, bad, f"missing_{key}.yaml"))


def test_cross_config_rejects_empty_z21(tmp_path):
    bad = copy.deepcopy(_BASE_CROSS_CFG)
    bad["z21"] = []
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_cross_config_rejects_bad_telescope_21(tmp_path):
    bad = copy.deepcopy(_BASE_CROSS_CFG)
    bad["telescope_21"] = "bogus"
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_cross_config_requires_ska_array_for_ska(tmp_path):
    bad = copy.deepcopy(_BASE_CROSS_CFG)
    bad["telescope_21"] = "ska"
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))
    bad["ska_array"] = "aast"
    load_config(_write(tmp_path, bad, "ok.yaml"))


def test_cross_config_warns_n_fields_ignored_for_non_ska(tmp_path):
    bad = copy.deepcopy(_BASE_CROSS_CFG)
    bad["n_fields"] = 3
    with pytest.warns(UserWarning, match="n_fields"):
        load_config(_write(tmp_path, bad))


def test_cross_config_rejects_null_telescopes(tmp_path):
    bad = copy.deepcopy(_BASE_CROSS_CFG)
    bad["telescopes"] = None
    # Supply 'ells' explicitly so this isolates the cross-specific
    # "telescopes must be non-null" check from the separate, generic
    # "ells is required when telescopes is null" one (both are real
    # rejections of this config; without 'ells' the KeyError from the
    # generic check would fire first).
    bad["ells"] = {"tau": {"start": 500, "stop": 5000, "step": 500}}
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_cross_config_rejects_multi_telescope_list(tmp_path):
    bad = copy.deepcopy(_BASE_CROSS_CFG)
    bad["telescopes"] = ["CMB-S4-LAT", "CMB-S4-LAT"]
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, bad))


def test_run_label_for_tau21():
    assert run_label(_BASE_CROSS_CFG, "tau21") == "unittest_cross_tau21"


def test_run_label_for_tau21_no_only_suffix():
    assert not run_label(_BASE_CROSS_CFG, "tau21").endswith("_only")


# --- shared build_ells/ells-vs-telescopes behavior (both auto and cross) ---

def test_build_ells_returns_none_when_telescopes_set_and_no_ells(tmp_path):
    cfg_dict = copy.deepcopy(_BASE_AUTO_CFG)
    cfg_dict["telescopes"] = ["CMB-S4-LAT", "CMB-S4-LAT", "CMB-S4-SAT"]
    del cfg_dict["ells"]
    cfg = load_config(_write(tmp_path, cfg_dict))
    assert build_ells(cfg) is None


def test_build_ells_warns_when_both_telescopes_and_ells_given(tmp_path):
    cfg_dict = copy.deepcopy(_BASE_AUTO_CFG)
    cfg_dict["telescopes"] = ["CMB-S4-LAT", "CMB-S4-LAT", "CMB-S4-SAT"]
    cfg = load_config(_write(tmp_path, cfg_dict))
    with pytest.warns(UserWarning, match="telescopes"):
        ells = build_ells(cfg)
    assert ells is not None


def test_build_ells_no_warning_when_telescopes_null(tmp_path, recwarn):
    cfg = load_config(_write(tmp_path, _BASE_AUTO_CFG))
    ells = build_ells(cfg)
    assert ells is not None
    assert len(recwarn) == 0


def test_validate_config_requires_ells_when_telescopes_null(tmp_path):
    bad = copy.deepcopy(_BASE_AUTO_CFG)
    del bad["ells"]
    with pytest.raises(KeyError):
        load_config(_write(tmp_path, bad))


def test_cross_ells_only_needs_tau_entry(tmp_path):
    # For a cross config, an explicit 'ells' override only needs a 'tau'
    # sub-entry -- ksz/bb are irrelevant and shouldn't be required.
    cfg_dict = copy.deepcopy(_BASE_CROSS_CFG)
    cfg_dict["ells"] = {"tau": {"start": 500, "stop": 5000, "step": 500}}
    cfg = load_config(_write(tmp_path, cfg_dict))
    with pytest.warns(UserWarning, match="telescopes"):
        ells = build_ells(cfg)
    assert len(ells) == 1
