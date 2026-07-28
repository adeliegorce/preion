import numpy as np
import yaml

VALID_DATA = ("bb", "ksz", "tau", "all")

_DEFAULTS = {
    "label": "cv_limited_new",
    "data": "all",
    "theta_true": [7.0, 1.5, 3.7, 0.10],
    "log_kappa": True,
    "niterations": 100000,
    "nwalkers": 8,
    "fsky": 1.0,
    "use_ksz_emulator": "RF",
    "overwrite": True,
    "plot": False,
    "telescopes": None,
    "output_dir": ".",
    "ells": {
        "tau": {"start": 100, "stop": 5000, "step": 100},
        "ksz": {"start": 1000, "stop": 8000, "step": 500},
        "bb": {"start": 10, "stop": 1000, "num": 50},
    },
}


def _deep_merge(defaults, overrides):
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    """Load a run config, filling in defaults that mirror the values that used
    to be hardcoded in run_mcmc_cv_limited_new.py."""
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(_DEFAULTS, user_cfg)
    if cfg["data"] not in VALID_DATA:
        raise ValueError(f"Invalid data option: {cfg['data']!r}, must be one of {VALID_DATA}.")
    return cfg


def build_ells(cfg):
    """Build the [ells_tau, ells_ksz, ells_bb] arrays from cfg['ells']."""
    tau_spec = cfg["ells"]["tau"]
    ksz_spec = cfg["ells"]["ksz"]
    bb_spec = cfg["ells"]["bb"]
    ells_tau = np.arange(tau_spec["start"], tau_spec["stop"], step=tau_spec["step"])
    ells_ksz = np.arange(ksz_spec["start"], ksz_spec["stop"], step=ksz_spec["step"])
    ells_bb = np.linspace(bb_spec["start"], bb_spec["stop"], bb_spec["num"])
    return [ells_tau, ells_ksz, ells_bb]


def run_label(cfg, data=None):
    """Reproduce the f'{label}_{data}_only' / f'{label}_{data}' naming used for
    backend/data filenames by run_mcmc_cv_limited_new.py."""
    data = data if data is not None else cfg["data"]
    if data not in VALID_DATA:
        raise ValueError(f"Invalid data option: {data!r}, must be one of {VALID_DATA}.")
    label = cfg["label"]
    if data in ("bb", "ksz", "tau"):
        return f"{label}_{data}_only"
    return f"{label}_{data}"
