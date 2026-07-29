import logging
import os

import numpy as np
import yaml

VALID_DATA = ("bb", "ksz", "tau", "all")

REQUIRED_KEYS = (
    "label", "data", "theta_true", "log_kappa", "niterations", "nwalkers",
    "fsky", "use_ksz_emulator", "overwrite", "plot", "progress",
    "telescopes", "output_dir", "ells",
)

_ELLS_NAMES = ("tau", "ksz", "bb")
_VALID_ELLS_KEYSETS = (
    frozenset({"start", "stop", "step"}),
    frozenset({"start", "stop", "num"}),
)


def _validate_config(cfg):
    """Check that `cfg` (as loaded from YAML) explicitly has every key in
    REQUIRED_KEYS (raising KeyError otherwise) and that each of
    ells.tau/ksz/bb has exactly the keys {start, stop, step} (for
    np.arange-style binning) or {start, stop, num} (for np.linspace-style
    binning), raising ValueError for any other combination (e.g. a typo'd or
    mixed key set)."""
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key!r}")
    for name in _ELLS_NAMES:
        if name not in cfg["ells"]:
            raise KeyError(f"Missing required config key: 'ells.{name}'")
        keys = frozenset(cfg["ells"][name].keys())
        if keys not in _VALID_ELLS_KEYSETS:
            raise ValueError(
                f"ells.{name} must have keys {{start, stop, step}} or "
                f"{{start, stop, num}}, got {sorted(keys)}."
            )


def load_config(path):
    """Load a run config. Every key in REQUIRED_KEYS must be explicitly
    present in the YAML file (raises KeyError otherwise, e.g. for a missing
    field) -- there is no defaulting/merging. See `_validate_config` for the
    additional check on the `ells` sub-dicts."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    _validate_config(cfg)
    if cfg["data"] not in VALID_DATA:
        raise ValueError(f"Invalid data option: {cfg['data']!r}, must be one of {VALID_DATA}.")
    return cfg


def build_ells(cfg):
    """Build the [ells_tau, ells_ksz, ells_bb] arrays from cfg['ells']. Each
    of tau/ksz/bb is built with np.arange if its spec has a 'step' key, or
    np.linspace if it has a 'num' key (as validated by `_validate_config`)."""
    ells = []
    for name in _ELLS_NAMES:
        spec = cfg["ells"][name]
        if "step" in spec:
            ells.append(np.arange(spec["start"], spec["stop"], step=spec["step"]))
        else:
            ells.append(np.linspace(spec["start"], spec["stop"], spec["num"]))
    return ells


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


def log_path(cfg):
    """Resolve the path to the logfile `setup_logging` writes to for `cfg`."""
    return os.path.abspath(os.path.join(cfg["output_dir"], f"{cfg['label']}_{cfg['data']}.log"))


def truncate_log_at_marker(cfg, marker):
    """If the logfile for `cfg` already exists and contains a line with
    `marker`, rewrite it to keep only the lines before the first such line
    (dropping that line and everything written after it in a previous run).
    No-op if the file doesn't exist or doesn't contain `marker`. Must be
    called before `setup_logging` opens its FileHandler on the same path,
    otherwise the handler's open file position won't reflect the rewrite.
    """
    path = log_path(cfg)
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if marker in line:
            with open(path, "w") as f:
                f.writelines(lines[:i])
            return


def setup_logging(cfg):
    """Configure the shared "preion.forecast" logger to write plain-text
    messages to {output_dir}/{label}_{data}.log instead of the console. Safe
    to call more than once for the same cfg (e.g. get_or_make_datapoints then
    run_mcmc in the same process) — only reconfigures if the resolved path
    actually changes, so a second call doesn't truncate what the first
    call already wrote.
    """
    os.makedirs(cfg["output_dir"], exist_ok=True)
    path = log_path(cfg)
    logger = logging.getLogger("preion.forecast")
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == path:
            return logger
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    mode = "w" if cfg["overwrite"] else "a"
    handler = logging.FileHandler(path, mode=mode)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
