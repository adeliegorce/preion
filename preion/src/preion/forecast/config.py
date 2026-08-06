import logging
import os
import warnings

import numpy as np
import yaml

VALID_DATA = ("bb", "ksz", "tau", "all", "tau21")
CROSS_DATA = ("tau21",)  # room for a future "bb21" once a noise model exists to port

REQUIRED_KEYS = (
    "label", "data", "theta_true", "log_kappa", "niterations", "nwalkers",
    "fsky", "use_ksz_emulator", "overwrite", "plot", "progress",
    "telescopes", "output_dir",
)

_CROSS_REQUIRED_KEYS = ("telescope_21", "sensitivity_case", "delta_nu", "z21")

_ELLS_NAMES = ("tau", "ksz", "bb")
_VALID_ELLS_KEYSETS = (
    frozenset({"start", "stop", "step"}),
    frozenset({"start", "stop", "num"}),
)


def _relevant_ells_names(cfg):
    return ("tau",) if cfg.get("data") in CROSS_DATA else _ELLS_NAMES


def _validate_config(cfg):
    """Check that `cfg` (as loaded from YAML) explicitly has every key in
    REQUIRED_KEYS (raising KeyError otherwise). `ells` is required only when
    `telescopes` is null (no telescope to derive a default multipole grid
    from); when present, each relevant name (tau/ksz/bb for an auto config,
    just tau for a cross one) must have exactly the keys {start, stop, step}
    (for np.arange-style binning) or {start, stop, num} (for np.linspace-style
    binning), raising ValueError for any other combination (e.g. a typo'd or
    mixed key set). Cross (`data in CROSS_DATA`) configs additionally require
    `_CROSS_REQUIRED_KEYS`, a valid `telescope_21`, `ska_array` when
    `telescope_21 == "ska"`, and a 1-element `telescopes` list."""
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key!r}")

    if "ells" in cfg:
        for name in _relevant_ells_names(cfg):
            if name not in cfg["ells"]:
                raise KeyError(f"Missing required config key: 'ells.{name}'")
            keys = frozenset(cfg["ells"][name].keys())
            if keys not in _VALID_ELLS_KEYSETS:
                raise ValueError(
                    f"ells.{name} must have keys {{start, stop, step}} or "
                    f"{{start, stop, num}}, got {sorted(keys)}."
                )
    elif cfg.get("telescopes") is None:
        raise KeyError(
            "Missing required config key: 'ells' (required when 'telescopes' is null)."
        )

    if cfg.get("data") in CROSS_DATA:
        for key in _CROSS_REQUIRED_KEYS:
            if key not in cfg:
                raise KeyError(f"Missing required config key for data={cfg['data']!r}: {key!r}")
        if not isinstance(cfg["z21"], list) or len(cfg["z21"]) == 0:
            raise ValueError("'z21' must be a non-empty list of redshifts.")
        if cfg["telescope_21"] not in ("hera", "ska", "mwa"):
            raise ValueError(
                f"telescope_21 must be one of 'hera'/'ska'/'mwa', got {cfg['telescope_21']!r}."
            )
        if cfg["telescope_21"] == "ska" and cfg.get("ska_array") not in ("aast", "aa4"):
            raise ValueError(
                "'ska_array' must be 'aast' or 'aa4' when telescope_21='ska', "
                f"got {cfg.get('ska_array')!r}."
            )
        if cfg.get("n_fields") is not None and cfg["telescope_21"] != "ska":
            warnings.warn(
                f"n_fields={cfg['n_fields']} was specified but telescope_21="
                f"{cfg['telescope_21']!r} is not 'ska'; it will be ignored."
            )
        telescopes = cfg.get("telescopes")
        if telescopes is None or len(telescopes) != 1:
            raise ValueError(
                "'telescopes' must be a 1-element list [tel_tau] when data='tau21' "
                "(there is no CV-limited/telescope-free mode for the cross covariance)."
            )


def load_config(path):
    """Load a run config. Every key in REQUIRED_KEYS must be explicitly
    present in the YAML file (raises KeyError otherwise, e.g. for a missing
    field) -- there is no defaulting/merging. See `_validate_config` for the
    additional checks on `ells` and (for `data in CROSS_DATA`) the
    cross-specific keys."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    _validate_config(cfg)
    if cfg["data"] not in VALID_DATA:
        raise ValueError(f"Invalid data option: {cfg['data']!r}, must be one of {VALID_DATA}.")
    return cfg


def build_ells(cfg):
    """Build the multipole-grid arrays from cfg['ells'] (one per name in
    `_relevant_ells_names(cfg)`: tau/ksz/bb for an auto config, just tau for
    a cross one). Each is built with np.arange if its spec has a 'step' key,
    or np.linspace if it has a 'num' key (as validated by `_validate_config`).

    Returns `None` if `cfg` has no 'ells' key at all -- telling the caller
    (`mcmc.py::get_or_make_datapoints`) to derive the grid from
    `cfg['telescopes']` via `get_lbins` instead. If 'ells' *is* present and
    `telescopes` is also not null, warns that the explicit 'ells' overrides
    the telescope-derived binning (easy to miss otherwise); no warning if
    `telescopes` is null, since 'ells' is required and expected there.
    """
    if "ells" not in cfg:
        return None
    if cfg.get("telescopes") is not None:
        warnings.warn(
            f"Config {cfg.get('label')!r} specifies both 'telescopes' and 'ells'; "
            "using the explicit 'ells' grid from the config file, not the "
            "telescope-derived binning from get_lbins()."
        )
    ells = []
    for name in _relevant_ells_names(cfg):
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


def cfg_title(cfg):
    """Return cfg['title'] if set, else fall back to run_label(cfg). 'title'
    is an optional config key (not in REQUIRED_KEYS) used to label figures/
    legend entries in preion-read-mcmc and preion-compare-mcmc."""
    return cfg.get("title", run_label(cfg))


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
