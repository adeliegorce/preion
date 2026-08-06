# Changelog

## 0.2.0

Added tau x 21cm cross-correlation forecasting alongside the existing
kSZ/tau/BB (auto) pipeline:

- New `data: tau21` config option, with `telescope_21`/`sensitivity_case`/
  `ska_array`/`n_fields` keys selecting the 21cm survey and
  `delta_nu`/`z21` selecting the cross-spectrum's frequency resolution and
  redshift(s).
- `preion.forecast.datapoints.make_cross_datapoints`/`load_cross_datapoints`.
- `preion.forecast.mcmc`'s `get_model`/`lnlike`/`lnprob`/`get_or_make_datapoints`
  are unified across both auto and cross configs (branching on `cfg['data']`)
  rather than duplicated; `plot_cross_obs` for mock-data plots. `run_mcmc`
  backfills `inv_cov_*` (and, for cross, `z21`/`delta_nu`) into `datapoints`
  if not already present, so datapoints obtained any way (not just via
  `get_or_make_datapoints`) work with it directly.
- `preion.forecast.read_mcmc.plot_cross_models`, `generate_prior_cache`
  (ported from the old `prior_distributions.py` script), and a
  `--with-prior-background` flag on `preion-compare-mcmc`.
- New optional `zend_prior`/`tau_prior` config keys, usable with **any**
  `data` value (not cross-only): a lower-limit prior on `zre-dz` and a
  Gaussian prior on the model's integrated optical depth.
- New optional `sensitivity_dir`/`lbin_edges` config keys; `ells` is now
  optional whenever `telescopes` is set (the grid then defaults to
  `get_lbins(telescope)`), for both auto and cross configs.
- `preion.forecast.utils.invert_covariance`: covariance inversion shared by
  auto and cross likelihoods, computed once per run instead of every MCMC
  step, and exact (not approximate) about excluding out-of-coverage,
  infinite-variance datapoints from the likelihood.
- Renamed the previously-auto-only, unmarked functions/notebook/config for
  symmetry with the new cross ones: `make_datapoints` -> `make_autos_datapoints`,
  `load_datapoints` -> `load_autos_datapoints`, `plot_obs` -> `plot_autos_obs`,
  `plot_models` -> `plot_autos_models`, `configs/config_tutorial_mcmc.yaml` ->
  `configs/config_tutorial_mcmc_autos.yaml`, `notebooks/forecast_tutorial.ipynb`
  -> `notebooks/forecast_tutorial_autos.ipynb`. The former `load_packaged_datapoints`
  convenience wrapper is gone entirely -- call `load_autos_datapoints`/
  `load_cross_datapoints` directly with the packaged `_PACKAGED_DATA_DIR`/
  `_PACKAGED_LABEL`/`_PACKAGED_CROSS_LABEL`/`_PACKAGED_CROSS_Z21` constants
  (see the tutorial notebooks for the exact call).
- Fixed two bugs in `theory.Pee_model.get_BB_21_cross` (unrelated to the
  cross-forecast port, found while adding test coverage for it): it used
  `self.tau_z_integ` (evaluated on an unrelated fixed z grid) instead of
  `self.xe2tau(zlin)` (its own local z grid), and returned `uK` instead of
  `uK**2` (contradicting its own docstring).
- Fixed a dormant bug in `make_autos_datapoints`/`make_cross_datapoints`
  where passing both `telescopes` and an explicit `ells` override together
  raised `IndexError` (a pre-existing variable-shadowing bug, exercised for
  the first time by the new "ells overrides the telescope-derived grid"
  feature).

## 0.1.0

Initial release: `preion.theory.Pee_model` (kSZ, tau-tau, 21cm, B-mode
power spectra) and the auto (kSZ/tau/BB) MCMC forecast pipeline
(`preion.forecast`).
