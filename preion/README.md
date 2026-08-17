# preion

A reionisation model based on the electron density power spectrum,
producing consistent kSZ, tau-tau, 21cm, and $B$-mode power spectra, plus a
secondary `preion.forecast` subpackage for running MCMC forecasts (both the
kSZ/tau/BB "auto" case and the tau x 21cm cross-correlation case) with
`emcee` and reading back the resulting chains. See `CHANGELOG.md` for
release notes.

## Installation

`preion` has two install flavours:

- **Theory only** — just `preion.theory.Pee_model` and its
  `numpy`/`scipy`/`astropy`/`camb` dependencies. 
- **Theory + forecast** — adds the `forecast` module (including `emcee`, `arviz`, `healpy`, `plancklens`, ...) needed for the MCMC pipeline.
  `plancklens` requires a C/Fortran toolchain (gcc, gfortran) on `PATH`.

### With `uv` (recommended)

Theory only:

```bash
cd preion
uv venv --python 3.11   # any Python >=3.9
source .venv/bin/activate
uv pip install -e .
```

Theory + forecast:

```bash
cd preion
uv venv --python 3.11   # any Python >=3.9
source .venv/bin/activate
uv pip install "numpy<2"  # for plancklens
uv pip install --no-build-isolation plancklens -e ".[forecast]"
```

Developper
```bash
cd preion
uv venv --python 3.11   # any Python >=3.9
source .venv/bin/activate
uv pip install -e ".[test]"
```

### With conda

Theory only:

```bash
conda create -n preion python=3.11 -c conda-forge
conda activate preion
cd preion
pip install -e ".[test]"
```

Theory + forecast:

```bash
conda create -n preion python=3.11 gcc gfortran -c conda-forge
conda activate preion
cd preion
pip install -e ".[forecast,test]"
```

### Installation notes

- `plancklens` (only pulled in by the `forecast` extra) comes straight from
[github.com/carronj/plancklens](https://github.com/carronj/plancklens) and
built during install. It has compiled extensions, so a C/Fortran toolchain
(gcc, gfortran) needs to be available. 

- `numpy` is pinned to `<2` because of `camb`'s matter-power interpolator and of `plancklens`

- If `plancklens` prints `could not load wigners.so fortran shared object` at
  import time, two things need fixing, both inside the active environment:

  1. `plancklens`'s `wigners` dependency was installed without its compiled
     extension (this can happen depending on the `numpy`/Python combination
     used to build it). Build it in place:
     ```bash
     uv pip install meson ninja   # build-time only, needed by f2py's meson backend on Python >=3.12
     cd .venv/lib/python*/site-packages/wigners
     f2py -c -m wigners wigners.f90
     cd -
     uv pip uninstall meson ninja  # optional cleanup, not needed at runtime
     ```
     This regenerates `wigners.cpython-*.so` next to `wigners.f90`.

  2. Even with that extension built, `plancklens/utils_spin.py` (as of commit
     `2ff8f1b`) imports it as `from plancklens.wigners import wigners` — a
     *nested* `plancklens.wigners` submodule — but `plancklens` actually
     installs `wigners` as a top-level package, so that nested path never
     exists. Add a shim module so the import resolves:
     ```bash
     cat > .venv/lib/python*/site-packages/plancklens/wigners.py << 'EOF'
     from wigners import wigners
     EOF
     ```

  With both in place, `import plancklens` should raise no warning
  (`plancklens.utils_spin.HASWIGNER` will be `True`).

- The `theory` module can use the `emul_sz` package to speed up the kSZ power spectrum calculation with an
emulator. `emul_sz` is not a declared dependency as it is not available on
PyPI and can't be installed directly. If the package is not installed, `preion`
falls back to the full physical kSZ computation. 
Install it separately, into the same environment, from
[git.ias.u-psud.fr/batman/emul_sz](https://git.ias.u-psud.fr/batman/emul_sz).

## Basic usage

### Theory module

Define a reionisation model and compute the corresponding observables

```python
from preion.theory import Pee_model
# define the reionisation model
model = Pee_model(zre_h=7.0, dz_h=1.5, alpha0=3.7, kappa=0.10)
# derive observables
tau_ps = model.get_tau(ells=[100, 500, 1000], signal='both', Dells=True)
ksz_ps = model.get_ksz(ells=[3000], signal='both', Dells=True)
bb_ps = model.get_B_modes(ells=[100, 500, 1000], Dells=True)
p21 = model.get_p21(np.logspace(-2, 0, 100), z=9.)
```

You can find more examples at `notebooks/theory_tutorial.ipynb`.

### Running and reading an MCMC forecast

The package can also be used to produce mock data points and fit them with the model.

1. Generate mock data points for $C_\ell^{\tau\tau}$, $C_\ell^\text{kSZ}$ and $C_\ell^{BB}$ given a reionisation model:

```python
from preion.forecast.datapoints import make_autos_datapoints

tau_ps, ksz_ps, bb_ps, cov_tau, cov_ksz, cov_bb = make_autos_datapoints(
    theta=[7.0, 1.5, 3.7, 0.10],
    telescopes=['CMB-S4-LAT', 'CMB-S4-LAT', 'CMB-S4-SAT'],
    ells=[tau_ells, ksz_ells, bb_ells],
)
```

The specs corresponding, e.g., to `CMB-S4-LAT` are defined in `preion.parameters`.

2. Run the forecast with `emcee`

```bash
preion-run-mcmc configs/config_tutorial_mcmc_autos.yaml
```

`preion-run-mcmc` writes mock data to `{output_dir}/data/` (if it is not pre-existing) and the emcee
chain to `{output_dir}/backends/`. 

3. Read and analyse the chains with `arviz`

```bash
preion-read-mcmc configs/config_tutorial_mcmc_autos.yaml
```

`preion-read-mcmc` reads the backends, prints
convergence diagnostics (autocorrelation time, burn-in, Gelman-Rubin R-hat)
and a bias/error summary table, and writes corner,
trace, and triangle plots to `{output_dir}/figures/`. Each figure is titled
with the config's `title` field, if set (falling back to its `run_label`,
e.g. `cv_limited_tau_only`, otherwise).

4. Compare several chains on one overlaid corner plot

```bash
preion-compare-mcmc config1.yaml config2.yaml config3.yaml -o comparison_corner.png
```

`preion-compare-mcmc` takes two or more already-run configs, loads each
chain, and overlays their corner plots on a single figure, saved to `-o/--output` (default `mcmc_comparison_corner.png`). The legend label for each chain is that
config's `title` field (or its `run_label` if `title` isn't set). All
configs must agree on `log_kappa` (kappa vs log-kappa axes aren't
comparable); a mismatched `theta_true` between configs is allowed but logs a
warning, since the truth markers are drawn from the first config only.

Both `preion-run-mcmc` and `preion-read-mcmc` write their status/timing/diagnostic
messages to a logfile at `{output_dir}/{label}_{data}.log` instead of the console
(`label` and `data` are defined in the YAML config). `preion-compare-mcmc` logs
to the console instead, since it spans multiple configs/output directories.


Both the run and the read-back step are driven by a YAML config to define the model parameters, forecast/MCMC options and output paths: see
`configs/config_tutorial_mcmc_autos.yaml` for an example.

See `notebooks/forecast_tutorial_autos.ipynb` for an example forecast.

#### Config options that apply to any forecast (auto or cross)

- `ells`: optional whenever `telescopes` is set (non-null) -- the multipole
  grid then defaults to `get_lbins(telescope, lbin_edges=cfg.get('lbin_edges'))`.
  An explicit `ells` in the same config overrides that default (and logs a
  warning, since the override is easy to miss). `ells` is still required
  when `telescopes` is null (cosmic-variance-limited, auto only).
- `lbin_edges`: optional override of the telescope-derived bin edges (a
  flat list for a cross config's single telescope, or a per-telescope list
  of lists for an auto config's three).
- `zend_prior`: optional float, a lower-limit prior on `zre - dz`.
- `tau_prior`: optional `{sigma: ...}` dict, a Gaussian prior on the
  model's integrated optical depth (`mu` is filled in automatically from
  the fiducial model, never stored in the YAML).

### Running and reading a cross-correlation (tau x 21cm) forecast

Set `data: tau21` in the config to fit a tau x 21cm cross-spectrum instead
(see `configs/config_tutorial_mcmc_cross.yaml` for a full example). This
adds a few cross-specific keys: `telescope_21` (`hera`/`ska`/`mwa`),
`sensitivity_case` (e.g. `moderate`/`optimistic`), `ska_array` (required
only for `telescope_21: ska`), `n_fields` (optional, `ska` only),
`delta_nu` (MHz, top-hat frequency-resolution window), and `z21` (one or
more 21cm-signal redshifts, fit simultaneously). Unlike the auto case,
`telescopes` must always name a single real CMB telescope (`[tel_tau]`,
not null) -- the cross covariance needs a concrete telescope for its
tau-tau reconstruction-noise curve, so there is no cosmic-variance-limited
mode here.

```python
from preion.forecast.datapoints import make_cross_datapoints

tau21, cl21, cov_tau21, tautau, tau = (
    make_cross_datapoints(
        theta=[7.0, 1.5, 3.7, 0.10],
        tel_tau='CMB-S4-LAT', telescope_21='hera', sensitivity_case='moderate',
        z21=[7.0], delta_nu=100.,
    )[key] for key in ('tau21', 'cl21', 'cov_tau21', 'tautau', 'tau')
)
```

```bash
preion-run-mcmc configs/config_tutorial_mcmc_cross.yaml
preion-read-mcmc configs/config_tutorial_mcmc_cross.yaml
preion-compare-mcmc config1.yaml config2.yaml --with-prior-background
```

`--with-prior-background` (cross configs only) overlays a grey
Latin-Hypercube-sampled prior background behind the chain contours,
generated via `preion.forecast.read_mcmc.generate_prior_cache` if not
already cached at `{output_dir}/prior_distributions_{label}.hdf5`.

See `notebooks/forecast_tutorial_cross.ipynb` for an example cross forecast.

## Tests

```bash
pytest
```

With a theory-only install, tests that need the `forecast` extra are
reported as **skipped** rather than failing; install `.[forecast,test]` to
run the full suite.

Tests marked `slow` exercise `Pee_model(run_camb=True)` (a few seconds each
via CAMB); skip them with `pytest -m "not slow"` for a quick check.

## Package layout

```
src/preion/
├── theory.py             # Pee_model: kSZ, tau-tau, 21cm, B-mode power spectra
├── parameters.py         # constants, telescope_specs
└── forecast/
    ├── utils.py            # noise/statistics helpers, sensitivity lookups
    ├── config.py           # shared YAML config loading (auto + cross)
    ├── datapoints.py       # mock data + covariance generation (auto + cross)
    ├── mcmc.py             # run_mcmc + CLI (auto + cross)
    ├── read_mcmc.py        # chain diagnostics/plots + CLI (auto + cross)
    └── sensitivity_21cm/   # packaged 21cm survey noise-curve lookup tables

configs/
├── config_tutorial_mcmc_autos.yaml    # example auto (kSZ/tau/BB) config
└── config_tutorial_mcmc_cross.yaml    # example cross (tau x 21cm) config

notebooks/
├── theory_tutorial.ipynb          # Pee_model observables, incl. cross-spectra
├── forecast_tutorial_autos.ipynb  # auto forecast walk-through
└── forecast_tutorial_cross.ipynb  # cross forecast walk-through
```
