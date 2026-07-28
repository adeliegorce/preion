# preion

A reionisation model based on the electron density power spectrum (`Pee_model`),
producing consistent kSZ, tau-tau, 21cm, and B-mode power spectra, plus a
secondary `preion.forecast` subpackage for running kSZ/tau/BB MCMC forecasts
with `emcee` and reading back the resulting chains.

## Installation

### With `uv` (recommended)

Requires a C/Fortran toolchain (gcc, gfortran) on `PATH` for `plancklens`'
compiled extensions.

```bash
cd preion
uv venv --python 3.11   # any Python >=3.9
source .venv/bin/activate
uv pip install -e ".[test]"
```

### With conda

```bash
conda activate preion2   # or any Python >=3.9 env with a working C/Fortran toolchain
cd preion
pip install -e ".[test]"
```

`plancklens` is pulled straight from
[github.com/carronj/plancklens](https://github.com/carronj/plancklens) and
built during install — it has compiled extensions, so a C/Fortran toolchain
(gcc, gfortran) needs to be available. `preion2` already has one.

**`numpy` is pinned to `<2`.** `camb`'s matter-power interpolator
(`results.get_matter_power_interpolator(...).P`), used inside
`Pee_model.run_camb`, raises `ValueError: setting an array element with a
sequence.` under numpy>=2's stricter `np.vectorize` dtype inference —
confirmed independent of the installed `camb` version. Don't relax this pin
without re-testing `Pee_model(run_camb=True)`.

### The `emul_sz` situation

`Pee_model(..., use_ksz_emulator="RF")` (or `"NN"`) can use the private
`emul_sz` package to speed up the kSZ power spectrum calculation. `emul_sz` is
**not** a declared dependency here — it lives in a private repository, not on
PyPI, so it can't be installed generically. If it isn't importable,
`Pee_model.__init__` catches the `ModuleNotFoundError`, prints a warning, and
falls back to the full physical kSZ computation instead — so the package works
correctly either way; you only need `emul_sz` if you want the emulator
speed-up. Install it separately, into the same environment, from
[git.ias.u-psud.fr/batman/emul_sz](https://git.ias.u-psud.fr/batman/emul_sz):

```bash
uv pip install git+https://git.ias.u-psud.fr/batman/emul_sz
# or, with conda/pip:
pip install git+https://git.ias.u-psud.fr/batman/emul_sz
```

## Basic usage

```python
from preion.theory import Pee_model

model = Pee_model(zre_h=7.0, dz_h=1.5, alpha0=3.7, kappa=0.10, run_camb=True)
tau_ps = model.get_tau(ells=[100, 500, 1000], signal='both', Dells=True)
```

```python
from preion.forecast.datapoints import make_datapoints

tau_ps, ksz_ps, bb_ps, cov_tau, cov_ksz, cov_bb = make_datapoints(
    theta=[7.0, 1.5, 3.7, 0.10],
    telescopes=None,  # cosmic-variance-limited
    ells=[tau_ells, ksz_ells, bb_ells],
)
```

### Running and reading an MCMC forecast

Both the run and the read-back step are driven by the same YAML config, so
they always agree on parameters and output paths — see
`configs/cv_limited_new.yaml` for the schema and defaults (label, `theta_true`,
`niterations`/`nwalkers`, telescopes, ell grids, `output_dir`, ...).

```bash
preion-run-mcmc configs/cv_limited_new.yaml --data tau
preion-read-mcmc configs/cv_limited_new.yaml --data tau --save-figures
```

`preion-run-mcmc` writes mock data to `{output_dir}/data/` and the emcee
chain to `{output_dir}/backends/`. `preion-read-mcmc` reads both back, prints
convergence diagnostics (autocorrelation time, burn-in, Gelman-Rubin R-hat)
and a bias/error summary table, and (with `--save-figures`) writes corner,
trace, and posterior-predictive plots to `{output_dir}/figures/`.

See `notebooks/theory_tutorial.ipynb` and `notebooks/forecast_tutorial.ipynb`
for worked, end-to-end examples of both halves of the package.

## Tests

```bash
pytest
```

Tests marked `slow` exercise `Pee_model(run_camb=True)` (a few seconds each
via CAMB); skip them with `pytest -m "not slow"` for a quick check.

## Package layout

```
src/preion/
├── theory.py             # Pee_model: kSZ, tau-tau, 21cm, B-mode power spectra
├── utils.py               # noise/statistics helpers + tau quadratic-estimator noise (plancklens-based)
├── parameters.py         # constants, telescope_specs
├── plotting.py            # vendored corner-plot fork (originally triangleme2.py / Foreman-Mackey's corner)
└── forecast/
    ├── config.py          # shared YAML config loading for mcmc.py / read_mcmc.py
    ├── datapoints.py      # make_datapoints: mock data + covariance generation
    ├── mcmc.py             # run_mcmc_cv_limited_new + `preion-run-mcmc` CLI
    └── read_mcmc.py        # chain diagnostics/plots + `preion-read-mcmc` CLI
```
