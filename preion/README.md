# preion

A reionisation model based on the electron density power spectrum,
producing consistent kSZ, tau-tau, 21cm, and $B$-mode power spectra, plus a
secondary `preion.forecast` subpackage for running kSZ/tau/BB MCMC forecasts
with `emcee` and reading back the resulting chains.

## Installation

`preion` has two install flavours:

- **Theory only** — just `preion.theory.Pee_model` and its
  `numpy`/`scipy`/`astropy`/`camb` dependencies. No compiled extensions, no
  toolchain required.
- **Theory + forecast** — adds the `forecast` extra (`emcee`, `arviz`,
  `healpy`, `plancklens`, ...) needed for the kSZ/tau/BB MCMC pipeline.
  `plancklens` has compiled extensions, so this flavour needs a C/Fortran
  toolchain (gcc, gfortran) on `PATH`.

### With `uv` (recommended)

Theory only:

```bash
cd preion
uv venv --python 3.11   # any Python >=3.9
source .venv/bin/activate
uv pip install -e ".[test]"
```

Theory + forecast:

```bash
cd preion
uv venv --python 3.11   # any Python >=3.9
source .venv/bin/activate
uv pip install "numpy<2"  # for plancklens
uv pip install --no-build-isolation-package plancklens -e ".[forecast,test]"
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

### Running and reading an MCMC forecast

The package can also be used to produce mock data points and fit them with the model.

1. Generate mock data points for $C_\ell^{\tau\tau}$, $C_\ell^\text{kSZ}$ and $C_\ell^{BB}$ given a reionisation model:

```python
from preion.forecast.datapoints import make_datapoints

tau_ps, ksz_ps, bb_ps, cov_tau, cov_ksz, cov_bb = make_datapoints(
    theta=[7.0, 1.5, 3.7, 0.10],
    telescopes=['CMB-S4-LAT', 'CMB-S4-LAT', 'CMB-S4-SAT'],
    ells=[tau_ells, ksz_ells, bb_ells],
)
```

The specs corresponding, e.g., to `CMB-S4-LAT` are defined in `preion.parameters`.

2. Run the forecast with `emcee`

```bash
preion-run-mcmc configs/cv_limited_new.yaml --data tau
```

`preion-run-mcmc` writes mock data to `{output_dir}/data/` (if it is not pre-existing) and the emcee
chain to `{output_dir}/backends/`. The `--data` flag only overrides the config in memory for that run;
it is not written back to the YAML file.

3. Read and analyse the chains with `arviz`

```bash
preion-read-mcmc configs/cv_limited_new.yaml --save-figures
```

Unlike `preion-run-mcmc`, `preion-read-mcmc` has no `--data` override: it reads the `data` field
directly from the YAML config to know which chain variant to load, so that field must match whatever
`data` (config field or `--data` override) was used to produce the chain — e.g. if you ran with
`--data tau` above, set `data: tau` in the config (or use a separate config per variant) before
reading it back. `preion-read-mcmc` reads the backends, prints
convergence diagnostics (autocorrelation time, burn-in, Gelman-Rubin R-hat)
and a bias/error summary table, and (with `--save-figures`) writes corner,
trace, and triangle plots to `{output_dir}/figures/`.

Both `preion-run-mcmc` and `preion-read-mcmc` write their status/timing/diagnostic
messages to a logfile at `{output_dir}/{label}_{data}.log` instead of the console
(`label` and `data` are the YAML config's `label` and `data` fields), so the
terminal only shows the `emcee`/`tqdm` progress bar (if `progress: true` in the
config). The logfile is truncated on each run if `overwrite: true`, appended to
otherwise.


Both the run and the read-back step are driven by the same YAML config, so
they always agree on parameters and output paths: see
`configs/cv_limited_new.yaml` for an example.

See `notebooks/theory_tutorial.ipynb` and `notebooks/forecast_tutorial.ipynb`
for worked, end-to-end examples of both halves of the package.

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
├── plotting.py            # vendored corner-plot fork (originally triangleme2.py / Foreman-Mackey's corner)
└── forecast/
    ├── utils.py            # noise/statistics helpers + tau quadratic-estimator noise (plancklens-based)
    ├── config.py          # shared YAML config loading + setup_logging (per-label logfile) for mcmc.py / read_mcmc.py
    ├── datapoints.py      # make_datapoints: mock data + covariance generation
    ├── mcmc.py             # get_or_make_datapoints, run_mcmc + `preion-run-mcmc` CLI
    └── read_mcmc.py        # chain diagnostics/plots + `preion-read-mcmc` CLI
```
