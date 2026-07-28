# preion

A reionisation model based on the electron density power spectrum,
producing consistent kSZ, tau-tau, 21cm, and $B$-mode power spectra, plus a
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
conda create -n preion python=3.11 gcc gfortran -c conda-forge
conda activate preion
cd preion
pip install -e ".[test]"
```

### Installation notes

- `plancklens` is pulled straight from
[github.com/carronj/plancklens](https://github.com/carronj/plancklens) and
built during install — it has compiled extensions, so a C/Fortran toolchain
(gcc, gfortran) needs to be available. 

- `numpy` is pinned to `<2` because of `camb`'s matter-power interpolator

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
chain to `{output_dir}/backends/`. 

3. Read and analyse the chains with `arviz`

```bash
preion-read-mcmc configs/cv_limited_new.yaml --data tau --save-figures
```

`preion-read-mcmc` reads the backends, prints
convergence diagnostics (autocorrelation time, burn-in, Gelman-Rubin R-hat)
and a bias/error summary table, and (with `--save-figures`) writes corner,
trace, and triangle plots to `{output_dir}/figures/`.


Both the run and the read-back step are driven by the same YAML config, so
they always agree on parameters and output paths: see
`configs/cv_limited_new.yaml` for an example.

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
