import matplotlib
matplotlib.use("Agg")

import os

import numpy as np
import pytest

pytest.importorskip("preion.forecast.read_mcmc")

from preion.forecast.read_mcmc import plot_corner_comparison, plot_cross_models, load_mock_data


def test_plot_corner_comparison_returns_figure_with_legend():
    rng = np.random.default_rng(0)
    chains = [rng.normal(size=(200, 3)) for _ in range(3)]
    labels = ["a", "b", "c"]
    chain_labels = ["chain 1", "chain 2", "chain 3"]

    fig = plot_corner_comparison(chains, labels, chain_labels, truths=[0., 0., 0.])

    assert fig.legends, "expected a legend to be added to the figure"
    legend_texts = {t.get_text() for t in fig.legends[0].get_texts()}
    assert legend_texts == set(chain_labels)


def test_plot_corner_comparison_default_colors_match_chain_count():
    rng = np.random.default_rng(1)
    chains = [rng.normal(size=(200, 2)) for _ in range(2)]
    labels = ["a", "b"]
    chain_labels = ["x", "y"]

    fig = plot_corner_comparison(chains, labels, chain_labels)

    legend_lines = fig.legends[0].get_lines()
    colors = [line.get_color() for line in legend_lines]
    assert colors == ["C0", "C1"]


class _FakeSampler:
    """Minimal stand-in for an emcee sampler, exposing only what
    plot_cross_models needs (get_blobs), so this test doesn't require a
    real chain file."""

    def __init__(self, tau21_models):
        self._tau21_models = tau21_models

    def get_blobs(self, flat=True, discard=0):
        dtype = [("tau21_models", float, self._tau21_models.shape[1:])]
        out = np.zeros(self._tau21_models.shape[0], dtype=dtype)
        out["tau21_models"] = self._tau21_models
        return out


def test_plot_cross_models_returns_figure():
    rng = np.random.default_rng(2)
    nz, nell, ndraws = 2, 5, 50
    ells = np.arange(500, 500 + nell * 500, 500)
    z21 = [7.0, 9.0]
    tau21_models = rng.normal(size=(ndraws, nz, nell))
    datapoints = {
        "ells": ells, "z21": z21,
        "tau21": rng.normal(size=(nz, nell)),
        "cov_tau21": np.array([np.diag(np.ones(nell)) for _ in range(nz)]),
    }
    sampler = _FakeSampler(tau21_models)

    fig = plot_cross_models(sampler, datapoints, burnin=0, n_draws=10)

    assert fig is not None
    assert len(fig.axes) == nz


def test_load_mock_data_dispatches_on_data_type(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "preion.forecast.read_mcmc.datapoints.load_autos_datapoints",
        lambda data_dir, label: calls.append(("autos", data_dir, label)) or {"tau": 1},
    )
    monkeypatch.setattr(
        "preion.forecast.read_mcmc.datapoints.load_cross_datapoints",
        lambda data_dir, label, z21: calls.append(("cross", data_dir, label, z21)) or {"tau21": 1},
    )

    cfg_auto = {"data": "all", "label": "run1", "output_dir": str(tmp_path)}
    out_auto = load_mock_data(cfg_auto)
    assert out_auto == {"tau": 1}
    assert calls[-1][0] == "autos"

    cfg_cross = {
        "data": "tau21", "label": "run2", "output_dir": str(tmp_path),
        "z21": [7.0], "delta_nu": 100.0,
    }
    out_cross = load_mock_data(cfg_cross)
    assert out_cross["tau21"] == 1
    assert out_cross["z21"] == [7.0]
    assert out_cross["delta_nu"] == 100.0
    assert calls[-1][0] == "cross"


@pytest.mark.slow
def test_generate_prior_cache_writes_expected_datasets(tmp_path):
    import h5py
    from preion.forecast.read_mcmc import generate_prior_cache

    cfg = {
        "label": "unittest_prior", "data": "tau21", "theta_true": [7.0, 1.5, 3.7, 0.10],
        "log_kappa": True, "niterations": 2, "nwalkers": 4, "fsky": 1.0,
        "use_ksz_emulator": False, "overwrite": True, "plot": False, "progress": False,
        "telescopes": ["CMB-S4-SAT"], "output_dir": str(tmp_path),
        "telescope_21": "hera", "sensitivity_case": "moderate",
        "delta_nu": 100.0, "z21": [7.0], "lbin_edges": [20, 100, 200, 330],
        "zend_prior": 4.5, "tau_prior": {"sigma": 0.007},
    }
    out_path = generate_prior_cache(cfg, nrand=20)
    assert os.path.exists(out_path)
    with h5py.File(out_path, "r") as f:
        n = f["params"].shape[0]
        assert n > 0
        assert f["tau21"].shape[0] == n
        assert f["cl21"].shape[0] == n
        assert f["tautau"].shape[0] == n
        assert f["tau"].shape[0] == n
        assert f["dksz"].shape[0] == n
        assert f["weight"].shape[0] == n
        # zend_prior cutoff: every surviving draw satisfies zre - dz > 4.5
        params = f["params"][:]
        assert np.all(params[:, 0] - params[:, 1] > cfg["zend_prior"])
        # tau_prior is set -- weight should vary (not all 1.0)
        weight = f["weight"][:]
        assert np.all(weight > 0.)
        assert not np.allclose(weight, 1.0)
