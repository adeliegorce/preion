import os

import numpy as np
import pytest

pytest.importorskip("preion.forecast.datapoints")

from preion.forecast.datapoints import make_datapoints


def test_packaged_datapoints_shapes(packaged_datapoints):
    # configs/config_tutorial_mcmc.yaml's ells grids: tau 49 points, ksz 14, bb 50.
    assert packaged_datapoints["tau"].shape == (49,)
    assert packaged_datapoints["ksz"].shape == (14,)
    assert packaged_datapoints["bb"].shape == (50,)
    assert packaged_datapoints["cov_tau"].shape == (49, 49)
    assert packaged_datapoints["cov_ksz"].shape == (14, 14)
    assert packaged_datapoints["cov_bb"].shape == (50, 50)
    assert packaged_datapoints["ells_tau"].shape == (49,)
    assert packaged_datapoints["ells_ksz"].shape == (14,)
    assert packaged_datapoints["ells_bb"].shape == (50,)
    for key in ["tau", "ksz", "bb", "cov_tau", "cov_ksz", "cov_bb"]:
        assert np.all(np.isfinite(packaged_datapoints[key]))


@pytest.mark.slow
def test_make_datapoints_shapes_no_telescope(theta_true, tiny_ells):
    tau_ps, ksz_ps, total_bb, cov_tau, cov_ksz, cov_bb = make_datapoints(
        theta_true, telescopes=None, ells=tiny_ells,
        use_ksz_emulator=False, randomness=False, save=None,
    )
    assert tau_ps.shape == (len(tiny_ells[0]),)
    assert ksz_ps.shape == (len(tiny_ells[1]),)
    assert total_bb.shape == (len(tiny_ells[2]),)
    assert cov_tau.shape == (len(tiny_ells[0]), len(tiny_ells[0]))
    assert cov_ksz.shape == (len(tiny_ells[1]), len(tiny_ells[1]))
    assert cov_bb.shape == (len(tiny_ells[2]), len(tiny_ells[2]))
    assert np.all(np.isfinite(tau_ps))
    assert np.all(np.isfinite(ksz_ps))
    assert np.all(np.isfinite(total_bb))

@pytest.mark.slow
def test_make_datapoints_save_writes_files(theta_true, tiny_ells, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    make_datapoints(
        theta_true, telescopes=None, ells=tiny_ells,
        use_ksz_emulator=False, randomness=False, save="unittest",
    )
    for suffix in ["bb_datapoints", "ksz_datapoints", "tau_datapoints", "cov_ksz", "cov_tau", "cov_bb"]:
        assert (tmp_path / "data" / f"unittest_{suffix}.txt").exists()
