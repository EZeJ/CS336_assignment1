from pathlib import Path
import numpy as np
import sys

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(TEST_ROOT))

from GP.aggregate_GP import aggregate_raw_logs, silu_fn, rms_inv_sqrt_fn


def test_aggregate_builds_datasets(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()

    # create mock epoch npz
    silu_vals = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    rms_vals = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    np.savez_compressed(raw_dir / "epoch_0000.npz", silu_in=silu_vals, gate_u=silu_vals, rms_mean_sq=rms_vals)

    outputs = aggregate_raw_logs(raw_dir, out_dir)

    # silu dataset
    silu_path = outputs["silu_in"]
    data = np.load(silu_path)
    assert data["inputs"].shape == (3, 1)
    np.testing.assert_allclose(data["targets"], silu_fn(silu_vals))

    # rms dataset
    rms_path = outputs["rms_mean_sq"]
    data_rms = np.load(rms_path)
    np.testing.assert_allclose(data_rms["targets"], rms_inv_sqrt_fn(rms_vals))


def test_skip_unknown_signal(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    np.savez_compressed(raw_dir / "epoch_0000.npz", foo=np.array([1, 2, 3], dtype=np.float32))
    outputs = aggregate_raw_logs(raw_dir, out_dir, signals=["foo"])
    assert outputs == {}
