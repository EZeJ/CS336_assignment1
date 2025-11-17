import numpy as np
import pytest
from GP.run_search import build_dataset_from_signal


def test_build_dataset_applies_target_fn():
    npz = {
        "silu_in": np.array([0.0, 1.0], dtype=np.float32),
        "rms_mean_sq": np.array([0.0, 1.0], dtype=np.float32),
        "other": np.array([2.0, -2.0], dtype=np.float32),
    }
    ds_silu = build_dataset_from_signal(npz, "silu_in")
    # SiLU: x * sigmoid(x); check positive side > 0.5
    assert ds_silu.targets.shape == (2,)
    assert ds_silu.targets[1] > 0.5

    ds_rms = build_dataset_from_signal(npz, "rms_mean_sq")
    # inv sqrt: 1/sqrt(x+eps); at x=0 should be large, at x=1 ~1
    assert ds_rms.targets[1] == pytest.approx(1.0, rel=1e-2)

    ds_other = build_dataset_from_signal(npz, "other")
    # default identity for unknown signals
    np.testing.assert_allclose(ds_other.targets, npz["other"])
