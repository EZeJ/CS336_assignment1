import sys
from pathlib import Path
import torch
import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(TEST_ROOT))

from GP.logger_GP import ActivationLogger
from llm_backbone.Transformers_cs336.modules.SwiGLU_GP import SwiGLUWithLogging
from llm_backbone.Transformers_cs336.modules.RMSNorm_GP import RMSNormWithLogging


def test_activation_logger_subsamples():
    logger = ActivationLogger(max_samples_per_call=5, rng_seed=0)
    tensor = torch.arange(100).float()
    logger.record("test", tensor, epoch=0)
    buffers = logger.get_epoch_buffers(0)
    assert "test" in buffers
    assert len(buffers["test"]) == 5  # subsampled to cap
    # ensure values come from original tensor
    assert buffers["test"].dtype == np.float32
    assert buffers["test"].min() >= 0
    assert buffers["test"].max() <= 99


def test_swiglu_with_logging_records_intermediates():
    torch.manual_seed(0)
    logger = ActivationLogger(max_samples_per_call=1000, rng_seed=0)
    module = SwiGLUWithLogging(d_model=4, d_ff=8, logger=logger)
    x = torch.randn(2, 3, 4)
    out = module(x, epoch=1)
    assert out.shape == (2, 3, 4)
    buffers = logger.get_epoch_buffers(1)
    assert "gate_u" in buffers or "gate_v" in buffers or "silu_in" in buffers
    assert buffers["gate_u"].size > 0
    assert buffers["gate_v"].size > 0
    assert buffers["silu_in"].size > 0


def test_rmsnorm_with_logging_records_mean_sq():
    torch.manual_seed(0)
    logger = ActivationLogger(max_samples_per_call=1000, rng_seed=0)
    module = RMSNormWithLogging(d_model=4, logger=logger)
    x = torch.randn(2, 3, 4)
    out = module(x, epoch=2)
    assert out.shape == (2, 3, 4)
    buffers = logger.get_epoch_buffers(2)
    assert "rms_mean_sq" in buffers
    assert buffers["rms_mean_sq"].size == 6  # 2*3 tokens logged


def test_flush_all_to_npz(tmp_path):
    logger = ActivationLogger(max_samples_per_call=10, rng_seed=0)
    logger.record("a", torch.tensor([1, 2, 3]), epoch=0)
    logger.record("a", torch.tensor([4, 5]), epoch=1)
    out_path = logger.flush_all_to_npz(tmp_path / "all.npz")
    data = np.load(out_path)
    np.testing.assert_array_equal(data["a"], np.array([1, 2, 3, 4, 5], dtype=np.float32))
    np.testing.assert_array_equal(data["a_epoch"], np.array([0, 0, 0, 1, 1], dtype=np.int32))
