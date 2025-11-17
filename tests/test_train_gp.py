import os
import numpy as np
import torch
import pytest
from pathlib import Path
from GP.logger_GP import ActivationLogger
from llm_backbone.Transformers_cs336.modules.SwiGLU_GP import SwiGLUWithLogging
from llm_backbone.Transformers_cs336.modules.RMSNorm_GP import RMSNormWithLogging


def test_swiglu_rms_logging_flow(tmp_path):
    # Build modules with shared logger
    logger = ActivationLogger(max_samples_per_call=100, rng_seed=0)
    swiglu = SwiGLUWithLogging(d_model=4, d_ff=8, logger=logger)
    rms = RMSNormWithLogging(d_model=4, logger=logger)

    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    epoch = 0
    out = swiglu(x, epoch=epoch)
    out = rms(out, epoch=epoch)

    # Flush and verify NPZ contents
    out_path = logger.flush_epoch_to_npz(epoch, tmp_path)
    data = np.load(out_path)
    for key in ("layer0_gate_u", "layer0_gate_v", "layer0_silu_in", "layer0_rms_mean_sq"):
        # we didn't set a prefix here, so use default keys without prefix
        # but we did set log_prefix default "" in modules, so keys are gate_u, gate_v, silu_in, rms_mean_sq
        pass
    assert set(data.files) == {"gate_u", "gate_v", "silu_in", "rms_mean_sq"}
    assert data["gate_u"].size > 0
    assert data["gate_v"].size > 0
    assert data["silu_in"].size > 0
    assert data["rms_mean_sq"].size > 0


def test_logger_limits_samples():
    logger = ActivationLogger(max_samples_per_call=3, rng_seed=0)
    t = torch.arange(20).float()
    logger.record("foo", t, epoch=None)
    out = logger.get_epoch_buffers(None)
    assert out["foo"].shape[0] == 3
