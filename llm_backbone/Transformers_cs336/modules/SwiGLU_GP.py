from __future__ import annotations

import torch
from torch import nn, Tensor
import llm_backbone.Transformers_cs336 as my_tf
from GP.logger_GP import ActivationLogger


class SwiGLUWithLogging(nn.Module):
    """
    SwiGLU variant that logs gate pre-activations for GP dataset collection.

    Logs:
        - gate_u: pre-activation before SiLU (w1 output)
        - gate_v: up-projection (w3 output)
        - silu_in: same as gate_u (for clarity)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        logger: ActivationLogger | None = None,
        log_prefix: str = "",
    ) -> None:
        super().__init__()
        self.inner = my_tf.modules.SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.logger = logger
        self.log_prefix = log_prefix

    def forward(self, x: Tensor, epoch: int | None = None) -> Tensor:
        # replicate inner forward to access intermediates
        u = self.inner.w1(x)
        v = self.inner.w3(x)

        if self.logger is not None:
            pref = self.log_prefix
            self.logger.record(f"{pref}gate_u", u, epoch=epoch)
            self.logger.record(f"{pref}gate_v", v, epoch=epoch)
            self.logger.record(f"{pref}silu_in", u, epoch=epoch)

        silu_u = my_tf.modules.silu(u)
        out = self.inner.w2(silu_u * v)
        return out
