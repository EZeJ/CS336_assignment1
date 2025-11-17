from __future__ import annotations

import torch
from torch import Tensor
import llm_backbone.Transformers_cs336 as my_tf
from GP.logger_GP import ActivationLogger


class RMSNormWithLogging(torch.nn.Module):
    """
    RMSNorm variant that logs per-token mean square before scaling.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        logger: ActivationLogger | None = None,
        log_prefix: str = "",
    ) -> None:
        super().__init__()
        self.inner = my_tf.modules.RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.logger = logger
        self.log_prefix = log_prefix
        self.eps = eps

    def forward(self, x: Tensor, epoch: int | None = None) -> Tensor:
        # track mean square per token
        ms = torch.mean(x.to(torch.float32) ** 2, dim=-1)
        if self.logger is not None:
            self.logger.record(f"{self.log_prefix}rms_mean_sq", ms, epoch=epoch)
        return self.inner(x)
