import torch
from torch import Tensor
import math
from einops import rearrange, einsum, reduce
from torch import nn
from torch.nn import functional as F, init

from .linear import Linear

class SwiGLU(nn.Module):
    r"""
    SwiGLU Feed-Forward Network (FFN) module.

    This module implements the SwiGLU activation mechanism used in transformer architectures. 
    It combines a SiLU activation function with a gated linear unit (GLU), following the form 
    proposed in recent transformer variants for more expressive and efficient representations.

    Definitions:
        - SiLU(x) = x * sigmoid(x)
        - GLU(x) = sigmoid(W₁ @ x) * (W₃ @ x)    (element-wise product, '*' in PyTorch)

    The SwiGLU FFN is defined as:
        FFN(x) = W₂ @ [SiLU(W₁ @ x) * (W₃ @ x)]

    where:
        - x ∈ R^{d_model}
        - W₁, W₃ ∈ R^{d_ff * d_model}
        - W₂ ∈ R^{d_model R d_ff}
        - Typically, d_ff ≈ (8/3) * d_model

    Args:
        d_model (int): Dimensionality of the input and output features.
        d_ff (int): Dimensionality of the intermediate (hidden) layer.
    
    Shape:
        - Input: (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, d_model)

    Example:
        >>> swiglu = SwiGLU(d_model=512, d_ff=1365)
        >>> x = torch.randn(32, 128, 512)
        >>> out = swiglu(x)  # shape: (32, 128, 512)
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int = None,
            device: torch.device = None,
            dtype: torch.dtype = None,

    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self._compute_d_ff(d_model)
        self.device = device
        self.dtype = dtype

        # Initialize weight matrices with Linear layers
        self.w1 = Linear(
            in_features=self.d_ff,
            out_features=d_model,
            device=self.device,
            dtype=self.dtype,
        )
        self.w2 = Linear(
            in_features=d_model,
            out_features=self.d_ff,
            device=self.device,
            dtype=self.dtype,
        )
        self.w3 = Linear(
            in_features=self.d_ff,
            out_features=d_model,
            device=self.device,
            dtype=self.dtype,
        )


    def _compute_d_ff(d_model: int) -> int:
        """
        Computes the hidden dimension d_ff as (8/3) * d_model,
        rounded up to the next multiple of 64 for hardware efficiency.
        """
        rough = (8 * d_model) / 3
        d_ff = math.ceil(rough / 64) * 64
        return int(d_ff)

    def silu(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the SwiGLU FFN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, d_model)
        """
        wx1 = self.w1(x)
        wx3 = self.w3(x)
        silu_wx1 = self.silu(wx1)
        swiglu = self.w2(silu_wx1 * wx3)
        return swiglu