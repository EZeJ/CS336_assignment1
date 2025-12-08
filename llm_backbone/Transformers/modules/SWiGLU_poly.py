import math
import torch
from torch import Tensor, nn
from .linear import Linear


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def expr_torch(x: Tensor) -> Tensor:
    """
    Polynomial surrogate for the SwiGLU FFN.

    This is the PyTorch polynomial expression evolved via GP in
    `experiments/silu_GP_comparision.py`, adapted here as a module-local
    function so it can be used inside the Transformer stack.
    """
    # Accept Python scalars / lists defensively, though we expect a Tensor.
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    return (-0.1248 * (
        (((-1.047 * x) +
          ((2.346 * (x + 3.576)) * ((2.346 * -0.9396) * (3.809 + 1.33)))) +
         (
             (
                 (-0.1248 * (
                     (
                         (
                             (x + (2.437 * x)) *
                             ((2.346 * -0.9396) * 3.221)
                         ) +
                         (
                             (2.346 * -0.9396) *
                             (
                                 ((3.809 + x) *
                                  (
                                      ((2.346 * ((-2.417 + x) * x)) *
                                       (2.346 * -0.9396)) +
                                      (2.437 * x)
                                  )
                                 ) +
                                 (3.809 + x)
                             )
                         ) +
                         -2.236
                     ) +
                     (
                         (
                             1.494 +
                             (
                                 ((3.809 + x) *
                                  (
                                      (2.346 *
                                       ((-2.417 + x) *
                                        (3.809 + (1.906 + 2.385)))) *
                                      (2.346 * -0.9396)
                                  ) +
                                  0.6062
                                 )
                             ) +
                             (
                                 (
                                     ((-2.417 + x) * x) +
                                     (2.346 * (3.809 + 1.33))
                                 ) +
                                 (x * x) +
                                 x
                             )
                         ) *
                         (3.809 +
                          ((x + (0.08281 + x)) + (3.809 + 1.33)))
                     )
                 )) *
                 (
                     ((x + x) * (2.346 * -0.9396)) +
                     ((x + x) + (2.437 * x))
                 )
             ) +
             (
                 (3.809 + x) *
                 (
                     ((x + (0.1279 + x)) * (2.346 * -0.9396)) +
                     (3.918 * -0.9396)
                 )
             ) +
             (
                 (-0.5138 * (x + ((0.3614 + x) + 0.5313))) +
                 ((2.346 * -0.9396) * (3.809 + 1.33))
             ) +
             (
                 -1.891 * (-2.07 + (3.809 + 3.848))
             )
         )
        ) *
        (
            ((x + x) * (2.346 * -0.9396)) +
            ((x + x) + (2.437 * x))
        )
    ))


class SwiGLU(nn.Module):
    r"""
    Polynomial SwiGLU Feed-Forward Network (FFN) module.

    This module preserves the interface of the original SwiGLU FFN but
    replaces the SiLU+GLU computation with a fixed polynomial surrogate
    `expr_torch(x)` learned via genetic programming.

    Args:
        d_model (int): Dimensionality of the input and output features.
        d_ff (int | None): Dimensionality of the intermediate (hidden) layer.
                           Kept for API compatibility but not used directly
                           in the polynomial.

    Shape:
        - Input:  (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self._compute_d_ff(d_model)
        self.device = device
        self.dtype = dtype

        # Keep Linear submodules for state_dict / logging compatibility,
        # even though the forward path uses the polynomial surrogate.
        self.w1 = Linear(in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=self.d_ff, out_features=d_model, device=device, dtype=dtype)

    @staticmethod
    def _compute_d_ff(d_model: int) -> int:
        """
        Computes the hidden dimension d_ff as (8/3) * d_model,
        rounded up to the next multiple of 64 for hardware efficiency.
        """
        rough = (8 * d_model) / 3
        d_ff = math.ceil(rough / 64) * 64
        return int(d_ff)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using the GP-learned polynomial surrogate.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, d_model)
        """
        return expr_torch(x)

