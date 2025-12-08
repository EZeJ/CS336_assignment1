import torch
from torch import Tensor
import math
from einops import rearrange, einsum, reduce
from torch.nn.parameter import Parameter
from torch.nn import functional as F, init


class RMSNorm(torch.nn.Module):
    r"""
        Root Mean Square Layer Normalization (RMSNorm) module.

        RMSNorm is an alternative to LayerNorm that normalizes inputs using the root mean square
        (RMS) of the hidden units, without subtracting the mean. This can be more stable and efficient 
        in certain transformer-based architectures.

        Args:
            d_model (int): Hidden dimension of the model. This is the size of the last dimension of the input tensor.
            eps (float, optional): A small epsilon value added for numerical stability. Default is 1e-5.
            device (torch.device, optional): The device to store the layer's parameters on. Default is None.
            dtype (torch.dtype, optional): The data type of the layer's parameters. Default is None.

        Shape:
            - Input: (batch_size, sequence_length, d_model)
            - Output: (batch_size, sequence_length, d_model) — same shape as input.

        Example:
            >>> norm = RMSNorm(d_model=512)
            >>> x = torch.randn(32, 128, 512)
            >>> output = norm(x)
            >>> print(output.shape)
            torch.Size([32, 128, 512])
    """
    def __init__(
        self, 
        d_model: int,
        eps: float = 1e-5,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(
            torch.ones(d_model, **factory_kwargs)
        )
        self.reset_parameters()
    # End of the constructor
    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}, weight={self.weight.shape}"
    # End of the extra_repr method

    # a private method to get the RMS value
    def _get_RMS(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Root Mean Square (RMS) of the input tensor.
        RMS(a) = sqrt([1/d_model * sum(a_i^2)_{from i to d_model}] + eps)
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: RMS value of the input tensor.
        """
        # [1/d_model * sum(a_i^2)_{from i to d_model}] is equivalent to calculating the mean of squares
        # We use ... to indicate that we only want to reduce the last dimension (d_model)
        mean_square = reduce(x ** 2, '... d_model -> ... 1', 'mean')
        return torch.sqrt(mean_square + self.eps)
    # End of the get_RMS method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor.
        RMSNorm(a_i) = a_i * g_i / RMS(a)
        RMS(a) = sqrt([1/d_model * sum(a_i^2)_(from i to d_model)] + eps)
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Normalized tensor of the same shape (batch_size, sequence_length, d_model).
        """
        # One way of calculating the RMSNorm
        # norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Calculate the RMS value
        # We use float32 to avoid overflow when calculating the square
        in_dtype = x.dtype 
        x = x.to(torch.float32)

        RMS = self._get_RMS(x)
        # Normalize the input tensor
        result = x / RMS
        # Scale the normalized tensor by the gain
        result = result * self.weight

        return result.to(in_dtype)