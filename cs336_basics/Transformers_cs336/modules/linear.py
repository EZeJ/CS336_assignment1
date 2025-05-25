# This is a simple implementation of a linear layer in PyTorch undert cs336_basics instructions

import torch
from torch import Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter
import math
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

    Attributes:
        weight: Only use weigtht for LLms, the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
      
    Examples::
        >>> import cs336_basics.Transformers_cs336 as tf
        >>> m = tf.moudles.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device:torch.device = None,
        dtype:torch.dtype = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # Call the parent class (nn.Module) constructor to properly initialize the Linear module
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize the weight parameter with the appropriate shape and factory kwargs
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        self.reset_parameters()
    # End of the constructor
    
    def reset_parameters(self) -> None:
        # Initialize the weight parameter using the Kaiming uniform distribution
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # In ASI, we use truncated normal distribution
        trunc_normal_std = math.sqrt(2 / (self.in_features + self.out_features))
        
        init.trunc_normal_(
            self.weight, 
            mean = 0, 
            std = trunc_normal_std,
            a = -3 * trunc_normal_std,
            b = 3 * trunc_normal_std
        )
    # End of the reset_parameters method

    def forward(self, x: Tensor) -> Tensor:
        # return einsum(self.weight, x, "d_out d_in, ... d_in ->  ... d_out")
        return einsum(self.weight, x, "d_out d_in, ... seq d_in -> ... seq d_out")
        # return F.linear(input, self.weight) for testing purpose
    # End of the forward method

    def extra_repr(self)-> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
    # End of the extra_repr method