import torch 
import torch.nn as nn
import math
from torch.nn import functional as F, init
from einops import rearrange, einsum, reduce
from torch import Tensor

class RotaryPositionalEmbedding(nn.Module):
    """
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device=None
    ) 
    Construct the RoPE module and create buffers if needed.
    theta: float Θ value for the RoPE 
    d_k: int dimension of query and key vectors
    max_seq_len: int Maximum sequence length that will be inputted device: 
    torch.device | None = None Device to store the buffer on
    """
    def __init__(
        self,
        theta: float,
        d_k: int,    
        max_seq_len:int,
        device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Create the positional encodings
        self.create_positional_encodings()

    def create_positional_encodings(self) -> None:
        # Create the whole R matrix
        R = torch.empty((self.max_seq_len, int(self.d_k/2), 2, 2))
        
        for i in range(self.max_seq_len):
            for k in range(int(self.d_k / 2)):
                angle_theta = i / (math.pow(self.theta, (2.0 * k / self.d_k)))
                cos_theata = math.cos(angle_theta)
                sin_theta = math.sin(angle_theta)

                R_i_k = torch.tensor([
                    [cos_theata, -sin_theta],
                    [sin_theta, cos_theata]
                ], device=self.device)
                R[i, k, :, :] = R_i_k
                if k >= i:
                    break

        self.register_buffer("R", R, persistent=False)
        return None

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        r"""
        def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor 
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. 
        Note that you should tolerate x with an arbitrary number of batch dimensions. 
        You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.
        """    
        re_in_query_or_key = rearrange(x, "batch pos (b1 b2) -> batch pos b1 b2", b2 = 2)
        rotated = einsum(
            re_in_query_or_key, self.R,
            "... pos b1 b2, pos b1 b2 b3 -> ... pos b1 b2",
        )
        put_it_back = rearrange(rotated, "batch pos b1 b2 -> batch pos (b1 b2)")
        return put_it_back


# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(
#         self,
#         theta: float,
#         d_k: int,
#         max_seq_len: int,
#         device: torch.device | None = None
#     ) -> None:
#         super().__init__()
#         self.theta = theta
#         self.d_k = d_k
#         self.max_seq_len = max_seq_len
#         self.device = device

#         inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k // 2,)
#         pos = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
#         angles = torch.einsum("i,j->ij", pos, inv_freq)  # (max_seq_len, d_k // 2)

#         cos = torch.cos(angles).to(torch.float32)  # (max_seq_len, d_k // 2)
#         sin = torch.sin(angles).to(torch.float32)  # (max_seq_len, d_k // 2)

#         self.register_buffer("cos", cos, persistent=False)
#         self.register_buffer("sin", sin, persistent=False)

#     def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
#         # x: (..., seq_len, d_k)
#         # token_positions: (..., seq_len)
#         cos = self.cos[token_positions]  # (..., seq_len, d_k // 2)
#         sin = self.sin[token_positions]  # (..., seq_len, d_k // 2)

#         x1, x2 = x[..., ::2], x[..., 1::2]  # Get even and odd parts
#         x_rotated = torch.stack([
#             x1 * cos - x2 * sin,
#             x1 * sin + x2 * cos
#         ], dim=-1)  # (..., seq_len, d_k // 2, 2)

#         return x_rotated.flatten(-2)  # (..., seq_len, d_k)



# import torch
# import torch.nn as nn
# from torch import Tensor
# from einops import rearrange

# class RotaryPositionalEmbedding(nn.Module):
#     def __init__(
#         self,
#         theta: float,
#         d_k: int,
#         max_seq_len: int,
#         device: torch.device | None = None
#     ) -> None:
#         super().__init__()
#         self.theta = theta
#         self.d_k = d_k
#         self.max_seq_len = max_seq_len
#         self.device = device

#         inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k // 2,)
#         pos = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
#         angles = torch.einsum("i,j->ij", pos, inv_freq)  # (max_seq_len, d_k // 2)

#         cos = torch.cos(angles).to(torch.float32)
#         sin = torch.sin(angles).to(torch.float32)

#         self.register_buffer("cos", cos, persistent=False)
#         self.register_buffer("sin", sin, persistent=False)

#     def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
#         # x: (..., seq_len, d_k)
#         # token_positions: (..., seq_len)

#         # Rearrange x to (..., seq_len, d_k // 2, 2) by grouping even/odd dims
#         x = rearrange(x, "... seq (d r) -> ... seq d r", r=2)  # r=2 means interleave as (even, odd)

#         cos = self.cos[token_positions]  # (..., seq_len, d_k // 2)
#         sin = self.sin[token_positions]  # (..., seq_len, d_k // 2)

#         # Reshape cos/sin to broadcast across r=2
#         cos = rearrange(cos, "... seq d -> ... seq d 1")
#         sin = rearrange(sin, "... seq d -> ... seq d 1")

#         # Apply rotation using complex-style multiplication
#         x_rotated = torch.stack([
#             x[..., 0] * cos[..., 0] - x[..., 1] * sin[..., 0],
#             x[..., 0] * sin[..., 0] + x[..., 1] * cos[..., 0]
#         ], dim=-1)  # (..., seq_len, d_k // 2, 2)

#         # Restore to original shape (..., seq_len, d_k)
#         out = rearrange(x_rotated, "... seq d r -> ... seq (d r)")
#         return out




import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Precompute angle frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))  # (d_k // 2,)
        pos = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
        angles = torch.einsum("i,j->ij", pos, inv_freq)  # (max_seq_len, d_k // 2)

        # Store cos/sin for all positions and dimensions
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)

        # Get cos/sin for given token positions
        # Repositioning cos/sin to match token positions by using positional indices
        cos = self.cos[token_positions]  # (..., seq_len, d_k // 2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k // 2)

        # Split x into even/odd channels: x[..., 0::2], x[..., 1::2]
        x = rearrange(x, "... seq (d r) -> ... seq d r", r=2)  # (..., seq_len, d_k//2, 2)
        x_even = x[..., 0]  # (..., seq_len, d_k//2)
        x_odd  = x[..., 1]  # (..., seq_len, d_k//2)

        # Apply RoPE rotation
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd  = x_even * sin + x_odd * cos

        # Combine back into (..., seq_len, d_k)
        out = rearrange([rotated_even, rotated_odd], "r ... seq d -> ... seq (d r)")
        return out

