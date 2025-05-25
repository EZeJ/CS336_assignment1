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


import torch
import torch.nn as nn
from einops import rearrange, einsum


class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE) via 2×2 matrix multiplication per head pair.

    Args:
        theta (float): Base Θ value for computing rotary frequencies.
        d_k (int): Dimensionality of queries/keys (must be even).
        max_seq_len (int): Maximum sequence length supported.
        device (torch.device|None): Device for buffer storage (defaults to CPU).
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute inverse frequencies for each pair dimension
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        )  # (d_k/2,)

        # Positions index
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (max_seq_len,)

        # Compute rotation angles: outer product -> (max_seq_len, d_k/2)
        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos()  # (max_seq_len, d_k/2)
        sin = angles.sin()  # (max_seq_len, d_k/2)

        # Build the 2×2 rotation matrices for each position and head pair:
        # rot_mats shape = (max_seq_len, d_k/2, 2, 2)
        rot_mats = torch.stack([
            torch.stack([cos, -sin], dim=-1),  # row 0
            torch.stack([sin,  cos], dim=-1),  # row 1
        ], dim=-2)

        # Register as non-learnable buffer
        self.register_buffer("rot_mats", rot_mats, persistent=False)

    # def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    #     """
    #     Apply RoPE to input tensor x.

    #     Args:
    #         x (Tensor): shape (..., seq_len, d_k)
    #         token_positions (LongTensor): shape (..., seq_len), values in [0, max_seq_len)

    #     Returns:
    #         Tensor of same shape as x, with rotary embeddings applied.
    #     """
    #     # Split into even/odd dims: (..., seq_len, d_k/2)
    #     x_even = x[..., 0::2]
    #     x_odd  = x[..., 1::2]

    #     # Gather the corresponding rotation matrices: (..., seq_len, d_k/2, 2, 2)
    #     rot = self.rot_mats[token_positions]

    #     # Pack x into last dimension: (..., seq_len, d_k/2, 2)
    #     x_pair = torch.stack([x_even, x_odd], dim=-1)

    #     # Matrix multiply each 2×2 rot matrix with its x_pair vector:
    #     # (..., seq_len, d_k/2, 2, 1)
    #     out_pair = torch.matmul(rot, x_pair.unsqueeze(-1))

    #     # Remove trailing singleton: (..., seq_len, d_k/2, 2)
    #     out_pair = out_pair.squeeze(-1)

    #     # Flatten the last two dims back to d_k: (..., seq_len, d_k)
    #     return out_pair.flatten(-2)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the last dim of x.

        Args:
            x (Tensor): shape (..., seq_len, d_k)
            token_positions (LongTensor): shape (..., seq_len), values in [0, max_seq_len)

        Returns:
            Tensor same shape as x, with rotary embeddings applied.
        """
        # Pack even/odd dims into head-pairs of size 2: (..., seq_len, d_k/2, 2)
        x_pair = rearrange(x, "... seq (h d) -> ... seq h d", d=2)

        # Gather the corresponding 2×2 rot matrices: (..., seq_len, d_k/2, 2, 2)
        rot = self.rot_mats[token_positions]

        # Matrix multiply each 2×2 matrix with its pair vector via einsum
        # 'row' and 'col' correspond to 2×2 dims:
        out_pair = einsum(
            rot,       # (..., seq, h, row, col)
            x_pair,    # (..., seq, h, col)
            "... seq h row col, ... seq h col -> ... seq h row"
        )

        # Flatten the last two dims back to d_k: (..., seq_len, d_k)
        return rearrange(out_pair, "... seq h d -> ... seq (h d)")


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

        out_pair = einsum(
            self.R[token_positions],       # (..., seq, h, row, col)
            re_in_query_or_key,    # (..., seq, h, col)
            "... seq h row col, ... seq h col -> ... seq h row"
        )
        
        put_it_back = rearrange(out_pair, "batch pos b1 b2 -> batch pos (b1 b2)")
        return put_it_back

