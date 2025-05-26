import torch 
import torch.nn as nn
from torch.nn import functional as F, init
from einops import rearrange, einsum, reduce
from torch import Tensor

class RotaryPositionalEmbedding(nn.Module):
    """
        Implements Rotary Positional Embedding (RoPE) for transformer-based models with matrices operation.

        RoPE is a method of encoding positional information using rotating vectors in 2D subspaces
        of the input embeddings. It allows attention mechanisms to naturally incorporate relative
        positions via complex-number-like rotations, leading to improved generalization and extrapolation
        capabilities.

        This module precomputes and stores a buffer of 2 by 2 rotation matrices for all positions up to
        `max_seq_len`, which are later used during the forward pass to apply rotation to the query or
        key vectors.

        Attributes:
            theta (float): Base for the geometric progression of rotation frequencies. Typically set to 10,000.
            d_k (int): Dimensionality of the input query/key vectors. Must be even.
            max_seq_len (int): Maximum sequence length supported for precomputed rotations.
            device (torch.device or None): The device on which rotation matrices are stored.
            R (Tensor): Buffer of shape (max_seq_len, d_k // 2, 2, 2), storing precomputed 2 by 2 rotation matrices.

        Methods:
            forward(x, token_positions):
                Applies rotary positional embeddings to the last dimension of the input tensor.
                Returns a tensor of the same shape with the RoPE applied.
        """
    def __init__(
        self,
        theta: float,
        d_k: int,    
        max_seq_len:int,
        device: torch.device | None = None
    ) -> None:
        r"""
        Initializes the RotaryPositionalEmbedding module.

        This constructor precomputes and stores 2 by 2 rotation matrices used to apply 
        rotary positional embeddings (RoPE) to query/key vectors in transformer models.

        Args:
            theta (float): The base of the geometric progression for computing rotational frequencies.
                        Typically set to 10,000 in transformer models.
            d_k (int): The dimension of the query/key vectors. Must be even.
            max_seq_len (int): The maximum sequence length this embedding will support.
            device (torch.device | None, optional): The device on which to store the precomputed
                                                    rotation matrices. If None, defaults to the
                                                    current PyTorch default device.

        Raises:
            AssertionError: If `d_k` is not an even number (required for alternating sine/cosine dimensions).
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Create the positional encodings
        self.create_positional_encodings()

    def create_positional_encodings(self) -> None:
       
        # Compute inverse frequencies for each pair dimension
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.d_k, 2, dtype=torch.float32, device=self.device) / self.d_k)
        )  # (d_k/2,))

        # Positions index
        positions = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device)  # (max_seq_len,)

        # Compute rotation angles: outer product -> (max_seq_len, d_k/2)
        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos()  # (max_seq_len, d_k/2)
        sin = angles.sin()  # (max_seq_len, d_k/2)

        # Build the 2×2 rotation matrices for each position and head pair:
        # rot_mats shape = (max_seq_len, d_k/2, 2, 2)
        R = torch.stack([
            torch.stack([cos, -sin], dim=-1),  # row 0
            torch.stack([sin,  cos], dim=-1),  # row 1
        ], dim=-2)

        self.register_buffer("R", R, persistent=False)
        return None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor x.

        Args:
            x (Tensor): shape (..., seq_len, d_k)
            token_positions (LongTensor): shape (..., seq_len), values in [batch, max_seq_len)

        Returns:
            Tensor of same shape as x, with rotary embeddings applied.
        """

        # It appears that using einops in our case is very much slower than using torch directly.
        # Method2     
        # Time consuming: 0.47s
        
        # re_in_query_or_key = rearrange(x, "... (b1 b2) -> ... b1 b2", b2 = 2)

        # out_pair = einsum(
        #     self.R[token_positions],       # (..., seq, h, row, col)
        #     re_in_query_or_key,            # (..., seq, h, col)
        #     "... seq h d_out d_in, ... seq h d_in -> ... seq h d_out"
        # )
        # return rearrange(out_pair, "... seq h d_out -> ... seq (h d_out)")
        
        # Method 2
        # Time consuming: 0.05s

        # Split into even/odd dims: (..., seq_len, d_k/2)
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        # Gather the corresponding rotation matrices: (..., seq_len, d_k/2, 2, 2)
        rot = self.R[token_positions]

        # Pack x into last dimension: (..., seq_len, d_k/2, 2)
        x_pair = torch.stack([x_even, x_odd], dim=-1)

        # Matrix multiply each 2×2 rot matrix with its x_pair vector:
        # (..., seq_len, d_k/2, 2, 1)
        out_pair = torch.matmul(rot, x_pair.unsqueeze(-1))

        # Remove trailing singleton: (..., seq_len, d_k/2, 2)
        out_pair = out_pair.squeeze(-1)

        # Flatten the last two dims back to d_k: (..., seq_len, d_k)
        return out_pair.flatten(-2)
    
