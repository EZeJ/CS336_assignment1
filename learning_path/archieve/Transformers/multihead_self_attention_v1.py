# import torch
# from torch import Tensor
# from einops import rearrange, reduce, einsum
# from jaxtyping import Float
# from torch.nn import init
# import torch.nn as nn
# from jaxtyping import Float, Int
# import cs336_basics.Transformers_cs336 as my_tf

# class MultiheadSelfAttention(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         num_heads: int,
#         max_seq_len: int = None, # Maximum sequence length to pre-cache if your implementation does that.
#         theta: float = None, # RoPE parameter.
#         token_positions: Float[Tensor, " ... sequence_length"] = None, # Tensor of shape (batch_size, sequence_length) with the token positions
#         mask: Float[Tensor, " ... queries keys"] = None, #ptional mask of shape [..., queries, keys]. Elements with a value of 0 will be masked (set to -inf before softmax).
#         device: torch.device = None, 
#     ) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.max_seq_len = max_seq_len
#         self.theta = theta
#         self.token_positions = token_positions
#         self.mask = mask
#         self.device = device

#         # Dimension equations:
#         # d_k = d_v = d model /h
#         # W_Q ∈ R ^ h * dk × d_model, 
#         # W_K ∈ R ^ h * dk × d_model, 
#         # W_V ∈ R ^ h * dv × d_model, 
#         # W_O ∈ R ^ dmodel × hdv
#         # Weight matrix A ∈ ℝ^{out_features × in_features}
#         # ------------------------------------------------
#         # Role              Matrix Dimension     PyTorch Name
#         # Output neurons    Rows                 out_features
#         # Input features    Columns              in_features
#         # ------------------------------------------------
#         #
#         # Each row in A defines the weights for producing one output feature 
#         # from all input features (i.e., a dot product with the input vector).

#         self.d_k = d_model // num_heads
#         self.d_v = d_model // num_heads

#         self.q_proj_weight = my_tf.modules.Linear(
#             in_features=d_model,
#             out_features=self.d_k * num_heads,
#             device=device
#         )
#         self.k_proj_weight = my_tf.modules.Linear(
#             in_features=d_model,
#             out_features=self.d_k * num_heads,
#             device=device
#         )
#         self.v_proj_weight = my_tf.modules.Linear(
#             in_features=d_model,
#             out_features=self.d_v * num_heads,
#             device=device
#         )
#         self.o_proj_weight = my_tf.modules.Linear(
#             in_features=self.d_v * num_heads,
#             out_features=d_model,
#             device=device
#         )

#     def forward(self, in_features: Float[Tensor, " ... sequence_length d_in"]) -> Float[Tensor, " ... sequence_length d_out"]:
#         # If we use RoPE
#         if self.token_positions is not None and self.theta is not None and self.max_seq_len is not None:
#             # Apply RoPE
#             in_features = my_tf.RoPE(
#                 theta=self.theta,
#                 d_k=self.d_k,
#                 max_seq_len=self.max_seq_len,
#                 device=self.device
#             )(in_features, self.token_positions)

#         # Calculate the attention scores
#         multi_head = my_tf.attention.scaled_dot_product_attention(
#             Q=self.q_proj_weight(in_features),
#             K=self.k_proj_weight(in_features),
#             V=self.v_proj_weight(in_features),
#             mask=self.mask
#         )
#         multi_head_selfAttention = self.o_proj_weight(multi_head)
#         return multi_head_selfAttention



import torch
from torch import nn, Tensor
from einops import rearrange, einsum
from jaxtyping import Float
import math
import cs336_basics.Transformers_cs336 as my_tf


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        token_positions: Float[Tensor, "batch seq_len"] = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions
        self.device = device

        # Projection weights
        self.q_proj_weight = my_tf.modules.Linear(d_model, self.d_k * num_heads, device=device)
        self.k_proj_weight = my_tf.modules.Linear(d_model, self.d_k * num_heads, device=device)
        self.v_proj_weight = my_tf.modules.Linear(d_model, self.d_v * num_heads, device=device)
        self.o_proj_weight = my_tf.modules.Linear(self.d_v * num_heads, d_model, device=device)

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        # [1, 1, seq_len, seq_len] - broadcastable to (B, H, S, S)
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, in_features: Float[Tensor, "batch seq_len d_model"]) -> Float[Tensor, "batch seq_len d_model"]:
        B, S, _ = in_features.shape

        # Project input to Q, K, V
        Q = self.q_proj_weight(in_features)  # [B, S, H * d_k]
        K = self.k_proj_weight(in_features)
        V = self.v_proj_weight(in_features)

        # Reshape to [B, H, S, d_k]
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE to Q and K only
        if self.token_positions is not None and self.theta is not None and self.max_seq_len is not None:
            rope = my_tf.RoPE(theta=self.theta, d_k=self.d_k, max_seq_len=self.max_seq_len, device=self.device)
            Q = rope(Q, self.token_positions)
            K = rope(K, self.token_positions)

        # Causal mask: shape [1, 1, S, S]
        mask = self._causal_mask(S, in_features.device)

        # Compute attention
        attn_output = my_tf.attention.scaled_dot_product_attention(
            Q=Q, K=K, V=V, mask=mask
        )  # [B, H, S, d_v]

        # Merge heads and project
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")
        return self.o_proj_weight(attn_output)