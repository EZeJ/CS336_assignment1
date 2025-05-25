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
            rope = my_tf.modules.RotaryPositionalEmbedding(theta=self.theta, d_k=self.d_k, max_seq_len=self.max_seq_len, device=self.device)
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