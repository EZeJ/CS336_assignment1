import torch
from torch import Tensor
from einops import rearrange, repeat, reduce, einsum
from jaxtyping import Float
import cs336_basics.Transformers_cs336 as my_tf


"""
    Adpater Instructions:
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
"""


class TransformerBlock(torch.nn.Module):
    """
    Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. 
    Your Transformer block should accept (at least) the following parameters.

    d_model: int Dimensionality of the Transformer block inputs. 
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer.

    """
    def __init__(
        self,
        d_model : int,
        num_heads : int,
        d_ff :int,
        max_seq_len: int = None,
        theta: float = None,
        device : torch.device = None,
    ) -> None:
        super().__init__()
        # Usage of MultiHeadSelfAttention
        self.d_model = d_model
        self.num_heads = num_heads
        # Usage of SwiGLU
        self.d_ff = d_ff
        # Usage of GPU allocation
        self.device = device
        # Usage of RoPE
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Build Transformer block
        self.RMSNorm_ln1 = my_tf.modules.RMSNorm(d_model, device=device)
        self.RMSNorM_ln2 = my_tf.modules.RMSNorm(d_model, device=device)
        self.multihead_self_attention = my_tf.attention.MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device
        )
        self.SwiGLU_ffn = my_tf.modules.SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device
        )

    def forward(
            self, in_features: Float[Tensor, "batch seq_len d_model"], 
            token_positions: Float[Tensor, "batch seq_len"]
        ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        Forward pass of the Transformer block.
        Args:
            x (Float[Tensor, "batch seq_len d_model"]): Input tensor to the Transformer block.
        Returns:
            Float[Tensor, "batch seq_len d_model"]: Output tensor after passing through the Transformer block.
        """
        # RMSNorm + MultiheadSelfAttention + Residual
        # y = in_features + MultiHeadSelfAttention(RMSNorm(in_features))
        self.residual = in_features
        self.x = self.RMSNorm_ln1(in_features)
        self.x = self.multihead_self_attention(self.x, token_positions) + self.residual
        # End of the first transformer block
        # RMSNorm + FeedForward SwiGLU + Residual
        # y = x + FeedForwardSwiGLU(RMSNorm(x))
        self.residual_2 = self.x
        self.x = self.RMSNorM_ln2(self.x)
        self.x = self.SwiGLU_ffn(self.x) + self.residual_2
        return self.x