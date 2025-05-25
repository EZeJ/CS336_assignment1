import torch
from torch import Tensor
from einops import rearrange, reduce, einsum, repeat
from jaxtyping import Float
import cs336_basics.Transformers_cs336 as my_tf

""" Upper Class Parameters
    Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.
    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
"""

class Transformer(torch.nn.Module):
    """
    Time to put it all together! Implement the Transformer language model as described in §3.1 and illustrated in Figure 1.
    At minimum, your implementation should accept all the aforementioned construction parameters for the Transformer block, 
    as well as these additional parameters:
    vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
    context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
    num_layers: int The number of Transformer blocks to use
    """
    def __init__(
        self,
        d_model : int,
        num_heads : int,
        d_ff :int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        max_seq_len: int = None,
        theta: float = None,
        device : torch.device = None,
    ) -> None:
        super().__init__()
        # Initialize the Parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device

        # Build the Embedding Layer
        self.embedding = my_tf.modules.Embedding(vocab_size, d_model, device=device)

        # Build the Transformer Layers with TransformerBlocks
        self.transformer_layers = torch.nn.ModuleList([
            my_tf.transformer.TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
            )
            for _ in range(num_layers)
        ])

        # Build the Final Layer Norm
        self.RMSNorm_ln_final = my_tf.modules.RMSNorm(d_model, device=device)

        # Build the Final Linear Layer
        self.lm_head = my_tf.modules.Linear(d_model, vocab_size, device=device)

    
    def forward(self, in_indices: Float[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len vocab_size"]:
        # 1. Token embeddings
        x = self.embedding(in_indices)  # [batch, seq_len, d_model]

        # 2. Create token position ids (used for RoPE)
        batch_size, seq_len = in_indices.shape
        pos_ids = torch.arange(seq_len, device=in_indices.device)
        pos_ids = repeat(pos_ids, 's -> b s', b=batch_size)

        # 3. Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, token_positions=pos_ids)

        # 4. Final RMSNorm
        x = self.RMSNorm_ln_final(x)

        # 5. Linear projection to vocab size
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits