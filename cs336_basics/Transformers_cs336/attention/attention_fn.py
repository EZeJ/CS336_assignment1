import torch
from torch import Tensor
from torch.nn import init
from einops import rearrange, repeat, reduce, einsum
import torch.nn as nn
from jaxtyping import Float, Int

def softmax(x : Tensor, dim_i : int) -> Tensor:
    """
    Computes the softmax of a tensor along a specified dimension.

    Args:
        x (Tensor): Input tensor.
        dim_i (int): Dimension along which to compute the softmax.

    Returns:
        Tensor: Softmax of the input tensor along the specified dimension.
    """

    # Note: This softmax function is doing computation on a selected dimension.
    x_max = torch.max(x, dim=dim_i, keepdim=True).values
    x_exp = torch.exp(x - x_max)  # Subtract max for numerical stability
    x_sum = torch.sum(x_exp, dim=dim_i, keepdim=True)
    return x_exp / x_sum


def scaled_dot_product_attention( 
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Computes scaled dot-product attention.

    This function implements the core attention mechanism used in Transformer models. 
    It computes attention scores between query and key vectors, applies optional masking, 
    normalizes the scores with softmax, and uses the resulting weights to produce a 
    weighted sum over the value vectors.

    Args:
        Q (Tensor): Query tensor of shape [..., queries, d_k], where '...' denotes any number 
                    of leading dimensions (e.g., batch size, heads).
        K (Tensor): Key tensor of shape [..., keys, d_k], matching the leading dimensions of Q.
        V (Tensor): Value tensor of shape [..., values, d_v], typically with values = keys.
        mask (Tensor or None, optional): Optional mask of shape [..., queries, keys]. Elements 
                    with a value of 0 will be masked (set to -inf before softmax).

    Returns:
        Tensor: Output tensor of shape [..., queries, d_v], the result of applying attention weights 
                to the value vectors.

    Notes:
        - The attention scores are scaled by sqrt(d_k) to improve gradient stability.
        - If a mask is provided, entries with mask == 0 are suppressed from attention.
        - The function assumes that Q, K, and V are already projected and aligned in dimensionality.
    """
    d_k = Q.shape[-1]
    Q_T_K = einsum(Q, K, "... q d, ... k d -> ... q k")
    Q_T_K_over_sqrt_d_k = Q_T_K / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype)) # Scale the scores
    if mask is not None:
        Q_T_K_over_sqrt_d_k = Q_T_K_over_sqrt_d_k.masked_fill(mask == 0, float("-inf"))

    # Apply softmax over the last dimension (keys) of the attention scores.
    softmax_QK_scores = softmax(Q_T_K_over_sqrt_d_k, dim_i=-1)  
    # After computing Q^T @ K (In mathematic notation), we obtain a tensor 
    # of shape [..., queries, keys], where each entry represents the similarity 
    # between a query and a key.
    #
    # We apply softmax along the last dimension (dim = -1) to normalize these
    # scores across all keys for each query. This converts raw similarity scores
    # into a probability distribution, so each query assigns attention weights 
    # summing to 1 across all keys.
    #
    # In other words:
    #   - For each query: softmax tells us how much attention to pay to each key.
    #   - Softmax on dim=-1 ensures this happens for each query independently.
    #
    # Do NOT apply softmax on dim=-2 (queries), as this would normalize across
    # queries for each key — which is not the intended behavior in attention.
    attn_scores = einsum(softmax_QK_scores, V, "... q k, ... k v -> ... q v")
    return attn_scores