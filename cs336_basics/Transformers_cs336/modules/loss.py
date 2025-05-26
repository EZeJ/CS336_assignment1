import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import repeat, reduce, einsum, rearrange
from cs336_basics.Transformers_cs336 import transformer as my_tf


"""Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) 
        with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
"""

def get_cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    # This function is given by a already falttened inputs and targets.
    # The orginal ouput of the model is of shape (batch_size, seq_len, vocab_size)
    # and the targets are of shape (batch_size, seq_len).
    # Flatten the inputs and targets by 
    # from einops import rearrange
    # inputs = rearrange(inputs, "b s v -> (b s) v")
    # targets = rearrange(targets, "b s -> (b s)")
    # So this function's input has shape (batch_size * seq_len, vocab_size)
    # and targets has shape (batch_size * seq_len,).
    # It is the same input format as the one used in the PyTorch's cross-entropy loss function.
    # That is, F.cross_entropy(inputs, targets, reduction='mean')
    
    # Subtract max for numerical stability
    x_max = inputs.max(dim=-1, keepdim=True).values
    logits = inputs - x_max

    # Compute log(sum(exp(logits)))
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))

    # Gather the logit corresponding to the target class
    target_logits = logits[torch.arange(inputs.shape[0]), targets]

    # More explaination of the target_logits indexing:
    # Here are two flattened tensors:
    # Inputs = 
    #       tensor([[0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
    #           [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
    #            [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
    #            [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
    #            [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
    #            [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
    #            [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
    #            [0.8541, 0.1803, 0.2054, 0.4775, 0.8199]])
    #
    # Targets = 
    #       tensor([1, 0, 2, 2, 4, 1, 4, 0])
    #
    # torch.arange(inputs.shape[0]) =
    #       tensor([0, 1, 2, 3, 4, 5, 6, 7])
    #
    # Inputs.shape[0] is the batch size.
    #
    # What logits[torch.arange(inputs.shape[0]), targets] does is,
    # we pick every row of the logits tensor, and for each row, we pick the column
    # corresponding to the target class.
    # same as a for loop as 
    #
    # target_logits[i] = 
    #       logits[i, targets[i]] for i in range(inputs.shape[0])
    #

    # Cross-entropy: - (target logit - log sum exp)
    loss = -target_logits + log_sum_exp
    return loss.mean()


def get_perplexity(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    loss = get_cross_entropy_loss(inputs, targets)
    return torch.exp(loss)