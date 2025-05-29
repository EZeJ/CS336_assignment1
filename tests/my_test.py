import torch
from torch import nn
import cs336_basics.Transformers_cs336 as my_tf
from cs336_basics.Transformers_cs336.modules.optimizer import SGD



def test_SGD_optimizer():
    """
    Test the SGD optimizer with a simple example.
    This function initializes a tensor of weights, computes a loss,
    and updates the weights using the SGD optimizer.
    """
    # Set random seed for reproducibility
    torch.manual_seed(0)
    # Initialize weights as a learnable parameter
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e2)

    # Training loop
    for t in range(100):
        opt.zero_grad()                        # Reset gradients
        loss = (weights**2).mean()             # Compute scalar loss
        print(loss.cpu().item())               # Print loss
        loss.backward()                        # Backpropagate
        opt.step()                             # Update parameters

    assert torch.allclose(weights, torch.zeros_like(weights), atol=1e-4), "Weights did not converge to zero"