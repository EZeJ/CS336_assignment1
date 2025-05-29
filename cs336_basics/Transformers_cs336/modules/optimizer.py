import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import repeat, reduce, einsum, rearrange
from cs336_basics.Transformers_cs336 import transformer as my_tf
from collections.abc import Callable
from typing import Optional
import math


def get_lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    A scheduler is simply a function that takes the current step t and other relevant parameters 
    (such as the initial and final learning rates), and returns the learning rate to use for the gradient 
    update at step t. The simplest schedule is the constant function, which will return the same learning rate given any t.

    The cosine annealing learning rate schedule takes (i) the current iteration t, (ii) the maximum learning 
    rate α max , (iii) the minimum (final) learning rate α min , (iv) the number of warm-up iterations T w 
    , and (v) the number of cosine annealing iterations T c . The learning rate at iteration t is defined as:
    """

    if t < warmup_iters:
        return t / warmup_iters * max_learning_rate
    elif t > cosine_cycle_iters:
        return min_learning_rate
    else:
        # when t >= warmup_iters and t <= cosine_cycle_iters:
        return (
            min_learning_rate
            + 0.5
            * (1 + math.cos(math.pi * (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
            * (max_learning_rate - min_learning_rate)
        )
    













class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number or default to 0.
                grad = p.grad.data     # Gradient of loss with respect to p.

                # Update weights with decaying learning rate.
                p.data -= lr / math.sqrt(t + 1) * grad

                # Update iteration count.
                state["t"] = t + 1

        return loss
    
class AdamW(torch.optim.Optimizer):
    """
    Implement the AdamW optimizer as a subclass of torch.optim.Optimizer. Your class should take the learning rate α 
    in __init__, as well as the β , ( and λ hyperparameters. To help you keep state, the base Optimizer class gives 
    you a dictionary self.state, which maps nn.Parameter objects to a dictionary that stores any information you need
    for that parameter (for AdamW, this would be the moment estimates). Implement [adapters.get_adamw_cls] and 
    make sure it passes uv run pytest -k test_adamw.


    # Example inputs:
        opt_class(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    """
    def __init__(
            self,
            params,
            lr=1e-3,
            weight_decay = 0.01,
            betas=(0.9, 0.999),
            eps=1e-8
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get state associated with p.
                state = self.state[p]
                t = state.get("t", 1) # start from 1 because of the power of betas 
                grad = p.grad.data # Compute the gradient of the loss at the current time step
                beta1, beta2 = betas

                # Initialize the momemtum vectors at t = 1.(Same as if they don't exist)
                if "m" not in state and "v" not in state:
                    # Initialize first and second moment estimates.
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                # Calculaiton of AdamW update.
                m = state["m"]
                # Update the first moment estimate
                m.mul_(beta1).add_(grad, alpha = 1 - beta1)
                v = state["v"]
                # Update the second moment estimate
                v.mul_(beta2).add_(grad * grad, alpha = 1 - beta2)

                # Compute adjusted α for iteration t
                lr_t = lr * math.sqrt( 1 - beta2 ** t) / (1 - beta1 ** t)
                
                # Update the weights with the AdamW update rule
                # We use addcdiv here to perform the update in a single step.
                # torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None) → Tensor
                # It performs the element-wise division of tensor1 by tensor2, multiplies the result by the scalar value and adds it to input.
                p.data.addcdiv_(m, v.sqrt() + eps, value = -lr_t)
                
                # Apply weight decay
                # We use mul_, it is In-place operation, Faster (no memory allocation), Common practice in custom optimizers
                # p.data = p.data - lr * weight_decay * p.data
                p.data.mul_(1 - lr * weight_decay)

                # Update iteration count.
                state["t"] = t + 1

        return loss