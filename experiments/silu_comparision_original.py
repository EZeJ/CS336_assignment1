import torch

def expr_torch(x):
    # Accept Python floats / lists as well
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    return (-0.1248 * (
        (((-1.047 * x) +
          ((2.346 * (x + 3.576)) * ((2.346 * -0.9396) * (3.809 + 1.33)))) +
         (
             (
                 (-0.1248 * (
                     (
                         (
                             (x + (2.437 * x)) *
                             ((2.346 * -0.9396) * 3.221)
                         ) +
                         (
                             (2.346 * -0.9396) *
                             (
                                 ((3.809 + x) *
                                  (
                                      ((2.346 * ((-2.417 + x) * x)) *
                                       (2.346 * -0.9396)) +
                                      (2.437 * x)
                                  )
                                 ) +
                                 (3.809 + x)
                             )
                         ) +
                         -2.236
                     ) +
                     (
                         (
                             1.494 +
                             (
                                 ((3.809 + x) *
                                  (
                                      (2.346 *
                                       ((-2.417 + x) *
                                        (3.809 + (1.906 + 2.385)))) *
                                      (2.346 * -0.9396)
                                  ) +
                                  0.6062
                                 )
                             ) +
                             (
                                 (
                                     ((-2.417 + x) * x) +
                                     (2.346 * (3.809 + 1.33))
                                 ) +
                                 (x * x) +
                                 x
                             )
                         ) *
                         (3.809 +
                          ((x + (0.08281 + x)) + (3.809 + 1.33)))
                     )
                 )) *
                 (
                     ((x + x) * (2.346 * -0.9396)) +
                     ((x + x) + (2.437 * x))
                 )
             ) +
             (
                 (3.809 + x) *
                 (
                     ((x + (0.1279 + x)) * (2.346 * -0.9396)) +
                     (3.918 * -0.9396)
                 )
             ) +
             (
                 (-0.5138 * (x + ((0.3614 + x) + 0.5313))) +
                 ((2.346 * -0.9396) * (3.809 + 1.33))
             ) +
             (
                 -1.891 * (-2.07 + (3.809 + 3.848))
             )
         )
        ) *
        (
            ((x + x) * (2.346 * -0.9396)) +
            ((x + x) + (2.437 * x))
        )
    ))


import numpy as np

def expr_numpy(x):
    x = np.asarray(x, dtype=np.float64)

    return (-0.1248 * (
        (((-1.047 * x) +
          ((2.346 * (x + 3.576)) * ((2.346 * -0.9396) * (3.809 + 1.33)))) +
         (
             (
                 (-0.1248 * (
                     (
                         (
                             (x + (2.437 * x)) *
                             ((2.346 * -0.9396) * 3.221)
                         ) +
                         (
                             (2.346 * -0.9396) *
                             (
                                 ((3.809 + x) *
                                  (
                                      ((2.346 * ((-2.417 + x) * x)) *
                                       (2.346 * -0.9396)) +
                                      (2.437 * x)
                                  )
                                 ) +
                                 (3.809 + x)
                             )
                         ) +
                         -2.236
                     ) +
                     (
                         (
                             1.494 +
                             (
                                 ((3.809 + x) *
                                  (
                                      (2.346 *
                                       ((-2.417 + x) *
                                        (3.809 + (1.906 + 2.385)))) *
                                      (2.346 * -0.9396)
                                  ) +
                                  0.6062
                                 )
                             ) +
                             (
                                 (
                                     ((-2.417 + x) * x) +
                                     (2.346 * (3.809 + 1.33))
                                 ) +
                                 (x * x) +
                                 x
                             )
                         ) *
                         (3.809 +
                          ((x + (0.08281 + x)) + (3.809 + 1.33)))
                     )
                 )) *
                 (
                     ((x + x) * (2.346 * -0.9396)) +
                     ((x + x) + (2.437 * x))
                 )
             ) +
             (
                 (3.809 + x) *
                 (
                     ((x + (0.1279 + x)) * (2.346 * -0.9396)) +
                     (3.918 * -0.9396)
                 )
             ) +
             (
                 (-0.5138 * (x + ((0.3614 + x) + 0.5313))) +
                 ((2.346 * -0.9396) * (3.809 + 1.33))
             ) +
             (
                 -1.891 * (-2.07 + (3.809 + 3.848))
             )
         )
        ) *
        (
            ((x + x) * (2.346 * -0.9396)) +
            ((x + x) + (2.437 * x))
        )
    ))


import math
import torch
from torch import nn, Tensor

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self._compute_d_ff(d_model)

        self.w1 = nn.Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = nn.Linear(self.d_ff, d_model, device=device, dtype=dtype)

    @staticmethod
    def _compute_d_ff(d_model: int) -> int:
        rough = (8 * d_model) / 3
        d_ff = math.ceil(rough / 64) * 64
        return int(d_ff)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        wx1 = self.w1(x)              # (B, T, d_ff)
        wx3 = self.w3(x)              # (B, T, d_ff)
        silu_wx1 = silu(wx1)          # (B, T, d_ff)  <-- FIX
        out = self.w2(silu_wx1 * wx3) # (B, T, d_model)
        return out

import time
import math
import torch
import numpy as np

# -----------------------------
# Config from your model
# -----------------------------
d_model = 256
d_ff = 1344
context_length = 256
batch_size = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# -----------------------------
# Assume these are already defined from previous step:
#   expr_torch(x: torch.Tensor) -> torch.Tensor
#   expr_numpy(x: np.ndarray) -> np.ndarray
# -----------------------------
# from your_poly_module import expr_torch, expr_numpy

# -----------------------------
# SwiGLU module (from above)
# -----------------------------
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self._compute_d_ff(d_model)

        self.w1 = torch.nn.Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = torch.nn.Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = torch.nn.Linear(self.d_ff, d_model, device=device, dtype=dtype)

    @staticmethod
    def _compute_d_ff(d_model: int) -> int:
        rough = (8 * d_model) / 3
        d_ff = math.ceil(rough / 64) * 64
        return int(d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wx1 = self.w1(x)
        wx3 = self.w3(x)
        silu_wx1 = wx1 * torch.sigmoid(wx1)  # SiLU(wx1)
        return self.w2(silu_wx1 * wx3)


# -----------------------------
# Utility: timing + error metrics
# -----------------------------
def rmse_and_se(a: torch.Tensor, b: torch.Tensor):
    """
    Returns:
      rmse: scalar
      se:   sum of squared errors (scalar)
    """
    diff = a - b
    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(mse)
    se = torch.sum(diff ** 2)
    return rmse.item(), se.item()


# -----------------------------
# 1) Create 1000 inputs
# Shape: (batch_size, context_length, d_model)
# -----------------------------
torch.manual_seed(0)
x = torch.randn(batch_size, context_length, d_model, device=device, dtype=dtype)
print(f"Input shape: {tuple(x.shape)}")

# -----------------------------
# 2) Run SwiGLU
# -----------------------------
swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
y_swiglu = swiglu(x)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

swiglu_time = t1 - t0
print(f"SwiGLU time: {swiglu_time:.4f} s, output shape = {tuple(y_swiglu.shape)}")

# -----------------------------
# 3) Polynomial (PyTorch)
# -----------------------------
if device.type == "cuda":
    torch.cuda.synchronize()

t0 = time.perf_counter()
y_poly_torch = expr_torch(x)  # <-- your big polynomial in torch
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()

poly_torch_time = t1 - t0
print(f"Poly (torch) time: {poly_torch_time:.4f} s")

# Error metrics between SwiGLU and Torch polynomial
rmse_torch, se_torch = rmse_and_se(y_swiglu, y_poly_torch)
print(f"[SwiGLU vs poly_torch] RMSE = {rmse_torch:.6e}, SE = {se_torch:.6e}")

# -----------------------------
# 4) Polynomial (NumPy)
#    Note: runs on CPU; we convert x to NumPy and back.
# -----------------------------
x_cpu = x.detach().cpu().numpy().astype(np.float32)

t0 = time.perf_counter()
y_poly_np = expr_numpy(x_cpu)  # <-- your big polynomial in numpy
t1 = time.perf_counter()
poly_numpy_time = t1 - t0
print(f"Poly (numpy) time: {poly_numpy_time:.4f} s")

# bring NumPy result back to torch for comparison
y_poly_np_torch = torch.from_numpy(y_poly_np).to(device=device, dtype=dtype)

rmse_np, se_np = rmse_and_se(y_swiglu, y_poly_np_torch)
print(f"[SwiGLU vs poly_numpy] RMSE = {rmse_np:.6e}, SE = {se_np:.6e}")
