import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def expr_torch(x):
    # Accept Python floats / lists as well
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    # return (-0.1248 * ((((-1.047 * x) + ((2.346 * (x + 3.576)) * ((2.346 * -0.9396) * (3.809 + 1.33)))) + ((((-0.1248 * (((((2.437 * x) * ((2.346 * -0.9396) * 3.221)) + (((2.346 * -0.9396) * (((3.809 + x) * (((2.346 * ((-2.417 + x) * x)) * -2.892) + (2.437 * x))) + (2.437 * x))) + -2.236)) + ((1.494 + (((3.809 + x) * (((2.346 * ((-2.417 + x) * (3.809 + (1.906 + 2.385)))) * (2.346 * -0.9396)) + (3.313 + -2.083))) + (((((-3.206 + x) * (x + (-0.8878 + x))) + (-2.417 + x)) + (x * x)) + (x * -2.666)))) * (3.809 + ((x + (0.1279 + x)) + (3.809 + 1.33))))) * (((x + x) * (2.346 * -0.9396)) + ((x + x) + (2.437 * x))))) + ((3.809 + x) * (((x + ((0.3614 + x) + 0.5313)) * (2.346 * -0.9396)) + (2.346 * -0.9396)))) + ((-0.5138 * (x + (((x + 0.6101) + 0.5313) + 0.5313))) + ((2.346 * -0.9396) * (3.809 + 1.33)))) + (-1.891 * (-2.07 + (3.809 + 3.848))))) * (((x + x) * (2.346 * -0.9396)) + ((x + x) + (2.437 * x)))))
    return (-0.1248 * ((((-1.047 * x) + ((2.346 * (x + 3.576)) * ((2.346 * -0.9396) * (3.809 + 1.33)))) + ((((-0.1248 * (((((2.437 * x) * ((2.346 * -0.9396) * 3.221)) + (((2.346 * -0.9396) * (((3.809 + x) * (((2.346 * ((-2.417 + x) * x)) * -2.892) + (2.437 * x))) + (2.437 * x))) + -2.236)) + ((1.494 + (((3.809 + x) * (((2.346 * ((-2.417 + x) * (3.809 + (1.906 + 2.385)))) * (2.346 * -0.9396)) + (3.313 + -2.083))) + (((((-3.206 + x) * (x + (-0.8878 + x))) + (-3.266 + x)) + (x * x)) + (x * -2.666)))) * (3.809 + ((x + (0.1279 + x)) + (3.809 + 1.33))))) * (((x + x) * (2.346 * -0.9396)) + ((x + x) + (2.437 * x))))) + ((3.809 + x) * (((x + ((0.3614 + x) + 0.5313)) * (2.346 * -0.9396)) + (2.346 * -0.9396)))) + ((-0.5138 * (x + (((x + 0.6101) + 0.5313) + 0.5313))) + ((2.346 * -0.9396) * (3.809 + 1.33)))) + (-1.891 * (-2.07 + (3.809 + 3.848))))) * (((x + x) * (2.346 * -0.9396)) + ((x + x) + (2.437 * x)))))


# -----------------------------
# Chebyshev approximation of SiLU
# (gate used inside SwiGLU), fit
# on x in [-6, 6] and converted
# to a power-series polynomial.
# Coefficients were generated via
# numpy.polynomial.Chebyshev.fit.
# -----------------------------
_CHEB_SILU_COEFFS_LIST = [
    1.58679809e-02,
    5.00000000e-01,
    2.19307009e-01,
    -1.06282049e-15,
    -9.91605229e-03,
    8.46262408e-17,
    2.77574206e-04,
    -2.73674466e-18,
    -3.00910333e-06,
    3.09898453e-20,
]

CHEB_SILU_COEFFS_TORCH = torch.tensor(_CHEB_SILU_COEFFS_LIST, dtype=torch.float32)


def cheb_silu_torch(x):
    """
    Chebyshev-based polynomial approximation to SiLU on [-6, 6].
    Evaluated via Horner's rule in torch.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    # Clamp to fit domain used for the approximation.
    x_clamped = torch.clamp(x, -6.0, 6.0)
    coeffs = CHEB_SILU_COEFFS_TORCH.to(device=x_clamped.device, dtype=x_clamped.dtype)

    y = torch.zeros_like(x_clamped)
    for c in reversed(coeffs):
        y = y * x_clamped + c
    return y


import math
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


def mean_max_rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """
    Mean/max relative error, matching GP's _safe_rel_err definition.
    """
    denom = torch.maximum(
        torch.abs(b),
        torch.tensor(eps, dtype=b.dtype, device=b.device),
    )
    rel = torch.abs(a - b) / denom
    mean_rel = torch.mean(rel)
    max_rel = torch.max(rel)
    return mean_rel.item(), max_rel.item()


# -----------------------------
# 1) Load real SiLU inputs from activations_all.npz
#    We gather all signals whose name ends with 'silu_in'
#    (e.g., layer0_silu_in, layer1_silu_in) and concatenate.
# -----------------------------
this_dir = Path(__file__).resolve().parent
npz_path = this_dir.parent / "GP" / "datasets" / "raw" / "20251116_210338" / "activations_all.npz"
data_npz = np.load(npz_path, allow_pickle=True)

silu_keys = [k for k in data_npz.files if k.endswith("silu_in") and not k.endswith("_epoch")]
if not silu_keys:
    raise RuntimeError(f"No *silu_in signals found in {npz_path}")

x_np = np.concatenate([data_npz[k].astype(np.float32) for k in silu_keys])
print("=== Loaded SiLU inputs ===")
print(f"  path        : {npz_path}")
print(f"  signals     : {silu_keys}")
print(f"  total points: {x_np.shape[0]:,}")

x_full = torch.from_numpy(x_np).to(device=device, dtype=dtype)
print(f"  tensor      : shape={tuple(x_full.shape)}, device={device}, dtype={dtype}")

# -----------------------------
# 2) Single-run summary on full dataset
# -----------------------------
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
y_silu_full = silu(x_full)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
silu_time_full = t1 - t0

if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
y_poly_full = expr_torch(x_full)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
poly_time_full = t1 - t0

if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
y_cheb_full = cheb_silu_torch(x_full)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
cheb_time_full = t1 - t0

rmse_poly_full, se_poly_full = rmse_and_se(y_silu_full, y_poly_full)
rmse_cheb_full, se_cheb_full = rmse_and_se(y_silu_full, y_cheb_full)
rmse_poly_vs_cheb_full, se_poly_vs_cheb_full = rmse_and_se(y_poly_full, y_cheb_full)

mean_rel_poly_full, max_rel_poly_full = mean_max_rel(y_poly_full, y_silu_full)
mean_rel_cheb_full, max_rel_cheb_full = mean_max_rel(y_cheb_full, y_silu_full)

# GP-style error-only loss (ignoring degree/multiplies/depth terms)
MEAN_REL_WEIGHT = 1.0
MAX_REL_WEIGHT = 0.5
loss_poly_full = MEAN_REL_WEIGHT * mean_rel_poly_full + MAX_REL_WEIGHT * max_rel_poly_full
loss_cheb_full = MEAN_REL_WEIGHT * mean_rel_cheb_full + MAX_REL_WEIGHT * max_rel_cheb_full

print("\n=== Single-run summary on full dataset ===")
print(f"  SiLU time      : {silu_time_full:.4f} s")
print(f"  GP poly time   : {poly_time_full:.4f} s")
print(f"  Cheb poly time : {cheb_time_full:.4f} s")
print("  Errors (RMSE / SE):")
print(f"    GP poly   : {rmse_poly_full:.4e} / {se_poly_full:.4e}")
print(f"    Chebyshev : {rmse_cheb_full:.4e} / {se_cheb_full:.4e}")
print(f"    GP vs Cheb: {rmse_poly_vs_cheb_full:.4e} / {se_poly_vs_cheb_full:.4e}")
print("  Relative errors (mean_rel / max_rel):")
print(f"    GP poly   : {mean_rel_poly_full:.4e} / {max_rel_poly_full:.4e}")
print(f"    Chebyshev : {mean_rel_cheb_full:.4e} / {max_rel_cheb_full:.4e}")
print("  GP-style error loss (mean_rel + 0.5 * max_rel):")
print(f"    GP poly   : {loss_poly_full:.4e}")
print(f"    Chebyshev : {loss_cheb_full:.4e}")

# -----------------------------
# 3) Multi-run evaluation on random subsets
# -----------------------------
num_trials = 4
sample_size = 250_000
sample_size = min(sample_size, x_np.shape[0])
rng = np.random.default_rng(seed=42)

rmse_results = {"gp": [], "cheb": []}
se_results = {"gp": [], "cheb": []}
mean_rel_results = {"gp": [], "cheb": []}
max_rel_results = {"gp": [], "cheb": []}

print(f"\n=== Multi-run subset evaluation ===")
print(f"  trials         : {num_trials}")
print(f"  sample size    : {sample_size:,} per trial")

for trial in range(num_trials):
    idx = rng.choice(x_np.shape[0], size=sample_size, replace=False)
    x_trial_np = x_np[idx]
    x_trial = torch.from_numpy(x_trial_np).to(device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y_silu = silu(x_trial)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    silu_time = t1 - t0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y_poly = expr_torch(x_trial)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    poly_time = t1 - t0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y_cheb = cheb_silu_torch(x_trial)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    cheb_time = t1 - t0

    rmse_poly, se_poly = rmse_and_se(y_silu, y_poly)
    rmse_cheb, se_cheb = rmse_and_se(y_silu, y_cheb)

    mean_rel_poly, max_rel_poly = mean_max_rel(y_poly, y_silu)
    mean_rel_cheb, max_rel_cheb = mean_max_rel(y_cheb, y_silu)

    rmse_results["gp"].append(rmse_poly)
    rmse_results["cheb"].append(rmse_cheb)
    se_results["gp"].append(se_poly)
    se_results["cheb"].append(se_cheb)
    mean_rel_results["gp"].append(mean_rel_poly)
    mean_rel_results["cheb"].append(mean_rel_cheb)
    max_rel_results["gp"].append(max_rel_poly)
    max_rel_results["cheb"].append(max_rel_cheb)

    print(
        f"  trial {trial+1}: "
        f"SiLU {silu_time:.4f}s, "
        f"GP {poly_time:.4f}s (RMSE {rmse_poly:.3e}, mean_rel {mean_rel_poly:.2e}), "
        f"Cheb {cheb_time:.4f}s (RMSE {rmse_cheb:.3e}, mean_rel {mean_rel_cheb:.2e})"
    )

rmse_gp = np.array(rmse_results["gp"])
rmse_cheb = np.array(rmse_results["cheb"])
se_gp = np.array(se_results["gp"])
se_cheb = np.array(se_results["cheb"])
mean_rel_gp = np.array(mean_rel_results["gp"])
mean_rel_cheb = np.array(mean_rel_results["cheb"])
max_rel_gp = np.array(max_rel_results["gp"])
max_rel_cheb = np.array(max_rel_results["cheb"])

print("\n=== Aggregate error statistics over trials ===")
print("  RMSE (mean ± std):")
print(f"    GP poly   : {rmse_gp.mean():.4e} ± {rmse_gp.std():.4e}")
print(f"    Chebyshev : {rmse_cheb.mean():.4e} ± {rmse_cheb.std():.4e}")
print("  SE (mean ± std):")
print(f"    GP poly   : {se_gp.mean():.4e} ± {se_gp.std():.4e}")
print(f"    Chebyshev : {se_cheb.mean():.4e} ± {se_cheb.std():.4e}")
print("  mean_rel (mean ± std):")
print(f"    GP poly   : {mean_rel_gp.mean():.4e} ± {mean_rel_gp.std():.4e}")
print(f"    Chebyshev : {mean_rel_cheb.mean():.4e} ± {mean_rel_cheb.std():.4e}")
print("  max_rel (mean ± std):")
print(f"    GP poly   : {max_rel_gp.mean():.4e} ± {max_rel_gp.std():.4e}")
print(f"    Chebyshev : {max_rel_cheb.mean():.4e} ± {max_rel_cheb.std():.4e}")

# -----------------------------
# 4) Boxplots for RMSE and SE
# -----------------------------
labels = ["GP poly", "Chebyshev"]

plt.figure(figsize=(6, 4))
plt.boxplot([rmse_gp, rmse_cheb], labels=labels, showmeans=True)
plt.ylabel("RMSE (SiLU vs approximation)")
plt.title(f"SiLU approximation RMSE over {num_trials} random subsets")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
rmse_plot_path = "silu_rmse_boxplot.png"
plt.savefig(rmse_plot_path, dpi=150)

plt.figure(figsize=(6, 4))
plt.boxplot([se_gp, se_cheb], labels=labels, showmeans=True)
plt.ylabel("SE (sum of squared errors)")
plt.yscale("log")
plt.title(f"SiLU approximation SE over {num_trials} random subsets")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
se_plot_path = "silu_se_boxplot.png"
plt.savefig(se_plot_path, dpi=150)

print("\nSaved boxplots:")
print(f"  RMSE: {rmse_plot_path}")
print(f"  SE  : {se_plot_path}")
