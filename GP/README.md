# GP Experiments

This folder will hold the genetic-programming pipeline for learning HE-friendly polynomial/piecewise surrogates (SiLU/SwiGLU gate and RMSNorm inv-sqrt).

Planned contents:
- `datasets/`: sampled activation traces and grids used for fitness evaluation.
- `search/`: GP/CGP/GGP implementations plus fitness/penalty functions (error, degree, multiply count, depth).
- `eval/`: scripts to benchmark candidate polynomials against baselines (Remez/Chebyshev) and produce reports.
- `integration/`: drop-in PyTorch modules that swap SiLU/RMSNorm with learned surrogates for simulation and fine-tuning.

Primary goal (from the proposal): evolve low-depth {+,\*}-only approximations that keep perplexity within ~10% while reducing multiplicative depth and HE latency.

## Current scaffolding
- `config.py`: GP hyperparameters and fitness weights.
- `datasets.py`: load NPZ (`inputs`, `targets`, optional `feature_names`) and split.
- `expr.py`: expression trees restricted to {const, var, add, mul}; random generation, mutation, crossover, degree/depth/multiply counts.
- `fitness.py`: compute composite loss (mean/max relative error + complexity penalties).
- `gp.py`: simple tournament-selection GP loop with elites.
- `search.py`: CLI runner.
- `run_search.py`: YAML-driven search runner that loads an activations NPZ and saves checkpoints per signal.
- `configs/default.yaml`: default hyperparameters.
- `utils.py`: YAML config loader, run-dir creation, checkpoint writer.

Example:
```bash
uv run python -m GP.search --data ./GP/datasets/silu_grid.npz --gen 200 --pop 128 --out ./GP/results/silu_best.json

uv run python -m GP.run_search --config ./GP/configs/default.yaml --npz ./GP/datasets/raw/20251116_210338/activations_all.npz --verbose --log-every 20
```
