# Execution Plan: HE-Friendly GP Surrogates for SiLU/SwiGLU and RMSNorm

Goal: Evolve {+,\*}-only polynomial/piecewise surrogates for SiLU/SwiGLU gating and RMSNorm inv-sqrt that keep LM perplexity within ~10% while reducing multiplicative depth and HE latency.

## Workstream 1: Data Collection
- Instrument `SwiGLU.forward` to log gate pre-activations (`u=xW_gate`, `v=xW_up`) and SiLU inputs; instrument `RMSNorm.forward` to log per-token mean square. Downsample per batch (e.g., cap 4k tokens/layer/step).
- Run short training/eval with `valid.yaml` to capture realistic distributions; save NPZ traces + percentiles manifests.
- Build GP fitness datasets:
  - Dense grids over domains derived from P0.5/P99.5 (expand ±20%).
  - Empirical samples from traces. Save as `GP/datasets/{silu,swiglu_gate,rmsinv}_grid.npz` and `_data.npz`.

## Workstream 2: Baselines
- Chebyshev/Remez polynomial fits for SiLU and inv-sqrt on chosen domains; record degree, max/mean rel error, multiply/depth.
- Optional piecewise (2–4 segments) least-squares fits to compare against GP.

## Workstream 3: GP Search
- Use `GP.search` on grid + empirical datasets with multi-objective loss (mean/max rel + degree/multiplies/depth penalties). Sweep seeds, population sizes, and max depths.
- Maintain Pareto front snapshots: (error vs. multiplies), (error vs. depth).
- Export best candidates to JSON (expr string + metrics).

## Workstream 4: Integration & Simulation
- Implement drop-in PyTorch modules:
  - `PolySiLU`: evaluates GP/Chebyshev polynomial.
  - `PolyRMSNorm`: replaces inv-sqrt with polynomial.
- Swap into the transformer; run validation perplexity (no fine-tune) to measure degradation.
- Fine-tune HE-approx model for a few epochs to recover accuracy; log PPL delta.

## Workstream 5: HE-Oriented Metrics
- Count multiplies and multiplicative depth analytically for each surrogate.
- Simulate CKKS cost: rotation/multiplication counts for a block; estimate ciphertext modulus budget.
- If toolchain available, run a single block under CKKS to measure wall-clock and depth feasibility.

## Experiments (minimum set)
1) **Approx quality:** GP vs. Chebyshev/Remez for SiLU and inv-sqrt (max/mean rel error on grid+data).  
2) **Model impact:** Perplexity on valid set with exact vs. surrogate, before/after brief fine-tune.  
3) **Complexity:** Multiply/depth counts; size of expressions.  
4) **Ablations:** GP family sweep (POPS: population/gen/depth/penalties); piecewise vs. single polynomial; RMSNorm surrogate vs. short Newton iteration (add/mul only).  
5) **HE cost:** Estimated depth/ops; optional CKKS timing for a block.

## Visualizations
- Error curves over input domain (SiLU, inv-sqrt): GP vs. Chebyshev vs. piecewise.
- Pareto fronts: (max rel error vs. multiplies), (max rel error vs. depth).
- Perplexity bars/lines: exact vs. surrogate (pre/post fine-tune).
- Expression size distributions across GP runs.
- HE-cost bar chart: multiplications/rotations/depth per surrogate; optional runtime if measured.

## Deliverables
- Datasets + manifests under `GP/datasets/`.
- Surrogate specs (JSON) and integration modules.
- Experiment scripts/notebooks to reproduce runs, plots saved to `GP/results/`.
- Report section summarizing error/PPL/HE metrics with figures.
