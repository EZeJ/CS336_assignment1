"""
Aggregate raw activation logs into GP-ready datasets.

Reads NPZ shards (per-epoch logs) containing signals like:
- silu_in: pre-activation for SiLU
- gate_u: gate pre-activation (same domain as silu_in)
- gate_v: up projection output (optionally used for joint fits)
- rms_mean_sq: per-token mean square before RMSNorm scale

Outputs NPZ datasets with fields:
- inputs: shape (N, 1)
- targets: shape (N,)
- feature_names: ["x"]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from typing import Callable, Dict


EPS = 1e-6


def silu_fn(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def rms_inv_sqrt_fn(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(x + EPS)


TARGET_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "silu_in": silu_fn,
    "gate_u": silu_fn,  # same as silu input
    "rms_mean_sq": rms_inv_sqrt_fn,
}


def aggregate_raw_logs(raw_dir: str | Path, out_dir: str | Path, signals=None) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if signals is None:
        signals = TARGET_FUNCS.keys()

    collected = {s: [] for s in signals if s in TARGET_FUNCS}

    for npz_path in sorted(raw_dir.glob("**/*.npz")):
        data = np.load(npz_path)
        for sig in signals:
            if sig in data and sig in TARGET_FUNCS:
                collected[sig].append(data[sig])

    outputs: dict[str, Path] = {}
    for sig, chunks in collected.items():
        if not chunks:
            continue
        x = np.concatenate(chunks).astype(np.float32)
        y = TARGET_FUNCS[sig](x).astype(np.float32)
        out_path = out_dir / f"{sig}_dataset.npz"
        np.savez_compressed(out_path, inputs=x[:, None], targets=y, feature_names=np.array(["x"]))
        outputs[sig] = out_path
    return outputs


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate raw GP logs into datasets.")
    p.add_argument("--raw", required=True, help="Directory with raw NPZ logs (epoch shards).")
    p.add_argument("--out", required=True, help="Output directory for aggregated datasets.")
    p.add_argument("--signals", nargs="*", default=None, help="Signals to aggregate (default: known signals).")
    return p.parse_args()


def main():
    args = parse_args()
    outputs = aggregate_raw_logs(args.raw, args.out, signals=args.signals)
    for sig, path in outputs.items():
        print(f"Saved {sig} dataset to {path}")


if __name__ == "__main__":
    main()
