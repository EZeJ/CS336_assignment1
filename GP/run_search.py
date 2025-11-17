"""
Run GP search over activations_all.npz and save checkpoints/results.

Usage:
    uv run python -m GP.run_search --config ./GP/configs/default.yaml --npz ./GP/datasets/raw/20251116_210338/activations_all.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from dataclasses import asdict
from GP.utils import load_yaml_config, make_run_dir, save_checkpoint
from GP.datasets import Dataset
from GP.gp import GPSearch
from GP.expr import Expr
from GP.aggregate_GP import TARGET_FUNCS


def _resolve_target_func(sig: str):
    """
    Map signal names (possibly prefixed with layer info) to target functions.
    Falls back to identity.
    """
    if sig in TARGET_FUNCS:
        return TARGET_FUNCS[sig]

    for base in TARGET_FUNCS:
        if sig.endswith(base):
            return TARGET_FUNCS[base]
    return lambda z: z


def build_dataset_from_signal(npz: dict, sig: str) -> Dataset:
    x = np.array(npz[sig]).astype(np.float32)
    fn = _resolve_target_func(sig)
    y = fn(x).astype(np.float32)
    return Dataset(inputs=x[:, None], targets=y, feature_names=["x"])


def collect_signals(npz: dict, target_key: str) -> list[str]:
    """
    Collect all signals whose name ends with target_key.
    For example, target_key='rms_mean_sq' will gather all layer-prefixed variants.
    """
    return [k for k in npz.files if k.endswith(target_key)]


def main():
    parser = argparse.ArgumentParser(description="Run GP search on activations NPZ.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--npz", required=True, help="Path to activations_all.npz.")
    parser.add_argument("--signals", nargs="*", default=None, help="Signals to fit; default = all non-epoch keys.")
    parser.add_argument("--target-key", default=None, help="If set, train on all signals ending with this key (e.g., rms_mean_sq).")
    parser.add_argument("--out-dir", default="./GP/checkpoints", help="Base checkpoints directory.")
    parser.add_argument("--verbose", action="store_true", help="Print per-generation best during search.")
    parser.add_argument("--log-every", type=int, default=10, help="Generations between verbose logs.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    npz_path = Path(args.npz)
    data = np.load(npz_path)
    if args.target_key:
        signals = collect_signals(data, args.target_key)
    else:
        signals = args.signals or [k for k in data.files if not k.endswith("_epoch")]

    run_dir = make_run_dir(args.out_dir, prefix="gp")
    # Save config snapshot
    save_checkpoint(run_dir, "config", asdict(cfg))

    results = []
    for sig in signals:
        ds = build_dataset_from_signal(data, sig)
        search = GPSearch(cfg, ds, verbose=args.verbose, log_every=args.log_every)
        best, _ = search.run()
        res = {
            "signal": sig,
            "expression": str(best.expr),
            "metrics": asdict(best.metrics),
        }
        results.append(res)
        save_checkpoint(run_dir, f"best_{sig}", res)
        print(f"[{sig}] best expr: {res['expression']}, metrics: {res['metrics']}")

    # Save aggregate results
    save_checkpoint(run_dir, "results", {"signals": results, "npz": str(npz_path)})


if __name__ == "__main__":
    main()
