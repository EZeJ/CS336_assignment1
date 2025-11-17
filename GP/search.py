"""
CLI entrypoint for running GP search on numpy datasets.

Example:
    uv run python -m GP.search --data ./GP/datasets/silu_grid.npz --gen 200 --pop 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict
from .config import GPConfig, FitnessWeights
from .datasets import load_dataset
from .gp import GPSearch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genetic programming search for polynomial surrogates.")
    p.add_argument("--data", required=True, help="Path to NPZ with inputs/targets.")
    p.add_argument("--pop", type=int, default=64, help="Population size.")
    p.add_argument("--gen", type=int, default=100, help="Number of generations.")
    p.add_argument("--depth", type=int, default=6, help="Maximum tree depth.")
    p.add_argument("--terms", type=int, default=32, help="Maximum terms for pruning.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--out", type=str, default=None, help="Path to save best expression and metrics as JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_dataset(args.data)

    cfg = GPConfig(
        population_size=args.pop,
        generations=args.gen,
        max_depth=args.depth,
        max_terms=args.terms,
        seed=args.seed,
        fitness_weights=FitnessWeights(),
    )

    search = GPSearch(cfg, ds)
    best, _ = search.run()

    result = {
        "expression": str(best.expr),
        "metrics": asdict(best.metrics),
        "config": asdict(cfg),
        "feature_names": ds.feature_names,
        "data_path": str(Path(args.data).resolve()),
    }

    print("Best expression:", result["expression"])
    print("Metrics:", json.dumps(result["metrics"], indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
