"""
Genetic programming tools for HE-friendly polynomial surrogates.

Modules:
- config: configs and defaults.
- datasets: utilities for loading/splitting GP fitness datasets.
- expr: expression trees restricted to {+, * , const, var}.
- fitness: metric computation (error, degree, multiply count, depth).
- gp: simple GP search loop (selection/mutation/crossover).
- search: CLI entrypoint to run a GP search over provided datasets.
"""

from .config import GPConfig, FitnessWeights
from .datasets import Dataset, load_dataset
from .expr import Expr, random_expr
from .fitness import compute_metrics
from .gp import GPSearch

__all__ = [
    "GPConfig",
    "FitnessWeights",
    "Dataset",
    "load_dataset",
    "Expr",
    "random_expr",
    "compute_metrics",
    "GPSearch",
]
