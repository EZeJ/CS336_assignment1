from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .config import FitnessWeights
from .expr import Expr
from .datasets import Dataset


@dataclass
class Metrics:
    loss: float
    mean_rel: float
    max_rel: float
    degree: int
    multiplies: int
    depth: int
    term_count: int


def _safe_rel_err(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    denom = np.maximum(np.abs(target), eps)
    return np.abs(pred - target) / denom


def compute_metrics(
    expr: Expr,
    dataset: Dataset,
    weights: FitnessWeights,
    max_terms: int | None = None,
    term_penalty: float = 1e3,
    eps: float = 1e-6,
) -> Metrics:
    inputs = {name: dataset.inputs[:, i] for i, name in enumerate(dataset.feature_names or ["x"])}
    pred = expr.eval(inputs)
    target = dataset.targets

    rel_err = _safe_rel_err(pred, target, eps=eps)
    mean_rel = float(np.mean(rel_err))
    max_rel = float(np.max(rel_err))

    degree = expr.degree()
    multiplies = expr.multiplies()
    depth = expr.depth()
    term_count = expr.term_count()

    # Original multi-objective loss (kept for reference):
    # loss = (
    #     weights.mean_rel * mean_rel
    #     + weights.max_rel * max_rel
    #     + weights.degree * degree
    #     + weights.multiplies * multiplies
    #     + weights.depth * depth
    # )
    # if max_terms is not None and term_count > max_terms:
    #     loss += term_penalty * (term_count - max_terms)

    # New accuracy-focused loss using RMSE + SE
    sq_err = (pred - target) ** 2
    rmse = float(np.sqrt(np.mean(sq_err)))
    se = float(np.sum(sq_err))
    loss = rmse + se

    return Metrics(
        loss=loss,
        mean_rel=mean_rel,
        max_rel=max_rel,
        degree=degree,
        multiplies=multiplies,
        depth=depth,
        term_count=term_count,
    )
