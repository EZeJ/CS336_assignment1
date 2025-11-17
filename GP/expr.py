from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable
import numpy as np
import random


# Allowed operators
OPS = ("add", "mul")


@dataclass
class Expr:
    """Expression tree limited to {+, * , const, var}."""

    op: str
    value: float | None = None
    name: str | None = None
    left: "Expr | None" = None
    right: "Expr | None" = None
    _cached_str: str = field(default=None, init=False, repr=False)

    # --- Constructors ---
    @staticmethod
    def var(name: str) -> "Expr":
        return Expr(op="var", name=name)

    @staticmethod
    def const(val: float) -> "Expr":
        return Expr(op="const", value=float(val))

    # --- Structural properties ---
    def degree(self) -> int:
        if self.op in ("const", "var"):
            return 0 if self.op == "const" else 1
        if self.op == "add":
            return max(self.left.degree(), self.right.degree())
        if self.op == "mul":
            return self.left.degree() + self.right.degree()
        raise ValueError(f"Unknown op {self.op}")

    def multiplies(self) -> int:
        if self.op in ("const", "var"):
            return 0
        return self.left.multiplies() + self.right.multiplies() + (1 if self.op == "mul" else 0)

    def depth(self) -> int:
        if self.op in ("const", "var"):
            return 1
        return 1 + max(self.left.depth(), self.right.depth())

    # --- Evaluation ---
    def eval(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        if self.op == "const":
            return np.asarray(self.value, dtype=np.float32)
        if self.op == "var":
            return np.asarray(inputs[self.name], dtype=np.float32)
        l = self.left.eval(inputs)
        r = self.right.eval(inputs)
        if self.op == "add":
            return l + r
        if self.op == "mul":
            return l * r
        raise ValueError(f"Unknown op {self.op}")

    # --- Mutation helpers ---
    def copy(self) -> "Expr":
        if self.op in ("const", "var"):
            return Expr(op=self.op, value=self.value, name=self.name)
        return Expr(op=self.op, left=self.left.copy(), right=self.right.copy())

    def nodes(self) -> list["Expr"]:
        if self.op in ("const", "var"):
            return [self]
        return [self] + self.left.nodes() + self.right.nodes()

    # --- Pretty printing ---
    def __str__(self) -> str:
        if self._cached_str is not None:
            return self._cached_str
        if self.op == "const":
            s = f"{self.value:.4g}"
        elif self.op == "var":
            s = self.name
        elif self.op == "add":
            s = f"({self.left} + {self.right})"
        elif self.op == "mul":
            s = f"({self.left} * {self.right})"
        else:
            raise ValueError(f"Unknown op {self.op}")
        self._cached_str = s
        return s


def random_const(rng: random.Random, const_range: float) -> Expr:
    return Expr.const(rng.uniform(-const_range, const_range))


def random_var(rng: random.Random, feature_names: list[str]) -> Expr:
    return Expr.var(rng.choice(feature_names))


def random_expr(
    rng: random.Random,
    max_depth: int,
    feature_names: list[str],
    const_range: float = 3.0,
) -> Expr:
    """Generate a random expression up to max_depth."""
    if max_depth <= 1:
        return random_var(rng, feature_names) if rng.random() < 0.5 else random_const(rng, const_range)

    op = rng.choice(OPS)
    left = random_expr(rng, max_depth - 1, feature_names, const_range)
    right = random_expr(rng, max_depth - 1, feature_names, const_range)
    return Expr(op=op, left=left, right=right)


def mutate(
    expr: Expr,
    rng: random.Random,
    feature_names: list[str],
    max_depth: int,
    const_range: float,
) -> Expr:
    """Replace a random subtree."""
    tree = expr.copy()
    nodes = tree.nodes()
    target = rng.choice(nodes)
    replacement = random_expr(rng, rng.randint(1, max_depth), feature_names, const_range)
    target.op = replacement.op
    target.value = replacement.value
    target.name = replacement.name
    target.left = replacement.left
    target.right = replacement.right
    target._cached_str = None
    return tree


def crossover(a: Expr, b: Expr, rng: random.Random) -> tuple[Expr, Expr]:
    """Swap random subtrees between parents."""
    a_copy, b_copy = a.copy(), b.copy()
    a_nodes = [n for n in a_copy.nodes() if n.op not in ("const", "var")]
    b_nodes = [n for n in b_copy.nodes() if n.op not in ("const", "var")]
    if not a_nodes or not b_nodes:
        return a_copy, b_copy

    a_target = rng.choice(a_nodes)
    b_target = rng.choice(b_nodes)

    # Swap by copying
    temp = b_target.copy()
    b_target.op, b_target.value, b_target.name, b_target.left, b_target.right = (
        a_target.op,
        a_target.value,
        a_target.name,
        a_target.left,
        a_target.right,
    )
    a_target.op, a_target.value, a_target.name, a_target.left, a_target.right = (
        temp.op,
        temp.value,
        temp.name,
        temp.left,
        temp.right,
    )
    a_target._cached_str = None
    b_target._cached_str = None
    return a_copy, b_copy
