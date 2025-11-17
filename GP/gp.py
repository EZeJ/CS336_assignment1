from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
import sys
from typing import List, Tuple
from .config import GPConfig
from .expr import Expr, random_expr, mutate, crossover
from .fitness import compute_metrics, Metrics
from .datasets import Dataset


@dataclass
class Individual:
    expr: Expr
    metrics: Metrics | None = None

    def __lt__(self, other: "Individual") -> bool:
        # For heapq ordering by loss
        return (self.metrics.loss if self.metrics else float("inf")) < (
            other.metrics.loss if other.metrics else float("inf")
        )


class GPSearch:
    def __init__(self, cfg: GPConfig, dataset: Dataset, verbose: bool = False, log_every: int = 10):
        self.cfg = cfg
        self.dataset = dataset
        self.rng = random.Random(cfg.seed)
        self.verbose = verbose
        self.log_every = log_every
        # Default names if none provided
        if not dataset.feature_names:
            dataset.feature_names = [f"x{i}" for i in range(dataset.inputs.shape[1])]

    def _init_population(self) -> List[Individual]:
        pop = [
            Individual(
                expr=random_expr(
                    rng=self.rng,
                    max_depth=self.cfg.max_depth,
                    feature_names=self.dataset.feature_names,
                    const_range=self.cfg.const_range,
                )
            )
            for _ in range(self.cfg.population_size)
        ]
        return pop

    def _eval(self, ind: Individual) -> Individual:
        if ind.metrics is None:
            ind.metrics = compute_metrics(
                ind.expr,
                self.dataset,
                self.cfg.fitness_weights,
                max_terms=self.cfg.max_terms,
            )
        return ind

    def _select(self, population: List[Individual]) -> Individual:
        tour = self.rng.sample(population, k=self.cfg.tournament_size)
        best = min(tour, key=lambda ind: ind.metrics.loss)
        return best

    def run(self) -> Tuple[Individual, List[Individual]]:
        if self.verbose:
            print(f"Initializing population of size {self.cfg.population_size}...")
            sys.stdout.flush()

        pop = [self._eval(ind) for ind in self._init_population()]
        best_overall = min(pop, key=lambda ind: ind.metrics.loss)

        if self.verbose:
            print(
                f"[init] best_loss={best_overall.metrics.loss:.6f} "
                f"expr={best_overall.expr}"
            )
            sys.stdout.flush()

        elite_count = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))

        for gen in range(self.cfg.generations):
            # Keep elites
            elites = heapq.nsmallest(elite_count, pop)

            new_pop: List[Individual] = [Individual(expr=e.expr.copy(), metrics=e.metrics) for e in elites]

            while len(new_pop) < self.cfg.population_size:
                if self.rng.random() < self.cfg.crossover_rate:
                    p1 = self._select(pop).expr
                    p2 = self._select(pop).expr
                    c1_expr, c2_expr = crossover(p1, p2, self.rng)
                    for child_expr in (c1_expr, c2_expr):
                        if self.rng.random() < self.cfg.mutation_rate:
                            child_expr = mutate(
                                child_expr,
                                rng=self.rng,
                                feature_names=self.dataset.feature_names,
                                max_depth=self.cfg.max_depth,
                                const_range=self.cfg.const_range,
                            )
                        new_pop.append(Individual(expr=child_expr))
                        if len(new_pop) >= self.cfg.population_size:
                            break
                else:
                    p = self._select(pop).expr
                    child_expr = mutate(
                        p,
                        rng=self.rng,
                        feature_names=self.dataset.feature_names,
                        max_depth=self.cfg.max_depth,
                        const_range=self.cfg.const_range,
                    )
                    new_pop.append(Individual(expr=child_expr))

            pop = [self._eval(ind) for ind in new_pop]
            gen_best = min(pop, key=lambda ind: ind.metrics.loss)
            if gen_best.metrics.loss < best_overall.metrics.loss:
                best_overall = gen_best

            if self.verbose and ((gen + 1) % max(1, self.log_every) == 0 or gen == self.cfg.generations - 1):
                print(
                    f"[gen {gen+1}/{self.cfg.generations}] "
                    f"best_loss={gen_best.metrics.loss:.6f} "
                    f"expr={gen_best.expr}"
                )
                sys.stdout.flush()

        return best_overall, pop
