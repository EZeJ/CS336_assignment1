from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
import sys
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from .config import GPConfig
from .expr import Expr, random_expr, mutate, crossover
from .fitness import compute_metrics, Metrics
from .datasets import Dataset


_DATASET_GLOBAL: Dataset | None = None
_WEIGHTS_GLOBAL = None
_MAX_TERMS_GLOBAL: int | None = None


def _init_worker(dataset: Dataset, weights, max_terms: int | None) -> None:
    global _DATASET_GLOBAL, _WEIGHTS_GLOBAL, _MAX_TERMS_GLOBAL
    _DATASET_GLOBAL = dataset
    _WEIGHTS_GLOBAL = weights
    _MAX_TERMS_GLOBAL = max_terms


def _compute_metrics_for_expr(expr: Expr) -> Metrics:
    assert _DATASET_GLOBAL is not None
    return compute_metrics(
        expr,
        _DATASET_GLOBAL,
        _WEIGHTS_GLOBAL,
        max_terms=_MAX_TERMS_GLOBAL,
    )


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
    def __init__(self, cfg: GPConfig, dataset: Dataset, verbose: bool = False, log_every: int = 10, n_jobs: int = 1):
        self.cfg = cfg
        self.dataset = dataset
        self.rng = random.Random(cfg.seed)
        self.verbose = verbose
        self.log_every = log_every
        self.n_jobs = max(1, n_jobs)
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

    def _evaluate_population_sequential(self, pop: List[Individual]) -> List[Individual]:
        return [self._eval(ind) for ind in pop]

    def _evaluate_population_parallel(self, pop: List[Individual], executor: ProcessPoolExecutor) -> List[Individual]:
        exprs = [ind.expr for ind in pop]
        metrics_list = list(executor.map(_compute_metrics_for_expr, exprs))
        for ind, m in zip(pop, metrics_list):
            ind.metrics = m
        return pop

    def run(self) -> Tuple[Individual, List[Individual]]:
        if self.verbose:
            print(f"Initializing population of size {self.cfg.population_size}...")
            sys.stdout.flush()

        if self.n_jobs == 1:
            pop = self._evaluate_population_sequential(self._init_population())
            best_overall = min(pop, key=lambda ind: ind.metrics.loss)

            if self.verbose:
                print(
                    f"[init] best_loss={best_overall.metrics.loss:.6f} "
                    f"expr={best_overall.expr}"
                )
                sys.stdout.flush()

            elite_count = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))

            for gen in range(self.cfg.generations):
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

                pop = self._evaluate_population_sequential(new_pop)
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

        else:
            with ProcessPoolExecutor(
                max_workers=self.n_jobs,
                initializer=_init_worker,
                initargs=(self.dataset, self.cfg.fitness_weights, self.cfg.max_terms),
            ) as ex:
                pop = self._evaluate_population_parallel(self._init_population(), ex)
                best_overall = min(pop, key=lambda ind: ind.metrics.loss)

                if self.verbose:
                    print(
                        f"[init] best_loss={best_overall.metrics.loss:.6f} "
                        f"expr={best_overall.expr}"
                    )
                    sys.stdout.flush()

                elite_count = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))

                for gen in range(self.cfg.generations):
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

                    pop = self._evaluate_population_parallel(new_pop, ex)
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
