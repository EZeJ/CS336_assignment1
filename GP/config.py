from dataclasses import dataclass


@dataclass
class FitnessWeights:
    """Weights for the composite fitness score."""

    mean_rel: float = 1.0
    max_rel: float = 0.5
    degree: float = 0.1
    multiplies: float = 0.05
    depth: float = 0.05


@dataclass
class GPConfig:
    """Genetic programming hyperparameters."""

    population_size: int = 64
    generations: int = 100
    tournament_size: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.4
    elite_fraction: float = 0.05
    max_depth: int = 6
    max_terms: int = 32
    const_range: float = 3.0
    seed: int = 42

    fitness_weights: FitnessWeights = FitnessWeights()
