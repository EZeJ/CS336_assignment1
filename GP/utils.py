from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict
from .config import GPConfig, FitnessWeights


def load_yaml_config(path: str | Path) -> GPConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    fw = cfg.get("fitness_weights", {})
    fitness_weights = FitnessWeights(**fw) if isinstance(fw, dict) else FitnessWeights()
    return GPConfig(
        population_size=cfg.get("population_size", 64),
        generations=cfg.get("generations", 100),
        tournament_size=cfg.get("tournament_size", 5),
        mutation_rate=cfg.get("mutation_rate", 0.3),
        crossover_rate=cfg.get("crossover_rate", 0.4),
        elite_fraction=cfg.get("elite_fraction", 0.05),
        max_depth=cfg.get("max_depth", 6),
        max_terms=cfg.get("max_terms", 32),
        const_range=cfg.get("const_range", 3.0),
        seed=cfg.get("seed", 42),
        target_key=cfg.get("target_key"),
        fitness_weights=fitness_weights,
    )


def load_run_yaml(path: str | Path) -> dict:
    """Load a run-level config YAML (npz path, combine, jobs, etc.)."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def make_run_dir(base: str | Path, prefix: str = "run") -> Path:
    """Create a date_time_id run directory under base."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(run_dir: Path, name: str, payload: Dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)
    return path
