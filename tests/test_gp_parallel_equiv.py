import numpy as np
from GP.gp import GPSearch
from GP.config import GPConfig
from GP.datasets import Dataset


def test_parallel_and_sequential_same_loss():
    x = np.linspace(-1, 1, 50, dtype=np.float32)
    y = x**2  # simple nonlinear target
    ds = Dataset(inputs=x[:, None], targets=y, feature_names=["x"])
    cfg = GPConfig(population_size=8, generations=5, max_depth=3, max_terms=20, seed=123)

    search_seq = GPSearch(cfg, ds, verbose=False, log_every=2, n_jobs=1)
    best_seq, _ = search_seq.run()

    search_par = GPSearch(cfg, ds, verbose=False, log_every=2, n_jobs=2)
    best_par, _ = search_par.run()

    assert abs(best_seq.metrics.loss - best_par.metrics.loss) < 1e-6
