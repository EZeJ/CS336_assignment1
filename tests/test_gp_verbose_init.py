from GP.gp import GPSearch
from GP.config import GPConfig
from GP.datasets import Dataset
import numpy as np


def test_gpsource_prints_init_best(capsys):
    # small dummy dataset
    x = np.linspace(-1, 1, 10, dtype=np.float32)
    y = x  # identity target
    ds = Dataset(inputs=x[:, None], targets=y, feature_names=["x"])
    # Ensure tournament_size <= population_size
    cfg = GPConfig(population_size=4, generations=0, max_depth=2, max_terms=10, tournament_size=2)

    search = GPSearch(cfg, ds, verbose=True, log_every=1)
    # Only initialization happens when generations=0
    search.run()

    captured = capsys.readouterr().out
    assert "Initializing population of size 4" in captured
    assert "[init] best_loss=" in captured
