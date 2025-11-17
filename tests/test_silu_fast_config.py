from pathlib import Path
from GP.utils import load_yaml_config


def test_silu_fast_config_values(tmp_path: Path):
    # Use the real config file
    cfg_path = Path("GP/configs/silu_fast.yaml")
    cfg = load_yaml_config(cfg_path)

    assert cfg.population_size == 64
    assert cfg.generations == 50
    assert cfg.max_depth == 6
    assert cfg.max_terms == 64
    assert cfg.target_key == "silu_in"

