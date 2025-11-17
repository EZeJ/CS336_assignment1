from pathlib import Path
from GP.utils import load_yaml_config
from GP.config import GPConfig


def test_load_yaml_config_reads_target_key(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "seed: 1\n"
        "population_size: 10\n"
        "generations: 20\n"
        "target_key: silu_in\n"
    )
    cfg = load_yaml_config(cfg_path)
    assert isinstance(cfg, GPConfig)
    assert cfg.target_key == "silu_in"
