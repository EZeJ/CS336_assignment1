from pathlib import Path
from GP.utils import load_yaml_config, load_run_yaml


def test_silu_complex_hparams():
    cfg_path = Path("GP/configs/silu_complex.yaml")
    cfg = load_yaml_config(cfg_path)
    assert cfg.population_size == 256
    assert cfg.generations == 1000
    assert cfg.max_depth == 18
    assert cfg.max_terms == 256
    assert cfg.target_key == "silu_in"


def test_silu_complex_train_config():
    run_path = Path("GP/train_configs/silu_complex_train.yaml")
    run_cfg = load_run_yaml(run_path)
    assert run_cfg["gp_config"].endswith("silu_complex.yaml")
    assert run_cfg["npz"].endswith("activations_all.npz")
    assert run_cfg["combine"] is True
    assert run_cfg["jobs"] == 4
    assert run_cfg["verbose"] is True
    assert run_cfg["log_every"] == 10
    assert run_cfg["target_key"] == "silu_in"

