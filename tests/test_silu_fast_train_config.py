from pathlib import Path
from GP.utils import load_run_yaml


def test_silu_fast_train_config_loads():
    cfg_path = Path("GP/train_configs/silu_fast_train.yaml")
    cfg = load_run_yaml(cfg_path)

    assert cfg["gp_config"].endswith("silu_fast.yaml")
    assert cfg["npz"].endswith("activations_all.npz")
    assert cfg["combine"] is True
    assert cfg["jobs"] == 2
    assert cfg["verbose"] is True
    assert cfg["log_every"] == 5
    assert cfg["target_key"] == "silu_in"
