from pathlib import Path
from GP.utils import load_run_yaml


def test_load_run_yaml_fields(tmp_path: Path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(
        "gp_config: ./GP/configs/silu_1000.yaml\n"
        "npz: ./GP/datasets/raw/some_run/activations_all.npz\n"
        "combine: true\n"
        "jobs: 4\n"
        "verbose: true\n"
        "log_every: 5\n"
        "out_dir: ./GP/checkpoints\n"
        "target_key: silu_in\n"
        "signals:\n"
        "  - layer0_silu_in\n"
    )
    cfg = load_run_yaml(cfg_path)
    assert cfg["gp_config"].endswith("silu_1000.yaml")
    assert cfg["npz"].endswith("activations_all.npz")
    assert cfg["combine"] is True
    assert cfg["jobs"] == 4
    assert cfg["verbose"] is True
    assert cfg["log_every"] == 5
    assert cfg["out_dir"].endswith("GP/checkpoints")
    assert cfg["target_key"] == "silu_in"
    assert cfg["signals"] == ["layer0_silu_in"]

