from pathlib import Path

from aegisfold.config import load_config


def test_default_config_loads() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.project_name == "AegisFold"
    assert config.random_seed == 42
    assert config.models.sequence == "facebook/esm2_t6_8M_UR50D"

