"""Typed project configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class PathConfig(BaseModel):
    """Project-relative storage locations."""

    model_config = ConfigDict(extra="forbid")

    data: Path = Path("data")
    artifacts: Path = Path("artifacts")
    reports: Path = Path("reports")


class ModelConfig(BaseModel):
    """Pretrained encoder identifiers and pooling strategy."""

    model_config = ConfigDict(extra="forbid")

    sequence: str = "facebook/esm2_t6_8M_UR50D"
    structure: str = "esm_if1_gvp4_t16_142M_UR50"
    pooling: str = "mean"


class ProjectConfig(BaseModel):
    """Top-level AegisFold configuration."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "AegisFold"
    random_seed: int = 42
    paths: PathConfig = PathConfig()
    models: ModelConfig = ModelConfig()


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a YAML configuration file."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ProjectConfig.model_validate(raw)

