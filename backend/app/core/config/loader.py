from __future__ import annotations

import pathlib
import tomllib

from app.core.config.models import AgentsConfig, SystemConfig


def load_system_config(path: pathlib.Path) -> SystemConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return SystemConfig.model_validate(data)


def load_agents_config(path: pathlib.Path) -> AgentsConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return AgentsConfig.model_validate(data)
