"""Configuration loading and validation."""
from __future__ import annotations
from pathlib import Path
import yaml

_DEFAULTS = {
    "problem": {"s": 4, "t": 4, "n": 17, "penalty_weight": 1.0},
    "evolution": {"max_generations": 100, "population_size": 20, "tournament_k": 3, "checkpoint_interval": 10},
    "proposer": {"type": "random"},
    "executor": {"timeout_seconds": 10},
    "logging": {"level": "INFO"},
    "seed": 42,
    "run_dir": "runs/",
}
_REQUIRED = ["problem"]

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    for key in _REQUIRED:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    return _deep_merge(_DEFAULTS, config)

def _deep_merge(defaults: dict, overrides: dict) -> dict:
    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
