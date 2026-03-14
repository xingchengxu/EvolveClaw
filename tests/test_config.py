"""Tests for configuration loading and deep merge."""
import tempfile
from pathlib import Path
import pytest
from evolveclaw_ramsey.utils.config import load_config, _deep_merge


def _write_yaml(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_load_config_with_defaults():
    path = _write_yaml("problem:\n  s: 5\n  t: 5\n")
    config = load_config(path)
    assert config["problem"]["s"] == 5
    assert config["problem"]["t"] == 5
    # Defaults filled in
    assert config["problem"]["n"] == 17
    assert config["evolution"]["max_generations"] == 100
    assert config["seed"] == 42


def test_load_config_overrides_defaults():
    path = _write_yaml("problem:\n  s: 3\n  t: 3\n  n: 10\nseed: 99\n")
    config = load_config(path)
    assert config["problem"]["n"] == 10
    assert config["seed"] == 99


def test_load_config_missing_problem_raises():
    path = _write_yaml("seed: 42\n")
    with pytest.raises(ValueError, match="Missing required config section"):
        load_config(path)


def test_deep_merge_nested():
    defaults = {"a": {"x": 1, "y": 2}, "b": 3}
    overrides = {"a": {"y": 99}, "c": 4}
    result = _deep_merge(defaults, overrides)
    assert result == {"a": {"x": 1, "y": 99}, "b": 3, "c": 4}


def test_deep_merge_override_non_dict():
    defaults = {"a": {"x": 1}}
    overrides = {"a": "replaced"}
    result = _deep_merge(defaults, overrides)
    assert result["a"] == "replaced"
