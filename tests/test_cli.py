"""Integration tests for CLI subcommands: run, eval, replay."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml
import pytest


def _write_config(tmpdir, **overrides):
    config = {
        "problem": {"s": 3, "t": 3, "n": 6, "penalty_weight": 1.0},
        "evolution": {"max_generations": 3, "population_size": 3, "tournament_k": 2, "checkpoint_interval": 3},
        "proposer": {"type": "random"},
        "executor": {"timeout_seconds": 5},
        "logging": {"level": "WARNING"},
        "seed": 42,
        "run_dir": str(tmpdir),
    }
    config.update(overrides)
    config_path = Path(tmpdir) / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


def test_cli_run():
    """Test that 'run' subcommand executes and produces output."""
    tmpdir = tempfile.mkdtemp()
    config_path = _write_config(tmpdir)
    result = subprocess.run(
        [sys.executable, "-m", "evolveclaw_ramsey", "run", "--config", config_path],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "Run complete" in result.stdout
    assert "Best score" in result.stdout


def test_cli_eval():
    """Test that 'eval' subcommand evaluates a strategy JSON."""
    tmpdir = tempfile.mkdtemp()
    strategy_path = Path(tmpdir) / "strategy.json"
    with open(strategy_path, "w") as f:
        json.dump({"name": "random", "edge_prob": 0.5}, f)
    result = subprocess.run(
        [sys.executable, "-m", "evolveclaw_ramsey", "eval",
         "--strategy", str(strategy_path), "--n", "10", "--s", "3", "--t", "3"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "Score:" in result.stdout
    assert "Violations:" in result.stdout


def test_cli_eval_best_json_format():
    """Test that 'eval' handles best.json format with nested strategy key."""
    tmpdir = tempfile.mkdtemp()
    strategy_path = Path(tmpdir) / "best.json"
    with open(strategy_path, "w") as f:
        json.dump({"generation": 5, "strategy": {"name": "random", "edge_prob": 0.4},
                    "score": 8.0, "violation_count": 2}, f)
    result = subprocess.run(
        [sys.executable, "-m", "evolveclaw_ramsey", "eval",
         "--strategy", str(strategy_path), "--n", "10", "--s", "3", "--t", "3"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "Score:" in result.stdout


def test_cli_replay():
    """Test that 'replay' subcommand replays a completed run."""
    tmpdir = tempfile.mkdtemp()
    config_path = _write_config(tmpdir)
    # First run to generate log
    run_result = subprocess.run(
        [sys.executable, "-m", "evolveclaw_ramsey", "run", "--config", config_path],
        capture_output=True, text=True, timeout=60,
    )
    assert run_result.returncode == 0
    # Find the run directory
    run_dirs = [d for d in Path(tmpdir).iterdir() if d.is_dir() and "test_config" in d.name]
    assert len(run_dirs) >= 1
    run_dir = str(run_dirs[0])
    # Replay
    replay_result = subprocess.run(
        [sys.executable, "-m", "evolveclaw_ramsey", "replay", "--run-dir", run_dir],
        capture_output=True, text=True, timeout=30,
    )
    assert replay_result.returncode == 0
    assert "Replaying run" in replay_result.stdout
    assert "Best score" in replay_result.stdout
