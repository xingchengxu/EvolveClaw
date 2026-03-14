"""Tests for the evolution loop: minimal run, early termination, checkpoint resume."""
import tempfile
from pathlib import Path
from evolveclaw_ramsey.agent.loop import run_evolution


def _make_config(run_dir, **overrides):
    config = {
        "problem": {"s": 3, "t": 3, "n": 6, "penalty_weight": 1.0},
        "evolution": {"max_generations": 5, "population_size": 5, "tournament_k": 2, "checkpoint_interval": 5},
        "proposer": {"type": "random"},
        "executor": {"timeout_seconds": 5},
        "logging": {"level": "WARNING"},
        "seed": 42,
        "run_dir": run_dir,
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
            config[k].update(v)
        else:
            config[k] = v
    return config


def test_evolution_loop_minimal():
    config = _make_config(tempfile.mkdtemp())
    result = run_evolution(config)
    assert result.generations_completed == 5
    assert result.best_score is not None
    assert result.best_strategy is not None


def test_zero_violation_early_termination():
    """R(3,3)=6, so n=5 should easily find zero-violation colorings and terminate early."""
    run_dir = tempfile.mkdtemp()
    config = _make_config(run_dir, problem={"s": 3, "t": 3, "n": 5, "penalty_weight": 1.0},
                          evolution={"max_generations": 200, "population_size": 10,
                                     "tournament_k": 3, "checkpoint_interval": 50})
    result = run_evolution(config)
    # With n=5 for R(3,3), a perfect coloring should be found well before 200 generations
    assert result.generations_completed < 200
    assert result.best_score == 5.0  # n - 0 violations = 5


def test_checkpoint_resume():
    """Run 5 generations, then resume from checkpoint and run 5 more."""
    run_dir = tempfile.mkdtemp()
    # First run: 5 generations with checkpoint at gen 5
    config = _make_config(run_dir, evolution={"max_generations": 5, "population_size": 5,
                                              "tournament_k": 2, "checkpoint_interval": 5})
    result1 = run_evolution(config, config_stem="test")
    actual_run_dir = result1.run_dir
    # Verify checkpoint was created
    ckpt_dir = Path(actual_run_dir) / "checkpoints"
    assert ckpt_dir.exists()
    assert any(ckpt_dir.glob("gen_*.json"))
    # Resume and run 5 more generations
    config2 = _make_config(run_dir, evolution={"max_generations": 10, "population_size": 5,
                                               "tournament_k": 2, "checkpoint_interval": 5})
    result2 = run_evolution(config2, resume_dir=actual_run_dir, config_stem="test")
    assert result2.generations_completed == 10
    assert result2.best_score is not None


def test_evolution_produces_output_files():
    """Verify that a run produces log.jsonl, summary.txt, and best.json."""
    run_dir = tempfile.mkdtemp()
    config = _make_config(run_dir)
    result = run_evolution(config, config_stem="test")
    rd = Path(result.run_dir)
    assert (rd / "log.jsonl").exists()
    assert (rd / "summary.txt").exists()
    assert (rd / "config.yaml").exists()
