"""Tests for Recorder: JSONL log, error log, best.json, summary.txt."""
import json
import tempfile
from pathlib import Path
import numpy as np
from evolveclaw_ramsey.harness.recorder import Recorder
from evolveclaw_ramsey.ramsey.scoring import ScoreResult
from evolveclaw_ramsey.ramsey.strategies import RandomStrategy


def _make_recorder():
    run_dir = tempfile.mkdtemp()
    return Recorder(run_dir), run_dir


def _make_score_result(score=10.0, violations=7, s_cliques=3, t_cliques=4, n=17):
    return ScoreResult(score=score, violation_count=violations,
                       s_cliques=s_cliques, t_cliques=t_cliques, n=n)


def test_log_generation_writes_jsonl():
    recorder, run_dir = _make_recorder()
    rng = np.random.default_rng(42)
    strategy = RandomStrategy(edge_prob=0.5, rng=rng)
    score_result = _make_score_result()
    recorder.log_generation(0, strategy, score_result, added=True)
    log_path = Path(run_dir) / "log.jsonl"
    assert log_path.exists()
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["generation"] == 0
    assert record["strategy_name"] == "random"
    assert record["score"] == 10.0
    assert record["violation_count"] == 7
    assert record["s_cliques"] == 3
    assert record["t_cliques"] == 4
    assert record["added_to_population"] is True
    assert "strategy_params" in record


def test_log_generation_multiple_entries():
    recorder, run_dir = _make_recorder()
    rng = np.random.default_rng(42)
    strategy = RandomStrategy(edge_prob=0.5, rng=rng)
    for gen in range(5):
        recorder.log_generation(gen, strategy, _make_score_result(score=float(gen)), added=False)
    log_path = Path(run_dir) / "log.jsonl"
    with open(log_path) as f:
        lines = f.readlines()
    assert len(lines) == 5


def test_log_error_writes_error_record():
    recorder, run_dir = _make_recorder()
    recorder.log_error(3, "Timeout")
    log_path = Path(run_dir) / "log.jsonl"
    with open(log_path) as f:
        record = json.loads(f.readline())
    assert record["generation"] == 3
    assert record["error"] == "Timeout"


def test_best_json_tracks_highest_score():
    recorder, run_dir = _make_recorder()
    rng = np.random.default_rng(42)
    strategy = RandomStrategy(edge_prob=0.5, rng=rng)
    recorder.log_generation(0, strategy, _make_score_result(score=5.0), added=True)
    recorder.log_generation(1, strategy, _make_score_result(score=12.0), added=True)
    recorder.log_generation(2, strategy, _make_score_result(score=8.0), added=True)
    best_path = Path(run_dir) / "best.json"
    assert best_path.exists()
    with open(best_path) as f:
        best = json.load(f)
    assert best["score"] == 12.0
    assert best["generation"] == 1


def test_write_summary():
    recorder, run_dir = _make_recorder()
    rng = np.random.default_rng(42)
    strategy = RandomStrategy(edge_prob=0.5, rng=rng)
    recorder.write_summary(strategy, best_score=15.0, generations=50)
    summary_path = Path(run_dir) / "summary.txt"
    assert summary_path.exists()
    text = summary_path.read_text()
    assert "15.0" in text
    assert "50" in text
    assert "random" in text


def test_save_config():
    recorder, run_dir = _make_recorder()
    config = {"problem": {"s": 4, "t": 4}, "seed": 42}
    recorder.save_config(config)
    config_path = Path(run_dir) / "config.yaml"
    assert config_path.exists()
