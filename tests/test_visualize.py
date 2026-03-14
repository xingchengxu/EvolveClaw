"""Tests for ASCII and matplotlib visualization."""
import json
import tempfile
from pathlib import Path
from evolveclaw_ramsey.harness.visualize import ascii_plot, matplotlib_plot, _load_scores


def _make_run_dir(records: list[dict]) -> str:
    run_dir = tempfile.mkdtemp()
    log_path = Path(run_dir) / "log.jsonl"
    with open(log_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return run_dir


def test_load_scores_skips_errors():
    run_dir = _make_run_dir([
        {"generation": 0, "score": 5.0},
        {"generation": 1, "error": "Timeout"},
        {"generation": 2, "score": 8.0},
    ])
    records = _load_scores(run_dir)
    assert len(records) == 2
    assert records[0]["score"] == 5.0
    assert records[1]["score"] == 8.0


def test_ascii_plot_empty():
    run_dir = _make_run_dir([])
    result = ascii_plot(run_dir)
    assert result == "No data to plot."


def test_ascii_plot_single_record():
    run_dir = _make_run_dir([{"generation": 0, "score": 5.0}])
    result = ascii_plot(run_dir)
    assert "Score Evolution" in result
    assert "5.0" in result


def test_ascii_plot_multiple_records():
    records = [{"generation": i, "score": float(i)} for i in range(10)]
    run_dir = _make_run_dir(records)
    result = ascii_plot(run_dir)
    assert "Score Evolution" in result
    assert "best-so-far" in result


def test_ascii_plot_constant_score():
    records = [{"generation": i, "score": 5.0} for i in range(5)]
    run_dir = _make_run_dir(records)
    result = ascii_plot(run_dir)
    assert "Score Evolution" in result


def test_matplotlib_plot_creates_image():
    records = [{"generation": i, "score": float(i)} for i in range(5)]
    run_dir = _make_run_dir(records)
    result = matplotlib_plot(run_dir)
    if result is not None:
        assert Path(result).exists()
        assert result.endswith(".png")


def test_matplotlib_plot_empty_returns_none():
    run_dir = _make_run_dir([])
    result = matplotlib_plot(run_dir)
    assert result is None
