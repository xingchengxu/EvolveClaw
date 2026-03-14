"""Tests for convergence statistics."""
from evolveclaw_ramsey.harness.stats import RunStats


def test_record_single_generation():
    stats = RunStats()
    gen = stats.record(0, [10.0, 8.0, 5.0], ["random", "paley", "random"])
    assert gen.best_score == 10.0
    assert gen.mean_score == (10.0 + 8.0 + 5.0) / 3
    assert gen.unique_strategies == 2
    assert gen.population_size == 3
    assert gen.diversity_ratio == 2 / 3


def test_record_with_explicit_unique_count():
    stats = RunStats()
    gen = stats.record(0, [10.0, 8.0], ["random", "random"], unique_count=5)
    assert gen.unique_strategies == 5


def test_to_dict_returns_latest():
    stats = RunStats()
    stats.record(0, [5.0], ["random"])
    stats.record(1, [10.0, 8.0], ["random", "paley"])
    d = stats.to_dict()
    assert d["best_score"] == 10.0
    assert d["unique_strategies"] == 2


def test_to_dict_empty():
    stats = RunStats()
    assert stats.to_dict() == {}


def test_convergence_summary():
    stats = RunStats()
    stats.record(0, [5.0], ["random"])
    stats.record(1, [5.0], ["random"])
    stats.record(2, [8.0], ["random"])
    stats.record(3, [8.0], ["random"])
    stats.record(4, [12.0], ["random"])
    summary = stats.convergence_summary()
    assert summary["total_generations"] == 5
    assert summary["initial_best_score"] == 5.0
    assert summary["final_best_score"] == 12.0
    assert summary["improvement_count"] == 2
    assert summary["last_improvement_gen"] == 4


def test_convergence_summary_no_improvement():
    stats = RunStats()
    stats.record(0, [5.0], ["random"])
    stats.record(1, [5.0], ["random"])
    summary = stats.convergence_summary()
    assert summary["improvement_count"] == 0
    assert summary["last_improvement_gen"] == 0


def test_record_empty_scores():
    stats = RunStats()
    gen = stats.record(0, [], [])
    assert gen.best_score == 0.0
    assert gen.mean_score == 0.0
    assert gen.diversity_ratio == 0.0
