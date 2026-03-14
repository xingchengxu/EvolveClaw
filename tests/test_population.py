"""Tests for Population eviction and deduplication."""
import numpy as np
from evolveclaw_ramsey.agent.population import Population
from evolveclaw_ramsey.ramsey.strategies import RandomStrategy, CyclicStrategy


def test_deduplication_rejects_identical_strategy():
    """Adding the same strategy object twice is rejected (same params_key)."""
    rng = np.random.default_rng(42)
    pop = Population(max_size=5)
    s1 = RandomStrategy(edge_prob=0.5, rng=rng)
    assert pop.add(s1, 10.0) is True
    # Same object added again — same params_key including seed
    assert pop.add(s1, 12.0) is False
    assert pop.size() == 1


def test_different_seeds_are_distinct():
    """Two strategies with same edge_prob but different seeds are distinct individuals."""
    rng = np.random.default_rng(42)
    pop = Population(max_size=5)
    s1 = RandomStrategy(edge_prob=0.5, rng=rng)
    s2 = RandomStrategy(edge_prob=0.5, rng=rng)
    assert pop.add(s1, 10.0) is True
    assert pop.add(s2, 12.0) is True
    assert pop.size() == 2


def test_eviction_replaces_worst():
    """When population is full, a better strategy evicts the worst."""
    rng = np.random.default_rng(42)
    pop = Population(max_size=2)
    s1 = RandomStrategy(edge_prob=0.3, rng=rng)
    s2 = RandomStrategy(edge_prob=0.5, rng=rng)
    s3 = RandomStrategy(edge_prob=0.7, rng=rng)
    pop.add(s1, 5.0)
    pop.add(s2, 8.0)
    assert pop.size() == 2
    # s3 with score 10 should evict worst (s1 with 5.0)
    assert pop.add(s3, 10.0) is True
    assert pop.size() == 2
    best, best_score = pop.best()
    assert best_score == 10.0


def test_eviction_rejects_worse_than_worst():
    """When population is full, a worse strategy is rejected."""
    rng = np.random.default_rng(42)
    pop = Population(max_size=2)
    s1 = RandomStrategy(edge_prob=0.3, rng=rng)
    s2 = RandomStrategy(edge_prob=0.5, rng=rng)
    s3 = RandomStrategy(edge_prob=0.7, rng=rng)
    pop.add(s1, 5.0)
    pop.add(s2, 8.0)
    # s3 with score 3.0 is worse than worst (5.0), should be rejected
    assert pop.add(s3, 3.0) is False
    assert pop.size() == 2


def test_tournament_select_returns_best_of_k():
    """Tournament selection picks the best among k random candidates."""
    rng = np.random.default_rng(42)
    pop = Population(max_size=10)
    for i in range(5):
        s = RandomStrategy(edge_prob=0.1 * (i + 1), rng=rng)
        pop.add(s, float(i))
    # With k=5 (all members), should always return the best
    selected, score = pop.tournament_select(k=5, rng=np.random.default_rng(0))
    assert score == 4.0


def test_serialization_roundtrip():
    """Population survives to_dict/from_dict roundtrip."""
    rng = np.random.default_rng(42)
    pop = Population(max_size=5)
    pop.add(RandomStrategy(edge_prob=0.5, rng=rng), 10.0)
    pop.add(CyclicStrategy(offsets=[1, 3], rng=rng), 8.0)
    d = pop.to_dict()
    restored = Population.from_dict(d, rng)
    assert restored.size() == 2
    _, best_score = restored.best()
    assert best_score == 10.0
