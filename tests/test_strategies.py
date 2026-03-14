import numpy as np
from evolveclaw_ramsey.ramsey.graph_repr import validate_adjacency
from evolveclaw_ramsey.ramsey.strategies import (
    RandomStrategy, PaleyStrategy, CyclicStrategy, PerturbedStrategy, strategy_from_dict,
)

def test_random_strategy_constructs_valid_graph():
    rng = np.random.default_rng(42)
    s = RandomStrategy(edge_prob=0.5, rng=rng)
    graph = s.construct(10)
    assert graph.shape == (10, 10)
    assert validate_adjacency(graph)

def test_random_strategy_mutate_changes_params():
    rng = np.random.default_rng(42)
    s = RandomStrategy(edge_prob=0.5, rng=rng)
    s2 = s.mutate(rng)
    assert isinstance(s2, RandomStrategy)

def test_random_strategy_serialization_roundtrip():
    rng = np.random.default_rng(42)
    s = RandomStrategy(edge_prob=0.3, rng=rng)
    d = s.to_dict()
    assert d["name"] == "random"
    s2 = strategy_from_dict(d, rng)
    assert isinstance(s2, RandomStrategy)
    assert s2.edge_prob == 0.3

def test_paley_strategy_on_valid_prime():
    rng = np.random.default_rng(42)
    s = PaleyStrategy(rng=rng)
    graph = s.construct(17)
    assert graph.shape == (17, 17)
    assert validate_adjacency(graph)

def test_paley_strategy_on_invalid_n():
    rng = np.random.default_rng(42)
    s = PaleyStrategy(rng=rng)
    graph = s.construct(10)
    assert graph.shape == (10, 10)
    assert validate_adjacency(graph)

def test_cyclic_strategy_constructs_valid_graph():
    rng = np.random.default_rng(42)
    s = CyclicStrategy(offsets=[1, 2, 5], rng=rng)
    graph = s.construct(10)
    assert graph.shape == (10, 10)
    assert validate_adjacency(graph)

def test_cyclic_strategy_edge_pattern():
    rng = np.random.default_rng(42)
    s = CyclicStrategy(offsets=[1], rng=rng)
    graph = s.construct(5)
    for i in range(5):
        j = (i + 1) % 5
        assert graph[i, j] == 1

def test_perturbed_strategy_constructs_valid_graph():
    rng = np.random.default_rng(42)
    base = RandomStrategy(edge_prob=0.5, rng=rng)
    s = PerturbedStrategy(base=base, flip_prob=0.1, rng=rng)
    graph = s.construct(10)
    assert graph.shape == (10, 10)
    assert validate_adjacency(graph)

def test_perturbed_strategy_serialization_roundtrip():
    rng = np.random.default_rng(42)
    base = RandomStrategy(edge_prob=0.5, rng=rng)
    s = PerturbedStrategy(base=base, flip_prob=0.1, rng=rng)
    d = s.to_dict()
    assert d["name"] == "perturbed"
    s2 = strategy_from_dict(d, rng)
    assert isinstance(s2, PerturbedStrategy)
    assert s2.flip_prob == 0.1

def test_cyclic_mutate_with_n_bounds_offsets():
    """CyclicStrategy.mutate(n=10) should produce offsets < 10."""
    rng = np.random.default_rng(42)
    s = CyclicStrategy(offsets=[1, 3], rng=rng)
    for seed in range(50):
        r = np.random.default_rng(seed)
        s2 = s.mutate(r, n=10)
        assert all(o < 10 for o in s2.offsets)

def test_cyclic_mutate_without_n_uses_default():
    """CyclicStrategy.mutate(n=None) should use default upper bound of 20."""
    rng = np.random.default_rng(42)
    s = CyclicStrategy(offsets=[1], rng=rng)
    s2 = s.mutate(rng, n=None)
    assert isinstance(s2, CyclicStrategy)
    assert all(o < 20 for o in s2.offsets)
