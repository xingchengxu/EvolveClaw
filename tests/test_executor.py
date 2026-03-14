"""Tests for strategy executor with timeout and validation."""
import numpy as np
from evolveclaw_ramsey.harness.executor import Executor
from evolveclaw_ramsey.ramsey.strategies import RandomStrategy, Strategy


class _SlowStrategy(Strategy):
    """Strategy that sleeps forever, for testing timeout."""
    name = "slow"
    def construct(self, n: int) -> np.ndarray:
        import time
        time.sleep(100)
        return np.zeros((n, n), dtype=np.int8)
    def mutate(self, rng, n=None):
        return self
    def to_dict(self):
        return {"name": "slow"}


class _CrashStrategy(Strategy):
    """Strategy that raises an exception during construction."""
    name = "crash"
    def construct(self, n: int) -> np.ndarray:
        raise RuntimeError("Intentional crash")
    def mutate(self, rng, n=None):
        return self
    def to_dict(self):
        return {"name": "crash"}


class _BadShapeStrategy(Strategy):
    """Strategy that returns wrong shape."""
    name = "badshape"
    def construct(self, n: int) -> np.ndarray:
        return np.zeros((n + 1, n + 1), dtype=np.int8)
    def mutate(self, rng, n=None):
        return self
    def to_dict(self):
        return {"name": "badshape"}


def test_execute_valid_strategy():
    rng = np.random.default_rng(42)
    executor = Executor(timeout_seconds=5.0)
    strategy = RandomStrategy(edge_prob=0.5, rng=rng)
    result = executor.execute(strategy, 10)
    assert result.error is None
    assert result.graph is not None
    assert result.graph.shape == (10, 10)
    assert result.elapsed_seconds > 0


def test_execute_timeout():
    executor = Executor(timeout_seconds=0.5)
    strategy = _SlowStrategy()
    result = executor.execute(strategy, 5)
    assert result.error == "Timeout"
    assert result.graph is None


def test_execute_crash():
    executor = Executor(timeout_seconds=5.0)
    strategy = _CrashStrategy()
    result = executor.execute(strategy, 5)
    assert result.error is not None
    assert "Intentional crash" in result.error
    assert result.graph is None


def test_execute_wrong_shape():
    executor = Executor(timeout_seconds=5.0)
    strategy = _BadShapeStrategy()
    result = executor.execute(strategy, 5)
    assert result.error is not None
    assert "shape" in result.error.lower()
    assert result.graph is None
