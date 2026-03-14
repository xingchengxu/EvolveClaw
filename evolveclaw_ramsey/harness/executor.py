"""Strategy executor with timeout and validation."""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import numpy as np
from evolveclaw_ramsey.ramsey.graph_repr import validate_adjacency
from evolveclaw_ramsey.ramsey.strategies import Strategy

@dataclass
class ExecutionResult:
    graph: np.ndarray | None
    elapsed_seconds: float
    error: str | None

class Executor:
    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, strategy: Strategy, n: int) -> ExecutionResult:
        start = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(strategy.construct, n)
                graph = future.result(timeout=self.timeout_seconds)
        except FuturesTimeoutError:
            elapsed = time.monotonic() - start
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error="Timeout")
        except Exception as e:
            elapsed = time.monotonic() - start
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error=str(e))
        elapsed = time.monotonic() - start
        if not isinstance(graph, np.ndarray):
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error="Output is not ndarray")
        if graph.shape != (n, n):
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error=f"Expected shape ({n},{n}), got {graph.shape}")
        if not validate_adjacency(graph):
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error="Output is not a valid adjacency matrix")
        return ExecutionResult(graph=graph, elapsed_seconds=elapsed, error=None)
