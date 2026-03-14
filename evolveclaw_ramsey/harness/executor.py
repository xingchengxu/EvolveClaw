"""Strategy executor with timeout and validation."""
from __future__ import annotations
import multiprocessing
import time
from dataclasses import dataclass
import numpy as np
from evolveclaw_ramsey.ramsey.graph_repr import validate_adjacency
from evolveclaw_ramsey.ramsey.strategies import Strategy


def _run_construct(strategy, n, result_queue):
    """Worker function that runs in a child process."""
    try:
        graph = strategy.construct(n)
        result_queue.put(("ok", graph))
    except Exception as e:
        result_queue.put(("error", str(e)))


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
        result_queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_run_construct, args=(strategy, n, result_queue), daemon=True
        )
        proc.start()
        proc.join(timeout=self.timeout_seconds)
        elapsed = time.monotonic() - start

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
            proc.close()
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error="Timeout")

        if result_queue.empty():
            proc.close()
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error="Process died without result")

        status, payload = result_queue.get_nowait()
        proc.close()
        if status == "error":
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error=payload)

        graph = payload
        if not isinstance(graph, np.ndarray):
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error="Output is not ndarray")
        if graph.shape != (n, n):
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error=f"Expected shape ({n},{n}), got {graph.shape}")
        if not validate_adjacency(graph):
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error="Output is not a valid adjacency matrix")
        return ExecutionResult(graph=graph, elapsed_seconds=elapsed, error=None)
