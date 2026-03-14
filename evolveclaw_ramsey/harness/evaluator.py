"""Evaluator: combines executor + scorer into a single evaluation call."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from evolveclaw_ramsey.harness.executor import Executor
from evolveclaw_ramsey.ramsey.scoring import RamseyScorer, ScoreResult
from evolveclaw_ramsey.ramsey.strategies import Strategy

@dataclass
class EvalResult:
    strategy: Strategy
    graph: np.ndarray | None
    score_result: ScoreResult | None
    error: str | None
    elapsed_seconds: float

class Evaluator:
    def __init__(self, scorer: RamseyScorer, executor: Executor):
        self.scorer = scorer
        self.executor = executor

    def evaluate(self, strategy: Strategy, n: int) -> EvalResult:
        exec_result = self.executor.execute(strategy, n)
        if exec_result.error:
            return EvalResult(strategy=strategy, graph=None, score_result=None,
                            error=exec_result.error, elapsed_seconds=exec_result.elapsed_seconds)
        score_result = self.scorer.score(exec_result.graph)
        return EvalResult(strategy=strategy, graph=exec_result.graph, score_result=score_result,
                        error=None, elapsed_seconds=exec_result.elapsed_seconds)
