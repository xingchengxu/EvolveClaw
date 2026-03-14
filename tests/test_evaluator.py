import numpy as np
from evolveclaw_ramsey.harness.evaluator import Evaluator, EvalResult
from evolveclaw_ramsey.harness.executor import Executor
from evolveclaw_ramsey.ramsey.scoring import RamseyScorer
from evolveclaw_ramsey.ramsey.strategies import RandomStrategy, CyclicStrategy

def test_evaluator_returns_eval_result():
    rng = np.random.default_rng(42)
    scorer = RamseyScorer(s=3, t=3)
    executor = Executor(timeout_seconds=5)
    evaluator = Evaluator(scorer=scorer, executor=executor)
    strategy = RandomStrategy(edge_prob=0.5, rng=rng)
    result = evaluator.evaluate(strategy, n=6)
    assert isinstance(result, EvalResult)
    assert result.graph is not None
    assert result.score_result is not None
    assert result.error is None

def test_evaluator_with_cyclic_strategy():
    rng = np.random.default_rng(42)
    scorer = RamseyScorer(s=3, t=3)
    executor = Executor(timeout_seconds=5)
    evaluator = Evaluator(scorer=scorer, executor=executor)
    strategy = CyclicStrategy(offsets=[1], rng=rng)
    result = evaluator.evaluate(strategy, n=5)
    assert result.score_result is not None
    assert result.score_result.violation_count == 0
    assert result.score_result.score == 5.0
