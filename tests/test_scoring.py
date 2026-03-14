import numpy as np
from evolveclaw_ramsey.ramsey.scoring import RamseyScorer, ScoreResult

def _complete_graph(n):
    m = np.ones((n, n), dtype=np.int8)
    np.fill_diagonal(m, 0)
    return m

def _empty_graph(n):
    return np.zeros((n, n), dtype=np.int8)

def test_score_result_fields():
    r = ScoreResult(score=5.0, violation_count=2, s_cliques=1, t_cliques=1, n=7)
    assert r.score == 5.0
    assert r.violation_count == 2

def test_count_cliques_complete_graph():
    scorer = RamseyScorer(s=3, t=3)
    k5 = _complete_graph(5)
    assert scorer.count_cliques(k5, 3) == 10

def test_count_cliques_empty_graph():
    scorer = RamseyScorer(s=3, t=3)
    empty = _empty_graph(5)
    assert scorer.count_cliques(empty, 3) == 0

def test_score_complete_graph():
    scorer = RamseyScorer(s=3, t=3)
    k5 = _complete_graph(5)
    result = scorer.score(k5)
    assert result.s_cliques == 10
    assert result.t_cliques == 0
    assert result.violation_count == 10
    assert result.score == 5 - 10 * 1.0

def test_score_empty_graph():
    scorer = RamseyScorer(s=3, t=3)
    empty = _empty_graph(5)
    result = scorer.score(empty)
    assert result.s_cliques == 0
    assert result.t_cliques == 10
    assert result.violation_count == 10

def test_score_known_r33_counterexample():
    c5 = np.zeros((5, 5), dtype=np.int8)
    for i in range(5):
        c5[i, (i + 1) % 5] = 1
        c5[(i + 1) % 5, i] = 1
    scorer = RamseyScorer(s=3, t=3)
    result = scorer.score(c5)
    assert result.s_cliques == 0
    assert result.t_cliques == 0
    assert result.violation_count == 0
    assert result.score == 5.0

def test_penalty_weight():
    scorer = RamseyScorer(s=3, t=3, penalty_weight=2.0)
    k5 = _complete_graph(5)
    result = scorer.score(k5)
    assert result.score == 5 - 10 * 2.0
