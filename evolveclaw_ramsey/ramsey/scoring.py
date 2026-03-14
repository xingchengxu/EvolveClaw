"""Ramsey graph scoring: count monochromatic cliques and compute violation scores."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from evolveclaw_ramsey.ramsey.graph_repr import complement

@dataclass
class ScoreResult:
    """Result of scoring a graph against Ramsey constraints."""
    score: float
    violation_count: int
    s_cliques: int
    t_cliques: int
    n: int

class RamseyScorer:
    """Score a graph by counting monochromatic cliques for R(s, t)."""
    def __init__(self, s: int, t: int, penalty_weight: float = 1.0):
        self.s = s
        self.t = t
        self.penalty_weight = penalty_weight

    def count_cliques(self, graph: np.ndarray, k: int) -> int:
        """Count the number of k-cliques in graph via brute-force enumeration."""
        n = graph.shape[0]
        count = 0
        for subset in combinations(range(n), k):
            if all(graph[i, j] == 1 for i, j in combinations(subset, 2)):
                count += 1
        return count

    def score(self, graph: np.ndarray) -> ScoreResult:
        """Compute violation score for a graph against R(s, t) constraints."""
        n = graph.shape[0]
        s_cliques = self.count_cliques(graph, self.s)
        comp = complement(graph)
        t_cliques = self.count_cliques(comp, self.t)
        violation_count = s_cliques + t_cliques
        score_val = n - violation_count * self.penalty_weight
        return ScoreResult(score=score_val, violation_count=violation_count,
                          s_cliques=s_cliques, t_cliques=t_cliques, n=n)
