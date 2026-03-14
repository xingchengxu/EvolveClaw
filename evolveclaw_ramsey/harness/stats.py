"""Convergence statistics: track per-generation metrics and population diversity."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    best_score: float
    mean_score: float
    unique_strategies: int
    population_size: int
    diversity_ratio: float  # unique_strategies / population_size


class RunStats:
    """Accumulates per-generation statistics over an evolution run."""

    def __init__(self):
        self.history: list[GenerationStats] = []

    def record(self, generation: int, scores: list[float],
               strategy_names: list[str]) -> GenerationStats:
        """Record stats for one generation."""
        best = max(scores) if scores else 0.0
        mean = sum(scores) / len(scores) if scores else 0.0
        unique = len(set(strategy_names))
        pop_size = len(scores)
        diversity = unique / pop_size if pop_size > 0 else 0.0
        stats = GenerationStats(
            generation=generation, best_score=best, mean_score=mean,
            unique_strategies=unique, population_size=pop_size,
            diversity_ratio=diversity,
        )
        self.history.append(stats)
        return stats

    def to_dict(self) -> dict:
        """Serialize stats for logging."""
        if not self.history:
            return {}
        latest = self.history[-1]
        return {
            "best_score": latest.best_score,
            "mean_score": round(latest.mean_score, 4),
            "unique_strategies": latest.unique_strategies,
            "diversity_ratio": round(latest.diversity_ratio, 4),
        }

    def convergence_summary(self) -> dict:
        """Compute summary statistics over the entire run."""
        if not self.history:
            return {}
        best_scores = [s.best_score for s in self.history]
        improvement_gens = []
        prev = best_scores[0]
        for i, s in enumerate(best_scores[1:], 1):
            if s > prev:
                improvement_gens.append(i)
                prev = s
        return {
            "total_generations": len(self.history),
            "final_best_score": best_scores[-1],
            "initial_best_score": best_scores[0],
            "improvement_count": len(improvement_gens),
            "last_improvement_gen": improvement_gens[-1] if improvement_gens else 0,
            "mean_diversity": round(
                sum(s.diversity_ratio for s in self.history) / len(self.history), 4
            ),
        }
