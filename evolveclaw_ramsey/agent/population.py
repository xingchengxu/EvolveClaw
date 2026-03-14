"""Population management for evolutionary strategy search."""
from __future__ import annotations
import numpy as np
from evolveclaw_ramsey.ramsey.strategies import Strategy, strategy_from_dict

class Population:
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._members: list[tuple[Strategy, float]] = []
        self._seen_keys: set[tuple] = set()

    def add(self, strategy: Strategy, score: float) -> bool:
        key = strategy.params_key()
        if key in self._seen_keys:
            return False
        if len(self._members) < self.max_size:
            self._members.append((strategy, score))
            self._seen_keys.add(key)
            self._members.sort(key=lambda x: x[1], reverse=True)
            return True
        if score > self._members[-1][1]:
            removed = self._members.pop()
            self._seen_keys.discard(removed[0].params_key())
            self._members.append((strategy, score))
            self._seen_keys.add(key)
            self._members.sort(key=lambda x: x[1], reverse=True)
            return True
        return False

    def tournament_select(self, k: int, rng: np.random.Generator) -> tuple[Strategy, float]:
        if not self._members:
            raise ValueError("Population is empty")
        k = min(k, len(self._members))
        indices = rng.choice(len(self._members), size=k, replace=False)
        candidates = [self._members[i] for i in indices]
        return max(candidates, key=lambda x: x[1])

    def best(self) -> tuple[Strategy, float]:
        if not self._members:
            raise ValueError("Population is empty")
        return self._members[0]

    def size(self) -> int:
        return len(self._members)

    def to_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "members": [{"strategy": s.to_dict(), "score": score} for s, score in self._members],
        }

    @classmethod
    def from_dict(cls, d: dict, rng: np.random.Generator) -> Population:
        pop = cls(max_size=d["max_size"])
        for entry in d["members"]:
            strategy = strategy_from_dict(entry["strategy"], rng)
            pop._members.append((strategy, entry["score"]))
            pop._seen_keys.add(strategy.params_key())
        pop._members.sort(key=lambda x: x[1], reverse=True)
        return pop
