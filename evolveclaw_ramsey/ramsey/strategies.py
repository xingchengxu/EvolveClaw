"""Graph construction strategies for Ramsey optimization."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Strategy(ABC):
    """Base class for graph construction strategies."""
    name: str

    @abstractmethod
    def construct(self, n: int) -> np.ndarray:
        ...
    @abstractmethod
    def mutate(self, rng: np.random.Generator, n: int | None = None) -> Strategy:
        ...
    @abstractmethod
    def to_dict(self) -> dict:
        ...
    def params_key(self) -> tuple:
        return (self.name,)

def _is_prime(n: int) -> bool:
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

def _quadratic_residues(p: int) -> set[int]:
    return {(x * x) % p for x in range(1, p)}

class RandomStrategy(Strategy):
    name = "random"
    def __init__(self, edge_prob: float, rng: np.random.Generator):
        self.edge_prob = edge_prob
        self._seed = int(rng.integers(0, 2**63))
    def construct(self, n: int) -> np.ndarray:
        local_rng = np.random.default_rng(self._seed)
        m = (local_rng.random((n, n)) < self.edge_prob).astype(np.int8)
        m = np.triu(m, 1)
        m = m + m.T
        return m
    def mutate(self, rng: np.random.Generator, n: int | None = None) -> Strategy:
        delta = rng.normal(0, 0.1)
        new_prob = max(0.05, min(0.95, self.edge_prob + delta))
        return RandomStrategy(edge_prob=new_prob, rng=rng)
    def to_dict(self) -> dict:
        return {"name": "random", "edge_prob": self.edge_prob, "seed": self._seed}
    def params_key(self) -> tuple:
        return ("random", round(self.edge_prob, 4), self._seed)

class PaleyStrategy(Strategy):
    name = "paley"
    def __init__(self, rng: np.random.Generator):
        pass  # No internal state — construct is fully determined by n
    def construct(self, n: int) -> np.ndarray:
        if not (_is_prime(n) and n % 4 == 1):
            # Deterministic fallback seeded from n — no internal RNG state
            local_rng = np.random.default_rng(n)
            m = (local_rng.random((n, n)) < 0.5).astype(np.int8)
            m = np.triu(m, 1)
            m = m + m.T
            return m
        qr = _quadratic_residues(n)
        m = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                diff = (i - j) % n
                if diff in qr:
                    m[i, j] = 1
                    m[j, i] = 1
        return m
    def mutate(self, rng: np.random.Generator, n: int | None = None) -> Strategy:
        return PerturbedStrategy(base=PaleyStrategy(rng=rng), flip_prob=0.05, rng=rng)
    def to_dict(self) -> dict:
        return {"name": "paley"}
    def params_key(self) -> tuple:
        return ("paley",)

class CyclicStrategy(Strategy):
    name = "cyclic"
    def __init__(self, offsets: list[int], rng: np.random.Generator):
        self.offsets = sorted(set(offsets))
    def construct(self, n: int) -> np.ndarray:
        offset_set = set(self.offsets)
        m = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                rev_diff = (i - j) % n
                if diff in offset_set or rev_diff in offset_set:
                    m[i, j] = 1
                    m[j, i] = 1
        return m
    def mutate(self, rng: np.random.Generator, n: int | None = None) -> Strategy:
        upper = max(2, n) if n else 20
        new_offsets = list(self.offsets)
        if len(new_offsets) > 1 and rng.random() < 0.3:
            idx = rng.integers(0, len(new_offsets))
            new_offsets.pop(idx)
        elif rng.random() < 0.5:
            new_offsets.append(int(rng.integers(1, upper)))
        else:
            if new_offsets:
                idx = rng.integers(0, len(new_offsets))
                new_offsets[idx] = int(rng.integers(1, upper))
        return CyclicStrategy(offsets=new_offsets, rng=rng)
    def to_dict(self) -> dict:
        return {"name": "cyclic", "offsets": self.offsets}
    def params_key(self) -> tuple:
        return ("cyclic", tuple(self.offsets))

class PerturbedStrategy(Strategy):
    name = "perturbed"
    def __init__(self, base: Strategy, flip_prob: float, rng: np.random.Generator):
        self.base = base
        self.flip_prob = flip_prob
        self._seed = int(rng.integers(0, 2**63))
    def construct(self, n: int) -> np.ndarray:
        local_rng = np.random.default_rng(self._seed)
        m = self.base.construct(n)
        flip_mask = (local_rng.random((n, n)) < self.flip_prob).astype(np.int8)
        flip_mask = np.triu(flip_mask, 1)
        flip_mask = flip_mask + flip_mask.T
        m = np.bitwise_xor(m, flip_mask).astype(np.int8)
        np.fill_diagonal(m, 0)
        return m
    def mutate(self, rng: np.random.Generator, n: int | None = None) -> Strategy:
        delta = rng.normal(0, 0.02)
        new_prob = max(0.01, min(0.5, self.flip_prob + delta))
        return PerturbedStrategy(base=self.base, flip_prob=new_prob, rng=rng)
    def to_dict(self) -> dict:
        return {"name": "perturbed", "base": self.base.to_dict(), "flip_prob": self.flip_prob, "seed": self._seed}
    def params_key(self) -> tuple:
        return ("perturbed", self.base.params_key(), round(self.flip_prob, 4), self._seed)

def strategy_from_dict(d: dict, rng: np.random.Generator) -> Strategy:
    name = d["name"]
    if name == "random":
        s = RandomStrategy(edge_prob=d["edge_prob"], rng=rng)
        if "seed" in d:
            s._seed = d["seed"]
        return s
    elif name == "paley":
        return PaleyStrategy(rng=rng)
    elif name == "cyclic":
        return CyclicStrategy(offsets=d["offsets"], rng=rng)
    elif name == "perturbed":
        base = strategy_from_dict(d["base"], rng)
        s = PerturbedStrategy(base=base, flip_prob=d["flip_prob"], rng=rng)
        if "seed" in d:
            s._seed = d["seed"]
        return s
    else:
        raise ValueError(f"Unknown strategy type: {name}")
