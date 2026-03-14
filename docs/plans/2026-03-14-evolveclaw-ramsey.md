# EvolveClaw-Ramsey Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal educational AlphaEvolve-style Ramsey optimization system that evolves graph construction strategies to find Ramsey-valid colorings of R(4,4) on n=17.

**Architecture:** Three-layer system: Domain (graph representation, scoring, strategies), Harness (executor, evaluator, recorder, checkpoint), Agent (population, proposer, evolution loop). CLI provides `run`, `eval`, and `replay` commands. Configuration via YAML. Optional LLM proposer with Anthropic/OpenAI integration.

**Tech Stack:** Python 3.10+, numpy, pyyaml, pytest. Optional: anthropic, openai.

**Spec:** `docs/superpowers/specs/2026-03-14-evolveclaw-ramsey-design.md`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `evolveclaw_ramsey/__init__.py`
- Create: `evolveclaw_ramsey/__main__.py`
- Create: `evolveclaw_ramsey/agent/__init__.py`
- Create: `evolveclaw_ramsey/harness/__init__.py`
- Create: `evolveclaw_ramsey/ramsey/__init__.py`
- Create: `evolveclaw_ramsey/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `configs/demo.yaml`
- Create: `configs/llm_demo.yaml`

**Step 1: Initialize git repo**

```bash
cd D:/project/github/EvolveClaw
git init
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "evolveclaw-ramsey"
version = "0.1.0"
description = "Minimal educational AlphaEvolve-style Ramsey optimization system"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
llm = [
    "anthropic>=0.40",
    "openai>=1.0",
]
dev = [
    "pytest>=7.0",
]

[project.scripts]
evolveclaw-ramsey = "evolveclaw_ramsey.cli:main"
```

**Step 3: Create requirements.txt**

```
numpy>=1.24
pyyaml>=6.0
pytest>=7.0
```

**Step 4: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
runs/
*.pdf
.env
.venv/
venv/
```

**Step 5: Create all __init__.py files**

`evolveclaw_ramsey/__init__.py`:
```python
"""EvolveClaw-Ramsey: Minimal educational AlphaEvolve-style Ramsey optimization."""
```

`evolveclaw_ramsey/__main__.py`:
```python
from evolveclaw_ramsey.cli import main

main()
```

`evolveclaw_ramsey/agent/__init__.py`, `evolveclaw_ramsey/harness/__init__.py`, `evolveclaw_ramsey/ramsey/__init__.py`, `evolveclaw_ramsey/utils/__init__.py`, `tests/__init__.py`:
```python
```
(empty files)

**Step 6: Create configs/demo.yaml**

```yaml
problem:
  s: 4
  t: 4
  n: 17
  penalty_weight: 1.0

evolution:
  max_generations: 100
  population_size: 20
  tournament_k: 3
  checkpoint_interval: 10

proposer:
  type: random

executor:
  timeout_seconds: 10

logging:
  level: INFO

seed: 42
run_dir: runs/
```

**Step 7: Create configs/llm_demo.yaml**

```yaml
problem:
  s: 4
  t: 4
  n: 17
  penalty_weight: 1.0

evolution:
  max_generations: 50
  population_size: 20
  tournament_k: 3
  checkpoint_interval: 10

proposer:
  type: llm
  llm_provider: anthropic
  llm_model: claude-sonnet-4-20250514
  llm_api_key_env: ANTHROPIC_API_KEY

executor:
  timeout_seconds: 10

logging:
  level: INFO

seed: 42
run_dir: runs/
```

**Step 8: Verify structure**

Run: `find . -name "*.py" -o -name "*.toml" -o -name "*.txt" -o -name "*.yaml" -o -name ".gitignore" | sort`

Expected: all files listed above present.

**Step 9: Install in dev mode and verify**

Run: `pip install -e ".[dev]" && python -c "import evolveclaw_ramsey; print('OK')"`

Expected: prints `OK`

**Step 10: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore evolveclaw_ramsey/ tests/__init__.py configs/
git commit -m "chore: scaffold evolveclaw-ramsey project structure"
```

---

## Task 2: Graph Representation (`ramsey/graph_repr.py`)

**Files:**
- Create: `evolveclaw_ramsey/ramsey/graph_repr.py`
- Create: `tests/test_graph_repr.py`

**Step 1: Write failing tests**

```python
# tests/test_graph_repr.py
import numpy as np
from evolveclaw_ramsey.ramsey.graph_repr import (
    validate_adjacency,
    complement,
    to_edge_list,
    from_edge_list,
)


def test_validate_adjacency_valid():
    m = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    assert validate_adjacency(m) is True


def test_validate_adjacency_not_square():
    m = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int8)
    assert validate_adjacency(m) is False


def test_validate_adjacency_nonzero_diagonal():
    m = np.array([[1, 1], [1, 0]], dtype=np.int8)
    assert validate_adjacency(m) is False


def test_validate_adjacency_not_symmetric():
    m = np.array([[0, 1], [0, 0]], dtype=np.int8)
    assert validate_adjacency(m) is False


def test_complement():
    m = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    c = complement(m)
    expected = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.int8)
    np.testing.assert_array_equal(c, expected)


def test_edge_list_roundtrip():
    m = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    edges = to_edge_list(m)
    assert set(edges) == {(0, 1), (1, 2)}
    m2 = from_edge_list(edges, 3)
    np.testing.assert_array_equal(m, m2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_repr.py -v`

Expected: FAIL (ImportError)

**Step 3: Implement graph_repr.py**

```python
# evolveclaw_ramsey/ramsey/graph_repr.py
"""Graph representation utilities for adjacency matrices."""

from __future__ import annotations

import numpy as np


def validate_adjacency(matrix: np.ndarray) -> bool:
    """Check that matrix is a valid adjacency matrix (square, symmetric, zero diagonal, binary)."""
    if matrix.ndim != 2:
        return False
    n, m = matrix.shape
    if n != m:
        return False
    if not np.all(np.diag(matrix) == 0):
        return False
    if not np.array_equal(matrix, matrix.T):
        return False
    if not np.all((matrix == 0) | (matrix == 1)):
        return False
    return True


def complement(matrix: np.ndarray) -> np.ndarray:
    """Return the complement graph: flip edges, keep diagonal zero."""
    c = 1 - matrix
    np.fill_diagonal(c, 0)
    return c.astype(np.int8)


def to_edge_list(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return sorted list of edges (i, j) where i < j and matrix[i][j] == 1."""
    edges = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] == 1:
                edges.append((i, j))
    return edges


def from_edge_list(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    """Build an n x n adjacency matrix from a list of edges."""
    matrix = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        matrix[i, j] = 1
        matrix[j, i] = 1
    return matrix
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_repr.py -v`

Expected: all 6 tests PASS

**Step 5: Commit**

```bash
git add evolveclaw_ramsey/ramsey/graph_repr.py tests/test_graph_repr.py
git commit -m "feat: add graph representation utilities with adjacency matrix ops"
```

---

## Task 3: Ramsey Scoring (`ramsey/scoring.py`)

**Files:**
- Create: `evolveclaw_ramsey/ramsey/scoring.py`
- Create: `tests/test_scoring.py`

**Step 1: Write failing tests**

```python
# tests/test_scoring.py
import numpy as np
from evolveclaw_ramsey.ramsey.scoring import RamseyScorer, ScoreResult


def _complete_graph(n: int) -> np.ndarray:
    """K_n: all edges present."""
    m = np.ones((n, n), dtype=np.int8)
    np.fill_diagonal(m, 0)
    return m


def _empty_graph(n: int) -> np.ndarray:
    """No edges."""
    return np.zeros((n, n), dtype=np.int8)


def test_score_result_fields():
    r = ScoreResult(score=5.0, violation_count=2, s_cliques=1, t_cliques=1, n=7)
    assert r.score == 5.0
    assert r.violation_count == 2


def test_count_cliques_complete_graph():
    """K_5 has C(5,3) = 10 triangles."""
    scorer = RamseyScorer(s=3, t=3)
    k5 = _complete_graph(5)
    assert scorer.count_cliques(k5, 3) == 10


def test_count_cliques_empty_graph():
    """Empty graph has 0 cliques of any size > 1."""
    scorer = RamseyScorer(s=3, t=3)
    empty = _empty_graph(5)
    assert scorer.count_cliques(empty, 3) == 0


def test_score_complete_graph():
    """K_5 for R(3,3): all edges present => many s-cliques, complement is empty => 0 t-cliques."""
    scorer = RamseyScorer(s=3, t=3)
    k5 = _complete_graph(5)
    result = scorer.score(k5)
    assert result.s_cliques == 10  # C(5,3)
    assert result.t_cliques == 0
    assert result.violation_count == 10
    assert result.score == 5 - 10 * 1.0  # n - violations * penalty


def test_score_empty_graph():
    """Empty graph for R(3,3): 0 s-cliques, complement is K_5 => 10 t-cliques."""
    scorer = RamseyScorer(s=3, t=3)
    empty = _empty_graph(5)
    result = scorer.score(empty)
    assert result.s_cliques == 0
    assert result.t_cliques == 10
    assert result.violation_count == 10


def test_score_known_r33_counterexample():
    """C_5 (5-cycle) is a valid R(3,3) counterexample on 5 vertices: no K_3 in G or complement."""
    # C_5 adjacency: 0-1, 1-2, 2-3, 3-4, 4-0
    c5 = np.zeros((5, 5), dtype=np.int8)
    for i in range(5):
        c5[i, (i + 1) % 5] = 1
        c5[(i + 1) % 5, i] = 1
    scorer = RamseyScorer(s=3, t=3)
    result = scorer.score(c5)
    assert result.s_cliques == 0
    assert result.t_cliques == 0
    assert result.violation_count == 0
    assert result.score == 5.0  # perfect score = n


def test_penalty_weight():
    scorer = RamseyScorer(s=3, t=3, penalty_weight=2.0)
    k5 = _complete_graph(5)
    result = scorer.score(k5)
    assert result.score == 5 - 10 * 2.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scoring.py -v`

Expected: FAIL (ImportError)

**Step 3: Implement scoring.py**

```python
# evolveclaw_ramsey/ramsey/scoring.py
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
        return ScoreResult(
            score=score_val,
            violation_count=violation_count,
            s_cliques=s_cliques,
            t_cliques=t_cliques,
            n=n,
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`

Expected: all 7 tests PASS

**Step 5: Commit**

```bash
git add evolveclaw_ramsey/ramsey/scoring.py tests/test_scoring.py
git commit -m "feat: add Ramsey scoring with clique counting and violation score"
```

---

## Task 4: Strategy Interface and Implementations (`ramsey/strategies.py`)

**Files:**
- Create: `evolveclaw_ramsey/ramsey/strategies.py`
- Create: `tests/test_strategies.py`

**Step 1: Write failing tests**

```python
# tests/test_strategies.py
import numpy as np
from evolveclaw_ramsey.ramsey.graph_repr import validate_adjacency
from evolveclaw_ramsey.ramsey.strategies import (
    RandomStrategy,
    PaleyStrategy,
    CyclicStrategy,
    PerturbedStrategy,
    strategy_from_dict,
)


def test_random_strategy_constructs_valid_graph():
    rng = np.random.default_rng(42)
    s = RandomStrategy(edge_prob=0.5, rng=rng)
    graph = s.construct(10)
    assert graph.shape == (10, 10)
    assert validate_adjacency(graph)


def test_random_strategy_mutate_changes_params():
    rng = np.random.default_rng(42)
    s = RandomStrategy(edge_prob=0.5, rng=rng)
    s2 = s.mutate(rng)
    # Mutation should produce a strategy (may or may not change edge_prob)
    assert isinstance(s2, RandomStrategy)


def test_random_strategy_serialization_roundtrip():
    rng = np.random.default_rng(42)
    s = RandomStrategy(edge_prob=0.3, rng=rng)
    d = s.to_dict()
    assert d["name"] == "random"
    s2 = strategy_from_dict(d, rng)
    assert isinstance(s2, RandomStrategy)
    assert s2.edge_prob == 0.3


def test_paley_strategy_on_valid_prime():
    """n=17 is prime, 17 % 4 == 1, so Paley construction is valid."""
    rng = np.random.default_rng(42)
    s = PaleyStrategy(rng=rng)
    graph = s.construct(17)
    assert graph.shape == (17, 17)
    assert validate_adjacency(graph)


def test_paley_strategy_on_invalid_n():
    """n=10 is not prime, so Paley falls back to random."""
    rng = np.random.default_rng(42)
    s = PaleyStrategy(rng=rng)
    graph = s.construct(10)
    assert graph.shape == (10, 10)
    assert validate_adjacency(graph)


def test_cyclic_strategy_constructs_valid_graph():
    rng = np.random.default_rng(42)
    s = CyclicStrategy(offsets=[1, 2, 5], rng=rng)
    graph = s.construct(10)
    assert graph.shape == (10, 10)
    assert validate_adjacency(graph)


def test_cyclic_strategy_edge_pattern():
    """With offsets [1], edges are (i, i+1 mod n) and (i+1 mod n, i)."""
    rng = np.random.default_rng(42)
    s = CyclicStrategy(offsets=[1], rng=rng)
    graph = s.construct(5)
    # Should be a 5-cycle
    for i in range(5):
        j = (i + 1) % 5
        assert graph[i, j] == 1


def test_perturbed_strategy_constructs_valid_graph():
    rng = np.random.default_rng(42)
    base = RandomStrategy(edge_prob=0.5, rng=rng)
    s = PerturbedStrategy(base=base, flip_prob=0.1, rng=rng)
    graph = s.construct(10)
    assert graph.shape == (10, 10)
    assert validate_adjacency(graph)


def test_perturbed_strategy_serialization_roundtrip():
    rng = np.random.default_rng(42)
    base = RandomStrategy(edge_prob=0.5, rng=rng)
    s = PerturbedStrategy(base=base, flip_prob=0.1, rng=rng)
    d = s.to_dict()
    assert d["name"] == "perturbed"
    s2 = strategy_from_dict(d, rng)
    assert isinstance(s2, PerturbedStrategy)
    assert s2.flip_prob == 0.1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategies.py -v`

Expected: FAIL (ImportError)

**Step 3: Implement strategies.py**

```python
# evolveclaw_ramsey/ramsey/strategies.py
"""Graph construction strategies for Ramsey optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isqrt

import numpy as np


class Strategy(ABC):
    """Base class for graph construction strategies."""

    name: str

    @abstractmethod
    def construct(self, n: int) -> np.ndarray:
        """Return an n x n symmetric binary adjacency matrix."""
        ...

    @abstractmethod
    def mutate(self, rng: np.random.Generator) -> Strategy:
        """Return a mutated copy with perturbed parameters."""
        ...

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        ...

    def params_key(self) -> tuple:
        """Return a hashable key for deduplication."""
        return (self.name,)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _quadratic_residues(p: int) -> set[int]:
    """Return the set of quadratic residues mod p (excluding 0)."""
    return {(x * x) % p for x in range(1, p)}


class RandomStrategy(Strategy):
    """Erdos-Renyi random graph with edge probability p."""

    name = "random"

    def __init__(self, edge_prob: float, rng: np.random.Generator):
        self.edge_prob = edge_prob
        self._rng = rng

    def construct(self, n: int) -> np.ndarray:
        child_rng = self._rng.spawn(1)[0]
        m = (child_rng.random((n, n)) < self.edge_prob).astype(np.int8)
        m = np.triu(m, 1)
        m = m + m.T
        return m

    def mutate(self, rng: np.random.Generator) -> Strategy:
        delta = rng.normal(0, 0.1)
        new_prob = max(0.05, min(0.95, self.edge_prob + delta))
        return RandomStrategy(edge_prob=new_prob, rng=rng)

    def to_dict(self) -> dict:
        return {"name": "random", "edge_prob": self.edge_prob}

    def params_key(self) -> tuple:
        return ("random", round(self.edge_prob, 4))


class PaleyStrategy(Strategy):
    """Paley graph construction: edge (i,j) if (i-j) is a quadratic residue mod n."""

    name = "paley"

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def construct(self, n: int) -> np.ndarray:
        if not (_is_prime(n) and n % 4 == 1):
            # Fall back to random
            fallback = RandomStrategy(edge_prob=0.5, rng=self._rng)
            return fallback.construct(n)
        qr = _quadratic_residues(n)
        m = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                diff = (i - j) % n
                if diff in qr:
                    m[i, j] = 1
                    m[j, i] = 1
        return m

    def mutate(self, rng: np.random.Generator) -> Strategy:
        # Paley has no params to mutate; return a perturbed version
        return PerturbedStrategy(base=PaleyStrategy(rng=rng), flip_prob=0.05, rng=rng)

    def to_dict(self) -> dict:
        return {"name": "paley"}

    def params_key(self) -> tuple:
        return ("paley",)


class CyclicStrategy(Strategy):
    """Cyclic graph: edge (i,j) if (j-i) mod n is in the offset set."""

    name = "cyclic"

    def __init__(self, offsets: list[int], rng: np.random.Generator):
        self.offsets = sorted(set(offsets))
        self._rng = rng

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

    def mutate(self, rng: np.random.Generator) -> Strategy:
        new_offsets = list(self.offsets)
        if len(new_offsets) > 1 and rng.random() < 0.3:
            # Remove a random offset
            idx = rng.integers(0, len(new_offsets))
            new_offsets.pop(idx)
        elif rng.random() < 0.5:
            # Add a random offset
            new_offsets.append(int(rng.integers(1, 20)))
        else:
            # Change a random offset
            if new_offsets:
                idx = rng.integers(0, len(new_offsets))
                new_offsets[idx] = int(rng.integers(1, 20))
        return CyclicStrategy(offsets=new_offsets, rng=rng)

    def to_dict(self) -> dict:
        return {"name": "cyclic", "offsets": self.offsets}

    def params_key(self) -> tuple:
        return ("cyclic", tuple(self.offsets))


class PerturbedStrategy(Strategy):
    """Takes a base strategy's output and randomly flips edges."""

    name = "perturbed"

    def __init__(self, base: Strategy, flip_prob: float, rng: np.random.Generator):
        self.base = base
        self.flip_prob = flip_prob
        self._rng = rng

    def construct(self, n: int) -> np.ndarray:
        child_rng = self._rng.spawn(1)[0]
        m = self.base.construct(n)
        flip_mask = (child_rng.random((n, n)) < self.flip_prob).astype(np.int8)
        flip_mask = np.triu(flip_mask, 1)
        flip_mask = flip_mask + flip_mask.T
        m = np.bitwise_xor(m, flip_mask).astype(np.int8)
        np.fill_diagonal(m, 0)
        return m

    def mutate(self, rng: np.random.Generator) -> Strategy:
        delta = rng.normal(0, 0.02)
        new_prob = max(0.01, min(0.5, self.flip_prob + delta))
        return PerturbedStrategy(base=self.base, flip_prob=new_prob, rng=rng)

    def to_dict(self) -> dict:
        return {
            "name": "perturbed",
            "base": self.base.to_dict(),
            "flip_prob": self.flip_prob,
        }

    def params_key(self) -> tuple:
        return ("perturbed", self.base.params_key(), round(self.flip_prob, 4))


_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "random": RandomStrategy,
    "paley": PaleyStrategy,
    "cyclic": CyclicStrategy,
    "perturbed": PerturbedStrategy,
}


def strategy_from_dict(d: dict, rng: np.random.Generator) -> Strategy:
    """Deserialize a strategy from a dict."""
    name = d["name"]
    if name == "random":
        return RandomStrategy(edge_prob=d["edge_prob"], rng=rng)
    elif name == "paley":
        return PaleyStrategy(rng=rng)
    elif name == "cyclic":
        return CyclicStrategy(offsets=d["offsets"], rng=rng)
    elif name == "perturbed":
        base = strategy_from_dict(d["base"], rng)
        return PerturbedStrategy(base=base, flip_prob=d["flip_prob"], rng=rng)
    else:
        raise ValueError(f"Unknown strategy type: {name}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategies.py -v`

Expected: all 10 tests PASS

**Step 5: Commit**

```bash
git add evolveclaw_ramsey/ramsey/strategies.py tests/test_strategies.py
git commit -m "feat: add strategy interface with Random, Paley, Cyclic, Perturbed implementations"
```

---

## Task 5: Utilities (`utils/config.py`, `utils/logging.py`)

**Files:**
- Create: `evolveclaw_ramsey/utils/config.py`
- Create: `evolveclaw_ramsey/utils/logging.py`

**Step 1: Implement config.py**

```python
# evolveclaw_ramsey/utils/config.py
"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml

_DEFAULTS = {
    "problem": {"s": 4, "t": 4, "n": 17, "penalty_weight": 1.0},
    "evolution": {
        "max_generations": 100,
        "population_size": 20,
        "tournament_k": 3,
        "checkpoint_interval": 10,
    },
    "proposer": {"type": "random"},
    "executor": {"timeout_seconds": 10},
    "logging": {"level": "INFO"},
    "seed": 42,
    "run_dir": "runs/",
}

_REQUIRED = ["problem"]


def load_config(path: str) -> dict:
    """Load a YAML config file and apply defaults for missing fields."""
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    for key in _REQUIRED:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    # Deep merge defaults
    merged = _deep_merge(_DEFAULTS, config)
    return merged


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge overrides into defaults."""
    result = dict(defaults)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
```

**Step 2: Implement logging.py**

```python
# evolveclaw_ramsey/utils/logging.py
"""Logging configuration for evolveclaw_ramsey."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "evolveclaw_ramsey"


def setup_logging(level: str = "INFO", run_dir: str | None = None) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if run_dir:
        log_path = Path(run_dir) / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the project logger."""
    return logging.getLogger(LOGGER_NAME)
```

**Step 3: Quick verification**

Run: `python -c "from evolveclaw_ramsey.utils.config import load_config; c = load_config('configs/demo.yaml'); print(c['problem']['s'])"`

Expected: prints `4`

**Step 4: Commit**

```bash
git add evolveclaw_ramsey/utils/config.py evolveclaw_ramsey/utils/logging.py
git commit -m "feat: add config loader with defaults and logging setup"
```

---

## Task 6: Executor (`harness/executor.py`)

**Files:**
- Create: `evolveclaw_ramsey/harness/executor.py`

**Step 1: Implement executor.py**

```python
# evolveclaw_ramsey/harness/executor.py
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
    """Result of executing a strategy."""
    graph: np.ndarray | None
    elapsed_seconds: float
    error: str | None


class Executor:
    """Execute a strategy's construct() with timeout and validation."""

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, strategy: Strategy, n: int) -> ExecutionResult:
        """Run strategy.construct(n) with timeout and output validation."""
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

        # Validate
        if not isinstance(graph, np.ndarray):
            return ExecutionResult(graph=None, elapsed_seconds=elapsed, error="Output is not ndarray")
        if graph.shape != (n, n):
            return ExecutionResult(
                graph=None, elapsed_seconds=elapsed,
                error=f"Expected shape ({n},{n}), got {graph.shape}",
            )
        if not validate_adjacency(graph):
            return ExecutionResult(
                graph=None, elapsed_seconds=elapsed,
                error="Output is not a valid adjacency matrix",
            )

        return ExecutionResult(graph=graph, elapsed_seconds=elapsed, error=None)
```

**Step 2: Quick verification**

Run: `python -c "from evolveclaw_ramsey.harness.executor import Executor; from evolveclaw_ramsey.ramsey.strategies import RandomStrategy; import numpy as np; e = Executor(); r = e.execute(RandomStrategy(0.5, np.random.default_rng(1)), 5); print(r.error, r.graph.shape)"`

Expected: `None (5, 5)`

**Step 3: Commit**

```bash
git add evolveclaw_ramsey/harness/executor.py
git commit -m "feat: add strategy executor with timeout and validation"
```

---

## Task 7: Evaluator (`harness/evaluator.py`)

**Files:**
- Create: `evolveclaw_ramsey/harness/evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write failing tests**

```python
# tests/test_evaluator.py
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
    # C_5 is a valid R(3,3) counterexample
    strategy = CyclicStrategy(offsets=[1], rng=rng)
    result = evaluator.evaluate(strategy, n=5)
    assert result.score_result is not None
    assert result.score_result.violation_count == 0
    assert result.score_result.score == 5.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluator.py -v`

Expected: FAIL (ImportError)

**Step 3: Implement evaluator.py**

```python
# evolveclaw_ramsey/harness/evaluator.py
"""Evaluator: combines executor + scorer into a single evaluation call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evolveclaw_ramsey.harness.executor import Executor, ExecutionResult
from evolveclaw_ramsey.ramsey.scoring import RamseyScorer, ScoreResult
from evolveclaw_ramsey.ramsey.strategies import Strategy


@dataclass
class EvalResult:
    """Full evaluation result combining execution and scoring."""
    strategy: Strategy
    graph: np.ndarray | None
    score_result: ScoreResult | None
    error: str | None
    elapsed_seconds: float


class Evaluator:
    """Execute a strategy and score the resulting graph."""

    def __init__(self, scorer: RamseyScorer, executor: Executor):
        self.scorer = scorer
        self.executor = executor

    def evaluate(self, strategy: Strategy, n: int) -> EvalResult:
        """Execute strategy and score the result."""
        exec_result = self.executor.execute(strategy, n)

        if exec_result.error:
            return EvalResult(
                strategy=strategy,
                graph=None,
                score_result=None,
                error=exec_result.error,
                elapsed_seconds=exec_result.elapsed_seconds,
            )

        score_result = self.scorer.score(exec_result.graph)

        return EvalResult(
            strategy=strategy,
            graph=exec_result.graph,
            score_result=score_result,
            error=None,
            elapsed_seconds=exec_result.elapsed_seconds,
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluator.py -v`

Expected: both tests PASS

**Step 5: Commit**

```bash
git add evolveclaw_ramsey/harness/evaluator.py tests/test_evaluator.py
git commit -m "feat: add evaluator combining executor and scorer"
```

---

## Task 8: Recorder (`harness/recorder.py`)

**Files:**
- Create: `evolveclaw_ramsey/harness/recorder.py`

**Step 1: Implement recorder.py**

```python
# evolveclaw_ramsey/harness/recorder.py
"""Experiment recorder: JSONL log, best.json, summary.txt, config snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from evolveclaw_ramsey.ramsey.scoring import ScoreResult
from evolveclaw_ramsey.ramsey.strategies import Strategy


class Recorder:
    """Persist experiment data to the run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / "log.jsonl"
        self._best_path = self.run_dir / "best.json"
        self._best_score: float | None = None

    def save_config(self, config: dict) -> None:
        """Save a frozen copy of the config."""
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def log_generation(
        self,
        gen: int,
        strategy: Strategy,
        score_result: ScoreResult,
        added: bool,
        extra: dict | None = None,
    ) -> None:
        """Append one generation record to log.jsonl."""
        record = {
            "generation": gen,
            "strategy_name": strategy.name,
            "strategy_params": strategy.to_dict(),
            "score": score_result.score,
            "violation_count": score_result.violation_count,
            "s_cliques": score_result.s_cliques,
            "t_cliques": score_result.t_cliques,
            "added_to_population": added,
        }
        if extra:
            record.update(extra)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Update best
        if self._best_score is None or score_result.score > self._best_score:
            self._best_score = score_result.score
            with open(self._best_path, "w") as f:
                json.dump(
                    {
                        "generation": gen,
                        "strategy": strategy.to_dict(),
                        "score": score_result.score,
                        "violation_count": score_result.violation_count,
                    },
                    f,
                    indent=2,
                )

    def log_error(self, gen: int, error: str) -> None:
        """Log an execution error."""
        record = {"generation": gen, "error": error}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def write_summary(self, best_strategy: Strategy, best_score: float, generations: int) -> None:
        """Write a human-readable summary."""
        summary = (
            f"EvolveClaw-Ramsey Run Summary\n"
            f"{'=' * 40}\n"
            f"Generations completed: {generations}\n"
            f"Best score: {best_score}\n"
            f"Best strategy: {best_strategy.name}\n"
            f"Best strategy params: {best_strategy.to_dict()}\n"
        )
        with open(self.run_dir / "summary.txt", "w") as f:
            f.write(summary)
```

**Step 2: Quick verification**

Run: `python -c "from evolveclaw_ramsey.harness.recorder import Recorder; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add evolveclaw_ramsey/harness/recorder.py
git commit -m "feat: add experiment recorder with JSONL log, best tracking, summary"
```

---

## Task 9: Checkpoint (`harness/checkpoint.py`)

**Files:**
- Create: `evolveclaw_ramsey/harness/checkpoint.py`

**Step 1: Implement checkpoint.py**

```python
# evolveclaw_ramsey/harness/checkpoint.py
"""Checkpoint save/load for population state and RNG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save(population_data: dict, generation: int, rng: np.random.Generator, run_dir: str) -> None:
    """Save population state and RNG to a checkpoint file."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng_state = rng.bit_generator.state
    # Convert numpy types to Python types for JSON serialization
    rng_state_serializable = _make_serializable(rng_state)

    data = {
        "generation": generation,
        "population": population_data,
        "rng_state": rng_state_serializable,
    }

    path = ckpt_dir / f"gen_{generation}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load(run_dir: str, generation: int | None = None) -> tuple[dict, int, dict]:
    """Load a checkpoint. Returns (population_data, generation, rng_state_dict).

    If generation is None, loads the latest checkpoint.
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory in {run_dir}")

    if generation is not None:
        path = ckpt_dir / f"gen_{generation}.json"
    else:
        files = sorted(ckpt_dir.glob("gen_*.json"))
        if not files:
            raise FileNotFoundError(f"No checkpoint files in {ckpt_dir}")
        path = files[-1]

    with open(path) as f:
        data = json.load(f)

    return data["population"], data["generation"], data["rng_state"]


def restore_rng(rng_state_dict: dict) -> np.random.Generator:
    """Restore a numpy Generator from a serialized state dict."""
    rng = np.random.default_rng()
    state = _restore_state(rng_state_dict)
    rng.bit_generator.state = state
    return rng


def _make_serializable(obj):
    """Convert numpy types to Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def _restore_state(obj):
    """Restore numpy types from JSON-deserialized data."""
    if isinstance(obj, dict):
        restored = {}
        for k, v in obj.items():
            if k == "s" and isinstance(v, dict) and "key" in v:
                # PCG64 state
                restored[k] = {
                    "state": v["state"] if isinstance(v["state"], int) else int(v["state"]),
                    "inc": v["inc"] if isinstance(v["inc"], int) else int(v["inc"]),
                    "has_uint32": v.get("has_uint32", 0),
                    "uinteger": v.get("uinteger", 0),
                }
            else:
                restored[k] = _restore_state(v)
        return restored
    elif isinstance(obj, list):
        return [_restore_state(v) for v in obj]
    return obj
```

**Step 2: Quick verification**

Run: `python -c "from evolveclaw_ramsey.harness.checkpoint import save, load; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add evolveclaw_ramsey/harness/checkpoint.py
git commit -m "feat: add checkpoint save/load for population and RNG state"
```

---

## Task 10: Population (`agent/population.py`)

**Files:**
- Create: `evolveclaw_ramsey/agent/population.py`

**Step 1: Implement population.py**

```python
# evolveclaw_ramsey/agent/population.py
"""Population management for evolutionary strategy search."""

from __future__ import annotations

import numpy as np

from evolveclaw_ramsey.ramsey.strategies import Strategy, strategy_from_dict


class Population:
    """A ranked collection of strategies with scores."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._members: list[tuple[Strategy, float]] = []
        self._seen_keys: set[tuple] = set()

    def add(self, strategy: Strategy, score: float) -> bool:
        """Add strategy if it scores better than worst or population not full.

        Returns True if added.
        """
        key = strategy.params_key()
        if key in self._seen_keys:
            return False

        if len(self._members) < self.max_size:
            self._members.append((strategy, score))
            self._seen_keys.add(key)
            self._members.sort(key=lambda x: x[1], reverse=True)
            return True

        # Check if better than worst
        if score > self._members[-1][1]:
            removed = self._members.pop()
            self._seen_keys.discard(removed[0].params_key())
            self._members.append((strategy, score))
            self._seen_keys.add(key)
            self._members.sort(key=lambda x: x[1], reverse=True)
            return True

        return False

    def tournament_select(self, k: int, rng: np.random.Generator) -> tuple[Strategy, float]:
        """Select the best of k randomly chosen members."""
        if not self._members:
            raise ValueError("Population is empty")
        k = min(k, len(self._members))
        indices = rng.choice(len(self._members), size=k, replace=False)
        candidates = [self._members[i] for i in indices]
        return max(candidates, key=lambda x: x[1])

    def best(self) -> tuple[Strategy, float]:
        """Return the highest-scoring member."""
        if not self._members:
            raise ValueError("Population is empty")
        return self._members[0]

    def size(self) -> int:
        return len(self._members)

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "max_size": self.max_size,
            "members": [
                {"strategy": s.to_dict(), "score": score}
                for s, score in self._members
            ],
        }

    @classmethod
    def from_dict(cls, d: dict, rng: np.random.Generator) -> Population:
        """Deserialize from checkpoint."""
        pop = cls(max_size=d["max_size"])
        for entry in d["members"]:
            strategy = strategy_from_dict(entry["strategy"], rng)
            pop._members.append((strategy, entry["score"]))
            pop._seen_keys.add(strategy.params_key())
        pop._members.sort(key=lambda x: x[1], reverse=True)
        return pop
```

**Step 2: Quick verification**

Run: `python -c "from evolveclaw_ramsey.agent.population import Population; p = Population(5); print('OK')" `

Expected: `OK`

**Step 3: Commit**

```bash
git add evolveclaw_ramsey/agent/population.py
git commit -m "feat: add population management with tournament selection and dedup"
```

---

## Task 11: Proposer (`agent/proposer.py`)

**Files:**
- Create: `evolveclaw_ramsey/agent/proposer.py`

**Step 1: Implement proposer.py**

```python
# evolveclaw_ramsey/agent/proposer.py
"""Strategy proposers: random mutation and optional LLM-based."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

import numpy as np

from evolveclaw_ramsey.ramsey.strategies import (
    Strategy,
    RandomStrategy,
    PaleyStrategy,
    CyclicStrategy,
    PerturbedStrategy,
    strategy_from_dict,
)
from evolveclaw_ramsey.utils.logging import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Proposer(ABC):
    """Base class for strategy proposers."""

    @abstractmethod
    def propose(
        self, parents: list[Strategy], scores: list[float], problem: dict
    ) -> Strategy:
        """Generate a new candidate strategy."""
        ...


class RandomMutationProposer(Proposer):
    """Propose new strategies via random mutation of parents."""

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def propose(
        self, parents: list[Strategy], scores: list[float], problem: dict
    ) -> Strategy:
        if not parents:
            return RandomStrategy(edge_prob=self._rng.random(), rng=self._rng)

        parent = parents[0]

        # With some probability, switch strategy type entirely
        if self._rng.random() < 0.15:
            n = problem.get("n", 17)
            choice = self._rng.integers(0, 4)
            if choice == 0:
                return RandomStrategy(edge_prob=self._rng.random(), rng=self._rng)
            elif choice == 1:
                return PaleyStrategy(rng=self._rng)
            elif choice == 2:
                num_offsets = int(self._rng.integers(1, n // 2))
                offsets = [int(x) for x in self._rng.integers(1, n, size=num_offsets)]
                return CyclicStrategy(offsets=offsets, rng=self._rng)
            else:
                return PerturbedStrategy(
                    base=parent, flip_prob=float(self._rng.uniform(0.01, 0.2)), rng=self._rng
                )

        # Otherwise, mutate the parent
        return parent.mutate(self._rng)


class LLMProposer(Proposer):
    """Propose new strategies by asking an LLM for suggestions."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        rng: np.random.Generator,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._rng = rng
        self._fallback = RandomMutationProposer(rng=rng)

    def propose(
        self, parents: list[Strategy], scores: list[float], problem: dict
    ) -> Strategy:
        prompt = self._build_prompt(parents, scores, problem)
        try:
            response_text = self._call_llm(prompt)
            strategy = self._parse_response(response_text)
            return strategy
        except Exception as e:
            logger.warning(f"LLM proposer failed, falling back to random: {e}")
            return self._fallback.propose(parents, scores, problem)

    def _build_prompt(
        self, parents: list[Strategy], scores: list[float], problem: dict
    ) -> str:
        parent_info = ""
        for p, s in zip(parents, scores):
            parent_info += f"  Strategy: {json.dumps(p.to_dict())}\n  Score: {s}\n\n"

        return f"""You are optimizing graph construction strategies for Ramsey number R({problem['s']},{problem['t']}).
Goal: Find a graph on {problem['n']} vertices with NO clique of size {problem['s']} and NO independent set of size {problem['t']}.
Higher score = better. Perfect score = {problem['n']} (zero violations).

Current parent strategies:
{parent_info}
Available strategy types and their JSON format:
1. {{"name": "random", "edge_prob": <float 0-1>}}
2. {{"name": "paley"}}
3. {{"name": "cyclic", "offsets": [<int>, ...]}}
4. {{"name": "perturbed", "base": <strategy dict>, "flip_prob": <float 0-0.5>}}

Suggest an improved strategy. Return ONLY a JSON object with the strategy specification, nothing else."""

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        elif self.provider == "openai":
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _parse_response(self, text: str) -> Strategy:
        # Try to extract JSON from the response
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        d = json.loads(text)
        return strategy_from_dict(d, self._rng)


def create_proposer(config: dict, rng: np.random.Generator) -> Proposer:
    """Factory function to create a proposer from config."""
    proposer_type = config.get("type", "random")

    if proposer_type == "random":
        return RandomMutationProposer(rng=rng)
    elif proposer_type == "llm":
        provider = config.get("llm_provider", "anthropic")
        model = config.get("llm_model", "claude-sonnet-4-20250514")
        api_key_env = config.get("llm_api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"LLM proposer requires API key. Set the {api_key_env} environment variable."
            )
        return LLMProposer(provider=provider, model=model, api_key=api_key, rng=rng)
    else:
        raise ValueError(f"Unknown proposer type: {proposer_type}")
```

**Step 2: Quick verification**

Run: `python -c "from evolveclaw_ramsey.agent.proposer import create_proposer; import numpy as np; p = create_proposer({'type': 'random'}, np.random.default_rng(42)); print(type(p).__name__)"`

Expected: `RandomMutationProposer`

**Step 3: Commit**

```bash
git add evolveclaw_ramsey/agent/proposer.py
git commit -m "feat: add proposer interface with random mutation and LLM implementations"
```

---

## Task 12: Evolution Loop (`agent/loop.py`)

**Files:**
- Create: `evolveclaw_ramsey/agent/loop.py`
- Create: `tests/test_loop.py`

**Step 1: Write failing test**

```python
# tests/test_loop.py
import tempfile

from evolveclaw_ramsey.agent.loop import run_evolution


def test_evolution_loop_minimal():
    """A 5-generation run with small graph completes without error."""
    config = {
        "problem": {"s": 3, "t": 3, "n": 6, "penalty_weight": 1.0},
        "evolution": {
            "max_generations": 5,
            "population_size": 5,
            "tournament_k": 2,
            "checkpoint_interval": 5,
        },
        "proposer": {"type": "random"},
        "executor": {"timeout_seconds": 5},
        "logging": {"level": "WARNING"},
        "seed": 42,
        "run_dir": tempfile.mkdtemp(),
    }
    result = run_evolution(config)
    assert result.generations_completed == 5
    assert result.best_score is not None
    assert result.best_strategy is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop.py -v`

Expected: FAIL (ImportError)

**Step 3: Implement loop.py**

```python
# evolveclaw_ramsey/agent/loop.py
"""Evolution loop: the core agent that evolves strategies."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evolveclaw_ramsey.agent.population import Population
from evolveclaw_ramsey.agent.proposer import create_proposer
from evolveclaw_ramsey.harness import checkpoint
from evolveclaw_ramsey.harness.evaluator import Evaluator
from evolveclaw_ramsey.harness.executor import Executor
from evolveclaw_ramsey.harness.recorder import Recorder
from evolveclaw_ramsey.ramsey.scoring import RamseyScorer
from evolveclaw_ramsey.ramsey.strategies import (
    RandomStrategy,
    PaleyStrategy,
    CyclicStrategy,
    Strategy,
)
from evolveclaw_ramsey.utils.logging import setup_logging, get_logger


@dataclass
class RunResult:
    """Result of an evolution run."""
    best_strategy: Strategy
    best_score: float
    run_dir: str
    generations_completed: int


def _make_run_dir(base: str, config_stem: str = "run") -> str:
    """Create a timestamped run directory."""
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{now}_{config_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def initialize_population(
    population: Population,
    config: dict,
    evaluator: Evaluator,
    rng: np.random.Generator,
) -> None:
    """Seed the population with diverse initial strategies."""
    n = config["problem"]["n"]
    pop_size = config["evolution"]["population_size"]

    initial_strategies: list[Strategy] = []

    # One random at 0.5
    initial_strategies.append(RandomStrategy(edge_prob=0.5, rng=rng))

    # One Paley (if valid for n)
    initial_strategies.append(PaleyStrategy(rng=rng))

    # One cyclic with random offsets
    num_offsets = max(1, n // 4)
    offsets = [int(x) for x in rng.integers(1, max(2, n // 2), size=num_offsets)]
    initial_strategies.append(CyclicStrategy(offsets=offsets, rng=rng))

    # Fill remaining with random at varied edge probabilities
    remaining = max(0, pop_size - len(initial_strategies))
    if remaining > 0:
        probs = np.linspace(0.2, 0.8, remaining)
        for p in probs:
            initial_strategies.append(RandomStrategy(edge_prob=float(p), rng=rng))

    for strategy in initial_strategies:
        result = evaluator.evaluate(strategy, n)
        if result.error is None and result.score_result is not None:
            population.add(strategy, result.score_result.score)


def run_evolution(config: dict, resume_dir: str | None = None) -> RunResult:
    """Run the evolution loop."""
    logger = setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
    )

    # Setup RNG
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    # Setup run directory
    run_dir = resume_dir or _make_run_dir(config.get("run_dir", "runs/"))

    # Setup logging with file handler
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        run_dir=run_dir,
    )
    logger = get_logger()

    # Setup components
    scorer = RamseyScorer(
        s=config["problem"]["s"],
        t=config["problem"]["t"],
        penalty_weight=config["problem"].get("penalty_weight", 1.0),
    )
    executor = Executor(config["executor"]["timeout_seconds"])
    evaluator = Evaluator(scorer=scorer, executor=executor)
    recorder = Recorder(run_dir)
    recorder.save_config(config)
    proposer = create_proposer(config["proposer"], rng)

    # Initialize or restore population
    start_gen = 0
    population = Population(config["evolution"]["population_size"])

    if resume_dir:
        try:
            pop_data, start_gen, rng_state = checkpoint.load(resume_dir)
            population = Population.from_dict(pop_data, rng)
            rng = checkpoint.restore_rng(rng_state)
            start_gen += 1
            logger.info(f"Resumed from generation {start_gen}")
        except FileNotFoundError:
            logger.warning("No checkpoint found, starting fresh")

    if population.size() == 0:
        logger.info("Initializing population...")
        initialize_population(population, config, evaluator, rng)
        logger.info(f"Population initialized with {population.size()} members")

    best_strat, best_score = population.best()
    logger.info(f"Initial best: score={best_score:.2f}, strategy={best_strat.name}")

    n = config["problem"]["n"]
    max_gen = config["evolution"]["max_generations"]
    tournament_k = config["evolution"]["tournament_k"]
    ckpt_interval = config["evolution"]["checkpoint_interval"]

    gen = start_gen
    for gen in range(start_gen, max_gen):
        # Select parent
        parent, parent_score = population.tournament_select(tournament_k, rng)

        # Propose candidate
        candidate = proposer.propose([parent], [parent_score], config["problem"])

        # Execute
        exec_result = executor.execute(candidate, n)
        if exec_result.error:
            recorder.log_error(gen, exec_result.error)
            logger.debug(f"Gen {gen}: execution error: {exec_result.error}")
            continue

        # Score
        score_result = scorer.score(exec_result.graph)

        # Update population
        added = population.add(candidate, score_result.score)

        # Record
        recorder.log_generation(gen, candidate, score_result, added)

        # Log progress
        current_best_strat, current_best_score = population.best()
        if score_result.score > best_score:
            best_score = score_result.score
            best_strat = current_best_strat
            logger.info(
                f"Gen {gen}: NEW BEST score={best_score:.2f} "
                f"violations={score_result.violation_count} "
                f"strategy={candidate.name}"
            )
        elif gen % 10 == 0:
            logger.info(
                f"Gen {gen}: score={score_result.score:.2f} "
                f"best={best_score:.2f}"
            )

        # Checkpoint
        if gen > 0 and gen % ckpt_interval == 0:
            checkpoint.save(population.to_dict(), gen, rng, run_dir)
            logger.debug(f"Checkpoint saved at generation {gen}")

        # Early termination if perfect score
        if score_result.violation_count == 0:
            logger.info(f"Gen {gen}: PERFECT SCORE! No violations found.")
            break

    # Final checkpoint and summary
    best_strat, best_score = population.best()
    checkpoint.save(population.to_dict(), gen, rng, run_dir)
    recorder.write_summary(best_strat, best_score, gen + 1)
    logger.info(f"Run complete. Best score: {best_score:.2f}, strategy: {best_strat.name}")

    return RunResult(
        best_strategy=best_strat,
        best_score=best_score,
        run_dir=run_dir,
        generations_completed=gen + 1,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_loop.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add evolveclaw_ramsey/agent/loop.py tests/test_loop.py
git commit -m "feat: add evolution loop with population init, checkpointing, early termination"
```

---

## Task 13: CLI (`cli.py`)

**Files:**
- Create: `evolveclaw_ramsey/cli.py`

**Step 1: Implement cli.py**

```python
# evolveclaw_ramsey/cli.py
"""Command-line interface for EvolveClaw-Ramsey."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evolveclaw_ramsey.agent.loop import run_evolution
from evolveclaw_ramsey.harness.evaluator import Evaluator
from evolveclaw_ramsey.harness.executor import Executor
from evolveclaw_ramsey.ramsey.scoring import RamseyScorer
from evolveclaw_ramsey.ramsey.strategies import strategy_from_dict
from evolveclaw_ramsey.utils.config import load_config


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the evolution loop."""
    config = load_config(args.config)
    resume_dir = args.resume if args.resume else None
    result = run_evolution(config, resume_dir=resume_dir)
    print(f"\nRun complete: {result.run_dir}")
    print(f"Best score: {result.best_score:.2f}")
    print(f"Best strategy: {result.best_strategy.name}")
    print(f"Generations: {result.generations_completed}")


def cmd_eval(args: argparse.Namespace) -> None:
    """Evaluate a single strategy."""
    # Load strategy from JSON file
    with open(args.strategy) as f:
        strategy_dict = json.load(f)

    rng = np.random.default_rng(42)
    strategy = strategy_from_dict(strategy_dict, rng)

    scorer = RamseyScorer(s=args.s, t=args.t)
    executor = Executor(timeout_seconds=10)
    evaluator = Evaluator(scorer=scorer, executor=executor)

    result = evaluator.evaluate(strategy, n=args.n)

    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)

    print(f"Strategy: {strategy.name}")
    print(f"n={args.n}, R({args.s},{args.t})")
    print(f"Score: {result.score_result.score:.2f}")
    print(f"Violations: {result.score_result.violation_count}")
    print(f"  s-cliques (K{args.s}): {result.score_result.s_cliques}")
    print(f"  t-cliques (K{args.t} in complement): {result.score_result.t_cliques}")
    print(f"Time: {result.elapsed_seconds:.3f}s")

    # ASCII adjacency matrix
    print(f"\nAdjacency matrix ({args.n}x{args.n}):")
    for i in range(args.n):
        row = ""
        for j in range(args.n):
            row += "#" if result.graph[i, j] == 1 else "."
        print(f"  {row}")


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay a past run from its log."""
    run_dir = Path(args.run_dir)
    log_path = run_dir / "log.jsonl"

    if not log_path.exists():
        print(f"No log.jsonl found in {run_dir}")
        sys.exit(1)

    print(f"Replaying run: {run_dir}")
    print("=" * 50)

    best_score = float("-inf")
    gen_count = 0
    error_count = 0

    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            if "error" in record:
                error_count += 1
                continue
            gen = record["generation"]
            score = record["score"]
            name = record["strategy_name"]
            violations = record["violation_count"]
            added = record.get("added_to_population", False)
            marker = " *NEW BEST*" if score > best_score else ""
            if score > best_score:
                best_score = score
            gen_count = gen + 1
            if added or marker:
                print(
                    f"Gen {gen:4d}: score={score:8.2f} violations={violations:3d} "
                    f"strategy={name:10s}{marker}"
                )

    print("=" * 50)
    print(f"Generations: {gen_count}, Errors: {error_count}")
    print(f"Best score: {best_score:.2f}")

    # Print summary if available
    summary_path = run_dir / "summary.txt"
    if summary_path.exists():
        print(f"\n{summary_path.read_text()}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="evolveclaw-ramsey",
        description="Minimal educational AlphaEvolve-style Ramsey optimization",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = subparsers.add_parser("run", help="Run the evolution loop")
    run_parser.add_argument("--config", required=True, help="Path to YAML config file")
    run_parser.add_argument("--resume", default=None, help="Run directory to resume from")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a single strategy")
    eval_parser.add_argument("--strategy", required=True, help="Path to strategy JSON file")
    eval_parser.add_argument("--n", type=int, required=True, help="Number of vertices")
    eval_parser.add_argument("--s", type=int, required=True, help="Clique size s")
    eval_parser.add_argument("--t", type=int, required=True, help="Clique size t")

    # replay
    replay_parser = subparsers.add_parser("replay", help="Replay a past run")
    replay_parser.add_argument("--run-dir", required=True, help="Path to run directory")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "replay":
        cmd_replay(args)


if __name__ == "__main__":
    main()
```

**Step 2: Verify CLI help works**

Run: `python -m evolveclaw_ramsey.cli --help`

Expected: help text showing `run`, `eval`, `replay` subcommands

**Step 3: Verify run subcommand with demo config**

Run: `cd D:/project/github/EvolveClaw && python -m evolveclaw_ramsey.cli run --config configs/demo.yaml`

Expected: evolution loop runs, prints progress, completes with a summary

**Step 4: Commit**

```bash
git add evolveclaw_ramsey/cli.py
git commit -m "feat: add CLI with run, eval, replay subcommands"
```

---

## Task 14: Demo Scripts

**Files:**
- Create: `scripts/run_demo.sh`
- Create: `scripts/run_search.sh`

**Step 1: Create run_demo.sh**

```bash
#!/usr/bin/env bash
# Quick demo: R(3,3) on n=5 (trivial, fast)
set -e
cd "$(dirname "$0")/.."

echo "=== EvolveClaw-Ramsey Demo ==="
echo "Running a quick demo: R(3,3) search on n=5"
echo ""

python -m evolveclaw_ramsey.cli run --config configs/demo.yaml

echo ""
echo "Done! Check the runs/ directory for results."
```

**Step 2: Create run_search.sh**

```bash
#!/usr/bin/env bash
# Full search: R(4,4) on n=17
set -e
cd "$(dirname "$0")/.."

echo "=== EvolveClaw-Ramsey Search ==="
echo "Running R(4,4) search on n=17 (this may take a while)"
echo ""

python -m evolveclaw_ramsey.cli run --config configs/demo.yaml

echo ""
echo "Done! Check the runs/ directory for results."
```

**Step 3: Make executable and commit**

```bash
chmod +x scripts/run_demo.sh scripts/run_search.sh
git add scripts/
git commit -m "feat: add demo and search shell scripts"
```

---

## Task 15: README

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

Write a comprehensive README covering all sections required by the prompt (Section VIII):
1. Project intro and relation to paper
2. Core ideas (AlphaEvolve, harness engineering)
3. References and inspirations (AlphaEvolve, OpenEvolve, google-research/ramsey, OpenClaw, OpenCode, nanobot, A3S-Code)
4. Project boundaries (educational simplifications)
5. Quick start (install, run demo, view results)
6. Repository structure
7. Technical details (candidate representation, scoring, evolution loop, harness layers)
8. Future extensions

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with project overview and technical details"
```

---

## Task 16: Research Notes

**Files:**
- Create: `research/notes.md`
- Create: `research/sources.md`

**Step 1: Create research/notes.md**

Summary of key findings from research:
- AlphaEvolve: evolves programs not solutions, uses LLM as mutator, MAP-Elites population
- OpenEvolve: faithful open-source reimplementation, 4-component architecture
- Ramsey paper results: improved bounds for R(3,13), R(3,18), R(4,13), R(4,14), R(4,15)
- Our simplifications: single population, synchronous loop, strategy objects instead of code strings

**Step 2: Create research/sources.md**

List all referenced materials:
- arXiv:2506.13131 (AlphaEvolve paper)
- arXiv:2603.09172 (Ramsey numbers paper)
- github.com/algorithmicsuperintelligence/openevolve
- github.com/google-research/google-research/tree/master/ramsey_number_bounds
- OpenClaw, OpenCode, nanobot, A3S-Code references

**Step 3: Commit**

```bash
git add research/
git commit -m "docs: add research notes and sources"
```

---

## Task 17: Integration Smoke Test

**Step 1: Run full test suite**

Run: `pytest tests/ -v`

Expected: all tests PASS

**Step 2: Run end-to-end demo**

Run: `python -m evolveclaw_ramsey.cli run --config configs/demo.yaml`

Expected: completes successfully, creates a run directory in `runs/`

**Step 3: Verify run artifacts**

Run: `ls runs/*/`

Expected: `config.yaml`, `log.jsonl`, `best.json`, `summary.txt`, `run.log`, `checkpoints/`

**Step 4: Test eval subcommand**

Run: `python -m evolveclaw_ramsey.cli eval --strategy runs/*/best.json --n 17 --s 4 --t 4`

Note: the best.json from `run` wraps the strategy in a top-level object. For eval, you may need to extract just the strategy field. If this doesn't work directly, create a test strategy file:

```bash
echo '{"name": "random", "edge_prob": 0.5}' > /tmp/test_strategy.json
python -m evolveclaw_ramsey.cli eval --strategy /tmp/test_strategy.json --n 17 --s 4 --t 4
```

Expected: prints score, violations, and ASCII adjacency matrix

**Step 5: Test replay subcommand**

Run: `python -m evolveclaw_ramsey.cli replay --run-dir runs/$(ls runs/ | head -1)`

Expected: prints generation-by-generation progress and summary

**Step 6: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: integration test fixes"
```

---

## Task Summary

| Task | Component | Estimated Steps |
|------|-----------|----------------|
| 1 | Project scaffolding | 10 |
| 2 | Graph representation | 5 |
| 3 | Ramsey scoring | 5 |
| 4 | Strategies | 5 |
| 5 | Config + logging utils | 4 |
| 6 | Executor | 3 |
| 7 | Evaluator | 5 |
| 8 | Recorder | 3 |
| 9 | Checkpoint | 3 |
| 10 | Population | 3 |
| 11 | Proposer | 3 |
| 12 | Evolution loop | 5 |
| 13 | CLI | 4 |
| 14 | Demo scripts | 3 |
| 15 | README | 2 |
| 16 | Research notes | 3 |
| 17 | Integration smoke test | 6 |

**Total: 17 tasks, ~72 steps**

Dependencies: Tasks 2-4 (domain) are independent. Task 5 (utils) is independent. Tasks 6-9 (harness) depend on domain. Tasks 10-11 (agent) depend on domain. Task 12 (loop) depends on everything. Task 13 (CLI) depends on loop. Tasks 14-16 (docs) are independent. Task 17 depends on everything.
