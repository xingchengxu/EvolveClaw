# EvolveClaw-Ramsey: Design Specification

**Date:** 2026-03-14
**Status:** Approved
**Project:** evolveclaw-ramsey — A minimal educational AlphaEvolve-style Ramsey optimization system

---

## 1. Overview

### What This Is

A minimal, educational implementation of the core ideas from Google DeepMind's AlphaEvolve paper, applied to Ramsey number lower bound search. The system evolves graph construction strategies using an evolutionary loop with optional LLM-guided mutations, evaluates candidates against Ramsey constraints, and tracks optimization progress through a structured harness.

### What This Is Not

- Not a production-grade or complete reimplementation of AlphaEvolve
- Not expected to reproduce the paper's results (improving R(3,13), R(4,13), etc.)
- Not a general-purpose evolutionary coding framework

### Goals

1. Demonstrate the "agent + search + evaluation harness + iterative improvement" closed loop
2. Provide a clear, runnable educational codebase
3. Embody Harness Engineering principles: reproducibility, modularity, logging, checkpointing
4. Serve as a foundation for extensions (stronger LLM proposers, larger Ramsey targets, parallel evaluation)

### Inspirations

| Project | What We Borrow |
|---------|---------------|
| **AlphaEvolve** (DeepMind) | Core idea: evolve algorithms/strategies, not raw solutions. Prompt → LLM → evaluate → store loop. |
| **OpenEvolve** | Architecture reference: population management, evaluator decoupling, checkpoint design. We simplify: single population (no islands), synchronous loop (no async), simple ranked list (no MAP-Elites). |
| **google-research/ramsey_number_bounds** | Problem domain: Ramsey graph evaluation, violation counting, construction heuristics. |
| **OpenClaw** | Agent design inspiration: skills system, modular integrations, daemon-style persistent operation. We adopt: modular proposer interface, pluggable strategy system. |
| **OpenCode** | Code agent architecture: client/server separation, multi-agent roles (build vs. plan). We adopt: CLI with distinct subcommands (run/eval/replay), clear separation of concerns. |
| **nanobot** | Educational minimalism: ~4000 lines, companion study guide. We adopt: minimal codebase, educational documentation, clear learning path. |
| **A3S-Code** | No public repository or documentation was found for this project during research. If it becomes publicly available, its autonomous code generation patterns may inform future extensions to the LLM proposer. Listed here for completeness per the original prompt requirements. |

---

## 2. Problem Domain

### Ramsey Numbers

R(s, t) is the minimum number of vertices n such that every 2-coloring of edges of the complete graph K_n contains either a red clique of size s or a blue clique of size t.

Equivalently: R(s, t) is the minimum n such that every graph on n vertices contains either a clique of size s or an independent set of size t.

### Target Task

**Default demo task: R(4, 4) on n = 17**

Search for a 2-coloring of K_17 (equivalently, a graph on 17 vertices) that contains no monochromatic K_4. Since R(4,4) = 18, such a coloring is known to exist, making this a feasible search target that demonstrates the system's ability to find valid constructions.

### Scoring Function

Given a graph G on n vertices with target (s, t):

```
violations = count_cliques(G, s) + count_cliques(complement(G), t)
score = n - violations * penalty_weight
```

- A perfect score equals n (no violations)
- Higher is better
- The penalty_weight is configurable (default: 1.0)

For clique detection: exact enumeration via `itertools.combinations` for s, t ≤ 6 and n ≤ 50. This is computationally feasible for the educational scope.

---

## 3. Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────┐
│                    CLI                       │
│         run | eval | replay                  │
├─────────────────────────────────────────────┤
│               Agent Layer                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐│
│  │Evolution │ │Proposer  │ │  Population  ││
│  │  Loop    │ │(Random/  │ │  (ranked     ││
│  │          │ │ LLM)     │ │   strategies)││
│  └──────────┘ └──────────┘ └──────────────┘│
├─────────────────────────────────────────────┤
│              Harness Layer                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐│
│  │Executor  │ │Evaluator │ │  Recorder    ││
│  │(timeout, │ │(exec +   │ │  (JSONL log, ││
│  │ validate)│ │ score)   │ │   checkpoint)││
│  └──────────┘ └──────────┘ └──────────────┘│
├─────────────────────────────────────────────┤
│             Ramsey Domain                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐│
│  │Graph     │ │Scoring   │ │  Strategies  ││
│  │Repr      │ │(clique   │ │  (Random,    ││
│  │(numpy)   │ │ detect)  │ │   Paley, ...) │
│  └──────────┘ └──────────┘ └──────────────┘│
└─────────────────────────────────────────────┘
```

### Data Flow

```
Config (YAML)
  │
  ▼
Evolution Loop
  │
  ├──► SELECT parent(s) from Population (tournament)
  │
  ├──► PROPOSE new Strategy via Proposer
  │         │
  │         ├── RandomMutationProposer: perturb params, flip type
  │         └── LLMProposer: send parent + score to LLM, parse response
  │
  ├──► EXECUTE via Executor: strategy.construct(n) → adjacency matrix
  │         │
  │         └── Timeout, exception handling, output validation
  │
  ├──► SCORE via RamseyScorer: count violations → compute score
  │
  ├──► UPDATE Population: insert if better than worst
  │
  ├──► RECORD via Recorder: append to log.jsonl, update best.json
  │
  └──► CHECKPOINT every k generations: save population state
```

---

## 4. Core Components

### 4.1 Strategy Interface (`ramsey/strategies.py`)

```python
class Strategy(ABC):
    """A callable graph construction strategy."""
    name: str        # e.g., "random", "paley", "cyclic"
    params: dict     # strategy-specific parameters

    @abstractmethod
    def construct(self, n: int) -> np.ndarray:
        """Return an n x n symmetric binary adjacency matrix."""
        ...

    @abstractmethod
    def mutate(self, rng: np.random.Generator) -> "Strategy":
        """Return a mutated copy with perturbed parameters."""
        ...

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict for checkpointing."""
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        """Deserialize from dict."""
        ...
```

Built-in implementations:

| Strategy | Parameters | Construction Method |
|----------|-----------|-------------------|
| `RandomStrategy` | `edge_prob: float` | Each edge exists with probability p (Erdos-Renyi) |
| `PaleyStrategy` | *(none)* | Quadratic residue graph: edge (i,j) if `(i-j) % n` is a quadratic residue mod n. Only valid when n is prime and n % 4 == 1. If n does not satisfy these conditions, falls back to `RandomStrategy`. The `construct(n)` argument determines the graph size; no separate `prime` parameter is stored. |
| `CyclicStrategy` | `offsets: list[int]` | Edge (i,j) exists if `(j-i) % n` in offsets |
| `PerturbedStrategy` | `base: Strategy, flip_prob: float` | Take base graph, randomly flip edges |

### 4.2 Graph Representation (`ramsey/graph_repr.py`)

- `np.ndarray` shape `(n, n)`, dtype `int8`
- Symmetric with zero diagonal
- Utility functions: `validate_adjacency(matrix)`, `complement(matrix)`, `to_edge_list(matrix)`, `from_edge_list(edges, n)`

### 4.3 Scoring (`ramsey/scoring.py`)

```python
class RamseyScorer:
    def __init__(self, s: int, t: int, penalty_weight: float = 1.0): ...

    def score(self, graph: np.ndarray) -> ScoreResult:
        """Return ScoreResult with violation_count, clique_details, score."""
        ...

    def count_cliques(self, graph: np.ndarray, k: int) -> int:
        """Count k-cliques in graph via itertools.combinations."""
        ...
```

`ScoreResult` is a dataclass:
```python
@dataclass
class ScoreResult:
    score: float
    violation_count: int
    s_cliques: int       # cliques of size s in G
    t_cliques: int       # cliques of size t in complement(G)
    n: int
```

### 4.4 Population (`agent/population.py`)

```python
class Population:
    def __init__(self, max_size: int = 20): ...

    def add(self, strategy: Strategy, score: float) -> bool:
        """Add if score > worst member or population not full. Returns True if added."""
        ...

    def tournament_select(self, k: int, rng) -> tuple[Strategy, float]:
        """Select best of k random members."""
        ...

    def best(self) -> tuple[Strategy, float]: ...
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> "Population": ...
```

Deduplication: strategies with identical `(name, params)` hash are not added twice.

### 4.5 Proposer (`agent/proposer.py`)

```python
class Proposer(ABC):
    @abstractmethod
    def propose(self, parents: list[Strategy], scores: list[float],
                problem: dict) -> Strategy:
        """Generate a new candidate strategy."""
        ...

class RandomMutationProposer(Proposer):
    """Mutate parent strategy with random parameter perturbation."""
    ...

class LLMProposer(Proposer):
    """Use an LLM to suggest improved strategies."""
    def __init__(self, provider: str, model: str, api_key: str): ...
    ...
```

The `LLMProposer` prompt template includes:
1. Ramsey problem definition
2. Available strategy types and their parameters
3. Parent strategy code representation and score
4. Best-so-far score for context
5. Instruction to return a JSON strategy specification

**LLM error handling:** The `LLMProposer` implements the following fallback chain:
- If the API call fails (network error, auth error, rate limit): log the error and fall back to `RandomMutationProposer.propose()` for this generation
- If the LLM returns unparseable JSON: log the raw response and fall back to random mutation
- If the LLM returns a strategy type not in the built-in set: log a warning and fall back to random mutation
- If the API key environment variable is not set at init time: raise a clear error immediately with instructions
- All fallback events are recorded in the generation log with `proposer_fallback: true`

A factory function `create_proposer(config: dict) -> Proposer` in `agent/proposer.py` handles instantiation based on the `proposer.type` config field.

### 4.6 Evolution Loop (`agent/loop.py`)

```python
def run_evolution(config: dict) -> RunResult:
    """Main evolution loop. Returns RunResult with best strategy and history."""

    population = Population(config["evolution"]["population_size"])
    proposer = create_proposer(config["proposer"])
    scorer = RamseyScorer(config["problem"]["s"], config["problem"]["t"])
    executor = Executor(config["executor"]["timeout_seconds"])
    recorder = Recorder(run_dir)

    # Initialize population with diverse random strategies
    initialize_population(population, config, scorer, executor)

    for gen in range(config["evolution"]["max_generations"]):
        # Select parent
        parent, parent_score = population.tournament_select(
            config["evolution"]["tournament_k"], rng)

        # Propose new candidate
        candidate = proposer.propose([parent], [parent_score], config["problem"])

        # Execute and evaluate
        exec_result = executor.execute(candidate, config["problem"]["n"])
        if exec_result.error:
            recorder.log_error(gen, exec_result.error)
            continue

        score_result = scorer.score(exec_result.graph)

        # Update population
        added = population.add(candidate, score_result.score)

        # Record
        recorder.log_generation(gen, candidate, score_result, added)

        # Checkpoint
        if gen % config["evolution"]["checkpoint_interval"] == 0:
            checkpoint.save(population, gen, rng, run_dir)

    recorder.write_summary(population)
    best_strategy, best_score = population.best()
    return RunResult(best_strategy, best_score, recorder.run_dir, gen + 1)
```

```python
@dataclass
class RunResult:
    best_strategy: Strategy
    best_score: float
    run_dir: str
    generations_completed: int
```

**`initialize_population`:** Creates `population_size` initial strategies with diversity:
- One `RandomStrategy` with `edge_prob=0.5`
- One `PaleyStrategy` (if n is valid for Paley)
- One `CyclicStrategy` with random offsets
- Remaining slots filled with `RandomStrategy` at varied edge probabilities (evenly spaced from 0.2 to 0.8)
- Each initial strategy is evaluated via `Evaluator.evaluate()` and added to the population with its score

**Seed propagation:** The config `seed` value is used to create a `numpy.random.Generator` via `np.random.default_rng(seed)`. This single RNG instance is passed to:
- `initialize_population()` for generating random initial strategies
- `population.tournament_select()` for random member selection
- `RandomMutationProposer.propose()` for mutation randomness
- Each `Strategy.construct()` call receives a child RNG via `rng.spawn(1)[0]`
- The `LLMProposer` is inherently non-deterministic; this is documented in run summaries
- Checkpoint saves `rng.bit_generator.state` and restores it on resume

### 4.7 Executor (`harness/executor.py`)

```python
class Executor:
    def __init__(self, timeout_seconds: float = 10.0): ...

    def execute(self, strategy: Strategy, n: int) -> ExecutionResult:
        """Run strategy.construct(n) with timeout and validation."""
        ...
```

**Timeout mechanism:** Uses `concurrent.futures.ThreadPoolExecutor` with `future.result(timeout=...)`. This works cross-platform (Unix and Windows) and can interrupt hung computations by abandoning the future. For truly blocking computations, an optional `concurrent.futures.ProcessPoolExecutor` mode can be used (configurable), which provides hard kill capability at the cost of process startup overhead. The default (thread-based) is sufficient for the educational scope where strategies are pure Python and typically fast.

Catches all exceptions. Validates output matrix shape, symmetry, and diagonal.

### 4.8 Evaluator (`harness/evaluator.py`)

```python
class Evaluator:
    def __init__(self, scorer: RamseyScorer, executor: Executor): ...

    def evaluate(self, strategy: Strategy, n: int) -> EvalResult:
        """Execute strategy and score the result."""
        ...
```

Combines executor + scorer into a single evaluation call. Returns `EvalResult` with full metadata.

**Role clarification:** The `Evaluator` is the public API used by the `eval` CLI subcommand and by `initialize_population()`. The evolution loop in `agent/loop.py` calls executor and scorer directly for finer-grained control (e.g., logging errors separately from scores). Both paths use the same `Executor` and `RamseyScorer` instances.

```python
@dataclass
class EvalResult:
    strategy: Strategy
    graph: np.ndarray | None    # None if execution failed
    score_result: ScoreResult | None
    error: str | None
    elapsed_seconds: float

@dataclass
class ExecutionResult:
    graph: np.ndarray | None    # None if execution failed
    elapsed_seconds: float
    error: str | None           # Exception message if failed
```

### 4.9 Recorder (`harness/recorder.py`)

```python
class Recorder:
    def __init__(self, run_dir: str): ...

    def log_generation(self, gen, strategy, score_result, added): ...
    def log_error(self, gen, error): ...
    def write_summary(self, population): ...
    def save_config(self, config): ...
```

Output files:
- `config.yaml` — frozen config at run start
- `log.jsonl` — one JSON line per generation
- `best.json` — current best strategy serialized
- `summary.txt` — human-readable run summary

### 4.10 Checkpoint (`harness/checkpoint.py`)

```python
def save(population, generation, rng_state, run_dir): ...
def load(run_dir, generation=None) -> tuple[Population, int, rng_state]: ...
```

Saves to `runs/<run_id>/checkpoints/gen_<N>.json`. Loading without a generation number loads the latest checkpoint.

### 4.11 Utilities

**`utils/config.py`:** Loads and validates YAML config files. Provides `load_config(path: str) -> dict` that reads the YAML file, applies defaults for missing optional fields, and validates required fields exist. No Pydantic — plain dict with manual validation.

**`utils/logging.py`:** Configures Python's `logging` module for the project. Provides `setup_logging(level: str, run_dir: str | None)` that sets up:
- Console handler with `INFO` level formatted output
- File handler writing to `runs/<run_id>/run.log` if `run_dir` is provided
- Logger name: `evolveclaw_ramsey`

### 4.12 Run Directory Naming

Run directories are named `<YYYYMMDD>_<HHMMSS>_<config_stem>`, e.g., `20260314_153000_demo`. The `config_stem` is the filename of the config file without extension. This provides chronological ordering and easy identification of which config was used.

---

## 5. CLI Design

```
python -m evolveclaw_ramsey.cli run --config configs/demo.yaml [--resume <run_id>]
python -m evolveclaw_ramsey.cli eval --strategy <strategy_json> --n 17 --s 4 --t 4
python -m evolveclaw_ramsey.cli replay --run-dir runs/<run_id>
```

### `run`
- Loads config from YAML
- If `--resume`, loads checkpoint from specified run directory
- Executes evolution loop
- Prints final summary to stdout

### `eval`
- Evaluates a single strategy (from JSON file or inline JSON)
- Prints score, violation count, and an ASCII adjacency matrix (`.` for 0, `#` for 1) to stdout
- Useful for debugging and testing individual strategies

### `replay`
- Reads `log.jsonl` from a completed run
- Prints generation-by-generation progress
- Shows best-of-run summary

---

## 6. Configuration

### `configs/demo.yaml`

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
  type: random  # "random" or "llm"
  # LLM settings (used when type: llm)
  # llm_provider: anthropic  # or "openai"
  # llm_model: claude-sonnet-4-20250514
  # llm_api_key_env: ANTHROPIC_API_KEY

executor:
  timeout_seconds: 10

logging:
  level: INFO

seed: 42
run_dir: runs/
```

### `configs/llm_demo.yaml`

Same as above but with `proposer.type: llm` and LLM settings uncommented.

---

## 7. Project Structure

```
evolveclaw-ramsey/
  README.md
  pyproject.toml
  requirements.txt
  .gitignore
  configs/
    demo.yaml
    llm_demo.yaml
  research/
    notes.md
    sources.md
  scripts/
    run_demo.sh
    run_search.sh
  evolveclaw_ramsey/
    __init__.py
    __main__.py        # enables `python -m evolveclaw_ramsey`
    cli.py
    agent/
      __init__.py
      loop.py
      proposer.py
      population.py
    harness/
      __init__.py
      executor.py
      evaluator.py
      recorder.py
      checkpoint.py
    ramsey/
      __init__.py
      graph_repr.py
      scoring.py
      strategies.py
    utils/
      __init__.py
      logging.py
      config.py
  tests/
    __init__.py
    test_scoring.py
    test_strategies.py
    test_evaluator.py
    test_loop.py
```

---

## 8. Dependencies

Both `pyproject.toml` (for `pip install -e .`) and `requirements.txt` (for direct `pip install -r`) are provided. The `pyproject.toml` defines a `[project.scripts]` entry `evolveclaw-ramsey = "evolveclaw_ramsey.cli:main"` for optional CLI installation. The package can also be run without installation via `python -m evolveclaw_ramsey.cli` from the project root.

### Required
- `numpy>=1.24` — graph representation and numerical operations
- `pyyaml>=6.0` — configuration parsing

### Optional (for LLM proposer)
- `anthropic>=0.40` — Anthropic Claude API
- `openai>=1.0` — OpenAI API

### Development
- `pytest>=7.0` — testing

---

## 9. Testing Strategy

| Test File | What It Tests |
|-----------|--------------|
| `test_scoring.py` | Clique counting, violation detection, score computation on known graphs |
| `test_strategies.py` | Each strategy type constructs valid adjacency matrices, mutation produces different params |
| `test_evaluator.py` | Evaluator correctly combines executor + scorer, handles errors |
| `test_loop.py` | A minimal 5-generation loop completes without errors, population improves or stays stable |

Tests use small graphs (n ≤ 10) for speed. No LLM calls in tests — only `RandomMutationProposer`.

---

## 10. Harness Engineering Principles

| Principle | Implementation |
|-----------|---------------|
| **Reproducible experiments** | Seeded RNG, frozen config snapshot in run directory |
| **Repeatable evaluation** | Deterministic scorer, isolated executor |
| **Clear module responsibilities** | Agent (search logic) vs. Harness (execution infrastructure) vs. Domain (Ramsey math) |
| **Complete logging** | JSONL per-generation log, human-readable summary |
| **Configuration-driven** | All parameters in YAML, no magic numbers |
| **Failure recovery** | Checkpoint every k generations, `--resume` support |
| **Independent components** | Evaluator, search, and replay can run independently via CLI |

---

## 11. Educational Simplifications vs. AlphaEvolve

| AlphaEvolve Feature | Our Simplification |
|---------------------|-------------------|
| MAP-Elites population with feature dimensions | Simple ranked list with deduplication |
| Island-based multi-population with migration | Single population |
| LLM ensemble (Flash + Pro) | Single optional LLM |
| Asynchronous parallel evaluation pipeline | Synchronous sequential loop |
| Diff-based code mutations | Strategy object mutations (param perturbation + type swapping) |
| Artifact side-channel feedback | Simple error string feedback |
| Multi-language support | Python only |
| Production-scale infrastructure | Single-machine, single-process |

---

## 12. Future Extension Points

- **Stronger LLM proposer**: Use few-shot examples, chain-of-thought, or multi-turn dialogue
- **Larger Ramsey targets**: R(5,5) lower bound exploration (n > 40)
- **Parallel evaluation**: Run multiple candidates concurrently via multiprocessing
- **MAP-Elites population**: Track diversity across feature dimensions (edge density, symmetry, etc.)
- **Smarter scoring**: Use approximate clique counting for larger graphs
- **Benchmark suite**: Multiple Ramsey targets with known best bounds for comparison
- **Visualization**: Plot score progression, graph adjacency matrices, strategy parameter distributions
