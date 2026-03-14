# EvolveClaw-Ramsey

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

A minimal, educational implementation of AlphaEvolve-style evolutionary search applied to Ramsey number lower bounds.

## Overview

**EvolveClaw-Ramsey** is a single-process Python system that uses evolutionary strategies to search for Ramsey-number counter-example colorings. It draws direct inspiration from Google DeepMind's [AlphaEvolve](https://arxiv.org/abs/2506.13131) paper (2025), which demonstrated that evolving *programs* (search heuristics) rather than raw solutions can push the boundaries of combinatorial mathematics.

The name combines **Evolve** (the evolutionary search core from AlphaEvolve) with **Claw** (a nod to the OpenClaw agent ecosystem), reflecting the project's dual heritage: AlphaEvolve's evolutionary methodology and modern AI-agent design patterns.

This project is deliberately small and transparent. It is designed for learning, not for breaking records.

## Core Ideas

### AlphaEvolve's Insight

AlphaEvolve showed that LLMs can act as mutation operators in an evolutionary loop: propose code changes, evaluate them automatically, and keep the best. The key is evolving the *search algorithm itself*, not individual candidate solutions. Applied to Ramsey theory, this approach improved lower bounds on several Ramsey numbers (arXiv:2603.09172).

### Our Minimal Implementation

EvolveClaw-Ramsey distills this into its simplest useful form:

- **Strategy objects** replace AlphaEvolve's evolved programs. Each strategy is a callable that decides how to color edges of a complete graph K_n.
- **Mutation operators** (random parameter perturbation, strategy type switching, edge flipping via PerturbedStrategy) stand in for LLM-proposed code diffs.
- **A synchronous evolution loop** replaces AlphaEvolve's async pipeline with a straightforward generate-evaluate-select cycle.
- **Violation-counting scorer** provides the fitness signal: fewer monochromatic s-cliques and t-cliques means a better coloring.

### Design Pattern Influences

Several open-source AI agent projects informed the engineering approach:

- **OpenClaw** -- personal AI assistant with gateway architecture and multi-platform skill system; its modular package design influenced the clean separation between `ramsey/`, `agent/`, and `harness/` packages.
- **OpenCode** -- open-source terminal coding agent with provider-agnostic design; its `LLMProvider` abstraction pattern is adopted in our `LLMProvider` ABC, making it trivial to add new LLM backends. Its executor patterns also informed the executor/evaluator split in the harness layer.
- **OpenEvolve** -- faithful AlphaEvolve reimplementation; its error artifact side-channel pattern (feeding execution failures back into prompts) is adopted in our proposer's `last_error` feedback mechanism.
- **nanobot** -- ultra-lightweight OpenClaw delivering core agent functionality in ~4000 lines of code; validated the philosophy of keeping this project small and readable.
- **A3S-Code** -- listed as a reference per project requirements.

### Harness Engineering

The harness layer (`evolveclaw_ramsey/harness/`) wraps the core search with operational concerns:

- **Executor** -- runs strategy objects with timeouts and error isolation.
- **Evaluator** -- scores candidates and ranks them.
- **Recorder** -- logs generation-by-generation results for analysis.
- **Checkpoint** -- saves and restores population state for resumable runs.

## References and Inspirations

| Project | What We Borrowed |
|---------|-----------------|
| [AlphaEvolve](https://arxiv.org/abs/2506.13131) | Core idea: evolutionary loop with LLM-as-mutator for combinatorial search |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | Population management, checkpoint design, error artifact feedback pattern; validated our simplifications |
| [google-research/ramsey](https://github.com/google-research/google-research/tree/master/ramsey_number_bounds) | Ramsey-specific evaluation and benchmark data |
| [OpenClaw](https://github.com/openclaw/openclaw) | Gateway/dispatcher architecture; inspired modular `ramsey/`, `agent/`, `harness/` package separation |
| [OpenCode](https://github.com/anomalyco/opencode) | Provider-agnostic LLM abstraction (`LLMProvider` ABC); informed executor/evaluator split and CLI patterns |
| [nanobot](https://github.com/HKUDS/nanobot) | Ultra-minimal agent philosophy (~4000 lines); validated "minimal yet functional" educational approach |
| [A3S-Code](https://github.com/A3S-Lab/Code) | Listed per project requirements |

## Project Boundaries

### Educational Simplifications

| AlphaEvolve (Production) | EvolveClaw-Ramsey (Educational) |
|--------------------------|-------------------------------|
| MAP-Elites + island model | Simple ranked population list |
| LLM ensemble (Flash + Pro) | Single optional LLM proposer |
| Asynchronous pipeline | Synchronous loop |
| Diff-based code mutations | Strategy object mutations |
| Distributed evaluation | Single-process execution |
| Production infrastructure | Educational single-file runs |

### What Is NOT Included

- No distributed computation or multi-node support.
- No production-grade LLM integration (the optional LLM proposer is a proof-of-concept).
- No MAP-Elites or island-based diversity maintenance.
- No claim of reproducing AlphaEvolve's Ramsey results. The known Ramsey bounds require far more compute and sophistication than this project provides.
- No GPU acceleration. All graph operations use NumPy on CPU.

### Honest Limitations

This project will not discover new Ramsey number bounds. It is a teaching tool that demonstrates the *shape* of AlphaEvolve's approach in a form that fits in a single Python package. The evolutionary loop works, the scoring is correct, but the search space for meaningful Ramsey problems is vast and the mutations here are simple.

## Quick Start

### Install

```bash
# Install core + dev dependencies (pytest, etc.)
pip install -e ".[dev]"
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

To use the LLM proposer (optional), also install the LLM dependencies and set your API key:

```bash
pip install -e ".[llm]"
```

Set your API key for the provider you want to use:

```bash
# Linux / macOS (bash/zsh):
export ANTHROPIC_API_KEY=your-key-here   # For Anthropic (default)
export OPENAI_API_KEY=your-key-here      # For OpenAI

# Windows (PowerShell):
$env:ANTHROPIC_API_KEY='your-key-here'
$env:OPENAI_API_KEY='your-key-here'
```

Without the LLM extra, configuring `type: llm` will fail at startup with a clear error message.

### Run Tests

```bash
python -m pytest -q
```

### Run the Demo

```bash
python -m evolveclaw_ramsey.cli run --config configs/demo.yaml
```

Or use the shell script:

```bash
bash scripts/run_demo.sh
```

### View Results

Results are written to the `runs/` directory. Each run produces:
- A log of generation-by-generation progress.
- Checkpoint files for resuming interrupted searches.
- The best coloring found and its violation count.

## Repository Structure

```
evolveclaw_ramsey/
  __init__.py
  __main__.py
  cli.py                    # CLI entry point
  ramsey/
    __init__.py
    graph_repr.py            # Adjacency matrix coloring representation
    scoring.py               # Monochromatic clique violation counter
    strategies.py            # Coloring strategies (Random, Paley, Cyclic, Perturbed)
  agent/
    __init__.py
    population.py            # Ranked population management
    proposer.py              # Mutation proposers (random, optional LLM)
    loop.py                  # Main evolution loop
  harness/
    __init__.py
    executor.py              # Strategy execution with timeout
    evaluator.py             # Candidate evaluation and ranking
    recorder.py              # Run logging and metrics
    checkpoint.py            # Save/restore population state
  utils/
    __init__.py
    config.py                # YAML config loading
    logging.py               # Logging setup
configs/
  demo.yaml                  # Quick demo configuration
  llm_demo.yaml              # LLM proposer configuration
scripts/
  run_demo.sh                # Quick demo launcher
  run_search.sh              # Full search launcher
research/
  notes.md                   # Research notes on AlphaEvolve and Ramsey theory
  sources.md                 # Reference links and bibliography
tests/
  test_graph_repr.py
  test_scoring.py
  test_strategies.py
  test_evaluator.py
  test_loop.py
```

## Technical Details

### Candidate Representation

Each candidate is a **strategy object** that produces a 2-coloring of the edges of the complete graph K_n. The coloring is stored as an n-by-n symmetric adjacency matrix with values in {0, 1}, where 0 and 1 represent the two colors (conventionally "red" and "blue").

### Scoring Logic

The scorer counts **monochromatic cliques**: for a target R(s, t), it counts the number of monochromatic s-cliques in color 0 and t-cliques in color 1. A coloring with zero violations is a valid counter-example proving R(s, t) > n.

The fitness function is:

```
score = n - violations * penalty_weight
```

where `violations = count_cliques(G, s) + count_cliques(complement(G), t)`. Higher scores are better; a perfect score of `n` (the graph size) means zero violations and a valid Ramsey counter-example was found.

### Evolution Loop

1. **Initialize** a population of random strategy objects.
2. **Evaluate** each candidate by executing its strategy and scoring the resulting coloring.
3. **Select** parents via tournament selection (pick `k` candidates, take the best).
4. **Mutate** selected parents to produce offspring (parameter perturbation, strategy type switching, or edge flipping).
5. **Replace** the worst members of the population with better offspring.
6. **Repeat** for `max_generations` or until a zero-violation coloring is found.

### Harness Layers

The harness wraps the core loop with operational infrastructure:

- **Executor** runs each strategy with a configurable timeout, catching errors and infinite loops.
- **Evaluator** takes executor output and computes fitness scores.
- **Recorder** writes per-generation statistics (best score, mean score, diversity metrics).
- **Checkpoint** serializes population state to disk at configurable intervals for crash recovery.

## Configuration

Configuration is via YAML files. See `configs/demo.yaml` for the default:

```yaml
problem:
  s: 4              # First Ramsey parameter
  t: 4              # Second Ramsey parameter
  n: 17             # Graph size (searching for R(s,t) > n)
  penalty_weight: 1.0

evolution:
  max_generations: 100
  population_size: 20
  tournament_k: 3
  checkpoint_interval: 10

proposer:
  type: random       # or "llm" with llm_demo.yaml

executor:
  timeout_seconds: 10

logging:
  level: INFO

seed: 42
run_dir: runs/
```

Key parameters:
- `s`, `t`: The Ramsey parameters. R(4,4) = 18, so n=17 should have valid counter-examples.
- `n`: The graph size. We are looking for a 2-coloring of K_n with no monochromatic s-clique or t-clique.
- `population_size`: Number of candidates maintained each generation.
- `tournament_k`: Number of candidates sampled for tournament selection.
- `checkpoint_interval`: Save population state every N generations.

## Future Extensions

- **LLM Proposer** -- Use an LLM (Claude, GPT-4) to propose strategy mutations, moving closer to AlphaEvolve's core mechanism.
- **Parallel Evaluation** -- Run candidate evaluations in parallel using multiprocessing.
- **MAP-Elites Population** -- Replace the ranked list with a MAP-Elites grid for better diversity maintenance.
- **Benchmark Suite** -- Systematic evaluation across R(3,k) and R(4,k) for various n values.
- **Visualization** -- Graph coloring visualizations and evolution progress plots.
- **Island Model** -- Multiple sub-populations with periodic migration to reduce premature convergence.

## License

This project is for educational and research purposes.
