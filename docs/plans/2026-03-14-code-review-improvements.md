# Code Review Improvements Design

**Date:** 2026-03-14
**Status:** Approved

## P1 — High Value

### 1. Proposer Unit Tests (`tests/test_proposer.py`)
- Test `RandomMutationProposer.propose()` with parents returns mutated strategy
- Test with no parents returns RandomStrategy
- Test 15% strategy type switching branch with fixed seed
- Test `LLMProposer._parse_response()` for valid JSON and markdown code blocks
- Test `LLMProposer` fallback when provider raises exception

### 2. Loop Test Expansion (`tests/test_loop.py`)
- Zero-violation early termination: R(3,3) n=5 with C5 coloring, verify `generations_completed < max_generations`
- Checkpoint resume: run 5 gens, resume from same dir, verify continuation

### 3. CLI Integration Tests (`tests/test_cli.py`)
- Test `run` subcommand with temp config
- Test `eval` subcommand with strategy JSON, verify stdout
- Test `replay` subcommand with prior run output

## P2 — Medium Value

### 4. Recorder Tests (`tests/test_recorder.py`)
- Verify JSONL log format with required fields
- Verify error log records
- Verify summary.txt generation
- Verify best.json tracks highest score

### 5. Module Docstrings
- Add docstrings to `__init__.py` files that lack them

### 6. LLMProposer API Key Validation
- Validate API key in `AnthropicProvider.__init__()` and `OpenAIProvider.__init__()`

## P3 — Nice to Have

### 7. Convergence Statistics (`evolveclaw_ramsey/harness/stats.py`)
- `RunStats` class tracking per-generation best/mean score, population diversity
- Integrate via `recorder.log_generation()` extra parameter

### 8. Evolution Visualization (`evolveclaw_ramsey/harness/visualize.py`)
- ASCII score curve (always available)
- Optional matplotlib plot (if installed)
- `--plot` flag on CLI `replay` command
