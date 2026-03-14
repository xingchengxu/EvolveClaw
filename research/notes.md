# Research Notes

## AlphaEvolve (Google DeepMind, 2025)

**Paper:** "AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery"

### Core Mechanism
- Evolutionary coding agent that uses LLMs to evolve programs (not raw solutions)
- Core loop: Prompt Sampler -> LLM Ensemble -> Evaluator Pool -> Program Database
- LLMs produce code diffs, not whole programs
- MAP-Elites population with island-based evolution for diversity

### Architecture (4 Components)
1. **Prompt Sampler** - selects programs from database, constructs context-rich prompts
2. **LLM Ensemble** - Gemini Flash (throughput) + Gemini Pro (quality)
3. **Evaluator Pool** - runs and scores proposed programs
4. **Program Database** - MAP-Elites + island model to prevent premature convergence

## Ramsey Number Results (arXiv:2603.09172)

Applied AlphaEvolve to Ramsey number lower bounds. Key results:
- R(3,13): 60 -> 61
- R(3,18): 99 -> 100
- R(4,13): 138 -> 139
- R(4,14): 147 -> 148
- R(4,15): 158 -> 159

Key insight: evolved the *search algorithm* (construction heuristic), not individual graphs.

## OpenEvolve

Open-source reimplementation of AlphaEvolve (~4.6k GitHub stars).
- Faithful replication of all 4 core components
- Process worker pattern, double-selection, lazy migration
- Multi-language support (Python, R, Rust)

## Our Simplifications

| AlphaEvolve | EvolveClaw-Ramsey |
|-------------|-------------------|
| MAP-Elites + islands | Simple ranked list |
| LLM ensemble | Single optional LLM |
| Async pipeline | Synchronous loop |
| Diff-based mutations | Strategy object mutations |
| Production infrastructure | Single-process educational |
