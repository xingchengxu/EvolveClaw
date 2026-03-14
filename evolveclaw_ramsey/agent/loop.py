"""Evolution loop: the core agent that evolves strategies."""
from __future__ import annotations
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from evolveclaw_ramsey.agent.population import Population
from evolveclaw_ramsey.agent.proposer import create_proposer
from evolveclaw_ramsey.harness import checkpoint
from evolveclaw_ramsey.harness.evaluator import Evaluator
from evolveclaw_ramsey.harness.executor import Executor
from evolveclaw_ramsey.harness.recorder import Recorder
from evolveclaw_ramsey.harness.stats import RunStats
from evolveclaw_ramsey.ramsey.scoring import RamseyScorer
from evolveclaw_ramsey.ramsey.strategies import RandomStrategy, PaleyStrategy, CyclicStrategy, Strategy
from evolveclaw_ramsey.utils.logging import setup_logging, get_logger

@dataclass
class RunResult:
    best_strategy: Strategy | None
    best_score: float
    run_dir: str
    generations_completed: int

def _make_run_dir(base: str, config_stem: str = "run") -> str:
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{now}_{config_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)

def initialize_population(population, config, evaluator, rng):
    n = config["problem"]["n"]
    pop_size = config["evolution"]["population_size"]
    initial_strategies = []
    initial_strategies.append(RandomStrategy(edge_prob=0.5, rng=rng))
    initial_strategies.append(PaleyStrategy(rng=rng))
    num_offsets = max(1, n // 4)
    offsets = [int(x) for x in rng.integers(1, max(2, n // 2), size=num_offsets)]
    initial_strategies.append(CyclicStrategy(offsets=offsets, rng=rng))
    remaining = max(0, pop_size - len(initial_strategies))
    if remaining > 0:
        probs = np.linspace(0.2, 0.8, remaining)
        for p in probs:
            initial_strategies.append(RandomStrategy(edge_prob=float(p), rng=rng))
    for strategy in initial_strategies:
        result = evaluator.evaluate(strategy, n)
        if result.error is None and result.score_result is not None:
            population.add(strategy, result.score_result.score)

def run_evolution(config, resume_dir=None, config_stem: str = "run"):
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)
    run_dir = resume_dir or _make_run_dir(config.get("run_dir", "runs/"), config_stem)
    setup_logging(level=config.get("logging", {}).get("level", "INFO"), run_dir=run_dir)
    logger = get_logger()

    scorer = RamseyScorer(s=config["problem"]["s"], t=config["problem"]["t"],
                         penalty_weight=config["problem"].get("penalty_weight", 1.0))
    executor = Executor(config["executor"]["timeout_seconds"])
    evaluator = Evaluator(scorer=scorer, executor=executor)

    start_gen = 0
    population = Population(config["evolution"]["population_size"])

    if resume_dir:
        try:
            pop_data, start_gen, rng_state = checkpoint.load(resume_dir)
            rng = checkpoint.restore_rng(rng_state)
            population = Population.from_dict(pop_data, rng)
            start_gen += 1
            logger.info(f"Resumed from generation {start_gen}")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load checkpoint, starting fresh: {e}")

    # Create proposer AFTER rng is potentially restored from checkpoint
    proposer = create_proposer(config["proposer"], rng)

    # Create recorder AFTER resume so it can pick up existing best score
    recorder = Recorder(run_dir, resume=resume_dir is not None)
    if not resume_dir:
        recorder.save_config(config)

    if population.size() == 0:
        logger.info("Initializing population...")
        initialize_population(population, config, evaluator, rng)
        logger.info(f"Population initialized with {population.size()} members")

    if population.size() == 0:
        logger.error("Population is empty after initialization — all candidates failed. Aborting.")
        checkpoint.save(population.to_dict(), 0, rng, run_dir)
        recorder.write_summary(None, 0.0, 0)
        return RunResult(best_strategy=None, best_score=0.0, run_dir=run_dir, generations_completed=0)

    best_strat, best_score = population.best()
    logger.info(f"Initial best: score={best_score:.2f}, strategy={best_strat.name}")

    n = config["problem"]["n"]
    max_gen = config["evolution"]["max_generations"]
    tournament_k = config["evolution"]["tournament_k"]
    ckpt_interval = config["evolution"]["checkpoint_interval"]

    gen = start_gen
    last_gen_run = -1  # tracks the last generation that entered the loop
    last_error = None
    had_success = False  # tracks whether any candidate was successfully scored
    run_stats = RunStats()
    for gen in range(start_gen, max_gen):
        last_gen_run = gen
        parent, parent_score = population.tournament_select(tournament_k, rng)
        candidate = proposer.propose([parent], [parent_score], config["problem"], last_error=last_error)
        exec_result = executor.execute(candidate, n)
        if exec_result.error:
            last_error = exec_result.error
            recorder.log_error(gen, exec_result.error)
            logger.debug(f"Gen {gen}: execution error: {exec_result.error}")
            continue
        last_error = None
        had_success = True
        score_result = scorer.score(exec_result.graph)
        added = population.add(candidate, score_result.score)
        gen_stats = run_stats.record(gen, population.scores(), population.type_counts())
        recorder.log_generation(gen, candidate, score_result, added, extra=run_stats.to_dict())
        current_best_strat, current_best_score = population.best()
        if score_result.score > best_score:
            best_score = score_result.score
            best_strat = current_best_strat
            logger.info(f"Gen {gen}: NEW BEST score={best_score:.2f} violations={score_result.violation_count} strategy={candidate.name}")
        elif gen % 10 == 0:
            logger.info(f"Gen {gen}: score={score_result.score:.2f} best={best_score:.2f}")
        if gen > 0 and gen % ckpt_interval == 0:
            checkpoint.save(population.to_dict(), gen, rng, run_dir)
        if score_result.violation_count == 0:
            logger.info(f"Gen {gen}: PERFECT SCORE! No violations found.")
            break

    best_strat, best_score = population.best()
    if had_success:
        checkpoint.save(population.to_dict(), gen, rng, run_dir)
    generations_completed = (last_gen_run + 1) if last_gen_run >= 0 else start_gen
    recorder.write_summary(best_strat, best_score, generations_completed)
    logger.info(f"Run complete. Best score: {best_score:.2f}, strategy: {best_strat.name}")
    return RunResult(best_strategy=best_strat, best_score=best_score, run_dir=run_dir, generations_completed=generations_completed)
