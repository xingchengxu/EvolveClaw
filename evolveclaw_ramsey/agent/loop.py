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
from evolveclaw_ramsey.ramsey.strategies import RandomStrategy, PaleyStrategy, CyclicStrategy, Strategy
from evolveclaw_ramsey.utils.logging import setup_logging, get_logger

@dataclass
class RunResult:
    best_strategy: Strategy
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

def run_evolution(config, resume_dir=None):
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)
    run_dir = resume_dir or _make_run_dir(config.get("run_dir", "runs/"))
    setup_logging(level=config.get("logging", {}).get("level", "INFO"), run_dir=run_dir)
    logger = get_logger()

    scorer = RamseyScorer(s=config["problem"]["s"], t=config["problem"]["t"],
                         penalty_weight=config["problem"].get("penalty_weight", 1.0))
    executor = Executor(config["executor"]["timeout_seconds"])
    evaluator = Evaluator(scorer=scorer, executor=executor)
    recorder = Recorder(run_dir)
    recorder.save_config(config)
    proposer = create_proposer(config["proposer"], rng)

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
        parent, parent_score = population.tournament_select(tournament_k, rng)
        candidate = proposer.propose([parent], [parent_score], config["problem"])
        exec_result = executor.execute(candidate, n)
        if exec_result.error:
            recorder.log_error(gen, exec_result.error)
            logger.debug(f"Gen {gen}: execution error: {exec_result.error}")
            continue
        score_result = scorer.score(exec_result.graph)
        added = population.add(candidate, score_result.score)
        recorder.log_generation(gen, candidate, score_result, added)
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
    checkpoint.save(population.to_dict(), gen, rng, run_dir)
    recorder.write_summary(best_strat, best_score, gen + 1)
    logger.info(f"Run complete. Best score: {best_score:.2f}, strategy: {best_strat.name}")
    return RunResult(best_strategy=best_strat, best_score=best_score, run_dir=run_dir, generations_completed=gen + 1)
