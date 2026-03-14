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

def cmd_run(args):
    config = load_config(args.config)
    resume_dir = args.resume if args.resume else None
    config_stem = Path(args.config).stem
    result = run_evolution(config, resume_dir=resume_dir, config_stem=config_stem)
    print(f"\nRun complete: {result.run_dir}")
    print(f"Best score: {result.best_score:.2f}")
    print(f"Best strategy: {result.best_strategy.name}")
    print(f"Generations: {result.generations_completed}")

def cmd_eval(args):
    with open(args.strategy) as f:
        strategy_dict = json.load(f)
    # Handle best.json format (has "strategy" key) or direct strategy dict
    if "strategy" in strategy_dict and "name" in strategy_dict["strategy"]:
        strategy_dict = strategy_dict["strategy"]
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
    print(f"\nAdjacency matrix ({args.n}x{args.n}):")
    for i in range(args.n):
        row = ""
        for j in range(args.n):
            row += "#" if result.graph[i, j] == 1 else "."
        print(f"  {row}")

def cmd_replay(args):
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
                print(f"Gen {gen:4d}: score={score:8.2f} violations={violations:3d} strategy={name:10s}{marker}")
    print("=" * 50)
    print(f"Generations: {gen_count}, Errors: {error_count}")
    print(f"Best score: {best_score:.2f}")
    summary_path = run_dir / "summary.txt"
    if summary_path.exists():
        print(f"\n{summary_path.read_text()}")

def main():
    parser = argparse.ArgumentParser(prog="evolveclaw-ramsey",
                                    description="Minimal educational AlphaEvolve-style Ramsey optimization")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the evolution loop")
    run_parser.add_argument("--config", required=True, help="Path to YAML config file")
    run_parser.add_argument("--resume", default=None, help="Run directory to resume from")
    eval_parser = subparsers.add_parser("eval", help="Evaluate a single strategy")
    eval_parser.add_argument("--strategy", required=True, help="Path to strategy JSON file")
    eval_parser.add_argument("--n", type=int, required=True, help="Number of vertices")
    eval_parser.add_argument("--s", type=int, required=True, help="Clique size s")
    eval_parser.add_argument("--t", type=int, required=True, help="Clique size t")
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
