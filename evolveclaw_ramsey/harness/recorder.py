"""Experiment recorder: JSONL log, best.json, summary.txt, config snapshot."""
from __future__ import annotations
import json
from pathlib import Path
import yaml
from evolveclaw_ramsey.ramsey.scoring import ScoreResult
from evolveclaw_ramsey.ramsey.strategies import Strategy

class Recorder:
    def __init__(self, run_dir: str, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.run_dir / "log.jsonl"
        self._best_path = self.run_dir / "best.json"
        self._best_score: float | None = None
        if resume and self._best_path.exists():
            try:
                with open(self._best_path) as f:
                    data = json.load(f)
                self._best_score = data.get("score")
            except (json.JSONDecodeError, KeyError):
                pass

    def save_config(self, config: dict) -> None:
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def log_generation(self, gen: int, strategy: Strategy, score_result: ScoreResult,
                      added: bool, extra: dict | None = None) -> None:
        record = {
            "generation": gen, "strategy_name": strategy.name,
            "strategy_params": strategy.to_dict(), "score": score_result.score,
            "violation_count": score_result.violation_count,
            "s_cliques": score_result.s_cliques, "t_cliques": score_result.t_cliques,
            "added_to_population": added,
        }
        if extra:
            record.update(extra)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self._best_score is None or score_result.score > self._best_score:
            self._best_score = score_result.score
            with open(self._best_path, "w") as f:
                json.dump({"generation": gen, "strategy": strategy.to_dict(),
                          "score": score_result.score, "violation_count": score_result.violation_count}, f, indent=2)

    def log_error(self, gen: int, error: str) -> None:
        record = {"generation": gen, "error": error}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def write_summary(self, best_strategy: Strategy | None, best_score: float, generations: int,
                      llm_stats: dict | None = None) -> None:
        strat_name = best_strategy.name if best_strategy else "none"
        strat_params = best_strategy.to_dict() if best_strategy else {}
        summary = (f"EvolveClaw-Ramsey Run Summary\n{'=' * 40}\n"
                  f"Generations completed: {generations}\nBest score: {best_score}\n"
                  f"Best strategy: {strat_name}\nBest strategy params: {strat_params}\n")
        if llm_stats:
            summary += (f"\nLLM Proposer Stats\n{'-' * 20}\n"
                       f"Total LLM calls: {llm_stats['llm_calls']}\n"
                       f"Successes: {llm_stats['llm_successes']}\n"
                       f"Failures: {llm_stats['llm_failures']}\n")
        with open(self.run_dir / "summary.txt", "w") as f:
            f.write(summary)
