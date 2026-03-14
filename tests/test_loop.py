import tempfile
from evolveclaw_ramsey.agent.loop import run_evolution

def test_evolution_loop_minimal():
    config = {
        "problem": {"s": 3, "t": 3, "n": 6, "penalty_weight": 1.0},
        "evolution": {"max_generations": 5, "population_size": 5, "tournament_k": 2, "checkpoint_interval": 5},
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
