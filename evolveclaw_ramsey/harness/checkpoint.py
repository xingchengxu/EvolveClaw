"""Checkpoint save/load for population state and RNG."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def save(population_data: dict, generation: int, rng: np.random.Generator, run_dir: str) -> None:
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng_state = rng.bit_generator.state
    rng_state_serializable = _make_serializable(rng_state)
    data = {"generation": generation, "population": population_data, "rng_state": rng_state_serializable}
    path = ckpt_dir / f"gen_{generation}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load(run_dir: str, generation: int | None = None) -> tuple[dict, int, dict]:
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
    rng = np.random.default_rng()
    state = _restore_state(rng_state_dict)
    rng.bit_generator.state = state
    return rng

def _make_serializable(obj):
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
    if isinstance(obj, dict):
        return {k: _restore_state(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_state(v) for v in obj]
    return obj
