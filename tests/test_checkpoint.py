"""Tests for checkpoint save/load round-trip."""
import json
from pathlib import Path
import numpy as np
import pytest
from evolveclaw_ramsey.harness.checkpoint import save, load, restore_rng


def test_save_load_roundtrip(tmp_path):
    """Checkpoint save then load returns identical population data and generation."""
    rng = np.random.default_rng(123)
    population_data = {
        "max_size": 5,
        "members": [
            {"strategy": {"name": "random", "edge_prob": 0.5}, "score": 10.0},
            {"strategy": {"name": "cyclic", "offsets": [1, 3]}, "score": 8.0},
        ],
    }
    save(population_data, generation=7, rng=rng, run_dir=str(tmp_path))
    loaded_pop, loaded_gen, loaded_rng_state, loaded_extra = load(str(tmp_path), generation=7)
    assert loaded_gen == 7
    assert loaded_pop["max_size"] == 5
    assert len(loaded_pop["members"]) == 2
    assert loaded_pop["members"][0]["score"] == 10.0
    assert loaded_pop["members"][1]["strategy"]["name"] == "cyclic"


def test_load_latest(tmp_path):
    """Loading without specifying generation returns the latest checkpoint."""
    rng = np.random.default_rng(42)
    pop = {"max_size": 3, "members": []}
    save(pop, generation=5, rng=rng, run_dir=str(tmp_path))
    save(pop, generation=10, rng=rng, run_dir=str(tmp_path))
    _, loaded_gen, _, _ = load(str(tmp_path))
    assert loaded_gen == 10


def test_rng_restore_produces_same_sequence(tmp_path):
    """Restored RNG produces the same random sequence as the original."""
    rng = np.random.default_rng(99)
    # Advance RNG a bit
    rng.random(100)
    # Save state
    pop = {"max_size": 1, "members": []}
    save(pop, generation=0, rng=rng, run_dir=str(tmp_path))
    # Generate sequence from current rng
    seq_original = rng.random(10).tolist()
    # Restore and generate same sequence
    _, _, rng_state, _ = load(str(tmp_path), generation=0)
    restored_rng = restore_rng(rng_state)
    seq_restored = restored_rng.random(10).tolist()
    assert seq_original == seq_restored


def test_load_missing_dir_raises():
    """Loading from nonexistent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load("/nonexistent/path/that/does/not/exist")
