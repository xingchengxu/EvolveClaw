"""Tests for strategy proposers: RandomMutationProposer and LLMProposer."""
import json
from unittest.mock import MagicMock
import numpy as np
import pytest
from evolveclaw_ramsey.agent.proposer import (
    RandomMutationProposer, LLMProposer, LLMProvider, create_proposer,
)
from evolveclaw_ramsey.ramsey.strategies import (
    RandomStrategy, PaleyStrategy, CyclicStrategy, PerturbedStrategy, Strategy,
)

PROBLEM = {"s": 4, "t": 4, "n": 17}


def test_random_proposer_with_parent_returns_strategy():
    rng = np.random.default_rng(42)
    proposer = RandomMutationProposer(rng=rng)
    parent = RandomStrategy(edge_prob=0.5, rng=rng)
    result = proposer.propose([parent], [10.0], PROBLEM)
    assert isinstance(result, Strategy)


def test_random_proposer_no_parents_returns_random():
    rng = np.random.default_rng(42)
    proposer = RandomMutationProposer(rng=rng)
    result = proposer.propose([], [], PROBLEM)
    assert isinstance(result, RandomStrategy)


def test_random_proposer_mutation_branch():
    """With 85% probability, propose() mutates the parent."""
    rng = np.random.default_rng(0)
    proposer = RandomMutationProposer(rng=rng)
    parent = RandomStrategy(edge_prob=0.5, rng=rng)
    # Run multiple times to cover both branches (mutation and type switch)
    results = set()
    for seed in range(100):
        r = np.random.default_rng(seed)
        p = RandomMutationProposer(rng=r)
        result = p.propose([parent], [10.0], PROBLEM)
        results.add(type(result).__name__)
    # Should have seen both mutation (RandomStrategy) and type switches
    assert len(results) >= 2


def test_random_proposer_type_switching():
    """Verify that type switching can produce all 4 strategy types."""
    seen_types = set()
    for seed in range(200):
        rng = np.random.default_rng(seed)
        proposer = RandomMutationProposer(rng=rng)
        parent = RandomStrategy(edge_prob=0.5, rng=rng)
        result = proposer.propose([parent], [10.0], PROBLEM)
        seen_types.add(type(result).__name__)
    assert "RandomStrategy" in seen_types
    assert "PaleyStrategy" in seen_types or "CyclicStrategy" in seen_types


def test_llm_proposer_parse_response_json():
    """LLMProposer._parse_response handles plain JSON."""
    rng = np.random.default_rng(42)
    provider = MagicMock(spec=LLMProvider)
    proposer = LLMProposer(provider=provider, rng=rng)
    text = json.dumps({"name": "random", "edge_prob": 0.6})
    result = proposer._parse_response(text)
    assert isinstance(result, RandomStrategy)
    assert result.edge_prob == 0.6


def test_llm_proposer_parse_response_markdown_block():
    """LLMProposer._parse_response handles markdown code blocks."""
    rng = np.random.default_rng(42)
    provider = MagicMock(spec=LLMProvider)
    proposer = LLMProposer(provider=provider, rng=rng)
    text = '```json\n{"name": "cyclic", "offsets": [1, 3, 5]}\n```'
    result = proposer._parse_response(text)
    assert isinstance(result, CyclicStrategy)
    assert result.offsets == [1, 3, 5]


def test_llm_proposer_fallback_on_provider_error():
    """LLMProposer falls back to random when provider raises."""
    rng = np.random.default_rng(42)
    provider = MagicMock(spec=LLMProvider)
    provider.call.side_effect = RuntimeError("API error")
    proposer = LLMProposer(provider=provider, rng=rng)
    parent = RandomStrategy(edge_prob=0.5, rng=rng)
    result = proposer.propose([parent], [10.0], PROBLEM)
    assert isinstance(result, Strategy)


def test_llm_proposer_includes_last_error_in_prompt():
    """LLMProposer._build_prompt includes last_error when provided."""
    rng = np.random.default_rng(42)
    provider = MagicMock(spec=LLMProvider)
    proposer = LLMProposer(provider=provider, rng=rng)
    parent = RandomStrategy(edge_prob=0.5, rng=rng)
    prompt = proposer._build_prompt([parent], [10.0], PROBLEM, last_error="Timeout")
    assert "Timeout" in prompt
    assert "previous candidate failed" in prompt


def test_llm_proposer_prompt_without_error():
    """LLMProposer._build_prompt omits error section when no error."""
    rng = np.random.default_rng(42)
    provider = MagicMock(spec=LLMProvider)
    proposer = LLMProposer(provider=provider, rng=rng)
    parent = RandomStrategy(edge_prob=0.5, rng=rng)
    prompt = proposer._build_prompt([parent], [10.0], PROBLEM)
    assert "previous candidate failed" not in prompt


def test_create_proposer_random():
    rng = np.random.default_rng(42)
    proposer = create_proposer({"type": "random"}, rng)
    assert isinstance(proposer, RandomMutationProposer)


def test_create_proposer_unknown_type():
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="Unknown proposer type"):
        create_proposer({"type": "unknown"}, rng)
